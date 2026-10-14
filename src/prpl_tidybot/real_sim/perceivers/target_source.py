"""Target-pose providers for perceivers.

A `TargetSource` answers `get_target()` with a `(x, y, z)` tuple in the env
world frame. Two implementations:

- `ConstantTargetSource` for fake / sim modes, where the target is a hard-coded
  pose threaded in from the Hydra config.
- `MarkerDetectorTargetSource` for real mode, which subscribes to the
  marker-detector publisher and reads the latest position of a specific ArUco
  marker (the `(x, y)` part) at a configured `z`.

A `YawTargetSource` additionally answers `get_target_and_yaw()` with the
target's rotation about z. `ConstantTargetSource` is one (yaw defaults to 0);
`MarkerDetectorCubeTargetSource` reads a cube that carries a different marker
on each face, taking position and yaw from whichever face the detector sees.

Splitting this out lets `BaseMotion3DPerceiver` stay frame-agnostic about where
the target came from — Hydra wires the right source per pipeline.

Perceivers that track several objects use a `CylinderTargets`: one call returns
the `(x, y, z)` of every cylinder, in the order of the cylinder specs the
pipeline was configured with (see `CylinderSpec`). `ConstantCylinderTargets`
reads each cylinder's fake pose from its spec; `MarkerDetectorCylinderTargets`
reads each cylinder's marker from the detector, sharing one client.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prpl_tidybot.marker_detector.client import MarkerDetectorClient
from prpl_tidybot.marker_detector.constants import CUBE_MARKER_IDS, TARGET_MARKER_IDS


@dataclass(frozen=True)
class CylinderSpec:
    """One physical cylinder as the pipeline config describes it.

    ``marker_id`` is the ArUco marker taped to the floor under it (real mode),
    ``fake_xy`` where fake mode pretends it stands. ``radius`` and ``height``
    are in metres; the standing cylinder's centre is at ``height / 2``.
    """

    radius: float
    height: float
    marker_id: int | None = None
    fake_xy: tuple[float, float] | None = None

    @property
    def center_z(self) -> float:
        """Height of the standing cylinder's centre above the floor."""
        return 0.5 * self.height

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any] | CylinderSpec) -> CylinderSpec:
        """Build from a config mapping (Hydra hands the yaml list through as dicts)."""
        if isinstance(spec, CylinderSpec):
            return spec
        fake_xy = spec.get("fake_xy")
        return cls(
            radius=float(spec["radius"]),
            height=float(spec["height"]),
            marker_id=None if spec.get("marker_id") is None else int(spec["marker_id"]),
            fake_xy=None if fake_xy is None else (float(fake_xy[0]), float(fake_xy[1])),
        )


def parse_cylinder_specs(
    specs: Sequence[Mapping[str, Any] | CylinderSpec],
) -> list[CylinderSpec]:
    """Normalise a config list of cylinders into `CylinderSpec`s."""
    if not specs:
        raise ValueError("At least one cylinder spec is required.")
    return [CylinderSpec.from_mapping(spec) for spec in specs]


class TargetSource(abc.ABC):
    """Provider of `(x, y, z)` task target positions in the env world frame."""

    @abc.abstractmethod
    def get_target(self) -> tuple[float, float, float]:
        """Return the latest target position."""

    def close(self) -> None:
        """Release any resources held by the source (e.g. detector sockets)."""


class YawTargetSource(TargetSource):
    """A `TargetSource` that also knows the target's rotation about z."""

    @abc.abstractmethod
    def get_target_and_yaw(self) -> tuple[float, float, float, float]:
        """Return the latest `(x, y, z, yaw)`, yaw in radians in the world frame."""

    def get_target(self) -> tuple[float, float, float]:
        x, y, z, _ = self.get_target_and_yaw()
        return (x, y, z)


class ConstantTargetSource(YawTargetSource):
    """Returns a fixed `(x, y, z)` (and `yaw`, default 0) every call.

    Used by fake / sim pipelines.
    """

    def __init__(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        self._target = (float(x), float(y), float(z), float(yaw))

    def get_target_and_yaw(self) -> tuple[float, float, float, float]:
        return self._target


class MarkerDetectorTargetSource(TargetSource):
    """Reads the latest position of an ArUco marker from `MarkerDetectorServer`.

    The marker detector publishes target marker positions as
    `{"targets": {aruco_id: (x, y)}}` (see `MarkerDetectorServer.get_data`).
    This source pairs that `(x, y)` with the constructor-supplied `target_z`.
    When the marker isn't in the latest payload (e.g. briefly occluded) the
    last successful detection is returned; if no detection has ever arrived,
    `get_target` raises so the rollout doesn't run on bogus data.
    """

    def __init__(
        self,
        marker_id: int,
        target_z: float,
        client: MarkerDetectorClient | None = None,
    ) -> None:
        self._marker_id = int(marker_id)
        self._target_z = float(target_z)
        self._client = client if client is not None else MarkerDetectorClient()
        self._last: tuple[float, float, float] | None = None

    def get_target(self) -> tuple[float, float, float]:
        payload = self._client.get_latest()
        targets = payload.get("targets") or {}
        if self._marker_id in targets:
            x, y = targets[self._marker_id]
            self._last = (float(x), float(y), self._target_z)
        if self._last is None:
            raise RuntimeError(
                f"MarkerDetectorTargetSource: marker id {self._marker_id} has "
                "never been reported by the detector"
            )
        return self._last

    def close(self) -> None:
        self._client.close()


class MarkerDetectorCubeTargetSource(YawTargetSource):
    """Reads a cube that carries a different ArUco marker on each face.

    Whichever configured face id the detector reports gives the cube's `(x, y)` (from
    `targets`) and yaw (from `target_yaws`); when several are reported, the first in
    `marker_ids` order wins. `yaw_offsets[id]` is the angle of that face's marker +x
    axis from the cube's x axis, subtracted from the measured yaw (a marker printed
    square to its face needs no offset for a cube, whose yaw only matters modulo 90
    degrees). As in `MarkerDetectorTargetSource`, a cube missing from the latest payload
    keeps its last detection and one never seen raises.
    """

    def __init__(
        self,
        target_z: float,
        marker_ids: Sequence[int] = CUBE_MARKER_IDS,
        yaw_offsets: Mapping[int, float] | None = None,
        client: MarkerDetectorClient | None = None,
    ) -> None:
        self._marker_ids = [int(marker_id) for marker_id in marker_ids]
        if not self._marker_ids:
            raise ValueError("A cube needs at least one face marker id.")
        unknown = sorted(set(self._marker_ids) - set(TARGET_MARKER_IDS))
        if unknown:
            raise ValueError(
                f"Cube marker id(s) {unknown} are not in TARGET_MARKER_IDS, so the "
                "detector would never report them."
            )
        self._target_z = float(target_z)
        self._yaw_offsets = {
            int(marker_id): float(offset)
            for marker_id, offset in (yaw_offsets or {}).items()
        }
        self._client = client if client is not None else MarkerDetectorClient()
        self._last: tuple[float, float, float, float] | None = None

    def get_target_and_yaw(self) -> tuple[float, float, float, float]:
        payload = self._client.get_latest()
        targets = payload.get("targets") or {}
        visible = [marker_id for marker_id in self._marker_ids if marker_id in targets]
        if visible:
            if "target_yaws" not in payload:
                raise RuntimeError(
                    "MarkerDetectorCubeTargetSource: the marker detector does not "
                    "publish target_yaws; update the detector on the perception host."
                )
            marker_id = visible[0]
            x, y = targets[marker_id]
            yaw = payload["target_yaws"][marker_id] - self._yaw_offsets.get(
                marker_id, 0.0
            )
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            self._last = (float(x), float(y), self._target_z, float(yaw))
        if self._last is None:
            raise RuntimeError(
                f"MarkerDetectorCubeTargetSource: the cube's marker ids "
                f"{self._marker_ids} have never been reported by the detector"
            )
        return self._last

    def close(self) -> None:
        self._client.close()


class CylinderTargets(abc.ABC):
    """Provider of every cylinder's `(x, y, z)` position, in spec order."""

    @abc.abstractmethod
    def get_targets(self) -> list[tuple[float, float, float]]:
        """Return the latest position of each cylinder."""

    def close(self) -> None:
        """Release any resources held by the source."""


class ConstantCylinderTargets(CylinderTargets):
    """Each cylinder stands at its spec's ``fake_xy`` with its centre at ``height / 2``.
    Used by fake / sim pipelines."""

    def __init__(self, cylinders: Sequence[Mapping[str, Any] | CylinderSpec]) -> None:
        self._targets: list[tuple[float, float, float]] = []
        for index, spec in enumerate(parse_cylinder_specs(cylinders)):
            if spec.fake_xy is None:
                raise ValueError(f"Cylinder {index} has no fake_xy for fake mode.")
            self._targets.append((spec.fake_xy[0], spec.fake_xy[1], spec.center_z))

    def get_targets(self) -> list[tuple[float, float, float]]:
        return list(self._targets)


class MarkerDetectorCylinderTargets(CylinderTargets):
    """Each cylinder's `(x, y)` comes from its spec's marker in one shared read of the
    marker detector, with z at the standing cylinder's centre.

    A marker missing from the latest payload keeps its last detection; a marker
    that has never been seen raises, so a rollout does not run on bogus data.
    """

    def __init__(
        self,
        cylinders: Sequence[Mapping[str, Any] | CylinderSpec],
        client: MarkerDetectorClient | None = None,
    ) -> None:
        self._client = client if client is not None else MarkerDetectorClient()
        self._specs = parse_cylinder_specs(cylinders)
        for index, spec in enumerate(self._specs):
            if spec.marker_id is None:
                raise ValueError(f"Cylinder {index} has no marker_id for real mode.")
        self._last: list[tuple[float, float, float] | None] = [None] * len(self._specs)

    def get_targets(self) -> list[tuple[float, float, float]]:
        payload = self._client.get_latest()
        detected = payload.get("targets") or {}
        for index, spec in enumerate(self._specs):
            if spec.marker_id in detected:
                x, y = detected[spec.marker_id]
                self._last[index] = (float(x), float(y), spec.center_z)
        missing = [
            spec.marker_id
            for spec, last in zip(self._specs, self._last)
            if last is None
        ]
        if missing:
            raise RuntimeError(
                f"MarkerDetectorCylinderTargets: marker id(s) {missing} have never "
                "been reported by the detector"
            )
        return [last for last in self._last if last is not None]

    def close(self) -> None:
        self._client.close()
