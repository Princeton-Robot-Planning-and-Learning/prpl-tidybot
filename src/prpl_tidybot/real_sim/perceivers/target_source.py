"""Target-pose providers for perceivers that emit a single task target point.

A `TargetSource` answers `get_target()` with a `(x, y, z)` tuple in the env
world frame. Two implementations:

- `ConstantTargetSource` for fake / sim modes, where the target is a hard-coded
  pose threaded in from the Hydra config.
- `MarkerDetectorTargetSource` for real mode, which subscribes to the
  marker-detector publisher and reads the latest position of a specific ArUco
  marker (the `(x, y)` part) at a configured `z`.

Splitting this out lets `BaseMotion3DPerceiver` stay frame-agnostic about where
the target came from — Hydra wires the right source per pipeline.
"""

import abc

from prpl_tidybot.marker_detector.client import MarkerDetectorClient


class TargetSource(abc.ABC):
    """Provider of `(x, y, z)` task target positions in the env world frame."""

    @abc.abstractmethod
    def get_target(self) -> tuple[float, float, float]:
        """Return the latest target position."""

    def close(self) -> None:
        """Release any resources held by the source (e.g. detector sockets)."""


class ConstantTargetSource(TargetSource):
    """Returns a fixed `(x, y, z)` every call. Used by fake / sim pipelines."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self._target = (float(x), float(y), float(z))

    def get_target(self) -> tuple[float, float, float]:
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
