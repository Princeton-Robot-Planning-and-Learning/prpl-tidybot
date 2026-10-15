"""Analytic grasp candidates for a cube, relative to its estimated pose.

The out-of-the-box pick grasps straight down at four fixed yaws with the gripper fully
open (85 mm). Beside a container wall that hand does not fit: the open fingers or the
knuckles behind them hit the wall before the pads reach the cube. The candidates here
widen the search along the two axes that matter for a cube near a wall:

* **Partial opening.** The fingers open only `opening_margins` wider than the cube
  rather than to 85 mm, so a finger between the cube and a wall needs a gap of roughly
  its own thickness plus the margin, not 30 mm.
* **Tilted approaches.** The approach axis is tilted by up to 30 degrees about the
  fingers' closing axis. Rotating about that axis keeps both pads parallel to the faces
  they squeeze, while swinging the wrist, camera and knuckles away from the wall on
  one side. With four face-aligned yaws and both tilt signs, the hand can lean away
  from a wall on any side.

A candidate is a cube-relative transform, so the same grasp can be re-targeted when a
closer look refines the cube's pose (the visual servo). Feasibility (inverse
kinematics, collisions in the planning scene) is not decided here; see
:mod:`prpl_tidybot.cube_pick.planning`.

Frames: the tool frame is the arm model's `tool_frame` link (z = approach, x = the
fingers' closing axis). The cube frame used here is the cube's centre with z up and x
along one of its horizontal edges (`CubeEstimate.upright_frame`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]

# Tool frame of a straight-down grasp, as columns in the cube's upright frame: the
# closing axis (tool x) along cube y, tool y along cube x, the approach (tool z) down.
# This is the rotation KINDER's pick uses (quaternion (0.707, 0.707, 0, 0) xyzw).
# Half the length of a Robotiq 2F-85 pad along the approach axis (KINDER's MuJoCo
# model, `pad_box1`), about the pinch point at its centre.
PAD_HALF_LENGTH = 0.0094

_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def pose_matrix(rotation: FloatArray, position: FloatArray) -> FloatArray:
    """A 4x4 transform from a rotation matrix and a translation."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


@dataclass(frozen=True)
class GraspCandidate:
    """One way to grasp the cube, relative to its upright frame."""

    cube_from_tool: FloatArray
    yaw_offset: float
    tilt: float
    opening: float
    depth: float
    source: str = "analytic"
    score: float = 1.0

    def tool_pose(self, world_from_cube: FloatArray) -> FloatArray:
        """World pose of the tool frame at the grasp."""
        return world_from_cube @ self.cube_from_tool

    def standoff_pose(self, world_from_cube: FloatArray, distance: float) -> FloatArray:
        """World pose of the tool frame `distance` back along the approach axis."""
        tool = self.tool_pose(world_from_cube)
        out = tool.copy()
        out[:3, 3] = tool[:3, 3] - tool[:3, 2] * distance
        return out

    @property
    def label(self) -> str:
        """Short human-readable description."""
        return (
            f"{self.source} yaw+{np.degrees(self.yaw_offset):.0f} tilt{np.degrees(self.tilt):+.0f} "
            f"open{self.opening * 1000:.0f}mm depth{self.depth * 1000:.0f}mm"
        )


@dataclass(frozen=True)
class CubeGraspSampler:
    """Enumerate grasp candidates in preference order.

    `tilts_deg` are tried smallest first (straight down is the most robust grasp when it
    fits); each nonzero tilt in both directions. `opening_margins` are total extra gap
    beyond the cube's width, narrowest first. `depths` place the pads' centre this far
    above the cube centre (the tool frame sits `tool_to_pinch` behind the pads).
    """

    cube_size: float
    tilts_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    opening_margins: tuple[float, ...] = (0.012, 0.008, 0.020)
    # How far down the cube's sides the pads reach (approved minimum 5 mm); the pads'
    # centre then sits at cube_size / 2 + PAD_HALF_LENGTH - overlap above the centre.
    # Shallower grasps keep the fingertips off the floor beneath a small cube.
    pad_overlaps: tuple[float, ...] = (0.025, 0.0188, 0.012, 0.007)
    depths: tuple[float, ...] | None = None
    tool_to_pinch: float = 0.0052
    max_opening: float = 0.085

    def grasp_depths(self) -> tuple[float, ...]:
        """Heights of the pads' centre above the cube centre, deepest first."""
        if self.depths is not None:
            return self.depths
        out = []
        for overlap in self.pad_overlaps:
            overlap = min(overlap, self.cube_size)
            depth = round(self.cube_size / 2 + PAD_HALF_LENGTH - overlap, 4)
            if depth not in out:
                out.append(depth)
        return tuple(out)

    def candidates(self) -> list[GraspCandidate]:
        """All candidates, most preferred first."""
        out = []
        for tilt_deg in self.tilts_deg:
            tilts = (
                [0.0]
                if tilt_deg == 0
                else [np.radians(tilt_deg), -np.radians(tilt_deg)]
            )
            for margin in self.opening_margins:
                opening = min(self.cube_size + margin, self.max_opening)
                for depth in self.grasp_depths():
                    for yaw_index in range(4):
                        yaw = yaw_index * np.pi / 2
                        for tilt in tilts:
                            out.append(self._candidate(yaw, tilt, opening, depth))
        return out

    def _candidate(
        self, yaw: float, tilt: float, opening: float, depth: float
    ) -> GraspCandidate:
        # Yaw about the cube's vertical, then tilt about the tool's own closing axis.
        rotation = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ _DOWN
            @ Rotation.from_euler("x", tilt).as_matrix()
        )
        # The pads' centre ("pinch") is `depth` above the cube centre; the tool frame
        # is `tool_to_pinch` behind it along the approach axis.
        pinch = np.array([0.0, 0.0, depth])
        position = pinch - rotation[:, 2] * self.tool_to_pinch
        return GraspCandidate(
            cube_from_tool=pose_matrix(rotation, position),
            yaw_offset=yaw,
            tilt=tilt,
            opening=opening,
            depth=depth,
        )


def upright_cube_frame(position: FloatArray, yaw: float) -> FloatArray:
    """The cube's upright frame: centre `position`, z up, x along an edge at `yaw`."""
    return pose_matrix(Rotation.from_euler("z", yaw).as_matrix(), np.asarray(position))


def nearest_equivalent_yaw(yaw: float, reference: float) -> float:
    """The member of {yaw + k * 90 deg} closest to `reference`."""
    k = np.round((reference - yaw) / (np.pi / 2))
    return float(yaw + k * np.pi / 2)


class GraspGenerator(Protocol):
    """Anything that proposes grasps for the perceived cube, best first.

    `cube_frame` is the cube's upright frame from perception (see
    :func:`upright_cube_frame`); candidates are returned relative to it, so the pick can
    re-target them when the visual servo refines the pose. `frame` is the wrist RGB-D
    frame the estimate came from, for generators that work on depth.
    """

    @property
    def name(self) -> str:
        """A short label for logs and results."""

    def candidates(
        self, *, cube_frame: FloatArray, frame: Any, mask: Any
    ) -> list[GraspCandidate]:
        """Grasp candidates, most preferred first."""


@dataclass(frozen=True)
class AnalyticGraspGenerator:
    """The analytic sampler behind the generator interface (ignores the depth)."""

    sampler: CubeGraspSampler
    name: str = "analytic"

    def candidates(
        self, *, cube_frame: FloatArray, frame: Any, mask: Any
    ) -> list[GraspCandidate]:
        del cube_frame, frame, mask
        return self.sampler.candidates()
