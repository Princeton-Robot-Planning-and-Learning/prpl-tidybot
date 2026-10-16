"""Toss at a bin wherever it stands: the real robot's toss parameters, their bounds, and
the base pose a toss is thrown from.

Tossing3D's toss skill takes ``(standoff_m, speed_deg_s, release_ms)``: how far the base
centre stands from the bin centre, the swing's peak joint speed and the cycle at which
the gripper opens. On this robot the swing is :func:`prpl_tidybot.toss.plan_toss_swing`
(a trapezoidal profile to a fixed release pose, at most 140 deg/s) with joint 1 at
heading 0, so the cube flies straight ahead along the base's heading. The real bounds
(:class:`RealTossBounds`) come from that swing and a ballistic model of the cube from
the release state (validated within ~0.1 m against a real throw):

* ``speed`` 100-140 deg/s: 140 is the most this arm has thrown (``TOSS_MAX_VEL``);
  below ~100 the cube does not reach a bin outside the base's own footprint.
* ``release_ms`` 500-850: across this window, at every speed above, the pinch point
  moves forward *and* up at release (forward speed >= 0.03 m/s, upward >= 0.17 m/s),
  so the cube is thrown ahead and never down at the floor or back at the robot. At
  140 deg/s and 850 ms it is almost straight up (it lands ~0.1 m ahead of release).
* ``standoff`` 0.9-1.45 m, base centre to bin centre: the span of rim-height landing
  distances that box produces.

Anything outside the box is rejected *before* the arm or base moves.

The base pose (:func:`aimed_toss_pose`) faces the bin centre from ``standoff`` along an
azimuth ``direction`` (map frame, from the bin centre to the robot). It must keep the
base's centre in the drivable box, the base's footprint (0.548 x 0.508 m, centred on
the base) clear of the bin, and the whole footprint on the robot's side of the
barrier: like Tossing3D's room-spanning barrier, the robot never crosses it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.bin_localizer import BinFit

# The TidyBot++ base, centre-referenced (m): along its heading, and across.
BASE_LENGTH = 0.548
BASE_WIDTH = 0.508

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class RealTossBounds:
    """The box of real toss parameters every one of which is a safe, forward, upward
    throw (see the module docstring for the derivation)."""

    standoff: tuple[float, float] = (0.9, 1.45)
    speed: tuple[float, float] = (100.0, 140.0)
    release_ms: tuple[float, float] = (500.0, 850.0)

    def validate(self, standoff: float, speed: float, release_ms: float) -> None:
        """Raise ``ValueError`` naming every parameter outside the box (or not a finite
        number)."""
        bad = []
        for name, value, (lo, hi) in (
            ("standoff", standoff, self.standoff),
            ("speed", speed, self.speed),
            ("release_ms", release_ms, self.release_ms),
        ):
            if not (isinstance(value, (int, float)) and math.isfinite(value)):
                bad.append(f"{name}={value!r} is not a number")
            elif not lo <= value <= hi:
                bad.append(f"{name}={value} outside [{lo}, {hi}]")
        if bad:
            raise ValueError("Toss rejected: " + "; ".join(bad) + ".")

    @classmethod
    def from_config(cls, cfg: Any) -> RealTossBounds:
        """From a ``{standoff: [lo, hi], speed: [...], release_ms: [...]}`` block."""
        return cls(
            standoff=(float(cfg["standoff"][0]), float(cfg["standoff"][1])),
            speed=(float(cfg["speed"][0]), float(cfg["speed"][1])),
            release_ms=(float(cfg["release_ms"][0]), float(cfg["release_ms"][1])),
        )


@dataclass(frozen=True)
class Barrier:
    """A rectangle on the floor (map frame) the robot never crosses: its long side
    (``length``) runs along ``yaw``.

    The robot works on its +x side.
    """

    x: float = 0.0
    y: float = 0.0
    length: float = 1.8
    thickness: float = 0.05
    yaw: float = math.pi / 2

    @property
    def robot_side_x(self) -> float:
        """The barrier's +x face: the robot's side starts here."""
        half = 0.5 * (
            abs(self.length * math.cos(self.yaw))
            + abs(self.thickness * math.sin(self.yaw))
        )
        return self.x + half

    def as_dict(self) -> dict[str, float]:
        """Plain floats: the skill server's ``observe()["barrier"]``."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "length": float(self.length),
            "thickness": float(self.thickness),
            "yaw": float(self.yaw),
        }

    @classmethod
    def from_config(cls, cfg: Any) -> Barrier:
        """From ``{center: [x, y], length, thickness, yaw}``."""
        return cls(
            x=float(cfg["center"][0]),
            y=float(cfg["center"][1]),
            length=float(cfg["length"]),
            thickness=float(cfg["thickness"]),
            yaw=float(cfg["yaw"]),
        )


@dataclass(frozen=True)
class Workspace:
    """Where the base may be: its centre inside ``floor_bounds`` (x_min, x_max, y_min,
    y_max) and, when there is a barrier, its whole footprint at least ``clearance``
    beyond the barrier's robot-side face."""

    floor_bounds: tuple[float, float, float, float] = (-1.5, 1.7, -1.5, 1.5)
    barrier: Barrier | None = None
    clearance: float = 0.05

    def pose_problem(
        self, pose: Sequence[float], bin_fit: BinFit | None = None
    ) -> str | None:
        """Why the base cannot stand at ``pose``, or None if it can."""
        x, y = pose[0], pose[1]
        fb = self.floor_bounds
        if not (fb[0] <= x <= fb[1] and fb[2] <= y <= fb[3]):
            return f"base centre ({x:+.3f}, {y:+.3f}) outside the drivable box {fb}"
        corners = footprint(pose)
        if self.barrier is not None:
            limit = self.barrier.robot_side_x + self.clearance
            if corners[:, 0].min() < limit:
                return (
                    f"base footprint reaches x = {corners[:, 0].min():+.3f}, across "
                    f"the barrier (must stay at x >= {limit:+.3f})"
                )
        if bin_fit is not None and _rectangles_overlap(
            corners, _bin_corners(bin_fit, self.clearance)
        ):
            return "base footprint overlaps the bin"
        return None

    def path_problem(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None = None,
        steps: int = 20,
    ) -> str | None:
        """Why the base cannot drive straight from ``start`` to ``goal`` (turning as it
        goes), or None: every pose after the start must be valid (the start is wherever
        the robot already is)."""
        dtheta = math.atan2(math.sin(goal[2] - start[2]), math.cos(goal[2] - start[2]))
        for k in range(1, steps + 1):
            f = k / steps
            pose = (
                start[0] + f * (goal[0] - start[0]),
                start[1] + f * (goal[1] - start[1]),
                start[2] + f * dtheta,
            )
            problem = self.pose_problem(pose, bin_fit)
            if problem is not None:
                return f"on the way to ({goal[0]:+.3f}, {goal[1]:+.3f}): {problem}"
        return None

    def valid(self, bin_fit: BinFit | None = None) -> Callable[[Sequence[float]], bool]:
        """A predicate for base poses (for ``top_down_pick.standoff_pose``)."""
        return lambda pose: self.pose_problem(pose, bin_fit) is None

    @classmethod
    def from_config(cls, cfg: Any, floor_bounds: Sequence[float]) -> Workspace:
        """From a ``barrier`` block (``enabled``, ``center``, ..., ``clearance``)."""
        fb = tuple(float(v) for v in floor_bounds)
        assert len(fb) == 4
        if cfg is None or not cfg.get("enabled", False):
            return cls(floor_bounds=fb)  # type: ignore[arg-type]
        return cls(
            floor_bounds=fb,  # type: ignore[arg-type]
            barrier=Barrier.from_config(cfg),
            clearance=float(cfg.get("clearance", 0.05)),
        )


def footprint(pose: Sequence[float]) -> NDArray[np.float64]:
    """The base's four footprint corners (4, 2) at map pose (x, y, heading)."""
    return _rectangle(pose[0], pose[1], pose[2], BASE_LENGTH, BASE_WIDTH)


def _rectangle(
    x: float, y: float, yaw: float, length: float, width: float
) -> NDArray[np.float64]:
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    return np.c_[
        x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]
    ]


def _bin_corners(bin_fit: BinFit, grow: float) -> NDArray[np.float64]:
    return _rectangle(
        bin_fit.x,
        bin_fit.y,
        bin_fit.yaw,
        bin_fit.length + 2 * grow,
        bin_fit.width + 2 * grow,
    )


def _rectangles_overlap(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def aimed_toss_pose(
    bin_fit: BinFit,
    standoff: float,
    direction: float,
    workspace: Workspace,
) -> Pose:
    """The base pose ``standoff`` from the bin centre along azimuth ``direction`` (map
    frame, bin centre -> robot), facing the bin.

    Raises ``ValueError`` if the base may not stand there.
    """
    x = bin_fit.x + standoff * math.cos(direction)
    y = bin_fit.y + standoff * math.sin(direction)
    heading = math.atan2(bin_fit.y - y, bin_fit.x - x)
    pose = (x, y, heading)
    problem = workspace.pose_problem(pose, bin_fit)
    if problem is not None:
        raise ValueError(
            f"No toss from direction {math.degrees(direction):+.0f} deg at "
            f"{standoff:.2f} m: {problem}."
        )
    return pose


def choose_toss_direction(
    bin_fit: BinFit,
    standoff: float,
    robot: Sequence[float],
    workspace: Workspace,
) -> float:
    """The azimuth (bin -> robot) nearest the robot's current bearing from the bin whose
    toss pose is valid, swinging round in 15-degree steps as
    ``top_down_pick.standoff_pose`` does.

    Raises ``ValueError`` if none is.
    """
    direct = math.atan2(robot[1] - bin_fit.y, robot[0] - bin_fit.x)
    for k in range(13):
        for sign in (1,) if k in (0, 12) else (1, -1):
            direction = direct + sign * math.radians(15 * k)
            try:
                aimed_toss_pose(bin_fit, standoff, direction, workspace)
            except ValueError:
                continue
            return math.atan2(math.sin(direction), math.cos(direction))
    raise ValueError(
        f"No valid toss pose {standoff:.2f} m from the bin at "
        f"({bin_fit.x:+.3f}, {bin_fit.y:+.3f})."
    )


def plan_aimed_toss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    bin_fit: BinFit,
    standoff: float,
    speed: float,
    release_ms: float,
    direction: float | None,
    robot: Sequence[float],
    workspace: Workspace,
    bounds: RealTossBounds,
) -> Pose:
    """Everything a toss is checked against before anything moves: the parameters
    against ``bounds``, the toss pose (along ``direction``, or the one nearest the
    robot's bearing when None) against ``workspace`` and the bin, and the straight drive
    there from ``robot``.

    Returns the toss pose; raises ``ValueError``.
    """
    bounds.validate(standoff, speed, release_ms)
    if direction is None:
        direction = choose_toss_direction(bin_fit, standoff, robot, workspace)
    pose = aimed_toss_pose(bin_fit, standoff, direction, workspace)
    problem = workspace.path_problem(robot, pose, bin_fit)
    if problem is not None:
        raise ValueError(f"Toss pose unreachable: {problem}.")
    return pose
