"""Plan executors for kinematic3d base-only trajectories.

A base-only trajectory is a sequence of (state, action) pairs whose
kinder 11-d action holds the arm joints and gripper at zero and encodes
a base delta in ``action[0:3]`` = ``(dx, dy, dtheta)``. Two tracking
strategies are exposed as separate subclasses:

* :class:`PurePursuitBaseMotion3DPlanExecutor` — across the planned
  polyline, project the robot's ``(x, y)`` onto the path and command a
  target at a fixed arc-length lookahead. The base OTG never decelerates
  between intermediate waypoints, so the base flows smoothly (issue
  #38). Used in real-mode rollouts.
* :class:`SettleBaseMotion3DPlanExecutor` — settle on each pair
  individually before advancing. Useful for diagnosing control issues
  or when waypoint-by-waypoint base motion is wanted.

Both subclasses share a small abstract base, :class:`BaseMotion3DPlanExecutor`,
that owns trajectory storage and the base-only validation. The shared
validator rejects any pair with a nontrivial arm joint or gripper
component at :meth:`set_trajectory` time — mixing base motion with arm
or gripper motion is the dispatcher's concern
(:class:`Kinematic3DPlanExecutor`), which segments mixed trajectories
and feeds each homogeneous segment to the appropriate sub-executor.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.structs import TidyBotAction

_BASE_MOTION_EPS = 1e-4
_ARM_MOTION_EPS = 1e-4


@dataclass(frozen=True)
class _Waypoint:
    """Pre-computed per-pair target the pure-pursuit logic reads each tick."""

    x: float
    y: float
    theta: float
    sim_action: NDArray[np.floating]


class BaseMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState], abc.ABC
):
    """Abstract base for kinematic3d base-only plan executors.

    Owns the trajectory storage, the base-only validation, and the
    convergence tolerance / iteration-cap parameters. Subclasses
    implement :meth:`_on_set_trajectory`, :meth:`step`, and :meth:`done`
    to provide the tracking strategy.
    """

    def __init__(
        self,
        robot_name: str = "robot",
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.1,
        max_iter_per_pair: int = 200,
    ) -> None:
        self._robot_name = robot_name
        self._position_tolerance = position_tolerance
        self._angle_tolerance = angle_tolerance
        self._max_iter_per_pair = max_iter_per_pair
        self._pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        for _, action in trajectory:
            _validate_base_only(action)
        self._pairs = list(trajectory)
        self._on_set_trajectory()

    @abc.abstractmethod
    def _on_set_trajectory(self) -> None:
        """Reset strategy-specific state at the start of a new trajectory."""

    @abc.abstractmethod
    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]: ...

    @abc.abstractmethod
    def done(self, sim_state: ObjectCentricState) -> bool: ...


class SettleBaseMotion3DPlanExecutor(BaseMotion3DPlanExecutor):
    """Settle-then-advance base executor.

    Grounds the per-pair target as ``state.base + delta`` on the first
    tick of each pair and reissues the same target every tick until
    either the perceived base lands within
    ``position_tolerance`` / ``angle_tolerance`` of it, or
    ``max_iter_per_pair`` ticks elapse. Then advances to the next pair.
    """

    def __init__(
        self,
        robot_name: str = "robot",
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.1,
        max_iter_per_pair: int = 200,
    ) -> None:
        super().__init__(
            robot_name=robot_name,
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
            max_iter_per_pair=max_iter_per_pair,
        )
        self._pair_idx: int = 0
        self._tick_on_pair: int = 0
        self._cached_target: TidyBotAction | None = None

    def _on_set_trajectory(self) -> None:
        self._pair_idx = 0
        self._tick_on_pair = 0
        self._cached_target = None

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if not self._pairs:
            raise RuntimeError(
                "SettleBaseMotion3DPlanExecutor.step called with no trajectory"
            )
        _, sim_action = self._pairs[self._pair_idx]
        if self._cached_target is None:
            self._cached_target = self._ground_target(sim_state, sim_action)
        self._tick_on_pair += 1
        return self._cached_target, sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        if not self._pairs:
            return True
        while self._pair_idx < len(self._pairs):
            if self._cached_target is None:
                # step() hasn't issued a command for this pair yet — can't
                # be converged on a target we never sent.
                return False
            converged = self._converged(sim_state, self._cached_target)
            exhausted = self._tick_on_pair >= self._max_iter_per_pair
            if not (converged or exhausted):
                return False
            self._pair_idx += 1
            self._tick_on_pair = 0
            self._cached_target = None
        return True

    def _ground_target(
        self,
        sim_state: ObjectCentricState,
        sim_action: NDArray[np.floating],
    ) -> TidyBotAction:
        robot = sim_state.get_object_from_name(self._robot_name)
        base_goal = SE2(
            x=sim_state.get(robot, "pos_base_x") + float(sim_action[0]),
            y=sim_state.get(robot, "pos_base_y") + float(sim_action[1]),
            theta=sim_state.get(robot, "pos_base_rot") + float(sim_action[2]),
        )
        # Arm + gripper held at the perceived state — this executor never
        # moves them; the trajectory validator rejects any pair that asks
        # for arm or gripper motion.
        arm_goal = [sim_state.get(robot, f"joint_{j + 1}") for j in range(7)]
        gripper_goal = float(sim_state.get(robot, "finger_state"))
        return TidyBotAction(
            arm_goal=arm_goal,
            base_pose_target_map=base_goal,
            gripper_goal=gripper_goal,
        )

    def _converged(self, sim_state: ObjectCentricState, target: TidyBotAction) -> bool:
        robot = sim_state.get_object_from_name(self._robot_name)
        target_map = target.base_pose_target_map
        dx = sim_state.get(robot, "pos_base_x") - target_map.x
        dy = sim_state.get(robot, "pos_base_y") - target_map.y
        if math.hypot(dx, dy) > self._position_tolerance:
            return False
        angle_err = abs(
            _wrap_angle(sim_state.get(robot, "pos_base_rot") - target_map.theta())
        )
        return angle_err <= self._angle_tolerance


class PurePursuitBaseMotion3DPlanExecutor(BaseMotion3DPlanExecutor):
    """Pure-pursuit base executor.

    Treats the planned trajectory as one polyline, flattens each pair
    into its starting base pose plus an implied final waypoint from the
    last pair's ``state + delta``, then per tick projects the robot's
    ``(x, y)`` onto the polyline and commands a target at a fixed
    arc-length lookahead. ``max_iter_per_pair * num_waypoints`` caps the
    total tick count to bound divergent rollouts.
    """

    def __init__(
        self,
        robot_name: str = "robot",
        lookahead_distance: float = 0.2,
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.1,
        max_iter_per_pair: int = 200,
    ) -> None:
        if lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be > 0")
        super().__init__(
            robot_name=robot_name,
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
            max_iter_per_pair=max_iter_per_pair,
        )
        self._lookahead_distance = lookahead_distance
        self._waypoints: list[_Waypoint] = []
        self._cumulative_arc: list[float] = []
        self._cursor_arc: float = 0.0
        self._tick_count: int = 0
        self._done_latched: bool = False

    def _on_set_trajectory(self) -> None:
        self._cursor_arc = 0.0
        self._tick_count = 0
        self._done_latched = False
        if self._pairs:
            self._waypoints = _waypoints_for_pairs(self._pairs, self._robot_name)
            self._cumulative_arc = _cumulative_arc_lengths(self._waypoints)
        else:
            self._waypoints = []
            self._cumulative_arc = []

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if not self._pairs:
            raise RuntimeError(
                "PurePursuitBaseMotion3DPlanExecutor.step called with no trajectory"
            )
        robot = sim_state.get_object_from_name(self._robot_name)
        robot_xy = (
            sim_state.get(robot, "pos_base_x"),
            sim_state.get(robot, "pos_base_y"),
        )
        self._cursor_arc = max(
            self._cursor_arc, _project_onto_path(self._waypoints, robot_xy)
        )
        target_arc = min(
            self._cursor_arc + self._lookahead_distance, self._cumulative_arc[-1]
        )
        target_x, target_y, target_theta = _interpolate_at_arc(
            self._waypoints, self._cumulative_arc, target_arc
        )
        # Arm + gripper held at the perceived state.
        arm_goal = [sim_state.get(robot, f"joint_{j + 1}") for j in range(7)]
        gripper_goal = float(sim_state.get(robot, "finger_state"))
        # Sim action recorded for the recorder: use the pair whose
        # end-of-segment matches the cursor's current position.
        sim_action = self._pair_for_cursor().sim_action
        action = TidyBotAction(
            arm_goal=arm_goal,
            base_pose_target_map=SE2(target_x, target_y, target_theta),
            gripper_goal=gripper_goal,
        )
        self._tick_count += 1
        return action, sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        if not self._pairs:
            return True
        if self._done_latched:
            return True
        if self._tick_count >= self._max_iter_per_pair * len(self._waypoints):
            self._done_latched = True
            return True
        robot = sim_state.get_object_from_name(self._robot_name)
        final = self._waypoints[-1]
        dx = sim_state.get(robot, "pos_base_x") - final.x
        dy = sim_state.get(robot, "pos_base_y") - final.y
        if math.hypot(dx, dy) > self._position_tolerance:
            return False
        angle_err = abs(_wrap_angle(sim_state.get(robot, "pos_base_rot") - final.theta))
        if angle_err > self._angle_tolerance:
            return False
        self._done_latched = True
        return True

    def _pair_for_cursor(self) -> _Waypoint:
        """Waypoint just past the cursor — its sim_action is the planner's
        "current" pair to record."""
        for i, arc in enumerate(self._cumulative_arc):
            if arc > self._cursor_arc:
                return self._waypoints[i]
        return self._waypoints[-1]


# ============================================================================
# Module-level helpers
# ============================================================================


def _validate_base_only(action: NDArray[np.floating]) -> None:
    arm_moves = bool(np.any(np.abs(action[3:10]) > _ARM_MOTION_EPS))
    gripper_moves = abs(float(action[10])) > _ARM_MOTION_EPS
    if arm_moves or gripper_moves:
        raise ValueError(
            "Base-motion plan executors require base-only pairs; got "
            f"arm_delta={action[3:10]}, gripper_cmd={action[10]}."
        )


def _waypoints_for_pairs(
    pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    robot_name: str,
) -> list[_Waypoint]:
    """Pure-pursuit waypoints for a run of base-motion pairs.

    Flattens each pair into its starting base pose, then appends the
    implied final waypoint reconstructed from the last pair's
    ``state + base_delta``.
    """
    waypoints: list[_Waypoint] = []
    for state, sim_action in pairs:
        robot = state.get_object_from_name(robot_name)
        waypoints.append(
            _Waypoint(
                x=state.get(robot, "pos_base_x"),
                y=state.get(robot, "pos_base_y"),
                theta=state.get(robot, "pos_base_rot"),
                sim_action=sim_action,
            )
        )
    final_state, final_action = pairs[-1]
    robot = final_state.get_object_from_name(robot_name)
    waypoints.append(
        _Waypoint(
            x=final_state.get(robot, "pos_base_x") + float(final_action[0]),
            y=final_state.get(robot, "pos_base_y") + float(final_action[1]),
            theta=final_state.get(robot, "pos_base_rot") + float(final_action[2]),
            sim_action=final_action,
        )
    )
    return waypoints


def _cumulative_arc_lengths(waypoints: list[_Waypoint]) -> list[float]:
    arc = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        arc.append(arc[-1] + math.hypot(dx, dy))
    return arc


def _project_onto_path(waypoints: list[_Waypoint], point: tuple[float, float]) -> float:
    """Arc length along the path of the point closest to ``point``.

    Walks each segment, finds the parametric closest-point projection,
    and returns the running arc length of the global closest. O(N) per
    call which is fine for the trajectory lengths we plan over.
    """
    best_arc = 0.0
    best_dist_sq = float("inf")
    px, py = point
    cumulative = 0.0
    for i in range(1, len(waypoints)):
        ax = waypoints[i - 1].x
        ay = waypoints[i - 1].y
        bx = waypoints[i].x
        by = waypoints[i].y
        seg_dx = bx - ax
        seg_dy = by - ay
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        seg_len = math.sqrt(seg_len_sq)
        if seg_len_sq == 0:
            t = 0.0
        else:
            t = max(
                0.0,
                min(1.0, ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq),
            )
        cx = ax + t * seg_dx
        cy = ay + t * seg_dy
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_arc = cumulative + t * seg_len
        cumulative += seg_len
    return best_arc


def _interpolate_at_arc(
    waypoints: list[_Waypoint], cumulative: list[float], arc: float
) -> tuple[float, float, float]:
    """Return the ``(x, y, theta)`` at arc-length ``arc`` along the path.

    Clamps to the endpoints if ``arc`` is outside ``[0, total_length]``.
    Theta is linearly interpolated with shortest-arc wrap so that
    +pi-epsilon / -pi-epsilon segments don't take the long way around.
    """
    if arc <= 0.0:
        wp = waypoints[0]
        return wp.x, wp.y, wp.theta
    if arc >= cumulative[-1]:
        wp = waypoints[-1]
        return wp.x, wp.y, wp.theta
    for i in range(1, len(waypoints)):
        if cumulative[i] >= arc:
            prev = waypoints[i - 1]
            nxt = waypoints[i]
            seg_len = cumulative[i] - cumulative[i - 1]
            t = (arc - cumulative[i - 1]) / seg_len if seg_len > 0 else 0.0
            x = prev.x + t * (nxt.x - prev.x)
            y = prev.y + t * (nxt.y - prev.y)
            theta = prev.theta + t * _wrap_angle(nxt.theta - prev.theta)
            return x, y, theta
    wp = waypoints[-1]
    return wp.x, wp.y, wp.theta


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
