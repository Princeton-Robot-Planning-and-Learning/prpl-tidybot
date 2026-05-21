"""Plan executor for kinder kinematic3d envs (PrplLab3D, BaseMotion3D, …).

The 11-d kinematic3d action layout splits into a base delta (action[0:3]),
an arm joint delta (action[3:10]), and a gripper command (action[10]).
:class:`Kinematic3DPlanExecutor` is a single closed-loop executor that
handles trajectories made of those pairs with two strict assumptions:

1. **No pair commands both base and arm motion at the same time.** The
   bilevel planner's skill controllers happen to produce pairs that
   move EITHER the base OR the arm, never both — the unified executor
   enforces this at :meth:`set_trajectory` time and ``raise``\\s if
   violated. Mixing the two complicates per-segment strategy choice
   (you can't pure-pursue an arm waypoint) and was never actually
   useful on the planner side.
2. **The arm is always followed waypoint-by-waypoint with a
   convergence check.** Pure-pursuit makes sense only in metric base
   space; it's nonsense in joint space, and on real hardware would
   skip past intermediate arm poses before the OTG had a chance to
   reach them. So arm segments always use settle-then-advance.

The base, in contrast, supports two strategies, configurable via the
constructor's ``base_strategy`` argument:

* ``"pure_pursuit"`` (default) — across a run of consecutive base
  pairs, project the robot's `(x, y)` onto the planned polyline and
  command a target at a fixed arc-length lookahead. The OTG never
  decelerates between intermediate waypoints, so the base flows
  smoothly (issue #38). Used in real-mode rollouts.
* ``"settle"`` — settle on each base pair individually before
  advancing, identical behaviour to arm segments. Useful for diagnosing
  control issues or when waypoint-by-waypoint base motion is wanted.

Internally the executor splits the trajectory into "segments" — each a
maximal run of same-kind (base or arm) pairs — and runs the appropriate
sub-strategy per segment in sequence. When a segment finishes (all its
pairs converged or max-iter'd out), the next segment's state is
snapshotted from the latest perceived state and started.
"""

from __future__ import annotations

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
    """Pre-computed per-pair targets the base pure-pursuit logic reads each tick."""

    x: float
    y: float
    theta: float
    sim_action: NDArray[np.floating]


@dataclass
class _Segment:
    """A maximal run of (state, action) pairs that all move the same component.

    ``kind`` is ``"base"`` (each pair has only base motion) or ``"arm"``
    (each pair has at most arm + gripper motion; base delta is zero).
    """

    kind: str
    pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]]


class Kinematic3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Unified executor for kinematic3d trajectories. See module docstring."""

    def __init__(
        self,
        robot_name: str = "robot",
        base_strategy: str = "pure_pursuit",
        lookahead_distance: float = 0.2,
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.1,
        joint_tolerance: float = 0.05,
        gripper_tolerance: float = 0.05,
        max_iter_per_pair: int = 200,
    ) -> None:
        if base_strategy not in ("pure_pursuit", "settle"):
            raise ValueError(
                "base_strategy must be 'pure_pursuit' or 'settle', "
                f"got {base_strategy!r}"
            )
        if lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be > 0")
        self._robot_name = robot_name
        self._base_strategy = base_strategy
        self._lookahead_distance = lookahead_distance
        self._position_tolerance = position_tolerance
        self._angle_tolerance = angle_tolerance
        self._joint_tolerance = joint_tolerance
        self._gripper_tolerance = gripper_tolerance
        self._max_iter_per_pair = max_iter_per_pair

        self._segments: list[_Segment] = []
        self._segment_idx: int = 0
        self._done_latched: bool = False

        # State that the current segment owns. Reset on every segment boundary.
        # Used by the settle path (arm segments + base-settle segments).
        self._pair_idx: int = 0
        self._tick_on_pair: int = 0
        self._cached_target: TidyBotAction | None = None
        # Used by the pure-pursuit path (base segments only).
        self._pp_waypoints: list[_Waypoint] = []
        self._pp_cumulative_arc: list[float] = []
        self._pp_cursor_arc: float = 0.0
        self._pp_tick_count: int = 0
        self._pp_done_latched: bool = False

    # ------------------------------------------------------------------ Public

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        for state, action in trajectory:
            del state
            self._validate_pair(action)
        self._segments = _build_segments(trajectory)
        self._segment_idx = 0
        self._done_latched = False
        self._enter_segment()

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if self._done_latched or self._segment_idx >= len(self._segments):
            raise RuntimeError(
                "Kinematic3DPlanExecutor.step called after the trajectory finished"
            )
        segment = self._segments[self._segment_idx]
        if segment.kind == "base" and self._base_strategy == "pure_pursuit":
            return self._step_pure_pursuit(sim_state)
        return self._step_settle(sim_state, segment)

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._done_latched:
            return True
        if not self._segments:
            self._done_latched = True
            return True
        # Advance segments while the current one is done.
        while self._segment_idx < len(self._segments):
            segment = self._segments[self._segment_idx]
            if not self._segment_done(sim_state, segment):
                return False
            self._segment_idx += 1
            if self._segment_idx < len(self._segments):
                self._enter_segment()
        self._done_latched = True
        return True

    # ---------------------------------------------------------------- Internal

    def _validate_pair(self, action: NDArray[np.floating]) -> None:
        base_moves = bool(np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS))
        arm_moves = bool(np.any(np.abs(action[3:10]) > _ARM_MOTION_EPS))
        if base_moves and arm_moves:
            raise ValueError(
                "Kinematic3DPlanExecutor requires each (state, action) pair to "
                "move ONLY the base OR the arm, not both. Got base_delta="
                f"{action[0:3]}, arm_delta={action[3:10]}."
            )

    def _enter_segment(self) -> None:
        """Reset per-segment state when crossing a boundary."""
        self._pair_idx = 0
        self._tick_on_pair = 0
        self._cached_target = None
        self._pp_waypoints = []
        self._pp_cumulative_arc = []
        self._pp_cursor_arc = 0.0
        self._pp_tick_count = 0
        self._pp_done_latched = False
        if self._segment_idx >= len(self._segments):
            return
        segment = self._segments[self._segment_idx]
        if segment.kind == "base" and self._base_strategy == "pure_pursuit":
            self._pp_waypoints = _base_waypoints_for_segment(segment, self._robot_name)
            self._pp_cumulative_arc = _cumulative_arc_lengths(self._pp_waypoints)

    # -- settle path (arm segments + base-settle segments) --------------------

    def _step_settle(
        self, sim_state: ObjectCentricState, segment: _Segment
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        _, sim_action = segment.pairs[self._pair_idx]
        if self._cached_target is None:
            self._cached_target = self._ground_target(sim_state, sim_action)
        self._tick_on_pair += 1
        return self._cached_target, sim_action

    def _settle_advance_if_ready(
        self, sim_state: ObjectCentricState, segment: _Segment
    ) -> None:
        """Advance the settle pair index while the current pair is converged."""
        while self._pair_idx < len(segment.pairs):
            if self._cached_target is None:
                # step() hasn't issued a command for this pair yet — can't
                # be converged on a target we never sent.
                return
            converged = self._converged(sim_state, self._cached_target)
            exhausted = self._tick_on_pair >= self._max_iter_per_pair
            if not (converged or exhausted):
                return
            self._pair_idx += 1
            self._tick_on_pair = 0
            self._cached_target = None

    # -- pure-pursuit path (base segments only) -------------------------------

    def _step_pure_pursuit(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        robot = sim_state.get_object_from_name(self._robot_name)
        robot_xy = (
            sim_state.get(robot, "pos_base_x"),
            sim_state.get(robot, "pos_base_y"),
        )
        self._pp_cursor_arc = max(
            self._pp_cursor_arc, _project_onto_path(self._pp_waypoints, robot_xy)
        )
        target_arc = min(
            self._pp_cursor_arc + self._lookahead_distance,
            self._pp_cumulative_arc[-1],
        )
        target_x, target_y, target_theta = _interpolate_at_arc(
            self._pp_waypoints, self._pp_cumulative_arc, target_arc
        )
        # Arm + gripper held at the perceived state — the segment validator
        # ensured no pair in this segment commands arm motion, so the planned
        # arm/gripper values don't change across the segment.
        arm_goal = [sim_state.get(robot, f"joint_{j + 1}") for j in range(7)]
        gripper_goal = float(sim_state.get(robot, "finger_state"))
        # Sim action recorded for the recorder: use the pair whose end-of-
        # segment matches the cursor's current position (best approximation
        # of "where in the planner's trajectory are we right now").
        sim_action = self._pp_pair_for_cursor().sim_action
        action = TidyBotAction(
            arm_goal=arm_goal,
            base_pose_target_map=SE2(target_x, target_y, target_theta),
            gripper_goal=gripper_goal,
        )
        self._pp_tick_count += 1
        return action, sim_action

    def _pp_pair_for_cursor(self) -> _Waypoint:
        """Waypoint just past the cursor — its sim_action is the planner's "current"
        pair to record."""
        for i, arc in enumerate(self._pp_cumulative_arc):
            if arc > self._pp_cursor_arc:
                return self._pp_waypoints[i]
        return self._pp_waypoints[-1]

    def _pure_pursuit_segment_done(self, sim_state: ObjectCentricState) -> bool:
        if self._pp_done_latched:
            return True
        if self._pp_tick_count >= self._max_iter_per_pair * len(self._pp_waypoints):
            self._pp_done_latched = True
            return True
        robot = sim_state.get_object_from_name(self._robot_name)
        final = self._pp_waypoints[-1]
        dx = sim_state.get(robot, "pos_base_x") - final.x
        dy = sim_state.get(robot, "pos_base_y") - final.y
        if math.hypot(dx, dy) > self._position_tolerance:
            return False
        angle_err = abs(_wrap_angle(sim_state.get(robot, "pos_base_rot") - final.theta))
        if angle_err > self._angle_tolerance:
            return False
        self._pp_done_latched = True
        return True

    # -- segment-level done dispatch -----------------------------------------

    def _segment_done(self, sim_state: ObjectCentricState, segment: _Segment) -> bool:
        if segment.kind == "base" and self._base_strategy == "pure_pursuit":
            return self._pure_pursuit_segment_done(sim_state)
        # Settle path.
        self._settle_advance_if_ready(sim_state, segment)
        return self._pair_idx >= len(segment.pairs)

    # -- grounding + convergence (shared by settle paths) ---------------------

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
        arm_goal = [
            sim_state.get(robot, f"joint_{j + 1}") + float(sim_action[3 + j])
            for j in range(7)
        ]
        gripper_goal = _gripper_target(sim_state.get(robot, "finger_state"), sim_action)
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
        if angle_err > self._angle_tolerance:
            return False
        for j in range(7):
            if (
                abs(sim_state.get(robot, f"joint_{j + 1}") - target.arm_goal[j])
                > self._joint_tolerance
            ):
                return False
        if (
            abs(sim_state.get(robot, "finger_state") - target.gripper_goal)
            > self._gripper_tolerance
        ):
            return False
        return True


# ============================================================================
# Module-level helpers
# ============================================================================


def _build_segments(
    trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
) -> list[_Segment]:
    """Split into maximal runs of same-kind pairs.

    A pair is "base" if its base delta is nontrivial; otherwise "arm".
    The validator in `Kinematic3DPlanExecutor.set_trajectory` already
    rejects pairs that move both, so the classification here is safe.
    Pairs that move neither (e.g. a pure gripper change, or a no-op)
    land in arm segments — the settle path handles them fine since
    the convergence check immediately passes for the unchanged dims.
    """
    segments: list[_Segment] = []
    current: _Segment | None = None
    for state, action in trajectory:
        kind = "base" if np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS) else "arm"
        if current is None or current.kind != kind:
            if current is not None:
                segments.append(current)
            current = _Segment(kind=kind, pairs=[])
        current.pairs.append((state, action))
    if current is not None:
        segments.append(current)
    return segments


def _base_waypoints_for_segment(segment: _Segment, robot_name: str) -> list[_Waypoint]:
    """Pure-pursuit waypoints for a base segment.

    The segment is a contiguous run of base-motion pairs; flatten each
    into its starting base pose, then append the implied final waypoint
    reconstructed from the last pair's `state + base_delta`.
    """
    waypoints: list[_Waypoint] = []
    for state, sim_action in segment.pairs:
        robot = state.get_object_from_name(robot_name)
        waypoints.append(
            _Waypoint(
                x=state.get(robot, "pos_base_x"),
                y=state.get(robot, "pos_base_y"),
                theta=state.get(robot, "pos_base_rot"),
                sim_action=sim_action,
            )
        )
    final_state, final_action = segment.pairs[-1]
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
    """Arc length along the path of the point closest to `point`.

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
    """Return the `(x, y, theta)` at arc-length `arc` along the path.

    Clamps to the endpoints if `arc` is outside [0, total_length]. Theta
    is linearly interpolated with shortest-arc wrap so that
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


def _gripper_target(current_finger: float, sim_action: NDArray[np.floating]) -> float:
    """Convert the kinder bipolar gripper command to a TidyBotAction absolute target."""
    gripper_cmd = float(sim_action[10])
    if gripper_cmd < -0.5:
        return 1.0  # close
    if gripper_cmd > 0.5:
        return 0.0  # open
    return float(current_finger)


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
