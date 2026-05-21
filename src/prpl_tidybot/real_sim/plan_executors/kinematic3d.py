"""Plan executors for kinder kinematic3d envs (PrplLab3D, BaseMotion3D, …).

Two concrete executors live here, both consuming the same 11-d
kinematic3d `(state, action)` trajectory and emitting absolute
`TidyBotAction` commands. The Hydra env yamls default to
:class:`PurePursuitKinematic3DPlanExecutor`; switch to
:class:`SettleKinematic3DPlanExecutor` per-rollout by overriding
``_target_`` on the command line, e.g.::

    python scripts/run_planner.py env=base_motion3d mode=real \\
      env.pipelines.real.plan_executor._target_=\\
        prpl_tidybot.real_sim.plan_executors.kinematic3d.SettleKinematic3DPlanExecutor

* :class:`PurePursuitKinematic3DPlanExecutor` projects the robot's
  current `(x, y)` onto the planner's base path and commands an
  absolute pose at a fixed arc-length lookahead. The OTG never sees a
  "we're done" target until the lookahead clamps to the final
  waypoint, so the base flows through intermediate waypoints (issue
  #38). Arm and gripper come from the closest waypoint's planned
  values unchanged.

* :class:`SettleKinematic3DPlanExecutor` snapshots the perceived state
  on first encounter with each waypoint, grounds the planned delta
  into an absolute `TidyBotAction`, and reissues that fixed target
  every tick until the perceived state is within tolerance — or
  `max_iter` ticks elapse — at which point advance to the next. This
  is the original per-waypoint convergence behaviour; kept for
  comparison and as a fallback when pure pursuit is unwanted (e.g.
  diagnosing where chunky motion is coming from).

The 11-d action layout is shared across kinematic3d envs (they all
use `Kinematic3DRobotActionSpace`), so each executor works for any of
them.
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


@dataclass(frozen=True)
class _Waypoint:
    """Pre-computed per-waypoint targets the pure-pursuit executor reads each tick."""

    x: float
    y: float
    theta: float
    arm: list[float]
    gripper: float
    sim_action: NDArray[np.floating]  # the planned sim_action recorded for this segment


class PurePursuitKinematic3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Pure-pursuit executor for kinematic3d state-action trajectories.

    Per tick:
      1. Project the robot's perceived `(x, y)` onto the trajectory's
         base path and advance a monotonic arc-length cursor to the
         projection (never retreats, so brief perception jitter back
         toward earlier waypoints doesn't make the lookahead regress).
      2. Command the absolute SE2 pose at ``cursor + lookahead_distance``
         along the path. Arm / gripper come from the closest waypoint.

    `done()` returns True when the perceived robot is within position +
    angle tolerance of the FINAL waypoint AND arm/gripper match its
    planned values — there's no per-intermediate-waypoint convergence
    check. `max_iter` caps total ticks across the whole trajectory as a
    safety net for unreachable goals.
    """

    def __init__(
        self,
        robot_name: str = "robot",
        lookahead_distance: float = 0.2,
        position_tolerance: float = 0.02,
        angle_tolerance: float = 0.05,
        joint_tolerance: float = 0.05,
        gripper_tolerance: float = 0.05,
        max_iter: int = 1000,
    ) -> None:
        if lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be > 0")
        self._robot_name = robot_name
        self._lookahead_distance = lookahead_distance
        # Tolerance defaults are set wider than the marker detector's typical
        # noise floor at the ceiling height — too-tight tolerances meant
        # done() never latched and the OTG kept chasing each tick's
        # recalibrated odom target (visible as oscillation around the goal
        # on real hardware). The defaults can still be overridden via the
        # Hydra env yaml for tighter sim-mode checks.
        self._position_tolerance = position_tolerance
        self._angle_tolerance = angle_tolerance
        self._joint_tolerance = joint_tolerance
        self._gripper_tolerance = gripper_tolerance
        self._max_iter = max_iter
        self._waypoints: list[_Waypoint] = []
        self._cumulative_arc: list[float] = []
        self._cursor_arc: float = 0.0
        self._tick_count: int = 0
        # Sticky done. Once `_at_final_waypoint` reports True even once,
        # latch the result so a subsequent noisier perception tick can't
        # un-converge us and re-engage the OTG. With the Runner's `while
        # not done` loop, latching here means the loop exits and the OTG
        # holds the last commanded pose — that, combined with looser
        # tolerances above, is what fixes the end-of-trajectory oscillation.
        self._done_latched: bool = False

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        self._waypoints = _waypoints_from_trajectory(trajectory, self._robot_name)
        self._cumulative_arc = _cumulative_arc_lengths(self._waypoints)
        self._cursor_arc = 0.0
        self._tick_count = 0
        self._done_latched = False

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if not self._waypoints:
            raise RuntimeError(
                "PurePursuitKinematic3DPlanExecutor.step called with empty trajectory"
            )
        robot = sim_state.get_object_from_name(self._robot_name)
        robot_xy = (
            sim_state.get(robot, "pos_base_x"),
            sim_state.get(robot, "pos_base_y"),
        )
        self._cursor_arc = max(self._cursor_arc, self._project_onto_path(robot_xy))
        target_arc = min(
            self._cursor_arc + self._lookahead_distance, self._cumulative_arc[-1]
        )
        target_x, target_y, target_theta = self._interpolate_at_arc(target_arc)
        ahead = self._next_waypoint_after_cursor()
        action = TidyBotAction(
            arm_goal=ahead.arm,
            base_pose_target_map=SE2(target_x, target_y, target_theta),
            gripper_goal=ahead.gripper,
        )
        self._tick_count += 1
        return action, ahead.sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._done_latched:
            return True
        if not self._waypoints:
            self._done_latched = True
            return True
        if self._tick_count >= self._max_iter:
            self._done_latched = True
            return True
        if self._at_final_waypoint(sim_state):
            self._done_latched = True
            return True
        return False

    def _project_onto_path(self, point: tuple[float, float]) -> float:
        """Arc length along the path of the point closest to `point`.

        Walks each segment, finds the parametric closest-point projection,
        and returns the running arc length of the global closest. O(N) per
        call which is fine for the trajectory lengths we plan over.
        """
        best_arc = 0.0
        best_dist_sq = float("inf")
        px, py = point
        for i in range(1, len(self._waypoints)):
            ax = self._waypoints[i - 1].x
            ay = self._waypoints[i - 1].y
            bx = self._waypoints[i].x
            by = self._waypoints[i].y
            seg_dx = bx - ax
            seg_dy = by - ay
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
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
                best_arc = self._cumulative_arc[i - 1] + t * math.sqrt(seg_len_sq)
        return best_arc

    def _interpolate_at_arc(self, arc: float) -> tuple[float, float, float]:
        """Return the (x, y, theta) at arc-length `arc` along the path.

        Clamps to the endpoints if `arc` is outside [0, total_length]. Theta
        is linearly interpolated with shortest-arc wrap so that
        +pi-epsilon / -pi-epsilon segments don't take the long way around.
        """
        if arc <= 0.0:
            wp = self._waypoints[0]
            return wp.x, wp.y, wp.theta
        if arc >= self._cumulative_arc[-1]:
            wp = self._waypoints[-1]
            return wp.x, wp.y, wp.theta
        for i in range(1, len(self._waypoints)):
            if self._cumulative_arc[i] >= arc:
                prev = self._waypoints[i - 1]
                nxt = self._waypoints[i]
                seg_len = self._cumulative_arc[i] - self._cumulative_arc[i - 1]
                t = (
                    (arc - self._cumulative_arc[i - 1]) / seg_len
                    if seg_len > 0
                    else 0.0
                )
                x = prev.x + t * (nxt.x - prev.x)
                y = prev.y + t * (nxt.y - prev.y)
                theta = prev.theta + t * _wrap_angle(nxt.theta - prev.theta)
                return x, y, theta
        wp = self._waypoints[-1]
        return wp.x, wp.y, wp.theta

    def _next_waypoint_after_cursor(self) -> _Waypoint:
        """The next planned waypoint strictly ahead of the cursor.

        Source of truth for arm/gripper targets and for the
        `current_sim_action` recorded with each tick. "Strictly ahead"
        (rather than "at or after") matters at the start of a segment:
        when the cursor sits exactly on waypoint[i], the planner's
        *target* for the next segment is waypoint[i+1], so that's what
        we should command. The final waypoint is the fallback once the
        cursor reaches the end of the path.
        """
        for i, arc in enumerate(self._cumulative_arc):
            if arc > self._cursor_arc:
                return self._waypoints[i]
        return self._waypoints[-1]

    def _at_final_waypoint(self, sim_state: ObjectCentricState) -> bool:
        robot = sim_state.get_object_from_name(self._robot_name)
        final = self._waypoints[-1]
        dx = sim_state.get(robot, "pos_base_x") - final.x
        dy = sim_state.get(robot, "pos_base_y") - final.y
        if math.hypot(dx, dy) > self._position_tolerance:
            return False
        angle_err = abs(_wrap_angle(sim_state.get(robot, "pos_base_rot") - final.theta))
        if angle_err > self._angle_tolerance:
            return False
        for j in range(7):
            if (
                abs(sim_state.get(robot, f"joint_{j + 1}") - final.arm[j])
                > self._joint_tolerance
            ):
                return False
        if (
            abs(sim_state.get(robot, "finger_state") - final.gripper)
            > self._gripper_tolerance
        ):
            return False
        return True


class SettleKinematic3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Settle-then-advance executor for kinematic3d state-action trajectories.

    Snapshots the perceived state on first encounter with each waypoint,
    grounds the planned 11-d delta into an absolute `TidyBotAction`
    target, and reissues that fixed target every tick until the
    perceived state is within tolerance — or `max_iter` ticks elapse —
    at which point advances to the next waypoint.

    This was the original behaviour before
    :class:`PurePursuitKinematic3DPlanExecutor` landed. Kept as the
    "drive to each waypoint and pause" fallback (e.g. when diagnosing
    where chunky motion comes from, or when intermediate-waypoint
    convergence is actually desired).
    """

    def __init__(
        self,
        robot_name: str = "robot",
        position_tolerance: float = 0.01,
        angle_tolerance: float = 0.01,
        joint_tolerance: float = 0.05,
        gripper_tolerance: float = 0.05,
        max_iter: int = 100,
    ) -> None:
        self._robot_name = robot_name
        self._position_tolerance = position_tolerance
        self._angle_tolerance = angle_tolerance
        self._joint_tolerance = joint_tolerance
        self._gripper_tolerance = gripper_tolerance
        self._max_iter = max_iter
        self._trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []
        self._index = 0
        self._tick_count_on_index = 0
        self._cached_target: TidyBotAction | None = None

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        self._trajectory = trajectory
        self._index = 0
        self._tick_count_on_index = 0
        self._cached_target = None

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        _, sim_action = self._trajectory[self._index]
        if self._cached_target is None:
            self._cached_target = self._ground_target(sim_state, sim_action)
        self._tick_count_on_index += 1
        return self._cached_target, sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        while self._index < len(self._trajectory):
            if self._cached_target is None:
                return False
            converged = self._converged(sim_state, self._cached_target)
            exhausted = self._tick_count_on_index >= self._max_iter
            if not (converged or exhausted):
                return False
            self._index += 1
            self._tick_count_on_index = 0
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


def _waypoints_from_trajectory(
    trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    robot_name: str,
) -> list[_Waypoint]:
    """Flatten `(state, action)` pairs into pure-pursuit waypoints.

    Includes the implied final waypoint reconstructed from the last
    pair's `state + action.base_delta` — the planner's
    `plan.states[-1]` would carry it exactly, but the (state, action)
    pair API drops it.
    """
    waypoints: list[_Waypoint] = []
    for state, sim_action in trajectory:
        robot = state.get_object_from_name(robot_name)
        waypoints.append(
            _Waypoint(
                x=state.get(robot, "pos_base_x"),
                y=state.get(robot, "pos_base_y"),
                theta=state.get(robot, "pos_base_rot"),
                arm=[state.get(robot, f"joint_{j + 1}") for j in range(7)],
                gripper=_gripper_target(state.get(robot, "finger_state"), sim_action),
                sim_action=sim_action,
            )
        )
    if trajectory:
        final_state, final_action = trajectory[-1]
        robot = final_state.get_object_from_name(robot_name)
        final_arm = [
            final_state.get(robot, f"joint_{j + 1}") + float(final_action[3 + j])
            for j in range(7)
        ]
        waypoints.append(
            _Waypoint(
                x=final_state.get(robot, "pos_base_x") + float(final_action[0]),
                y=final_state.get(robot, "pos_base_y") + float(final_action[1]),
                theta=final_state.get(robot, "pos_base_rot") + float(final_action[2]),
                arm=final_arm,
                gripper=_gripper_target(
                    final_state.get(robot, "finger_state"), final_action
                ),
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
