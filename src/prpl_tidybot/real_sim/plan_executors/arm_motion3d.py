"""Plan executors for kinematic3d arm + gripper trajectories.

An arm-only trajectory is a sequence of (state, action) pairs whose
kinder 11-d action holds the base delta at zero, encodes arm joint
deltas in ``action[3:10]``, and a gripper command in ``action[10]``.
The mapping from kinder gripper command to TidyBotAction gripper_goal
is: ``cmd < -0.5`` → close (1.0), ``cmd > 0.5`` → open (0.0), otherwise
hold the perceived finger state.

Currently one concrete subclass is exposed:

* :class:`StreamingArmMotion3DPlanExecutor` — discrete-waypoint
  crossover advance. Each pair's intended absolute joint target
  ``state.joints + action.arm_delta`` is precomputed; per tick, the
  cursor advances through any waypoints already within
  ``advance_radius`` of the perceived joints, and the OTG target is
  set to the waypoint at the new cursor. By re-targeting *before* the
  OTG has decelerated to zero at the current waypoint, Ruckig's
  mid-flight replan keeps the arm in cruise — the planned via-points
  are visited within tolerance without per-waypoint accel/decel
  cycles. See the class docstring for full details.

The abstract base :class:`ArmMotion3DPlanExecutor` owns the trajectory
storage and the arm-only validation (rejects any pair with nontrivial
base motion at :meth:`set_trajectory` time). Mixed base + arm/gripper
trajectories are the dispatcher's concern
(:class:`Kinematic3DPlanExecutor`), which segments mixed trajectories
and feeds each homogeneous segment to the appropriate sub-executor.
"""

from __future__ import annotations

import abc
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.structs import TidyBotAction

JointPositions = list[float]
# Matches pybullet-helpers' create_joint_distance_fn return type. list (not
# Sequence) in the argument slot so that callable is assignable here without
# contravariance complaints; Sequence-typed callers still match via the
# contravariant direction.
JointDistanceFn = Callable[[JointPositions, JointPositions], float]

_BASE_MOTION_EPS = 1e-4


class ArmMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState], abc.ABC
):
    """Abstract base for kinematic3d arm + gripper plan executors.

    Owns trajectory storage and the arm-only validation. Subclasses
    implement :meth:`_on_set_trajectory`, :meth:`step`, and :meth:`done`
    to provide the tracking strategy.
    """

    def __init__(self, robot_name: str = "robot") -> None:
        self._robot_name = robot_name
        self._pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        for _, action in trajectory:
            _validate_arm_only(action)
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


class StreamingArmMotion3DPlanExecutor(ArmMotion3DPlanExecutor):
    """Discrete-waypoint crossover advance for arm + gripper trajectories.

    Each (state, action) pair's intended absolute joint target is
    ``state.joints + action.arm_delta`` (computed wrap-aware by the
    injected ``distance_fn`` at convergence-check time; the raw sum
    stored here may lie outside ``[-pi, pi]`` for circular joints —
    that's fine, the distance function handles it). On each tick:

    1. While the perceived joints are within ``advance_radius`` (in the
       ``distance_fn`` metric) of the current target, advance the cursor
       to the next pair. Multiple waypoints can be passed in a single
       tick if they're all already within radius — the cursor jumps
       straight to the furthest one.
    2. Command the new cursor's target as the arm goal in the resulting
       :class:`TidyBotAction`. Same target is re-issued every tick as a
       heartbeat (the underlying Kinova controller's watchdog freezes
       the OTG after ``2.5 * POLICY_CONTROL_PERIOD`` without a fresh
       command — see ``arm_controller.py:104``).

    Because the cursor advances *before* the OTG has decelerated to
    zero at the current waypoint, Ruckig's mid-flight replan uses the
    current nonzero velocity as initial conditions and the arm rounds
    through the waypoint at speed. The contract this requires from the
    planner: **adjacent waypoints in the trajectory must be spaced
    further apart in the distance metric than** ``advance_radius`` —
    otherwise the while-loop can skip a waypoint without the arm having
    visited it. With waypoints intended as via-points (obstacle
    avoidance, etc.), that gap is what guarantees the via-point is
    actually reached.

    The base + gripper components of the commanded ``TidyBotAction``
    hold at the perceived state for arm pairs that don't move them
    (``action[10]`` in ``[-0.5, 0.5]`` means "hold gripper"); explicit
    gripper open/close commands surface as ``gripper_goal=0.0/1.0``.

    Why this strategy works only because of the OTG: if the underlying
    controller were a pre-computed-trajectory follower, re-targeting
    mid-flight would abort the in-flight motion and restart from rest,
    defeating the whole point. The Kinova controller's Ruckig-based
    online retargeting is what makes mid-flight cursor advance smooth.
    """

    def __init__(
        self,
        distance_fn: JointDistanceFn,
        robot_name: str = "robot",
        advance_radius: float = 0.2,
        arrival_tolerance: float = 0.1,
        max_iter_total: int = 2000,
        gripper_dwell_ticks: int = 0,
    ) -> None:
        super().__init__(robot_name=robot_name)
        if advance_radius <= 0:
            raise ValueError("advance_radius must be > 0")
        if arrival_tolerance <= 0:
            raise ValueError("arrival_tolerance must be > 0")
        if max_iter_total <= 0:
            raise ValueError("max_iter_total must be > 0")
        if gripper_dwell_ticks < 0:
            raise ValueError("gripper_dwell_ticks must be >= 0")
        self._distance_fn = distance_fn
        self._advance_radius = advance_radius
        self._arrival_tolerance = arrival_tolerance
        self._max_iter_total = max_iter_total
        self._gripper_dwell_ticks = gripper_dwell_ticks

        self._targets: list[JointPositions] = []
        self._cursor: int = 0
        self._tick_count: int = 0
        self._done_latched: bool = False
        self._gripper_cursor: int = -1
        self._gripper_ticks_remaining: int = 0
        self._last_gripper_goal: float | None = None

    def _on_set_trajectory(self) -> None:
        self._targets = [
            _absolute_target(state, action, self._robot_name)
            for state, action in self._pairs
        ]
        self._cursor = 0
        self._tick_count = 0
        self._done_latched = False
        self._gripper_cursor = -1
        self._gripper_ticks_remaining = 0
        self._last_gripper_goal = None

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if not self._pairs:
            raise RuntimeError(
                "StreamingArmMotion3DPlanExecutor.step called with no trajectory"
            )
        perceived = _perceived_joints(sim_state, self._robot_name)
        self._advance_cursor(perceived)
        target = self._targets[self._cursor]
        _, sim_action = self._pairs[self._cursor]
        # Remember the most recent explicit open/close so that subsequent "hold"
        # ticks (e.g. the entire retract phase) re-issue the same gripper goal.
        # The planning sim's finger_state may not reflect the real gripper state
        # (kinder does not update finger_state after close actions), so we cannot
        # rely on the planned state; tracking the last command is authoritative.
        if _is_gripper_cmd(sim_action):
            self._last_gripper_goal = 1.0 if float(sim_action[10]) < -0.5 else 0.0
        action = _build_tidybot_action(
            sim_state, target, sim_action, self._robot_name, self._last_gripper_goal
        )
        self._tick_count += 1
        # Advance past a gripper pair after gripper_dwell_ticks extra ticks.
        # With gripper_dwell_ticks=0 the cursor advances on the very next tick
        # after the command is issued (original behaviour, correct for sim/fake
        # where FakeInterface stores the target immediately).  In real mode set
        # gripper_dwell_ticks to something like 20 (≈5 s at 0.25 s/tick) so the
        # arm stays at the grasp position while the Kinova gripper physically
        # closes around the object before retract begins.
        if _is_gripper_cmd(sim_action) and self._cursor + 1 < len(self._targets):
            if self._cursor != self._gripper_cursor:
                self._gripper_cursor = self._cursor
                self._gripper_ticks_remaining = self._gripper_dwell_ticks
            if self._gripper_ticks_remaining > 0:
                self._gripper_ticks_remaining -= 1
            else:
                self._cursor += 1
        return action, sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._done_latched:
            return True
        if not self._pairs:
            self._done_latched = True
            return True
        if self._tick_count >= self._max_iter_total:
            self._done_latched = True
            return True
        # Require the cursor to have reached the last waypoint before declaring
        # done. Without this guard, done() returns True immediately when the
        # final target (the retract/home position) happens to equal the robot's
        # initial perceived position — the merged arm segment (approach +
        # gripper + retract) starts and ends at home, so the distance check
        # fires before a single step() is ever called.
        if self._cursor < len(self._targets) - 1:
            return False
        perceived = _perceived_joints(sim_state, self._robot_name)
        final_target = self._targets[-1]
        if self._distance_fn(perceived, final_target) <= self._arrival_tolerance:
            self._done_latched = True
            return True
        return False

    def _advance_cursor(self, perceived: JointPositions) -> None:
        while (
            self._cursor + 1 < len(self._targets)
            and not _is_gripper_cmd(self._pairs[self._cursor][1])
            and self._distance_fn(perceived, self._targets[self._cursor])
            <= self._advance_radius
        ):
            self._cursor += 1


# ============================================================================
# Module-level helpers
# ============================================================================


def _is_gripper_cmd(action: NDArray[np.floating]) -> bool:
    """True when action[10] encodes an explicit open or close command (|cmd| > 0.5)."""
    return abs(float(action[10])) > 0.5


def _validate_arm_only(action: NDArray[np.floating]) -> None:
    base_moves = bool(np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS))
    if base_moves:
        raise ValueError(
            "Arm-motion plan executors require arm-only pairs; got "
            f"base_delta={action[0:3]}."
        )


def _perceived_joints(sim_state: ObjectCentricState, robot_name: str) -> JointPositions:
    robot = sim_state.get_object_from_name(robot_name)
    return [sim_state.get(robot, f"joint_{j + 1}") for j in range(7)]


def _absolute_target(
    state: ObjectCentricState,
    sim_action: NDArray[np.floating],
    robot_name: str,
) -> JointPositions:
    """Per-pair absolute joint target = state.joints + arm_delta.

    The raw sum may fall outside ``[-pi, pi]`` for circular joints;
    that's fine — downstream distance/equality checks must be
    wrap-aware (the convergence check goes through ``distance_fn``,
    which is, and the Kinova arm controller's
    :func:`execute_action_angular` path also wraps the target itself
    inside the inner loop, so the OTG receives a wrap-safe value
    either way).
    """
    robot = state.get_object_from_name(robot_name)
    return [
        float(state.get(robot, f"joint_{j + 1}")) + float(sim_action[3 + j])
        for j in range(7)
    ]


def _build_tidybot_action(
    sim_state: ObjectCentricState,
    arm_target: JointPositions,
    sim_action: NDArray[np.floating],
    robot_name: str,
    last_gripper_goal: float | None = None,
) -> TidyBotAction:
    """Pack the commanded arm goal + held base pose + gripper into a TidyBotAction.

    For "hold" gripper commands (|action[10]| <= 0.5), uses ``last_gripper_goal``
    as the hold target when provided (the last explicit open/close command issued
    by the executor), falling back to perceived finger_state when no explicit
    command has been issued yet. This ensures the gripper stays closed throughout
    retract after a gripper-close pair instead of reverting to perceived state.
    """
    robot = sim_state.get_object_from_name(robot_name)
    base_goal = SE2(
        x=float(sim_state.get(robot, "pos_base_x")),
        y=float(sim_state.get(robot, "pos_base_y")),
        theta=float(sim_state.get(robot, "pos_base_rot")),
    )
    hold_finger = (
        last_gripper_goal
        if last_gripper_goal is not None
        else float(sim_state.get(robot, "finger_state"))
    )
    gripper_goal = _gripper_target(hold_finger, sim_action)
    return TidyBotAction(
        arm_goal=list(arm_target),
        base_pose_target_map=base_goal,
        gripper_goal=gripper_goal,
    )


def _gripper_target(current_finger: float, sim_action: NDArray[np.floating]) -> float:
    """Convert the kinder bipolar gripper command to a TidyBotAction absolute target."""
    gripper_cmd = float(sim_action[10])
    if gripper_cmd < -0.5:
        return 1.0  # close
    if gripper_cmd > 0.5:
        return 0.0  # open
    return current_finger
