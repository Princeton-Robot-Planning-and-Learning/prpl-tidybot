"""Plan executor for kinder kinematic3d envs (PrplLab3D, BaseMotion3D, …).

Preserves the settle-then-advance behaviour the old grounder + RealTidyBotEnv
inner loop produced: snapshot the perceived state on first encounter with a
waypoint, ground the planned 11-d delta into an absolute `TidyBotAction`
target, and reissue that fixed target every tick until the perceived state is
within tolerance — or `max_iter` ticks elapse — at which point advance to the
next waypoint. The 11-d action layout is shared across kinematic3d envs (they
all use `Kinematic3DRobotActionSpace`), so one executor works for any of them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.structs import TidyBotAction


class Kinematic3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Settle-then-advance executor for kinematic3d state-action trajectories.

    Convergence is checked against the absolute `TidyBotAction` target the
    executor itself computed, not against the planner's expected next state —
    this matches the behaviour of the old `RealTidyBotEnv._converged`. The
    tolerances and `max_iter` are properties of the executor (rather than the
    env) because "I'm done tracking this trajectory" is executor-logic.

    Gripper convention: kinder uses bipolar `< -0.5` close / `> 0.5` open;
    `TidyBotAction` uses absolute 0..1 with 1 = closed. The "no change" branch
    passes through the perceiver-written `finger_state`, which is in the same
    convention as `TidyBotAction.gripper_goal`.
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
            # First tick on this waypoint: snapshot the perceived state and
            # ground the planned delta into an absolute TidyBotAction target.
            # We then reissue that fixed target each subsequent tick until
            # convergence — matching the per-action settle loop the old
            # RealTidyBotEnv.step did inline.
            self._cached_target = self._ground_target(sim_state, sim_action)
        self._tick_count_on_index += 1
        return self._cached_target, sim_action

    def done(self, sim_state: ObjectCentricState) -> bool:
        # Advance past every waypoint that is either already converged or
        # has exhausted its iteration budget. Looping (rather than a single
        # if) handles the rare case where multiple successive waypoints are
        # trivially satisfied by the current state.
        while self._index < len(self._trajectory):
            if self._cached_target is None:
                # Haven't issued a command for this waypoint yet — caller
                # should step() before checking convergence.
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

        gripper_cmd = float(sim_action[10])
        if gripper_cmd < -0.5:
            gripper_goal = 1.0  # close
        elif gripper_cmd > 0.5:
            gripper_goal = 0.0  # open
        else:
            gripper_goal = sim_state.get(robot, "finger_state")

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


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
