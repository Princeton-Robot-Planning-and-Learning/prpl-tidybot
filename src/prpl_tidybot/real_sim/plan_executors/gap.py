"""Executors for the gaps a magic skill leaves in a planned trajectory.

A magic skill shows up in a bilevel-planning trajectory as a single
``SkillCall`` pair: the planner did not simulate the skill's policy and
instead recorded the state its option model predicts. Whatever follows in
the plan was refined against that predicted state, so before execution
resumes the robot has to be brought to (at least) the predicted base pose
and arm configuration.

:class:`SettleGapExecutor` does only that settling: it synthesises a base
pair and an arm/gripper pair whose targets are the predicted configuration
and tracks them with the usual sub-executors. On its own it is the whole
gap handling for fake mode, where nothing physically performs the skill.

:class:`TeleopGapExecutor` is the real-mode version: it first emits a
``TeleopHandoff`` so the env releases the arm to a human (the Kinova's own
gamepad teleoperation) and waits for them to report done, then settles from
wherever the operator left the robot to the predicted configuration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from kinder_models.kinematic3d.constants import GRIPPER_OPEN_THRESHOLD
from kinder_models.structs import SkillCall
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import ArmMotion3DPlanExecutor
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    BaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.kinematic3d import (
    Kinematic3DPlanExecutor,
    SimAction,
)
from prpl_tidybot.structs import RealAction, TeleopHandoff

_BASE_MOTION_EPS = 1e-4


class SettleGapExecutor(PlanExecutor[SimAction, RealAction, ObjectCentricState]):
    """Drive the robot to a SkillCall's predicted configuration.

    ``set_trajectory`` accepts exactly one ``(state, SkillCall)`` pair. The
    base is moved first (skipped when the predicted base pose already
    matches the pair's state), then the arm joints and gripper are settled
    to the predicted values. Every ``step`` reports the ``SkillCall`` as
    the sim action being tracked, so the gap stays visible in recorded
    trajectories.
    """

    def __init__(
        self,
        base_executor: BaseMotion3DPlanExecutor | None = None,
        arm_executor: ArmMotion3DPlanExecutor | None = None,
        robot_name: str = "robot",
    ) -> None:
        self._inner = Kinematic3DPlanExecutor(
            base_executor=base_executor, arm_executor=arm_executor
        )
        self._robot_name = robot_name
        self._call: SkillCall[ObjectCentricState] | None = None

    def set_trajectory(
        self, trajectory: Sequence[tuple[ObjectCentricState, SimAction]]
    ) -> None:
        if len(trajectory) != 1:
            raise ValueError(
                "SettleGapExecutor expects a single (state, SkillCall) pair; got "
                f"{len(trajectory)} pair(s)."
            )
        state, call = trajectory[0]
        if not isinstance(call, SkillCall):
            raise ValueError(
                "SettleGapExecutor expects a single (state, SkillCall) pair; got "
                f"a {type(call).__name__} action."
            )
        self._call = call
        self._inner.set_trajectory(
            _settle_pairs(state, call.predicted_state, self._robot_name)
        )

    def step(self, sim_state: ObjectCentricState) -> tuple[RealAction, SimAction]:
        if self._call is None:
            raise RuntimeError("SettleGapExecutor.step called with no trajectory")
        real_action, _ = self._inner.step(sim_state)
        return real_action, self._call

    def done(self, sim_state: ObjectCentricState) -> bool:
        return self._inner.done(sim_state)


class TeleopGapExecutor(PlanExecutor[SimAction, RealAction, ObjectCentricState]):
    """Hand a magic gap to a human teleoperator, then settle to the prediction.

    The first tick emits a :class:`TeleopHandoff` naming the skill; the env
    blocks on it until the operator reports done. Every later tick goes to a
    :class:`SettleGapExecutor` whose trajectory is built from the first state
    perceived after the hand-back, so the settle drives the robot from where
    the operator actually left it to the predicted base pose, arm
    configuration, and gripper state.
    """

    def __init__(
        self,
        base_executor: BaseMotion3DPlanExecutor | None = None,
        arm_executor: ArmMotion3DPlanExecutor | None = None,
        robot_name: str = "robot",
    ) -> None:
        self._settle = SettleGapExecutor(
            base_executor=base_executor,
            arm_executor=arm_executor,
            robot_name=robot_name,
        )
        self._call: SkillCall[ObjectCentricState] | None = None
        self._handed_off = False
        self._settling = False

    def set_trajectory(
        self, trajectory: Sequence[tuple[ObjectCentricState, SimAction]]
    ) -> None:
        if len(trajectory) != 1 or not isinstance(trajectory[0][1], SkillCall):
            raise ValueError(
                "TeleopGapExecutor expects a single (state, SkillCall) pair; got "
                f"{len(trajectory)} pair(s)."
            )
        self._call = trajectory[0][1]
        self._handed_off = False
        self._settling = False

    def step(self, sim_state: ObjectCentricState) -> tuple[RealAction, SimAction]:
        if self._call is None:
            raise RuntimeError("TeleopGapExecutor.step called with no trajectory")
        if not self._handed_off:
            self._handed_off = True
            return TeleopHandoff(prompt=self._handoff_prompt()), self._call
        self._start_settling(sim_state)
        return self._settle.step(sim_state)

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._call is None or not self._handed_off:
            return False
        self._start_settling(sim_state)
        return self._settle.done(sim_state)

    def _start_settling(self, sim_state: ObjectCentricState) -> None:
        """Build the settle trajectory from the first post-handoff state."""
        if self._settling:
            return
        assert self._call is not None
        self._settle.set_trajectory([(sim_state, self._call)])
        self._settling = True

    def _handoff_prompt(self) -> str:
        assert self._call is not None
        return (
            f"\nMAGIC GAP: {self._call}. The arm is released for teleoperation.\n"
            "Perform the skill with the gamepad, then press Enter to hand the "
            "arm back; the robot will then settle to the planned post-skill "
            "configuration: "
        )


def _settle_pairs(
    state: ObjectCentricState, predicted: ObjectCentricState, robot_name: str
) -> list[tuple[ObjectCentricState, SimAction]]:
    """Kinder-action pairs whose absolute targets are ``predicted``'s configuration.

    The sub-executors read each pair's target as ``state`` plus the action's
    delta, so the deltas are taken from ``state`` to ``predicted``. The
    gripper command closes when the predicted finger state is past the open
    threshold and opens otherwise.
    """
    robot = state.get_object_from_name(robot_name)
    predicted_robot = predicted.get_object_from_name(robot_name)
    pairs: list[tuple[ObjectCentricState, SimAction]] = []

    base_delta = np.array(
        [
            predicted.get(predicted_robot, f) - state.get(robot, f)
            for f in ("pos_base_x", "pos_base_y", "pos_base_rot")
        ],
        dtype=float,
    )
    if np.any(np.abs(base_delta) > _BASE_MOTION_EPS):
        base_action = np.zeros(11)
        base_action[0:3] = base_delta
        pairs.append((state, base_action))

    arm_action = np.zeros(11)
    arm_action[3:10] = [
        predicted.get(predicted_robot, f"joint_{j + 1}")
        - state.get(robot, f"joint_{j + 1}")
        for j in range(7)
    ]
    closed = predicted.get(predicted_robot, "finger_state") > GRIPPER_OPEN_THRESHOLD
    arm_action[10] = -1.0 if closed else 1.0
    pairs.append((state, arm_action))
    return pairs
