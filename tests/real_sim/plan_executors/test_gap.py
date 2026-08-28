"""Tests for the SettleGapExecutor."""

from typing import Sequence

import numpy as np
import pytest
from kinder_models.structs import SkillCall
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    StreamingArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.gap import SettleGapExecutor
from prpl_tidybot.structs import TidyBotObservation

_CARRY_JOINTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _l1_distance(q1: Sequence[float], q2: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.array(q1) - np.array(q2))))


def _make_state(
    *,
    base_xytheta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    arm_conf: list[float] | None = None,
    gripper: float = 0.0,
):
    obs = TidyBotObservation(
        arm_conf=arm_conf or [0.0] * 7,
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=base_xytheta[0], y=base_xytheta[1], theta=base_xytheta[2]),
        gripper=gripper,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


def _executor() -> SettleGapExecutor:
    return SettleGapExecutor(
        base_executor=PurePursuitBaseMotion3DPlanExecutor(position_tolerance=1e-3),
        arm_executor=StreamingArmMotion3DPlanExecutor(
            distance_fn=_l1_distance, arrival_tolerance=1e-3
        ),
    )


def _pick_call(pre_state, predicted) -> SkillCall:
    robot = pre_state.get_object_from_name("robot")
    return SkillCall("Pick", (robot,), np.array([0.8, 0.0]), predicted)


def test_settles_base_then_arm_to_predicted_configuration():
    """The gap executor first drives the base to the predicted pose, then the arm and
    gripper, reporting the SkillCall as the tracked sim action throughout."""
    s0 = _make_state()
    predicted = _make_state(
        base_xytheta=(1.0, 0.0, 0.0), arm_conf=_CARRY_JOINTS, gripper=0.6
    )
    call = _pick_call(s0, predicted)
    executor = _executor()
    executor.set_trajectory([(s0, call)])

    assert not executor.done(s0)
    real_action, sim_action = executor.step(s0)
    assert sim_action is call
    # Base phase: pure pursuit commands a lookahead along the path to x=1.
    assert real_action.base_pose_target_map.x == pytest.approx(0.2)
    assert real_action.arm_goal == pytest.approx([0.0] * 7)

    # Base arrived; the arm phase commands the predicted joints and a
    # closed gripper (the predicted finger state is past the open threshold).
    at_base = _make_state(base_xytheta=(1.0, 0.0, 0.0))
    assert not executor.done(at_base)
    real_action, sim_action = executor.step(at_base)
    assert sim_action is call
    assert real_action.arm_goal == pytest.approx(_CARRY_JOINTS)
    assert real_action.gripper_goal == 1.0
    assert real_action.base_pose_target_map.x == pytest.approx(1.0)

    assert executor.done(predicted)


def test_skips_base_phase_when_base_already_matches():
    """With no predicted base motion, the first tick already commands the arm."""
    s0 = _make_state()
    predicted = _make_state(arm_conf=_CARRY_JOINTS, gripper=0.0)
    executor = _executor()
    executor.set_trajectory([(s0, _pick_call(s0, predicted))])
    real_action, _ = executor.step(s0)
    assert real_action.arm_goal == pytest.approx(_CARRY_JOINTS)
    assert real_action.gripper_goal == 0.0
    assert executor.done(predicted)


def test_rejects_anything_but_a_single_skill_call():
    """The executor only knows how to settle one gap at a time."""
    s0 = _make_state()
    predicted = _make_state(arm_conf=_CARRY_JOINTS)
    call = _pick_call(s0, predicted)
    executor = _executor()
    with pytest.raises(ValueError, match="single"):
        executor.set_trajectory([(s0, call), (s0, call)])
    with pytest.raises(ValueError, match="single"):
        executor.set_trajectory([(s0, np.zeros(11))])
    with pytest.raises(RuntimeError, match="no trajectory"):
        executor.step(s0)
