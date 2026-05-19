"""Tests for real_sim/action_grounders/kinematic3d.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.action_grounders.kinematic3d import (
    Kinematic3DActionGrounder,
)
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.structs import TidyBotObservation


def _make_state(
    *,
    base_xytheta: tuple[float, float, float] = (1.0, 2.0, 0.5),
    arm_conf: list[float] | None = None,
    gripper: float = 0.4,
):
    """Build a sim state by running the perceiver on a hand-built obs."""
    obs = TidyBotObservation(
        arm_conf=arm_conf or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=base_xytheta[0], y=base_xytheta[1], theta=base_xytheta[2]),
        gripper=gripper,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


def test_base_delta_becomes_absolute_target():
    """Base components of the sim action add componentwise to the current world pose."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.1
    action[1] = -0.2
    action[2] = 0.05

    real_action = Kinematic3DActionGrounder()(action, state)

    assert real_action.base_pose_target_map.x == pytest.approx(1.1)
    assert real_action.base_pose_target_map.y == pytest.approx(1.8)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.55)


def test_arm_delta_summed_with_current_joints():
    """Arm deltas at action[3..10] are added per-joint to the current arm conf."""
    state = _make_state(arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.zeros(11)
    action[3:10] = [0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.04]

    real_action = Kinematic3DActionGrounder()(action, state)

    expected = [0.11, 0.22, 0.33, 0.39, 0.48, 0.57, 0.74]
    assert real_action.arm_goal == pytest.approx(expected)


def test_gripper_close_command():
    """Gripper command <-0.5 becomes TidyBotAction.gripper_goal=1.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = -1.0
    real_action = Kinematic3DActionGrounder()(action, state)
    assert real_action.gripper_goal == 1.0


def test_gripper_open_command():
    """Gripper command >0.5 becomes TidyBotAction.gripper_goal=0.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 1.0
    real_action = Kinematic3DActionGrounder()(action, state)
    assert real_action.gripper_goal == 0.0


def test_gripper_no_change_passes_through_current():
    """Gripper command in [-0.5, 0.5] passes through the current finger_state."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 0.0
    real_action = Kinematic3DActionGrounder()(action, state)
    assert real_action.gripper_goal == pytest.approx(0.4)
