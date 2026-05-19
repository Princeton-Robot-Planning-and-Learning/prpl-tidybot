"""Tests for structs.py."""

import numpy as np
import pytest
import spatialmath

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


def test_tidybot_observation():
    """Tests for TidyBotObservation()."""
    obs = TidyBotObservation(
        arm_conf=[0.0] * 7,
        base_pose=spatialmath.SE2(x=0, y=0, theta=0),
        map_base_pose=spatialmath.SE2(x=0, y=0, theta=0),
        gripper=0.0,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    assert np.allclose(obs.arm_conf, [0.0] * 7)
    assert np.allclose(obs.base_pose.A, spatialmath.SE2(x=0, y=0, theta=0).A)
    assert np.isclose(obs.gripper, 0.0)
    with pytest.raises(AssertionError):
        TidyBotObservation(
            arm_conf=[0.0] * 7,
            base_pose=spatialmath.SE2(x=0, y=0, theta=0),
            map_base_pose=spatialmath.SE2(x=0, y=0, theta=0),
            gripper=0.0,
            wrist_camera=np.zeros((1, 1, 1), dtype=np.uint8),
            base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
        )


def test_tidybot_action():
    """Tests for TidyBotAction()."""
    arm_goal = [1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.2]
    base_pose_target_map = spatialmath.SE2(x=1.0, y=-2.0, theta=0.5)
    action = TidyBotAction(
        arm_goal=arm_goal,
        base_pose_target_map=base_pose_target_map,
        gripper_goal=1.0,
    )
    assert np.allclose(action.arm_goal, arm_goal)
    assert np.allclose(action.base_pose_target_map.A, base_pose_target_map.A)
    assert action.gripper_goal == 1.0
