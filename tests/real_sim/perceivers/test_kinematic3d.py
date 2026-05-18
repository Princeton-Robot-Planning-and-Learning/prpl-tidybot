"""Tests for real_sim/perceivers/kinematic3d.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.structs import TidyBotObservation


def _make_obs() -> TidyBotObservation:
    return TidyBotObservation(
        arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=2.0, y=1.0, theta=-0.4),
        gripper=0.6,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )


def test_step_populates_robot_features():
    """The perceiver lifts obs.{map_base_pose, arm_conf, gripper} into the kinematic3d
    robot feature dict; grasp_active is 0 and grasp_tf is identity."""
    perceiver = PrplLab3DPerceiver()
    state = perceiver.step(_make_obs(), {})
    robot = state.get_object_from_name("robot")
    assert state.get(robot, "pos_base_x") == 2.0
    assert state.get(robot, "pos_base_y") == 1.0
    assert state.get(robot, "pos_base_rot") == pytest.approx(-0.4)
    assert state.get(robot, "joint_1") == 0.1
    assert state.get(robot, "joint_7") == 0.7
    assert state.get(robot, "finger_state") == 0.6
    assert state.get(robot, "grasp_active") == 0.0
    assert state.get(robot, "grasp_tf_qw") == 1.0
    for f in ("grasp_tf_x", "grasp_tf_y", "grasp_tf_z", "grasp_tf_qx"):
        assert state.get(robot, f) == 0.0


def test_reset_matches_step():
    """Reset() and step() are equivalent for this stateless perceiver."""
    perceiver = PrplLab3DPerceiver()
    obs = _make_obs()
    assert perceiver.reset(obs, {}) == perceiver.step(obs, {})
