"""Tests for arm_interface.py."""

import numpy as np

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface


def test_fake_arm_interface_defaults():
    """FakeArmInterface() starts at a zero arm conf and closed gripper."""
    arm = FakeArmInterface()
    assert np.allclose(arm.get_arm_state(), [0.0] * 7)
    assert arm.get_gripper_state() == 0.0


def test_fake_arm_interface_execute_action_sets_arm_and_gripper_atomically():
    """A single execute_action call stores both the joint conf and gripper value.

    The atomic-pair API replaced separate execute_action / execute_gripper_action
    methods because the latter pair, when called back-to-back against the Kinova
    controller, had the gripper call overwrite the arm call's `qpos`.
    """
    arm = FakeArmInterface()
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    arm.execute_action(target, 1.0)
    assert np.allclose(arm.get_arm_state(), target)
    assert arm.get_gripper_state() == 1.0
    arm.execute_action(target, 0.3)
    assert arm.get_gripper_state() == 0.3
