"""Tests for arm_interface.py."""

import numpy as np

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface


def test_fake_arm_interface_defaults():
    """FakeArmInterface() starts at a zero arm conf and closed gripper."""
    arm = FakeArmInterface()
    assert np.allclose(arm.get_arm_state(), [0.0] * 7)
    assert arm.get_gripper_state() == 0.0


def test_fake_arm_interface_execute_action():
    """execute_action() stores the commanded joint conf."""
    arm = FakeArmInterface()
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    arm.execute_action(target)
    assert np.allclose(arm.get_arm_state(), target)


def test_fake_arm_interface_execute_gripper_action():
    """execute_gripper_action() stores the commanded gripper value."""
    arm = FakeArmInterface()
    arm.execute_gripper_action(1.0)
    assert arm.get_gripper_state() == 1.0
    arm.execute_gripper_action(0.3)
    assert arm.get_gripper_state() == 0.3
