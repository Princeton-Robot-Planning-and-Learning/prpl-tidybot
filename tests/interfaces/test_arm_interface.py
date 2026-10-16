"""Tests for arm_interface.py."""

from unittest import mock

import numpy as np
import pytest

from prpl_tidybot.interfaces.arm_interface import ArmMotionAborted, FakeArmInterface
from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface


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


def test_fake_arm_interface_toss_records_the_call_and_leaves_the_gripper_open():
    """A fake toss records its parameters and ends with the gripper open (0.0), as
    the real toss does once the gripper has released the object."""
    arm = FakeArmInterface()
    arm.execute_action([0.1] * 7, 1.0)
    arm.toss(heading=15.0, max_vel=60.0, gripper_release_ms=900)
    assert arm.toss_calls == [
        {"heading": 15.0, "max_vel": 60.0, "gripper_release_ms": 900}
    ]
    assert arm.get_gripper_state() == 0.0


def test_fake_arm_interface_toss_defaults_are_jimmys():
    """With no arguments the fake records Jimmy's heading-0, 140 deg/s, 600 ms toss."""
    arm = FakeArmInterface()
    arm.toss()
    assert arm.toss_calls == [
        {"heading": 0.0, "max_vel": 140.0, "gripper_release_ms": 600}
    ]


def test_fake_arm_interface_toss_rejects_what_the_real_arm_would():
    """The fake validates the swing like the server does, so a bad speed fails in
    fake mode before it can reach the robot."""
    arm = FakeArmInterface()
    with pytest.raises(ValueError):
        arm.toss(max_vel=200.0)
    assert not arm.toss_calls


def test_real_arm_interface_toss_forwards_to_the_server():
    """RealArmInterface.toss is one RPC to the arm server with the same arguments."""
    arm = RealArmInterface.__new__(RealArmInterface)
    arm.arm = mock.Mock()
    arm.toss(heading=-10.0, max_vel=80.0, gripper_release_ms=700)
    arm.arm.toss.assert_called_once_with(
        heading=-10.0, max_vel=80.0, gripper_release_ms=700
    )


def test_fake_arm_interface_angular_moves_only_while_released():
    """Like the real arm, the onboard-controller moves need the torque loop released;
    they land on the goal (plus any configured offset) and move the gripper."""
    arm = FakeArmInterface()
    with pytest.raises(RuntimeError, match="release"):
        arm.move_angular([0.1] * 7)
    arm.release()
    assert np.allclose(arm.move_angular([0.1] * 7), [0.1] * 7)
    assert arm.set_gripper_angular(1.0) == 1.0
    arm.angular_offset = 0.02
    assert np.allclose(arm.move_angular([0.2] * 7), [0.22] * 7)
    arm.resume()
    with pytest.raises(RuntimeError, match="release"):
        arm.set_gripper_angular(0.0)


@pytest.mark.parametrize("call", ["move_angular", "set_gripper_angular"])
def test_ctrl_c_during_an_onboard_move_stops_the_arm(monkeypatch, call):
    """The onboard controller keeps moving after the client dies, so Ctrl-C sends a
    Stop to the arm over a fresh Kortex session, then raises ArmMotionAborted."""
    stops = []
    monkeypatch.setattr(
        "prpl_tidybot.interfaces.real_arm_interface.stop_arm_motion",
        lambda: stops.append(1),
    )
    arm = RealArmInterface.__new__(RealArmInterface)
    arm.arm = mock.Mock()
    arm.arm.move_joints_angular.side_effect = KeyboardInterrupt
    arm.arm.gripper_angular.side_effect = KeyboardInterrupt
    with pytest.raises(ArmMotionAborted):
        if call == "move_angular":
            arm.move_angular([0.0] * 7)
        else:
            arm.set_gripper_angular(1.0)
    assert stops == [1]


@pytest.mark.parametrize("left_running", [True, False])
def test_connecting_warns_when_a_previous_client_left_the_torque_loop_running(
    monkeypatch, caplog, left_running
):
    """A loop already running when a client connects means the last client died without
    close(); say so before reset() takes the loop over."""
    server_arm = mock.Mock()
    server_arm.is_cyclic_running.return_value = left_running
    manager = mock.Mock()
    manager.Arm.return_value = server_arm
    monkeypatch.setattr(
        "prpl_tidybot.interfaces.real_arm_interface.ArmManager",
        lambda **kwargs: manager,
    )
    with caplog.at_level("WARNING"):
        RealArmInterface(reset_arm=False)
    server_arm.reset.assert_called_once_with(reset_arm=False)
    warned = any("already running" in r.getMessage() for r in caplog.records)
    assert warned == left_running
