"""Tests for control/gripper_movement.py."""

from spatialmath import SE2

from prpl_tidybot.control.gripper_movement import reach_target_gripper
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.structs import TidyBotAction


def test_reach_target_gripper_converges():
    """Executing the target gripper value before polling makes the helper return True on
    the first iteration."""
    interface = FakeInterface()
    target = 0.6
    interface.execute_gripper_action(
        TidyBotAction(
            arm_goal=[0.0] * 7,
            base_local_goal=SE2(x=0, y=0, theta=0),
            gripper_goal=target,
        )
    )
    assert reach_target_gripper(interface, target, max_iter=5, control_period=0.0)


def test_reach_target_gripper_times_out_without_execute():
    """Without executing, the fake's gripper stays at 0 and the helper exhausts max_iter
    against a non-zero target."""
    interface = FakeInterface()
    assert not reach_target_gripper(interface, 1.0, max_iter=3, control_period=0.0)
