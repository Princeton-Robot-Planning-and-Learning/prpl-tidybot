"""Tests for control/arm_movement.py."""

from spatialmath import SE2

from prpl_tidybot.control.arm_movement import reach_target_arm_conf
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.structs import TidyBotAction


def test_reach_target_arm_conf_converges():
    """Executing the target arm conf before polling makes the helper return True on the
    first iteration."""
    interface = FakeInterface()
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    interface.execute_arm_action(
        TidyBotAction(
            arm_goal=target,
            base_local_goal=SE2(x=0, y=0, theta=0),
            gripper_goal=0.0,
        )
    )
    assert reach_target_arm_conf(interface, target, max_iter=5, control_period=0.0)


def test_reach_target_arm_conf_times_out_without_execute():
    """Without executing, the fake's arm stays at zeros and the helper exhausts
    max_iter."""
    interface = FakeInterface()
    target = [0.5] * 7
    assert not reach_target_arm_conf(interface, target, max_iter=3, control_period=0.0)
