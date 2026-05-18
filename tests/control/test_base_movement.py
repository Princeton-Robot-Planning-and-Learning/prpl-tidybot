"""Tests for control/base_movement.py."""

from spatialmath import SE2

from prpl_tidybot.control.base_movement import reach_target_pose
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.structs import TidyBotAction


def test_reach_target_pose_converges():
    """Executing the target before polling makes the helper return True on the first
    iteration (FakeInterface stores commanded poses)."""
    interface = FakeInterface()
    target = SE2(x=1.0, y=-0.5, theta=0.2)
    interface.execute_base_action(
        TidyBotAction(arm_goal=[0.0] * 7, base_local_goal=target, gripper_goal=0.0)
    )
    assert reach_target_pose(interface, target, max_iter=5, control_period=0.0)


def test_reach_target_pose_times_out_without_execute():
    """Without executing the action, the fake's base stays at the origin and the helper
    exhausts max_iter."""
    interface = FakeInterface()
    target = SE2(x=2.0, y=2.0, theta=0.0)
    assert not reach_target_pose(interface, target, max_iter=3, control_period=0.0)
