"""Tests for real_env.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.structs import TidyBotAction


def test_real_tidybot_env_reset():
    """Reset() returns the current observation from the underlying Interface."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.base_state = SE2(x=1.0, y=0.0, theta=0.0)
    env = RealTidyBotEnv(interface, control_period=0.0)
    obs, info = env.reset()
    assert np.allclose(obs.arm_conf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(obs.base_pose.A, SE2(x=1.0, y=0.0, theta=0.0).A)
    assert not info


def test_step_raises_if_called_before_reset():
    """The converter is initialized in reset(); step() before reset() should raise."""
    env = RealTidyBotEnv(FakeInterface(), control_period=0.0)
    action = TidyBotAction(
        arm_goal=[0.0] * 7,
        base_pose_target_map=SE2(x=0.0, y=0.0, theta=0.0),
        gripper_goal=0.0,
    )
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)


def test_real_tidybot_env_step_converges_to_action_targets():
    """Step() runs the convergence loop until all components match within tolerance and
    returns the resulting observation; reward / terminated / truncated are fixed."""
    interface = FakeInterface()
    env = RealTidyBotEnv(interface, control_period=0.0)
    env.reset()
    arm_goal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    base_goal = SE2(x=2.0, y=-1.0, theta=0.4)
    action = TidyBotAction(
        arm_goal=arm_goal,
        base_pose_target_map=base_goal,
        gripper_goal=0.8,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    assert np.allclose(obs.arm_conf, arm_goal)
    assert np.allclose(obs.map_base_pose.A, base_goal.A)
    assert obs.gripper == 0.8
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert not info


def test_step_returns_after_max_iter_when_target_unreachable():
    """When a base sub-interface that ignores commands prevents convergence, step()
    bails after max_iter rather than looping forever."""

    class _StuckBase(FakeBaseInterface):
        def execute_action(self, action: SE2) -> None:
            del action

    interface = FakeInterface()
    interface.base_interface = _StuckBase()
    env = RealTidyBotEnv(interface, control_period=0.0, max_iter=3)
    env.reset()
    action = TidyBotAction(
        arm_goal=[0.0] * 7,
        base_pose_target_map=SE2(x=1.0, y=0.0, theta=0.0),
        gripper_goal=0.0,
    )

    obs, _, _, _, _ = env.step(action)

    assert obs.map_base_pose.x == 0.0
