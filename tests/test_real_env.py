"""Tests for real_env.py."""

import numpy as np
from spatialmath import SE2

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.structs import TidyBotAction


def test_real_tidybot_env_reset():
    """Reset() returns the current observation from the underlying Interface."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.base_state = SE2(x=1.0, y=0.0, theta=0.0)
    env = RealTidyBotEnv(interface)
    obs, info = env.reset()
    assert np.allclose(obs.arm_conf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(obs.base_pose.A, SE2(x=1.0, y=0.0, theta=0.0).A)
    assert not info


def test_real_tidybot_env_step_executes_action():
    """Step() issues each component of the TidyBotAction via the Interface and returns
    the resulting observation; reward / terminated / truncated are fixed."""
    interface = FakeInterface()
    env = RealTidyBotEnv(interface)
    env.reset()
    arm_goal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    base_goal = SE2(x=2.0, y=-1.0, theta=0.4)
    action = TidyBotAction(
        arm_goal=arm_goal,
        base_local_goal=base_goal,
        gripper_goal=0.8,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    assert np.allclose(obs.arm_conf, arm_goal)
    assert np.allclose(obs.base_pose.A, base_goal.A)
    assert obs.gripper == 0.8
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert not info
