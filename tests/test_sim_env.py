"""Tests for sim_env.py."""

import numpy as np
import pytest

from prpl_tidybot.sim_env import PrplLab3DSimEnv


@pytest.fixture(name="env")
def _env():
    env = PrplLab3DSimEnv()
    yield env
    env.close()


def test_reset_returns_devectorized_state(env):
    """Reset() returns an ObjectCentricState (not a raw numpy vector) plus info dict,
    with the robot at the env's home pose."""
    state, info = env.reset(seed=0)
    robot = state.get_object_from_name("robot")
    assert state.get(robot, "pos_base_x") == pytest.approx(0.3)
    assert state.get(robot, "pos_base_y") == pytest.approx(0.0)
    assert state.get(robot, "pos_base_rot") == pytest.approx(np.pi / 2)
    assert isinstance(info, dict)


def test_step_returns_devectorized_state(env):
    """Step() routes through the wrapped kinder env and returns a devectorized state
    whose robot has advanced by the commanded delta."""
    state, _ = env.reset(seed=0)
    robot = state.get_object_from_name("robot")
    base_x_0 = state.get(robot, "pos_base_x")

    action = np.zeros(11)
    action[0] = 0.05
    state, reward, terminated, truncated, info = env.step(action)
    base_x_1 = state.get(robot, "pos_base_x")

    assert base_x_1 == pytest.approx(base_x_0 + 0.05)
    assert float(reward) == reward  # any SupportsFloat is fine
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
