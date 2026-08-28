"""Tests for sim_env.py."""

import numpy as np
import pytest
from kinder_models.structs import SkillCall

from prpl_tidybot.sim_env import KinderSimEnv


@pytest.fixture(name="env")
def _env():
    env = KinderSimEnv("kinder/PrplLab3D-o1-v0")
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


def test_step_with_skill_call_teleports_to_predicted_state():
    """A SkillCall action is carried out by teleporting the sim to the call's predicted
    state, with no reward and no simulation."""
    env = KinderSimEnv("kinder/PrplLab3D-o1-v0", allow_state_access=True)
    try:
        state, _ = env.reset(seed=0)
        robot = state.get_object_from_name("robot")
        predicted = state.copy()
        predicted.set(robot, "pos_base_x", 1.5)
        call = SkillCall("Pick", (robot,), np.array([0.8, 0.0]), predicted)
        next_state, reward, terminated, truncated, _ = env.step(call)
        assert next_state.get(robot, "pos_base_x") == pytest.approx(1.5)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
    finally:
        env.close()
