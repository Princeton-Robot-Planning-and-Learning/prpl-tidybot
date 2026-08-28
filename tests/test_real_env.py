"""Tests for real_env.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.structs import TeleopHandoff, TidyBotAction


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


def test_step_issues_action_targets_and_returns_fresh_obs():
    """Step() issues one command to each sub-interface and returns the observation taken
    after `control_period`.

    Convergence (settle loops, tolerances, max_iter) lives in the PlanExecutor — not
    here — so with FakeInterface the post-command observation already reflects the
    commanded state. Reward / terminated / truncated are fixed.
    """
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


def test_render_routes_through_renderer_when_provided():
    """When a renderer is supplied, render() returns its frame instead of the base
    camera image; without one, it falls back to interface.get_base_image()."""

    class _StubRenderer:
        """Returns a constant frame; tracks no state besides the stored buffer."""

        def __init__(self) -> None:
            self.frame = np.full((4, 4, 3), 7, dtype=np.uint8)

        def render(self) -> np.ndarray:
            """Return the stored frame."""
            return self.frame

        def close(self) -> None:
            """No resources to release."""

    renderer = _StubRenderer()
    env = RealTidyBotEnv(
        FakeInterface(),
        control_period=0.0,
        renderer=renderer,
    )
    frame = env.render()
    assert frame is renderer.frame

    # No renderer -> fall back to the (fake) base camera image.
    fallback = RealTidyBotEnv(FakeInterface(), control_period=0.0).render()
    assert isinstance(fallback, np.ndarray)
    assert fallback.shape == (360, 640, 3)  # BASE_CAMERA_DIMS


class _HandoffCountingArm(FakeArmInterface):
    """Fake arm that records release / resume calls."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def release(self) -> None:
        self.events.append("release")

    def resume(self) -> None:
        self.events.append("resume")


def test_teleop_handoff_releases_prompts_and_resumes():
    """A TeleopHandoff action releases the arm, blocks on the prompt, re-acquires the
    arm, and returns the observation taken after the hand-back."""
    interface = FakeInterface()
    arm = _HandoffCountingArm()
    interface.arm_interface = arm
    prompts: list[str] = []

    def _prompt(msg: str) -> str:
        prompts.append(msg)
        # The operator "moved" the arm while it was released.
        assert arm.events == ["release"]
        arm.arm_state = [0.5] * 7
        arm.gripper_state = 1.0
        return ""

    env = RealTidyBotEnv(interface, control_period=0.0, prompt_fn=_prompt)
    env.reset()
    obs, reward, terminated, truncated, _ = env.step(
        TeleopHandoff(prompt="go", countdown_seconds=0.0)
    )

    assert prompts == ["go"]
    assert arm.events == ["release", "resume"]
    assert np.allclose(obs.arm_conf, [0.5] * 7)
    assert obs.gripper == 1.0
    assert reward == 0.0 and not terminated and not truncated
    # The next regular step re-calibrates from the post-handoff observation.
    obs2, _, _, _, _ = env.step(
        TidyBotAction(
            arm_goal=[0.1] * 7,
            base_pose_target_map=SE2(x=0.0, y=0.0, theta=0.0),
            gripper_goal=0.0,
        )
    )
    assert np.allclose(obs2.arm_conf, [0.1] * 7)


def test_teleop_handoff_resumes_even_if_prompt_raises():
    """The arm is always re-acquired, even when the operator aborts the prompt."""
    interface = FakeInterface()
    arm = _HandoffCountingArm()
    interface.arm_interface = arm

    def _abort(_msg: str) -> str:
        raise KeyboardInterrupt

    env = RealTidyBotEnv(interface, control_period=0.0, prompt_fn=_abort)
    env.reset()
    with pytest.raises(KeyboardInterrupt):
        env.step(TeleopHandoff(prompt="go"))
    assert arm.events == ["release", "resume"]
