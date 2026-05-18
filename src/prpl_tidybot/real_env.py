"""Gymnasium environment for the real TidyBot++."""

from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import RenderFrame

from prpl_tidybot.interfaces.interface import Interface
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++.

    Each step issues the action's base, arm, and gripper components via the
    Interface and returns the resulting observation. Reward is always 0 and
    terminated / truncated are always False: the real environment has no
    task semantics, and waiting-for-motion-to-settle is the Interface's
    job (FakeInterface stores commanded values immediately; a real-hardware
    Interface should block in execute_*_action or expose its own wait API).
    """

    def __init__(self, interface: Interface) -> None:
        self._interface = interface

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TidyBotObservation, dict[str, Any]]:
        super().reset(seed=seed)
        return self._interface.get_observation(), {}

    def step(
        self, action: TidyBotAction
    ) -> tuple[TidyBotObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        self._interface.execute_action(action)
        return self._interface.get_observation(), 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None
