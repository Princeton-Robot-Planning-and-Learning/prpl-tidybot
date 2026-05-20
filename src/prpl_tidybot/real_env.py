"""Gymnasium environment for the real TidyBot++."""

import time
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import RenderFrame

from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import Interface
from prpl_tidybot.rendering import Renderer
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++.

    `step` issues one command to each sub-interface and returns a fresh
    observation after `control_period`. Re-projection of the action's map-frame
    base target into the base controller's odom frame happens here because the
    map/odom transform is a property of this env's perception pipeline (marker
    detector + base controller). Trajectory tracking — including convergence
    tolerances and any inner settle loop — lives in the configured
    `PlanExecutor`, not here.

    Reward is always 0 and terminated / truncated are always False: the real
    environment has no task semantics. Convergence is the executor's call.
    """

    def __init__(
        self,
        interface: Interface,
        control_period: float = POLICY_CONTROL_PERIOD,
        renderer: Renderer | None = None,
    ) -> None:
        self._interface = interface
        self._control_period = control_period
        self._converter: CoordFrameConverter | None = None
        self._renderer = renderer

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TidyBotObservation, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._interface.get_observation()
        self._converter = CoordFrameConverter(obs.map_base_pose, obs.base_pose)
        return obs, {}

    def step(
        self, action: TidyBotAction
    ) -> tuple[TidyBotObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._converter is None:
            raise RuntimeError("RealTidyBotEnv.step called before reset")
        obs = self._interface.get_observation()
        self._converter.update(obs.map_base_pose, obs.base_pose)
        target_odom = self._converter.convert_pose(action.base_pose_target_map)
        self._interface.base_interface.execute_action(target_odom)
        self._interface.arm_interface.execute_action(action.arm_goal)
        self._interface.arm_interface.execute_gripper_action(action.gripper_goal)
        time.sleep(self._control_period)
        obs = self._interface.get_observation()
        return obs, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self._renderer is not None:
            return self._renderer.render()
        return self._interface.get_base_image()
