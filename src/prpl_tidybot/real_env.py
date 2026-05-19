"""Gymnasium environment for the real TidyBot++."""

import math
import time
from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import RenderFrame

from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import Interface
from prpl_tidybot.rendering import Renderer
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD


class RealTidyBotEnv(gymnasium.Env[TidyBotObservation, TidyBotAction]):
    """Gymnasium environment for the real TidyBot++.

    `step` closed-loops over the action's absolute targets until they are all
    within tolerance or `max_iter` is hit. Each inner iteration refreshes the
    observation, recalibrates the map/odom converter from the new observation
    pair, re-projects the base's map-frame target into the odom frame, and re-
    issues commands to each sub-interface. Re-projecting on every tick is the
    load-bearing trick: the convergence criterion is in the map frame but the
    base controller takes odom-frame commands, and the map/odom transform
    drifts as the marker detector reports fresh readings.

    Reward is always 0 and terminated / truncated are always False: the real
    environment has no task semantics. Convergence is what `step` returns on.
    """

    def __init__(
        self,
        interface: Interface,
        position_tolerance: float = 0.01,
        angle_tolerance: float = 0.01,
        joint_tolerance: float = 0.05,
        gripper_tolerance: float = 0.05,
        max_iter: int = 100,
        control_period: float = POLICY_CONTROL_PERIOD,
        renderer: Renderer | None = None,
    ) -> None:
        self._interface = interface
        self._position_tolerance = position_tolerance
        self._angle_tolerance = angle_tolerance
        self._joint_tolerance = joint_tolerance
        self._gripper_tolerance = gripper_tolerance
        self._max_iter = max_iter
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
        for _ in range(self._max_iter):
            self._converter.update(obs.map_base_pose, obs.base_pose)
            target_odom = self._converter.convert_pose(action.base_pose_target_map)
            self._interface.base_interface.execute_action(target_odom)
            self._interface.arm_interface.execute_action(action.arm_goal)
            self._interface.arm_interface.execute_gripper_action(action.gripper_goal)
            time.sleep(self._control_period)
            obs = self._interface.get_observation()
            if self._converged(obs, action):
                break
        return obs, 0.0, False, False, {}

    def _converged(self, obs: TidyBotObservation, action: TidyBotAction) -> bool:
        target_map = action.base_pose_target_map
        position_err = float(
            np.linalg.norm(
                [
                    obs.map_base_pose.x - target_map.x,
                    obs.map_base_pose.y - target_map.y,
                ]
            )
        )
        if position_err > self._position_tolerance:
            return False
        angle_err = abs(_wrap_angle(obs.map_base_pose.theta() - target_map.theta()))
        if angle_err > self._angle_tolerance:
            return False
        if any(
            abs(obs.arm_conf[i] - action.arm_goal[i]) > self._joint_tolerance
            for i in range(7)
        ):
            return False
        if abs(obs.gripper - action.gripper_goal) > self._gripper_tolerance:
            return False
        return True

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self._renderer is not None:
            return self._renderer.render()
        return self._interface.get_base_image()


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
