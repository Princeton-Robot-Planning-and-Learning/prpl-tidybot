"""Perceiver for the kinder/PrplLab3D-o{1,2}-v0 (kinematic3d) env."""

from typing import Any

from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.prpl3d import PrplLab3DEnvConfig
from prpl_utils.real_sim import Perceiver
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.structs import TidyBotObservation

_DEFAULT_CUBE_HALF_EXTENTS = PrplLab3DEnvConfig().block_half_extents


class PrplLab3DPerceiver(Perceiver[TidyBotObservation, ObjectCentricState]):
    """Build a PrplLab3D ObjectCentricState from a TidyBotObservation.

    Uses the Kinematic3DEnvTypeFeatures schema. No velocity fields (the env
    is kinematic); a `grasp_active` / `grasp_tf_*` block tracks an
    optionally-grasped object — reported as "nothing grasped" here.
    Non-robot detection currently returns placeholder cube(s) at the
    origin; replace when real perception lands.
    """

    def __init__(self, robot_name: str = "robot", num_cubes: int = 1) -> None:
        self._robot_name = robot_name
        self._num_cubes = num_cubes

    def reset(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        return self._build_state(obs, info)

    def step(self, obs: TidyBotObservation, info: dict[str, Any]) -> ObjectCentricState:
        return self._build_state(obs, info)

    def _build_state(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        del info  # currently unused
        state_dict: dict[Object, dict[str, float]] = {}
        robot = Object(self._robot_name, Kinematic3DRobotType)
        state_dict[robot] = {
            "pos_base_x": obs.map_base_pose.x,
            "pos_base_y": obs.map_base_pose.y,
            "pos_base_rot": obs.map_base_pose.theta(),
            "joint_1": obs.arm_conf[0],
            "joint_2": obs.arm_conf[1],
            "joint_3": obs.arm_conf[2],
            "joint_4": obs.arm_conf[3],
            "joint_5": obs.arm_conf[4],
            "joint_6": obs.arm_conf[5],
            "joint_7": obs.arm_conf[6],
            "finger_state": obs.gripper,
            "grasp_active": 0.0,
            "grasp_tf_x": 0.0,
            "grasp_tf_y": 0.0,
            "grasp_tf_z": 0.0,
            "grasp_tf_qx": 0.0,
            "grasp_tf_qy": 0.0,
            "grasp_tf_qz": 0.0,
            "grasp_tf_qw": 1.0,
        }
        hx, hy, hz = _DEFAULT_CUBE_HALF_EXTENTS
        for i in range(self._num_cubes):
            cube = Object(f"cube{i}", Kinematic3DCuboidType)
            state_dict[cube] = {
                "pose_x": 0.0,
                "pose_y": 0.0,
                "pose_z": 0.0,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "pose_qw": 1.0,
                "grasp_active": 0.0,
                "object_type": 0.0,
                "half_extent_x": hx,
                "half_extent_y": hy,
                "half_extent_z": hz,
            }
        return create_state_from_dict(state_dict, Kinematic3DEnvTypeFeatures)
