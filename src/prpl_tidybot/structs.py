"""Data structures."""

from dataclasses import dataclass

import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS


@dataclass(frozen=True)
class TidyBotObservation:
    """Raw observations from the TidyBot environment."""

    arm_conf: list[float]  # 7-DOF joints
    base_pose: spatialmath.SE2  # base pose for the robot
    map_base_pose: spatialmath.SE2  # base pose for the robot in the map frame
    gripper: float  # 1 = closed, 0 = open
    wrist_camera: Image  # see WRIST_CAMERA_DIMS
    base_camera: Image  # see BASE_CAMERA_DIMS

    def __post_init__(self) -> None:
        assert self.wrist_camera.shape == WRIST_CAMERA_DIMS
        assert self.base_camera.shape == BASE_CAMERA_DIMS


@dataclass(frozen=True)
class TidyBotAction:
    """Low-level joint and base commands for the real TidyBot environment."""

    arm_goal: list[float]  # absolute arm position
    base_pose_target_map: spatialmath.SE2  # absolute base pose target in the map frame
    gripper_goal: float  # absolute
