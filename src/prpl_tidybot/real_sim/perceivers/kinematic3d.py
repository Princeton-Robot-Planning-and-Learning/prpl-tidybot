"""Perceivers for kinematic3d kinder envs.

Each subclass turns a `TidyBotObservation` into an `ObjectCentricState`
that matches a particular kinder env's type schema. The robot
proprioception is identical across kinematic3d envs (same
`Kinematic3DRobotType` features), so it lives on the base class; only
the non-robot object detection differs per env.
"""

import abc
from typing import Any

from kinder.envs.kinematic3d.base_motion3d import BaseMotion3DObjectCentricState
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CYLINDER_OBJECT_TYPE,
    CylinderShelf3DEnvConfig,
    CylinderShelf3DObjectCentricState,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DFixtureType,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.prpl3d import (
    PrplLab3DEnvConfig,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.shelf3d import (
    Shelf3DEnvConfig,
    Shelf3DObjectCentricState,
)
from prpl_utils.real_sim import Perceiver
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.real_sim.perceivers.target_source import TargetSource
from prpl_tidybot.structs import TidyBotObservation

_DEFAULT_CUBE_HALF_EXTENTS = PrplLab3DEnvConfig().block_half_extents
_DEFAULT_SHELF_CUBE_HALF_EXTENTS = Shelf3DEnvConfig().block_half_extents
_DEFAULT_CYLINDER_SHELF_CONFIG = CylinderShelf3DEnvConfig()


class KinematicRobotPerceiverBase(
    Perceiver[TidyBotObservation, ObjectCentricState], abc.ABC
):
    """Build a kinematic3d ObjectCentricState from a TidyBotObservation.

    Subclasses implement `_detect_objects` to add non-robot scene content
    (placeholder cubes, target points, etc.) on top of the shared robot
    proprioception this class writes.
    """

    def __init__(self, robot_name: str = "robot") -> None:
        self._robot_name = robot_name

    @property
    @abc.abstractmethod
    def _state_cls(self) -> type[ObjectCentricState]:
        """Env-specific ObjectCentricState subclass to construct.

        Bilevel-planning env models for each kinder env assert that the
        state is an instance of the env's specific state class (e.g.
        `BaseMotion3DObjectCentricState`), so we have to construct that
        same subclass here rather than a plain ObjectCentricState.
        """

    @abc.abstractmethod
    def _detect_objects(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> dict[Object, dict[str, float]]:
        """Return the non-robot objects in the scene with their features.

        Currently returns hardcoded placeholders; replace when real
        perception lands.
        """

    def reset(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        return self._build_state(obs, info)

    def step(self, obs: TidyBotObservation, info: dict[str, Any]) -> ObjectCentricState:
        return self._build_state(obs, info)

    def _build_state(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}
        robot = Object(self._robot_name, Kinematic3DRobotType)
        state_dict[robot] = self._build_robot_features(obs)
        state_dict.update(self._detect_objects(obs, info))
        return create_state_from_dict(
            state_dict, Kinematic3DEnvTypeFeatures, state_cls=self._state_cls
        )

    def _build_robot_features(self, obs: TidyBotObservation) -> dict[str, float]:
        # Real-side grasp tracking lands with real perception; report
        # "nothing grasped" for now.
        return {
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


class PrplLab3DPerceiver(KinematicRobotPerceiverBase):
    """Perceiver for kinder/PrplLab3D-o{1,2}-v0.

    Non-robot detection currently emits placeholder cubes at the origin
    with the env's default half-extents.
    """

    def __init__(self, robot_name: str = "robot", num_cubes: int = 1) -> None:
        super().__init__(robot_name=robot_name)
        self._num_cubes = num_cubes

    @property
    def _state_cls(self) -> type[ObjectCentricState]:
        return PrplLab3DObjectCentricState

    def _detect_objects(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> dict[Object, dict[str, float]]:
        del obs, info
        hx, hy, hz = _DEFAULT_CUBE_HALF_EXTENTS
        return {
            Object(f"cube{i}", Kinematic3DCuboidType): {
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
            for i in range(self._num_cubes)
        }


class BaseMotion3DPerceiver(KinematicRobotPerceiverBase):
    """Perceiver for kinder/BaseMotion3D-v0.

    Emits a single `target` of `Kinematic3DPointType` whose `(x, y, z)` comes
    from the supplied `TargetSource`. In fake / sim modes the source is a
    `ConstantTargetSource` threaded in from the env yaml; in real mode it's a
    `MarkerDetectorTargetSource` that subscribes to the marker-detector
    publisher.

    The target is queried once per :meth:`reset` and cached for every
    :meth:`step` call within that episode. A `MarkerDetectorTargetSource`
    query blocks up to the publisher's refresh interval; doing it every
    inner tick stretched the gap between consecutive base commands past the
    base controller's "no command in 2.5 * POLICY_CONTROL_PERIOD" timeout,
    producing chunky individual motions. The target marker is stationary
    during a rollout by design, so episode-scoped caching is safe.
    """

    def __init__(
        self,
        target_source: TargetSource,
        robot_name: str = "robot",
    ) -> None:
        super().__init__(robot_name=robot_name)
        self._target_source = target_source
        self._cached_target: tuple[float, float, float] | None = None

    @property
    def _state_cls(self) -> type[ObjectCentricState]:
        return BaseMotion3DObjectCentricState

    def reset(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        self._cached_target = self._target_source.get_target()
        return super().reset(obs, info)

    def _detect_objects(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> dict[Object, dict[str, float]]:
        del obs, info
        if self._cached_target is None:
            # Defensive — reset() should have run first; populate lazily so
            # callers that call step() before reset() still get a target.
            self._cached_target = self._target_source.get_target()
        x, y, z = self._cached_target
        return {
            Object("target", Kinematic3DPointType): {
                "x": x,
                "y": y,
                "z": z,
            }
        }


class Shelf3DPerceiver(KinematicRobotPerceiverBase):
    """Perceiver for kinder/KinematicShelf3D-o{N}-v0.

    Emits one cube (cube0) whose `(x, y, z)` comes from a TargetSource
    plus a shelf fixture at a fixed pose threaded in from the env yaml.
    The cube target follows the same pattern as
    :class:`BaseMotion3DPerceiver`: in fake / sim modes the source is a
    `ConstantTargetSource`; in real mode it's a
    `MarkerDetectorTargetSource` for the ArUco that marks the cube. The
    marker detector only provides `(x, y, z)` so the cube's orientation
    defaults to identity (z-up, lying flat); the shelf orientation does
    the same. The cube target is cached once per :meth:`reset` for the
    same control-loop-latency reason as in
    :class:`BaseMotion3DPerceiver` — the cube is stationary during a
    rollout by design.
    """

    def __init__(
        self,
        target_source: TargetSource,
        shelf_pose: tuple[float, float, float],
        cube_half_extents: tuple[
            float, float, float
        ] = _DEFAULT_SHELF_CUBE_HALF_EXTENTS,
        robot_name: str = "robot",
    ) -> None:
        super().__init__(robot_name=robot_name)
        self._target_source = target_source
        self._shelf_pose = tuple(shelf_pose)
        self._cube_half_extents = tuple(cube_half_extents)
        self._cached_cube_target: tuple[float, float, float] | None = None

    @property
    def _state_cls(self) -> type[ObjectCentricState]:
        return Shelf3DObjectCentricState

    def reset(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        self._cached_cube_target = self._target_source.get_target()
        return super().reset(obs, info)

    def _detect_objects(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> dict[Object, dict[str, float]]:
        del obs, info
        if self._cached_cube_target is None:
            # Defensive — reset() should have run first; populate lazily
            # so callers that call step() before reset() still get a cube.
            self._cached_cube_target = self._target_source.get_target()
        cube_x, cube_y, cube_z = self._cached_cube_target
        shelf_x, shelf_y, shelf_z = self._shelf_pose
        hx, hy, hz = self._cube_half_extents
        return {
            Object("shelf", Kinematic3DFixtureType): {
                "pose_x": shelf_x,
                "pose_y": shelf_y,
                "pose_z": shelf_z,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "pose_qw": 1.0,
            },
            Object("cube0", Kinematic3DCuboidType): {
                "pose_x": cube_x,
                "pose_y": cube_y,
                "pose_z": cube_z,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "pose_qw": 1.0,
                "grasp_active": 0.0,
                "object_type": 0.0,
                "half_extent_x": hx,
                "half_extent_y": hy,
                "half_extent_z": hz,
            },
        }


class CylinderShelf3DPerceiver(KinematicRobotPerceiverBase):
    """Perceiver for kinder/KinematicCylinderShelf3D-o{N}-v0.

    Emits one cylinder (cylinder0) whose `(x, y, z)` comes from a
    TargetSource plus a shelf fixture at a fixed pose threaded in from the
    env yaml — the same pattern as :class:`Shelf3DPerceiver`, with the
    cube swapped for a vertical cylinder. The cylinder's radius and height
    default to the env config's first cylinder; override per-lab once the
    real objects are measured. The marker detector only provides
    `(x, y, z)` so the cylinder orientation defaults to identity
    (standing upright), which for a rotationally symmetric cylinder is
    the only orientation the planner needs. The target is cached once per
    :meth:`reset` for the same control-loop-latency reason as in
    :class:`BaseMotion3DPerceiver` — the cylinder is stationary during a
    rollout by design.
    """

    def __init__(
        self,
        target_source: TargetSource,
        shelf_pose: tuple[float, float, float],
        cylinder_radius: float = _DEFAULT_CYLINDER_SHELF_CONFIG.cylinder_radius,
        cylinder_height: float = _DEFAULT_CYLINDER_SHELF_CONFIG.cylinder_heights[0],
        robot_name: str = "robot",
    ) -> None:
        super().__init__(robot_name=robot_name)
        self._target_source = target_source
        self._shelf_pose = tuple(shelf_pose)
        self._cylinder_radius = cylinder_radius
        self._cylinder_height = cylinder_height
        self._cached_cylinder_target: tuple[float, float, float] | None = None

    @property
    def _state_cls(self) -> type[ObjectCentricState]:
        return CylinderShelf3DObjectCentricState

    def reset(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> ObjectCentricState:
        self._cached_cylinder_target = self._target_source.get_target()
        return super().reset(obs, info)

    def _detect_objects(
        self, obs: TidyBotObservation, info: dict[str, Any]
    ) -> dict[Object, dict[str, float]]:
        del obs, info
        if self._cached_cylinder_target is None:
            # Defensive — reset() should have run first; populate lazily
            # so callers that call step() before reset() still get a
            # cylinder.
            self._cached_cylinder_target = self._target_source.get_target()
        cyl_x, cyl_y, cyl_z = self._cached_cylinder_target
        shelf_x, shelf_y, shelf_z = self._shelf_pose
        return {
            Object("shelf", Kinematic3DFixtureType): {
                "pose_x": shelf_x,
                "pose_y": shelf_y,
                "pose_z": shelf_z,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "pose_qw": 1.0,
            },
            Object("cylinder0", Kinematic3DCuboidType): {
                "pose_x": cyl_x,
                "pose_y": cyl_y,
                "pose_z": cyl_z,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "pose_qw": 1.0,
                "grasp_active": 0.0,
                "object_type": CYLINDER_OBJECT_TYPE,
                "half_extent_x": self._cylinder_radius,
                "half_extent_y": self._cylinder_radius,
                "half_extent_z": self._cylinder_height / 2,
            },
        }
