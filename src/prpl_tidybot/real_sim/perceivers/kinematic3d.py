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
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.prpl3d import (
    PrplLab3DEnvConfig,
    PrplLab3DObjectCentricState,
)
from prpl_utils.real_sim import Perceiver
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.real_sim.perceivers.target_source import TargetSource
from prpl_tidybot.structs import TidyBotObservation

_DEFAULT_CUBE_HALF_EXTENTS = PrplLab3DEnvConfig().block_half_extents


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
