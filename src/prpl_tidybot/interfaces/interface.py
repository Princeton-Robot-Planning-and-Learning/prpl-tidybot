"""The top-level interface composing arm, base, and camera.

Keep all real-world code behind this abstraction so that the rest of the package
is testable without the real robot.
"""

import abc

import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.interfaces.arm_interface import (
    ArmInterface,
    FakeArmInterface,
    RealArmInterface,
)
from prpl_tidybot.interfaces.base_interface import (
    BaseInterface,
    FakeBaseInterface,
    RealBaseInterface,
)
from prpl_tidybot.interfaces.camera_interface import (
    CameraInterface,
    FakeCameraInterface,
    RealCameraInterface,
)
from prpl_tidybot.structs import TidyBotObservation


class Interface(abc.ABC):
    """A generic interface to the TidyBot++, real or fake.

    The component sub-interfaces are exposed as attributes so that callers
    (e.g. `RealTidyBotEnv.step`) can address each component directly when the
    composite action's components need different handling — most notably, the
    base target is a map-frame pose that needs converting to odom before
    `base_interface.execute_action` sees it.
    """

    arm_interface: ArmInterface
    base_interface: BaseInterface
    camera_interface: CameraInterface

    @abc.abstractmethod
    def get_base_state(self) -> spatialmath.SE2:
        """Get the base pose in the odom frame."""

    @abc.abstractmethod
    def get_map_base_state(self) -> spatialmath.SE2:
        """Get the base pose in the map frame."""

    @abc.abstractmethod
    def get_arm_state(self) -> list[float]:
        """Get the 7-DOF arm joint positions."""

    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """Get the gripper state (1 is open, 0 is closed)."""

    @abc.abstractmethod
    def get_wrist_image(self) -> Image:
        """Get the current wrist image."""

    @abc.abstractmethod
    def get_base_image(self) -> Image:
        """Get the current base image."""

    def get_observation(self) -> TidyBotObservation:
        """Build a full TidyBotObservation from the component getters."""
        return TidyBotObservation(
            arm_conf=self.get_arm_state(),
            base_pose=self.get_base_state(),
            map_base_pose=self.get_map_base_state(),
            gripper=self.get_gripper_state(),
            wrist_camera=self.get_wrist_image(),
            base_camera=self.get_base_image(),
        )


class FakeInterface(Interface):
    """A fake interface composing fake arm, base, and camera interfaces."""

    def __init__(self) -> None:
        self.arm_interface = FakeArmInterface()
        self.base_interface = FakeBaseInterface()
        self.camera_interface = FakeCameraInterface()

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_base_state()

    def get_map_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_map_base_state()

    def get_arm_state(self) -> list[float]:
        return self.arm_interface.get_arm_state()

    def get_gripper_state(self) -> float:
        return self.arm_interface.get_gripper_state()

    def get_wrist_image(self) -> Image:
        return self.camera_interface.get_wrist_image()

    def get_base_image(self) -> Image:
        return self.camera_interface.get_base_image()


class RealInterface(Interface):
    """The real and sole interface to the TidyBot++ robot.

    Each component (arm, base, camera) defaults to a skeleton whose
    methods raise `NotImplementedError`; implement them piece-by-piece
    against the real hardware drivers.

    Any component can be swapped at construction time — e.g. pass a
    `FakeArmInterface` when running an env (like BaseMotion3D) whose
    rollout doesn't need real arm reads or writes. This is wired
    through Hydra in `conf/env/<env>.yaml`'s `real` pipeline; see
    `conf/env/base_motion3d.yaml`.
    """

    def __init__(
        self,
        arm_interface: ArmInterface | None = None,
        base_interface: BaseInterface | None = None,
        camera_interface: CameraInterface | None = None,
    ) -> None:
        self.arm_interface = arm_interface or RealArmInterface()
        self.base_interface = base_interface or RealBaseInterface()
        self.camera_interface = camera_interface or RealCameraInterface()

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_base_state()

    def get_map_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_map_base_state()

    def get_arm_state(self) -> list[float]:
        return self.arm_interface.get_arm_state()

    def get_gripper_state(self) -> float:
        return self.arm_interface.get_gripper_state()

    def get_wrist_image(self) -> Image:
        return self.camera_interface.get_wrist_image()

    def get_base_image(self) -> Image:
        return self.camera_interface.get_base_image()
