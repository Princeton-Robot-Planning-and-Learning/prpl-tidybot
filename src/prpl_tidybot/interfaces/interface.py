"""The top-level interface composing arm, base, and camera.

Keep all real-world code behind this abstraction so that the rest of the package
is testable without the real robot.
"""

import abc

import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


class Interface(abc.ABC):
    """A generic interface to the TidyBot++, real or fake."""

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

    @abc.abstractmethod
    def execute_base_action(self, action: TidyBotAction) -> None:
        """Execute the base component of a TidyBotAction in the local frame."""

    @abc.abstractmethod
    def execute_arm_action(self, action: TidyBotAction) -> None:
        """Execute the arm component of a TidyBotAction."""

    @abc.abstractmethod
    def execute_gripper_action(self, action: TidyBotAction) -> None:
        """Execute the gripper component of a TidyBotAction."""

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

    def execute_base_action(self, action: TidyBotAction) -> None:
        self.base_interface.execute_action(action.base_local_goal)

    def execute_arm_action(self, action: TidyBotAction) -> None:
        self.arm_interface.execute_action(action.arm_goal)

    def execute_gripper_action(self, action: TidyBotAction) -> None:
        self.arm_interface.execute_gripper_action(action.gripper_goal)
