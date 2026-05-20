"""Arm interface."""

import abc


class ArmInterface(abc.ABC):
    """Arm interface."""

    @abc.abstractmethod
    def get_arm_state(self) -> list[float]:
        """Get the current 7-DOF joint positions."""

    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """Get the current gripper state (1 is open, 0 is closed)."""

    @abc.abstractmethod
    def execute_action(self, action: list[float]) -> None:
        """Execute an absolute joint-space action on the arm."""

    @abc.abstractmethod
    def execute_gripper_action(self, action: float) -> None:
        """Execute a gripper action (1 is open, 0 is closed)."""


class FakeArmInterface(ArmInterface):
    """Fake arm interface that stores commanded values in memory."""

    def __init__(self) -> None:
        self.arm_state: list[float] = [0.0] * 7
        self.gripper_state: float = 0.0

    def get_arm_state(self) -> list[float]:
        return list(self.arm_state)

    def get_gripper_state(self) -> float:
        return self.gripper_state

    def execute_action(self, action: list[float]) -> None:
        self.arm_state = list(action)

    def execute_gripper_action(self, action: float) -> None:
        self.gripper_state = action


class RealArmInterface(ArmInterface):
    """Real arm interface backed by the Kinova arm RPC server."""

    def __init__(self) -> None:
        # Deferred import: arm_server pulls in pinocchio/kortex which are only
        # available on hardware; importing here keeps the module loadable elsewhere.
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.third_party.arm_server import ArmManager
        from prpl_tidybot.third_party.constants import (
            ARM_RPC_HOST,
            ARM_RPC_PORT,
            RPC_AUTHKEY,
        )

        self.manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.manager.connect()
        self.arm = self.manager.Arm()  # type: ignore # pylint: disable=no-member
        self.arm.reset()

    def get_arm_state(self) -> list[float]:
        return self.arm.get_joint_angles()

    def get_gripper_state(self) -> float:
        return self.arm.get_gripper_position()

    def execute_action(self, action: list[float]) -> None:
        self.arm.execute_action_angular(
            qpos=action, gripper_pos=self.arm.get_gripper_position()
        )

    def execute_gripper_action(self, action: float) -> None:
        self.arm.execute_action_angular(
            qpos=self.arm.get_joint_angles(), gripper_pos=action
        )

    def close(self) -> None:
        """Tear down the RPC connection and stop the low-level arm control loop."""
        self.arm.close()
