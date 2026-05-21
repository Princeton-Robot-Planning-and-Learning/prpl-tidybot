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
