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
    """Skeleton real arm interface. Every method is a placeholder that
    raises until the hardware driver gets wired up."""

    def get_arm_state(self) -> list[float]:
        raise NotImplementedError(
            "RealArmInterface.get_arm_state: read the 7-DOF joint positions "
            "from the real arm (e.g. via the Kinova Kortex SDK)."
        )

    def get_gripper_state(self) -> float:
        raise NotImplementedError(
            "RealArmInterface.get_gripper_state: read the gripper position "
            "from the real arm (return 1=closed, 0=open)."
        )

    def execute_action(self, action: list[float]) -> None:
        raise NotImplementedError(
            "RealArmInterface.execute_action: command the real arm to the "
            "absolute joint configuration in `action`."
        )

    def execute_gripper_action(self, action: float) -> None:
        raise NotImplementedError(
            "RealArmInterface.execute_gripper_action: command the gripper "
            "to absolute position `action` (1=closed, 0=open)."
        )
