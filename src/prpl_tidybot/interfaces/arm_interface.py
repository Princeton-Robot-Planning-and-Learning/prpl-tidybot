"""Arm interface."""

import abc


class ArmInterface(abc.ABC):
    """Arm interface.

    Arm and gripper are commanded together in a single atomic call. The
    underlying Kinova controller accepts both at once via
    `execute_action_angular(qpos, gripper_pos)`; sending them as two
    separate calls (which is what we used to do) is wrong because the
    second call overrides the first — a gripper-only update would queue
    `(current_joints, gripper)` and cancel any in-flight arm motion.
    """

    @abc.abstractmethod
    def get_arm_state(self) -> list[float]:
        """Get the current 7-DOF joint positions."""

    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """Get the current gripper state (1 is open, 0 is closed)."""

    @abc.abstractmethod
    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        """Send a single arm + gripper command (1 is open, 0 is closed)."""

    def close(self) -> None:
        """Tear down hardware connections; real-resource subclasses override."""


class FakeArmInterface(ArmInterface):
    """Fake arm interface that stores commanded values in memory."""

    def __init__(self) -> None:
        self.arm_state: list[float] = [0.0] * 7
        self.gripper_state: float = 0.0

    def get_arm_state(self) -> list[float]:
        return list(self.arm_state)

    def get_gripper_state(self) -> float:
        return self.gripper_state

    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        self.arm_state = list(arm_goal)
        self.gripper_state = gripper_goal
