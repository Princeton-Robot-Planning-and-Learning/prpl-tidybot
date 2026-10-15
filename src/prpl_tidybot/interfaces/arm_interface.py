"""Arm interface."""

import abc
from typing import Any

from prpl_tidybot.toss import (
    TOSS_GRIPPER_RELEASE_MS,
    TOSS_MAX_VEL,
    plan_toss_swing,
    toss_windup_pose,
)


class ArmMotionAborted(KeyboardInterrupt):
    """Ctrl-C during an onboard-controller move; the arm was sent a Stop."""


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

    def release(self) -> None:
        """Give the arm back to its onboard controller for manual teleoperation.

        After this, ``execute_action`` must not be called until ``resume``.
        Real-resource subclasses override; the default is a no-op.
        """

    def resume(self) -> None:
        """Re-take control of the arm from wherever it currently is.

        Real-resource subclasses override; the default is a no-op.
        """

    def move_angular(self, goal: list[float]) -> list[float]:
        """Move to ``goal`` (radians) under the arm's onboard position controller and
        return the measured joints. Only between ``release`` and ``resume``."""
        raise NotImplementedError(f"{type(self).__name__} has no onboard controller.")

    def set_gripper_angular(self, position: float, timeout: float = 4.0) -> float:
        """Move the gripper under the onboard controller; returns where it stopped.
        Only between ``release`` and ``resume``."""
        raise NotImplementedError(f"{type(self).__name__} has no onboard controller.")

    def toss(
        self,
        heading: float = 0.0,
        max_vel: float = TOSS_MAX_VEL,
        gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
    ) -> None:
        """Throw whatever the gripper holds (see ``prpl_tidybot.toss``).

        Moves to the wind-up pose with joint 1 at ``heading`` (Kortex degrees), swings
        to the release pose at up to ``max_vel`` deg/s and opens the gripper
        ``gripper_release_ms`` ms into the swing. Blocks until the arm is back under the
        normal controller, holding the release pose with the gripper open.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot toss.")

    def close(self) -> None:
        """Tear down hardware connections; real-resource subclasses override."""


class FakeArmInterface(ArmInterface):
    """Fake arm interface that stores commanded values in memory."""

    def __init__(self) -> None:
        self.arm_state: list[float] = [0.0] * 7
        self.gripper_state: float = 0.0
        self.toss_calls: list[dict[str, Any]] = []
        self.released = False
        # Added to every joint by move_angular, to model an arm that stops short.
        self.angular_offset = 0.0

    def release(self) -> None:
        self.released = True

    def resume(self) -> None:
        self.released = False

    def move_angular(self, goal: list[float]) -> list[float]:
        if not self.released:
            raise RuntimeError("release() the torque loop before an angular move")
        self.arm_state = [float(q) + self.angular_offset for q in goal]
        return list(self.arm_state)

    def set_gripper_angular(self, position: float, timeout: float = 4.0) -> float:
        if not self.released:
            raise RuntimeError("release() the torque loop before an angular move")
        self.gripper_state = float(position)
        return self.gripper_state

    def get_arm_state(self) -> list[float]:
        return list(self.arm_state)

    def get_gripper_state(self) -> float:
        return self.gripper_state

    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        self.arm_state = list(arm_goal)
        self.gripper_state = gripper_goal

    def toss(
        self,
        heading: float = 0.0,
        max_vel: float = TOSS_MAX_VEL,
        gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
    ) -> None:
        """Record the call and open the gripper; the arm conf is left unchanged.

        The swing is planned (and discarded) so that parameters the arm server would
        reject fail here too.
        """
        plan_toss_swing(
            toss_windup_pose(heading),
            heading,
            max_vel=max_vel,
            gripper_release_ms=gripper_release_ms,
        )
        self.toss_calls.append(
            {
                "heading": float(heading),
                "max_vel": float(max_vel),
                "gripper_release_ms": int(gripper_release_ms),
            }
        )
        self.gripper_state = 0.0
