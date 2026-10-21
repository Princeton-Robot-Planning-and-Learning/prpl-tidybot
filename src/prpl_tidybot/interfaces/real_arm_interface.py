"""Real arm interface backed by the Kinova arm RPC server."""

from prpl_tidybot.interfaces.arm_interface import ArmInterface, ArmMotionAborted
from prpl_tidybot.third_party.arm_server import ArmManager
from prpl_tidybot.third_party.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from prpl_tidybot.third_party.kinova import stop_arm_motion
from prpl_tidybot.toss import TOSS_GRIPPER_RELEASE_MS, TOSS_MAX_VEL


class RealArmInterface(ArmInterface):
    """Real arm interface backed by the Kinova arm RPC server."""

    def __init__(self, reset_arm: bool = True) -> None:
        self.manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.manager.connect()
        self.arm = self.manager.Arm()  # type: ignore # pylint: disable=no-member
        self.arm.reset(reset_arm=reset_arm)
        # Whether the arm server's cyclic loop runs: only then does it update the
        # joint angles and gripper position that get_arm_state and get_gripper_state
        # return (reset() ends by resuming it). Bookkeeping on this side only.
        self._cyclic = True

    def readings_are_live(self) -> bool:
        """True while the arm server's cyclic loop runs; while it is released (an
        angular or gripper move under the onboard controller, the toss) the readings
        hold their last value."""
        return self._cyclic

    def get_arm_state(self) -> list[float]:
        return self.arm.get_joint_angles()

    def get_gripper_state(self) -> float:
        return self.arm.get_gripper_position()

    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        self.arm.execute_action_angular(qpos=arm_goal, gripper_pos=gripper_goal)

    def release(self) -> None:
        """Stop the low-level loop; the Kinova returns to high-level servoing, where its
        onboard controller and any gamepad plugged into the base take over."""
        self._cyclic = False
        self.arm.release()

    def resume(self) -> None:
        """Restart the low-level loop from the arm's current configuration."""
        self.arm.resume()
        self._cyclic = True

    def move_angular(self, goal: list[float]) -> list[float]:
        """One blocking ``move_angular`` on the arm server. The onboard controller
        keeps going if this process dies, so Ctrl-C stops the arm directly."""
        try:
            return list(self.arm.move_joints_angular(list(goal)))
        except KeyboardInterrupt as e:
            stop_arm_motion()
            raise ArmMotionAborted("Ctrl-C during an angular move; arm stopped") from e

    def set_gripper_angular(self, position: float, timeout: float = 4.0) -> float:
        """Gripper move on the arm server; Ctrl-C stops the arm as above."""
        try:
            return float(self.arm.gripper_angular(position, timeout))
        except KeyboardInterrupt as e:
            stop_arm_motion()
            raise ArmMotionAborted("Ctrl-C during a gripper move; arm stopped") from e

    def toss(
        self,
        heading: float = 0.0,
        max_vel: float = TOSS_MAX_VEL,
        gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
    ) -> dict[str, float] | None:
        """Run the toss on the arm server (``Arm.toss``); blocks until it is done.
        Returns the swing's start time (after the wind-up)."""
        # The arm server releases the cyclic loop for the toss and always resumes it
        # after (Arm.toss's finally).
        self._cyclic = False
        try:
            result = self.arm.toss(
                heading=heading, max_vel=max_vel, gripper_release_ms=gripper_release_ms
            )
        finally:
            self._cyclic = True
        return dict(result) if isinstance(result, dict) else None

    def close(self) -> None:
        """Tear down the RPC connection and stop the low-level arm control loop."""
        self.arm.close()
