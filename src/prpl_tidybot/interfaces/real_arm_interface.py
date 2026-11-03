# Hardware imports remain lazy for workstation simulation.
# pylint: disable=import-outside-toplevel
"""Real arm interface backed by the Kinova arm RPC server."""

from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.third_party.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY


class RealArmInterface(ArmInterface):
    """Real arm interface backed by the Kinova arm RPC server."""

    def __init__(
        self, reset_arm: bool = True, *, control_limits: dict | None = None
    ) -> None:
        from prpl_tidybot.third_party.arm_server import ArmManager

        self.manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.manager.connect()
        self.arm = (
            getattr(self.manager, "Arm")()
            if control_limits is None
            else getattr(self.manager, "Arm")(control_limits=control_limits)
        )
        self.arm.reset(reset_arm=reset_arm)

    def get_arm_state(self) -> list[float]:
        return self.arm.get_joint_angles()

    def get_gripper_state(self) -> float:
        return self.arm.get_gripper_position()

    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        self.arm.execute_action_angular(qpos=arm_goal, gripper_pos=gripper_goal)

    def release(self) -> None:
        """Stop the low-level loop; the Kinova returns to high-level servoing, where its
        onboard controller and any gamepad plugged into the base take over."""
        self.arm.release()

    def resume(self) -> None:
        """Restart the low-level loop from the arm's current configuration."""
        self.arm.resume()

    def close(self) -> None:
        """Tear down the RPC connection and stop the low-level arm control loop."""
        self.arm.close()
