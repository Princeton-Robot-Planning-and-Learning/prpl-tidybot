"""Real arm interface backed by the Kinova arm RPC server."""

from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.third_party.arm_server import ArmManager
from prpl_tidybot.third_party.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY


class RealArmInterface(ArmInterface):
    """Real arm interface backed by the Kinova arm RPC server."""

    def __init__(self, reset_arm: bool = True) -> None:
        self.manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.manager.connect()
        self.arm = self.manager.Arm()  # type: ignore # pylint: disable=no-member
        self.arm.reset(reset_arm=reset_arm)

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
