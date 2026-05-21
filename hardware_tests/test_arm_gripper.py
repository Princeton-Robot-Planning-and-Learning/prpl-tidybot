"""Hardware integration test: close then open the gripper.

Run on the robot (the arm server must already be up). Connects via RealArmInterface,
closes the gripper over N_STEPS steps, then opens it over N_STEPS steps, printing the
gripper state at each step.

python hardware_tests/test_arm_gripper.py
"""

import sys
import time

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD

N_STEPS = 10

GRIPPER_CLOSED = 1.0
GRIPPER_OPEN = 0.0


def main() -> int:
    """Close then open the gripper over N_STEPS steps each."""
    print("Connecting to the real arm interface...")
    arm = RealArmInterface(reset_arm=False)
    try:
        print("Closing gripper...")
        for i in range(N_STEPS):
            gripper = arm.get_gripper_state()
            print(f"step {i + 1:02d}/{N_STEPS}  gripper={gripper:.3f}")
            arm.execute_gripper_action(GRIPPER_CLOSED)
            time.sleep(POLICY_CONTROL_PERIOD)

        print("Opening gripper...")
        for i in range(N_STEPS):
            gripper = arm.get_gripper_state()
            print(f"step {i + 1:02d}/{N_STEPS}  gripper={gripper:.3f}")
            arm.execute_gripper_action(GRIPPER_OPEN)
            time.sleep(POLICY_CONTROL_PERIOD)

        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
