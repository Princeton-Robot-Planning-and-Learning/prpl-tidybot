"""Hardware integration test: read arm joint angles and gripper position.

Run on the robot (the arm server must already be up). Connects via RealArmInterface,
resets the arm to the retract configuration, then prints joint angles and gripper
position at policy rate for 50 steps.

python hardware_tests/test_arm_state_read.py
"""

import sys
import time

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD

N_STEPS = 50


def main() -> int:
    """Connect to the real arm, reset it, and stream state for N_STEPS steps."""
    print("Connecting to the real arm interface...")
    arm = RealArmInterface()
    try:
        for i in range(N_STEPS):
            joints = arm.get_arm_state()
            gripper = arm.get_gripper_state()
            joints_str = "  ".join(f"{j:+.3f}" for j in joints)
            print(
                f"step {i + 1:02d}/{N_STEPS}  joints=[{joints_str}]  grip={gripper:.3f}"
            )
            time.sleep(POLICY_CONTROL_PERIOD)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
