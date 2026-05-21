"""Hardware integration test: command the real arm to a target joint configuration.

Run on the robot (the arm server must already be up). Connects via RealArmInterface,
prints the current joint angles, then commands the arm to the retract configuration for
N_STEPS steps at policy rate, printing the per-step state.

python hardware_tests/test_arm_retract.py
"""

import sys
import time

import numpy as np

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD, RETRACT_ARM_CONF

N_STEPS = 20


def main() -> int:
    """Move the arm to the retract configuration over N_STEPS steps."""
    target = RETRACT_ARM_CONF.tolist()
    print("Connecting to the real arm interface...")
    arm = RealArmInterface()
    try:
        for i in range(N_STEPS):
            joints = arm.get_arm_state()
            joints_str = "  ".join(f"{j:+.3f}" for j in joints)
            err = np.linalg.norm(np.array(joints) - np.array(target))
            print(f"step {i + 1:02d}/{N_STEPS}  joints=[{joints_str}]  err={err:.4f}")
            arm.execute_action(target)
            time.sleep(POLICY_CONTROL_PERIOD)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
