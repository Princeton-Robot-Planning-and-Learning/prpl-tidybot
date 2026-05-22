"""Hardware integration test: move the arm to the home pose via IK.

Run on the robot (the arm server must already be up). Uses IKSolver to compute the home
joint configuration from the home end-effector position and quaternion, then commands
the arm to that configuration for N_STEPS steps at policy rate.

python hardware_tests/test_arm_ik_home.py
"""

import sys
import time

import numpy as np

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD, RETRACT_ARM_CONF

HOME_POS = np.array([0.456, 0.0, 0.434])
HOME_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # (x, y, z, w)

N_STEPS = 20


def main() -> int:
    """Solve IK for the home pose and command the arm there over N_STEPS steps."""
    print("Solving IK for home pose...")
    from prpl_tidybot.third_party.ik_solver import (  # pylint: disable=import-outside-toplevel
        IKSolver,
    )

    ik_solver = IKSolver()  # type: ignore[no-untyped-call]
    target = ik_solver.solve(  # type: ignore[no-untyped-call]
        HOME_POS, HOME_QUAT, RETRACT_ARM_CONF
    ).tolist()
    target_str = "  ".join(f"{j:+.3f}" for j in target)
    print(f"Home joint angles: [{target_str}]")

    print("Connecting to the real arm interface...")
    arm = RealArmInterface()
    try:
        for i in range(N_STEPS):
            joints = arm.get_arm_state()
            joints_str = "  ".join(f"{j:+.3f}" for j in joints)
            err = np.linalg.norm(np.array(joints) - np.array(target))
            print(f"step {i + 1:02d}/{N_STEPS}  joints=[{joints_str}]  err={err:.4f}")
            arm.execute_action(target, arm.get_gripper_state())
            time.sleep(POLICY_CONTROL_PERIOD)
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
