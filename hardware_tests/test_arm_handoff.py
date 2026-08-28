"""Hardware integration test: release the arm for gamepad teleop, then re-acquire it.

Run on the robot (the arm server must already be up). Connects via RealArmInterface
without resetting the arm, prints the joint state, releases the arm to its onboard
controller (the Kinova returns to high-level servoing, so a gamepad plugged into the
base drives it), waits for Enter, re-acquires the arm from wherever it was left, and
holds that configuration for a few ticks while printing the joint state again.

python hardware_tests/test_arm_handoff.py
"""

import sys
import time

import numpy as np

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD

N_HOLD_STEPS = 20


def _print_state(label: str, arm: RealArmInterface) -> None:
    joints = np.round(arm.get_arm_state(), 3)
    print(f"{label}: joints={joints.tolist()} gripper={arm.get_gripper_state():.3f}")


def main() -> int:
    """Release, wait for the operator, re-acquire, hold."""
    print("Connecting to the real arm interface...")
    arm = RealArmInterface(reset_arm=False)
    try:
        _print_state("before release", arm)
        arm.release()
        input(
            "Arm released: move it with the gamepad (and open/close the gripper), "
            "then press Enter to hand it back: "
        )
        arm.resume()
        _print_state("after resume", arm)
        hold_joints = arm.get_arm_state()
        hold_gripper = arm.get_gripper_state()
        print(f"Holding for {N_HOLD_STEPS} ticks...")
        for i in range(N_HOLD_STEPS):
            arm.execute_action(hold_joints, hold_gripper)
            time.sleep(POLICY_CONTROL_PERIOD)
            if i % 5 == 4:
                _print_state(f"tick {i + 1:02d}", arm)
        drift = np.abs(np.array(arm.get_arm_state()) - np.array(hold_joints))
        print(f"max joint drift while holding: {drift.max():.4f} rad")
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
