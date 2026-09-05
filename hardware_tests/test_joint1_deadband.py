"""Hardware probe: does commanding a small joint-1 change actually move joint 1?

The lateral grasp corrections (marker-based and SAM) steer by offsetting joint
1 by ~0.02-0.04 rad. If the compliant arm's joint 1 has a deadband larger than
that, those commands do nothing — which would explain why the corrections never
visibly move joint 1 and why flipping their sign changes nothing.

This holds the other six joints at their current values and commands joint 1
to +delta, then -delta, then back, reading the achieved joint-1 angle after
each so the tracked motion (or lack of it) is explicit. Run on the robot with
the arm server up:

    python hardware_tests/test_joint1_deadband.py            # delta 0.05 rad
    python hardware_tests/test_joint1_deadband.py --delta 0.1
"""

import argparse
import sys
import time

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD


def _settle(arm: RealArmInterface, goal: list[float], gripper: float) -> list[float]:
    """Command ``goal`` for a second, then return the achieved joint angles."""
    for _ in range(10):
        arm.execute_action(goal, gripper)
        time.sleep(POLICY_CONTROL_PERIOD)
    return arm.get_arm_state()


def main() -> int:
    """Command joint 1 up, down, and back; report the achieved motion each time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    arm = RealArmInterface()
    try:
        gripper = arm.get_gripper_state()
        start = arm.get_arm_state()
        print(f"start joint 1 = {start[0]:+.4f} rad")
        for label, offset in (
            ("+delta", args.delta),
            ("-delta", -args.delta),
            ("back", 0.0),
        ):
            goal = list(start)
            goal[0] = start[0] + offset
            achieved = _settle(arm, goal, gripper)
            moved = achieved[0] - start[0]
            print(
                f"{label}: commanded joint 1 {goal[0]:+.4f} "
                f"(offset {offset:+.4f}) -> achieved {achieved[0]:+.4f} "
                f"(moved {moved:+.4f} rad = {abs(moved) / max(abs(offset), 1e-9) * 100:.0f}% "
                "of commanded)"
            )
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
