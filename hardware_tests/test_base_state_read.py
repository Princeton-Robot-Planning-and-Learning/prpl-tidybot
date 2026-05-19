"""Hardware integration test: read the base state from the real robot.

Run on the robot (the base controller and marker detector servers must already be up).
Prints the odom-frame and map-frame poses and asks the operator to confirm they look
reasonable.

python hardware_tests/test_base_state_read.py
"""

import sys

from prpl_tidybot.interfaces.base_interface import RealBaseInterface


def main() -> int:
    print("Connecting to the real base interface...")
    base = RealBaseInterface()
    try:
        odom_pose = base.get_base_state()
        map_pose = base.get_map_base_state()
    finally:
        base.close()

    print(
        f"odom-frame pose: x={odom_pose.x:.3f} y={odom_pose.y:.3f} "
        f"theta={odom_pose.theta():.3f}"
    )
    print(
        f"map-frame pose:  x={map_pose.x:.3f} y={map_pose.y:.3f} "
        f"theta={map_pose.theta():.3f}"
    )

    answer = input("Do these poses match the robot's actual position? [y/N] ")
    if answer.strip().lower() == "y":
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
