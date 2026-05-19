"""Hardware integration test: drive the real base to three map-frame waypoints.

Run on the robot (the base controller and marker detector servers must already be up).
Uses `RealTidyBotEnv` with a `FakeArmInterface`, so only the base is exercised against
the hardware; each `env.step` converges the base in the map frame via the env's closed
loop (recalibrate map/odom from each fresh observation, re-project the target into odom,
command, check tolerance).

Before running, confirm ~1.5 m of clear floor in every direction from the robot.

python hardware_tests/test_base_map_target.py
"""

import math
import sys
import time

from spatialmath import SE2

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.structs import TidyBotAction

TARGETS_MAP = [
    SE2(0.5, 0.5, math.pi / 2),
    SE2(-0.5, 0.5, 0.0),
    SE2(0.5, -0.5, math.pi),
]
DWELL_BETWEEN_WAYPOINTS_S = 1.0


def main() -> int:
    """Drive the base to each map-frame waypoint in turn, printing the tracking error at
    each stop."""
    print("Connecting to the real base interface (arm and cameras are faked)...")
    interface = RealInterface(
        arm_interface=FakeArmInterface(),
        camera_interface=FakeCameraInterface(),
    )
    env = RealTidyBotEnv(interface)
    try:
        obs, _ = env.reset()
        arm_goal = interface.get_arm_state()
        gripper_goal = interface.get_gripper_state()
        print(
            f"Start map pose: x={obs.map_base_pose.x:+.3f} "
            f"y={obs.map_base_pose.y:+.3f} theta={obs.map_base_pose.theta():+.3f}"
        )

        for i, target in enumerate(TARGETS_MAP, start=1):
            print(
                f"Waypoint {i}/{len(TARGETS_MAP)}: "
                f"x={target.x:+.3f} y={target.y:+.3f} theta={target.theta():+.3f}"
            )
            action = TidyBotAction(
                arm_goal=arm_goal,
                base_pose_target_map=target,
                gripper_goal=gripper_goal,
            )
            obs, _, _, _, _ = env.step(action)
            err_xy = math.hypot(
                obs.map_base_pose.x - target.x, obs.map_base_pose.y - target.y
            )
            err_theta = obs.map_base_pose.theta() - target.theta()
            print(
                f"  reached: x={obs.map_base_pose.x:+.3f} "
                f"y={obs.map_base_pose.y:+.3f} theta={obs.map_base_pose.theta():+.3f}"
            )
            print(f"  err_xy={err_xy:.3f} m  err_theta={err_theta:+.3f} rad")
            time.sleep(DWELL_BETWEEN_WAYPOINTS_S)

        return 0
    finally:
        interface.base_interface.close()


if __name__ == "__main__":
    sys.exit(main())
