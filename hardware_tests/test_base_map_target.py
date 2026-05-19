"""Hardware integration test: drive the real base to three map-frame waypoints.

Run on the robot (the base controller and marker detector servers must already be up).
Before any motion is commanded, the test opens a top-down plot of the robot's current
map pose and the planned waypoints and waits for the operator to confirm the plan
visually — this catches map-frame misregistration before the robot drives.

Before running, confirm ~1.5 m of clear floor in every direction from the robot.

python hardware_tests/test_base_map_target.py
"""

import math
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
HEADING_ARROW_LEN_M = 0.15


def _draw_pose(ax: Axes, pose: SE2, label: str, color: str) -> None:
    ax.plot(pose.x, pose.y, "o", color=color)
    dx = HEADING_ARROW_LEN_M * math.cos(pose.theta())
    dy = HEADING_ARROW_LEN_M * math.sin(pose.theta())
    ax.arrow(
        pose.x,
        pose.y,
        dx,
        dy,
        head_width=0.05,
        length_includes_head=True,
        color=color,
    )
    ax.annotate(
        f"{label}\n({pose.x:+.2f}, {pose.y:+.2f}, "
        f"{math.degrees(pose.theta()):+.0f}°)",
        (pose.x, pose.y),
        textcoords="offset points",
        xytext=(8, 8),
        color=color,
        fontsize=9,
    )


def _show_plan(start: SE2, targets: list[SE2]) -> None:
    """Open a non-blocking top-down plot of the start pose, target waypoints, and the
    planned traversal order.

    Lets the operator visually verify map alignment before any motion is commanded.
    """
    plt.ion()
    _, ax = plt.subplots(figsize=(7, 7))
    _draw_pose(ax, start, "start", "tab:blue")
    for i, target in enumerate(targets, start=1):
        _draw_pose(ax, target, f"target {i}", "tab:red")

    xs = [start.x] + [t.x for t in targets]
    ys = [start.y] + [t.y for t in targets]
    ax.plot(xs, ys, "--", color="gray", alpha=0.6, label="planned order")

    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title("Robot start pose and planned waypoints (map frame)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def main() -> int:
    """Drive the base to each map-frame waypoint in turn, after the operator confirms
    the plot of the start pose and waypoints looks right."""
    print("Connecting to the real base interface (arm and cameras are faked)...")
    interface = RealInterface(
        arm_interface=FakeArmInterface(),
        camera_interface=FakeCameraInterface(),
    )
    env = RealTidyBotEnv(interface)
    try:
        obs, _ = env.reset()
        start = obs.map_base_pose
        print(
            f"Start map pose: x={start.x:+.3f} y={start.y:+.3f} "
            f"theta={start.theta():+.3f}"
        )

        _show_plan(start, TARGETS_MAP)
        answer = input(
            "Does the plot match the robot's actual position and the intended "
            "targets? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted by user before motion.")
            return 1

        arm_goal = interface.get_arm_state()
        gripper_goal = interface.get_gripper_state()

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
