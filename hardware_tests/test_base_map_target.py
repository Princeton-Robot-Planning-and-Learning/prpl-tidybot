"""Hardware integration test: drive the real base to three map-frame waypoints.

Run on the robot (the base controller and marker detector servers must already be up).
Before any motion is commanded, the test writes a top-down PNG of the robot's current
map pose and the planned waypoints into the project tree, prints the absolute path, and
waits for the operator to confirm the plan looks right — this catches map-frame
misregistration before the robot drives. The PNG is the workaround for running inside
tmux (no interactive matplotlib window); open it from VS Code Remote (or similar) on the
project tree. The file is deleted once the operator answers the prompt.

Before running, confirm ~1.5 m of clear floor in every direction from the robot.

python hardware_tests/test_base_map_target.py
"""

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from spatialmath import SE2  # noqa: E402

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
_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_PNG_PATH = _REPO_ROOT / "test_base_map_target_plan.png"


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


def _save_plan_png(start: SE2, targets: list[SE2], path: Path) -> None:
    """Render a top-down view of the start pose, target waypoints, and planned traversal
    order to a PNG so the operator can open it from outside the tmux pane before
    confirming."""
    fig, ax = plt.subplots(figsize=(7, 7))
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
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

        print("Planned waypoints (map frame):")
        for i, target in enumerate(TARGETS_MAP, start=1):
            print(
                f"  target {i}: x={target.x:+.3f} y={target.y:+.3f} "
                f"theta={target.theta():+.3f} ({math.degrees(target.theta()):+.0f}°)"
            )

        _save_plan_png(start, TARGETS_MAP, PLAN_PNG_PATH)
        try:
            print(f"Saved plan visualization to {PLAN_PNG_PATH}")
            answer = input(
                "Open the PNG and confirm the start pose and targets are correct "
                "relative to the robot's actual position. Proceed with motion? [y/N] "
            )
        finally:
            PLAN_PNG_PATH.unlink(missing_ok=True)
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
