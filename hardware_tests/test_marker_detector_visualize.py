"""Hardware integration test: visualise the marker-detector's view of the scene.

Run on the robot (the marker-detector server must already be up). Subscribes to the
marker detector, pulls the latest payload, and writes a top-down PNG of the robot pose
plus every detected target marker into the project tree. The operator opens the PNG
(e.g. from VS Code Remote attached to the project) and confirms that the rendered
geometry matches the actual scene before continuing.

python hardware_tests/test_marker_detector_visualize.py
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from prpl_tidybot.marker_detector.client import MarkerDetectorClient
from prpl_tidybot.marker_detector.constants import FLOOR_LENGTH, FLOOR_WIDTH

HEADING_ARROW_LEN_M = 0.20
_REPO_ROOT = Path(__file__).resolve().parent.parent
PNG_PATH = _REPO_ROOT / "test_marker_detector_visualize.png"


def _draw_robot(ax: Axes, x: float, y: float, theta: float) -> None:
    ax.plot(x, y, "o", color="tab:blue", markersize=10)
    ax.arrow(
        x,
        y,
        HEADING_ARROW_LEN_M * math.cos(theta),
        HEADING_ARROW_LEN_M * math.sin(theta),
        head_width=0.05,
        length_includes_head=True,
        color="tab:blue",
    )
    ax.annotate(
        f"robot 0\n({x:+.2f}, {y:+.2f}, {math.degrees(theta):+.0f}°)",
        (x, y),
        textcoords="offset points",
        xytext=(8, 8),
        color="tab:blue",
        fontsize=9,
    )


def _draw_target(ax: Axes, aruco_id: int, x: float, y: float) -> None:
    ax.plot(x, y, "s", color="tab:red", markersize=8)
    ax.annotate(
        f"target id={aruco_id}\n({x:+.2f}, {y:+.2f})",
        (x, y),
        textcoords="offset points",
        xytext=(8, 8),
        color="tab:red",
        fontsize=9,
    )


def _save_snapshot_png(payload: dict, path: Path) -> None:
    """Render the detector payload to a top-down PNG."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.add_patch(
        Rectangle(
            (-FLOOR_WIDTH / 2, -FLOOR_LENGTH / 2),
            FLOOR_WIDTH,
            FLOOR_LENGTH,
            edgecolor="gray",
            facecolor="none",
            linewidth=1.0,
            label="floor extent",
        )
    )

    poses = payload.get("poses") or {}
    targets = payload.get("targets") or {}

    if 0 in poses:
        x, y, theta = poses[0]
        _draw_robot(ax, float(x), float(y), float(theta))

    for aruco_id, (x, y) in targets.items():
        _draw_target(ax, int(aruco_id), float(x), float(y))

    pad = 0.3
    ax.set_xlim(-FLOOR_WIDTH / 2 - pad, FLOOR_WIDTH / 2 + pad)
    ax.set_ylim(-FLOOR_LENGTH / 2 - pad, FLOOR_LENGTH / 2 + pad)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title("Marker-detector snapshot (map frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Pull one payload from the detector, render it, and ask the operator to
    confirm."""
    print("Connecting to the marker detector...")
    client = MarkerDetectorClient()
    try:
        payload = client.get_latest()
    finally:
        client.close()

    poses = payload.get("poses") or {}
    targets = payload.get("targets") or {}

    print("Detector payload:")
    if 0 in poses:
        x, y, theta = poses[0]
        print(
            f"  robot 0: x={x:+.3f} y={y:+.3f} theta={theta:+.3f} "
            f"({math.degrees(theta):+.0f}°)"
        )
    else:
        print("  robot 0: not detected")
    if targets:
        for aruco_id, (x, y) in sorted(targets.items()):
            print(f"  target id={aruco_id}: x={x:+.3f} y={y:+.3f}")
    else:
        print("  no target markers detected")

    _save_snapshot_png(payload, PNG_PATH)
    try:
        print(f"Saved snapshot to {PNG_PATH}")
        answer = input(
            "Open the PNG and confirm the robot and target markers are placed "
            "correctly relative to the actual scene. Does it look right? [y/N] "
        )
    finally:
        PNG_PATH.unlink(missing_ok=True)

    if answer.strip().lower() == "y":
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
