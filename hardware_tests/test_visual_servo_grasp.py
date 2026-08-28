"""Hardware integration test: visually servo the gripper onto a cylinder, step by step.

Run on the robot (the arm server must already be up) with the arm already at a
pre-grasp pose in front of a standing cylinder, gripper open, both cylinder
edges visible in the wrist camera. Each iteration captures a wrist frame,
runs the edge detector, writes the annotated frame under
hardware_tests/artifacts/, prints the lateral error and the tool-frame step
the servo would take, and asks before commanding it. Use it to confirm the
sign of the lateral axis (`--lateral-sign -1` if the first aligning step
moves the wrong way) and the gain before enabling the executor in a
rollout.

    python hardware_tests/test_visual_servo_grasp.py [--dry-run] [--lateral-sign -1]

The script never closes the gripper; it stops after the approach distance
so you can check the fingers straddle the cylinder before a real grasp.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.visual_servo.cylinder_edges import (
    EdgeDetectorParams,
    detect_cylinder_edges,
    render_edge_overlay,
)
from prpl_tidybot.visual_servo.image_sources import KinovaWristCameraSource
from prpl_tidybot.visual_servo.tool_frame import ToolFrameStepper, tool_delta

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
TICKS_PER_STEP = 10


def main() -> int:
    """Interactive align-then-approach with a confirmation before every step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="never move the arm")
    parser.add_argument("--lateral-sign", type=float, default=1.0)
    parser.add_argument("--lateral-gain", type=float, default=0.0003)
    parser.add_argument("--lateral-tolerance-px", type=float, default=8.0)
    parser.add_argument("--lateral-max-step", type=float, default=0.01)
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--approach-step", type=float, default=0.01)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    params = EdgeDetectorParams()
    stepper = ToolFrameStepper()
    camera = KinovaWristCameraSource()
    print("Connecting to the real arm interface (no reset)...")
    arm = RealArmInterface(reset_arm=False)
    advanced = 0.0
    index = 0
    try:
        while True:
            image = camera.get_image()
            if image is None:
                time.sleep(0.05)
                continue
            edges = detect_cylinder_edges(image, params)
            overlay = render_edge_overlay(image, edges, params, label=f"step {index}")
            out = ARTIFACT_DIR / f"visual_servo_{index:03d}.png"
            cv.imwrite(str(out), cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
            if edges is None:
                answer = input(f"No edges found (overlay: {out}). Retry? [Y/n]: ")
                if answer.strip().lower() in ("n", "no"):
                    return 1
                continue
            error = edges.lateral_error_px
            aligned = abs(error) <= args.lateral_tolerance_px
            lateral = 0.0
            if not aligned:
                lateral = float(
                    np.clip(
                        -args.lateral_sign * args.lateral_gain * error,
                        -args.lateral_max_step,
                        args.lateral_max_step,
                    )
                )
            forward = 0.0
            if aligned:
                forward = min(args.approach_step, args.approach_distance - advanced)
            if aligned and forward <= 1e-6:
                print(f"Aligned and approach complete ({advanced:.3f} m). Done.")
                return 0
            delta = tool_delta("x", lateral)
            delta[2] += forward
            joints = arm.get_arm_state()
            target = stepper.step(joints, delta)
            print(
                f"step {index}: error {error:+.1f}px width {edges.width_px:.0f}px "
                f"-> tool delta {np.round(delta, 4)} m, joint delta "
                f"{np.round(np.array(target) - np.array(joints), 3)} (overlay: {out})"
            )
            answer = input("Execute this step? [Enter=yes / s=skip capture / q=quit]: ")
            if answer.strip().lower() == "q":
                return 0
            if answer.strip().lower() == "s":
                index += 1
                continue
            if not args.dry_run:
                gripper = arm.get_gripper_state()
                for _ in range(TICKS_PER_STEP):
                    arm.execute_action(target, gripper)
                    time.sleep(POLICY_CONTROL_PERIOD)
            advanced += forward
            index += 1
    finally:
        camera.close()
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
