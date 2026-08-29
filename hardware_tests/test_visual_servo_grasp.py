"""Hardware integration test: visually servo the gripper onto a cylinder, step by step.

Run on the robot (the arm server must already be up) with the arm already at a
pre-grasp pose in front of a standing cylinder, gripper open, both cylinder
edges visible in the wrist camera. Each iteration captures a wrist frame,
runs the edge detector, writes the annotated frame under
hardware_tests/artifacts/, prints the lateral error and the tool-frame step
the servo would take, and asks before commanding it. Alignment acts on the
camera; once the cylinder has been within tolerance on two consecutive
captures the approach is open loop (straight forward steps, frames still
saved for the record). Use it to confirm the sign of the lateral axis
(`--lateral-sign -1` if the first aligning step moves the wrong way) and
the gain before enabling the executor in a rollout.

    python hardware_tests/test_visual_servo_grasp.py [--dry-run] [--lateral-sign -1]

Steps are chained from the last *commanded* joint target rather than the
perceived joints: the compliant controller holds a few hundredths of a
radian short of any target, so commands computed from the perceived joints
would never accumulate past that deadband and the arm would not move.

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


def _wrap(angles: np.ndarray) -> np.ndarray:
    """Wrap joint differences to [-pi, pi] so continuous joints print sensibly."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def _fit_approach_total(
    samples: list[tuple[float, float]],
    fallback: float,
    camera_to_grasp_offset: float,
    disabled: bool,
) -> float:
    """Approach length from the width growth over the baseline (1/w linear in the
    displacement), or the fallback when disabled or the fit is unusable."""
    if disabled or len(samples) < 3:
        print(
            f"Range estimate skipped ({len(samples)} samples); using {fallback:.3f} m."
        )
        return fallback
    slope, intercept = np.polyfit(
        [d for d, _ in samples], [1.0 / w for _, w in samples], 1
    )
    if slope >= 0:
        print(f"Range estimate rejected (width did not grow); using {fallback:.3f} m.")
        return fallback
    camera_to_axis = float(-intercept / slope)
    total = camera_to_axis - camera_to_grasp_offset
    print(
        f"Range estimate: camera-to-axis {camera_to_axis:.3f} m from "
        f"{[(round(d, 3), round(w, 1)) for d, w in samples]} -> approach {total:.3f} m "
        f"(fixed default {fallback:.3f} m)."
    )
    if not 0.04 <= total <= 0.20:
        print("  implausible; using the fixed default.")
        return fallback
    return total


def main() -> int:
    """Interactive align-then-approach with a confirmation before every step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="never move the arm")
    parser.add_argument("--lateral-sign", type=float, default=1.0)
    parser.add_argument("--lateral-gain", type=float, default=0.0003)
    parser.add_argument("--lateral-tolerance-px", type=float, default=8.0)
    parser.add_argument("--lateral-max-step", type=float, default=0.01)
    parser.add_argument("--lateral-min-step", type=float, default=0.006)
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--approach-step", type=float, default=0.01)
    parser.add_argument("--range-baseline", type=float, default=0.04)
    parser.add_argument("--camera-to-grasp-offset", type=float, default=0.108)
    parser.add_argument("--no-range-estimate", action="store_true")
    parser.add_argument("--max-width-change-frac", type=float, default=0.3)
    parser.add_argument("--width-consensus-ticks", type=int, default=3)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    params = EdgeDetectorParams()
    stepper = ToolFrameStepper()
    camera = KinovaWristCameraSource()
    print("Connecting to the real arm interface (no reset)...")
    arm = RealArmInterface(reset_arm=False)
    advanced = 0.0
    index = 0
    aligned_captures = 0
    approaching = False
    approach_total: float | None = None
    range_samples: list[tuple[float, float]] = []
    last_width: float | None = None
    rejected_widths: list[float] = []
    commanded: list[float] = list(arm.get_arm_state())
    try:
        while True:
            image = camera.get_image()
            if image is None:
                time.sleep(0.05)
                continue
            edges = detect_cylinder_edges(image, params)
            # A cylinder cut off by the image border has no usable width: it
            # skips the jump check and leaves the width reference alone. A
            # width that jumps from the reference is ignored, unless the
            # same new width comes back on width_consensus_ticks consecutive
            # captures: then the reference was the wrong one (a partial
            # detection of a white can on a light floor, say) and the new
            # width is adopted.
            if edges is not None and not edges.clipped and last_width is not None:
                change = abs(edges.width_px - last_width) / last_width
                if change > args.max_width_change_frac:
                    rejected_widths.append(edges.width_px)
                    recent = rejected_widths[-args.width_consensus_ticks :]
                    consistent = len(recent) == args.width_consensus_ticks and all(
                        abs(w - np.median(recent)) / np.median(recent)
                        <= args.max_width_change_frac
                        for w in recent
                    )
                    if consistent:
                        print(
                            f"step {index}: adopting width {edges.width_px:.0f}px "
                            f"after {len(recent)} consistent detections "
                            f"(reference was {last_width:.0f}px)"
                        )
                        rejected_widths.clear()
                    else:
                        print(
                            f"step {index}: ignoring a detection of width "
                            f"{edges.width_px:.0f}px ({100 * change:.0f}% change "
                            f"from {last_width:.0f}px)"
                        )
                        edges = None
                else:
                    rejected_widths.clear()
            if edges is not None and not edges.clipped:
                last_width = edges.width_px
            phase = "approach" if approaching else "align"
            overlay = render_edge_overlay(
                image, edges, params, label=f"step {index} {phase}"
            )
            out = ARTIFACT_DIR / f"visual_servo_{index:03d}.png"
            cv.imwrite(str(out), cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
            cv.imwrite(
                str(ARTIFACT_DIR / f"visual_servo_{index:03d}_raw.png"),
                cv.cvtColor(image, cv.COLOR_RGB2BGR),
            )
            if approaching:
                # Open loop from here; the camera only feeds the range estimate
                # over the first range_baseline metres.
                if approach_total is None:
                    if (
                        edges is not None
                        and not edges.clipped
                        and advanced <= args.range_baseline + 1e-9
                    ):
                        range_samples.append((advanced, edges.width_px))
                    if advanced >= args.range_baseline - 1e-9 or args.no_range_estimate:
                        approach_total = _fit_approach_total(
                            range_samples,
                            args.approach_distance,
                            args.camera_to_grasp_offset,
                            args.no_range_estimate,
                        )
                total = (
                    approach_total
                    if approach_total is not None
                    else max(args.approach_distance, args.range_baseline)
                )
                forward = min(args.approach_step, total - advanced)
                if forward <= 1e-6:
                    print(f"Approach complete ({advanced:.3f} m). Done.")
                    return 0
                delta = tool_delta("z", forward)
                width = "-" if edges is None else f"{edges.width_px:.0f}px"
                report = (
                    f"open-loop forward {forward:.3f} m ({advanced:.3f} of {total:.3f}; "
                    f"width {width})"
                )
            else:
                if edges is None:
                    answer = input(f"No edges found (overlay: {out}). Retry? [Y/n]: ")
                    if answer.strip().lower() in ("n", "no"):
                        return 1
                    continue
                error = edges.servo_error_px
                if edges.clipped:
                    side = "left" if edges.clipped_left else "right"
                    print(
                        f"step {index}: cylinder cut off by the {side} edge of the "
                        "image; stepping that way"
                    )
                if abs(error) <= args.lateral_tolerance_px:
                    aligned_captures += 1
                    print(
                        f"step {index}: error {error:+.1f}px within tolerance "
                        f"({aligned_captures}/2)"
                    )
                    if aligned_captures >= 2:
                        approaching = True
                        print("Aligned; switching to the open-loop approach.")
                    index += 1
                    continue
                aligned_captures = 0
                direction = -args.lateral_sign * np.sign(error)
                magnitude = min(
                    max(args.lateral_gain * abs(error), args.lateral_min_step),
                    args.lateral_max_step,
                )
                delta = tool_delta("x", float(direction * magnitude))
                forward = 0.0
                report = f"error {error:+.1f}px width {edges.width_px:.0f}px"
            perceived = arm.get_arm_state()
            target = stepper.step(commanded, delta)
            lag = np.round(_wrap(np.array(commanded) - np.array(perceived)), 3)
            print(
                f"step {index}: {report} -> tool delta {np.round(delta, 4)} m, joint "
                f"delta {np.round(np.array(target) - np.array(commanded), 3)} from the "
                f"last target (perceived lag {lag}) (overlay: {out})"
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
            commanded = target
            advanced += forward
            index += 1
    finally:
        camera.close()
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
