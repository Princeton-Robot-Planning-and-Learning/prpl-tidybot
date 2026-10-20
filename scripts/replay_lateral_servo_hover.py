"""Replay the top-down pick's lateral servo (TopDownLateralServo) offline on the hover
frame of recorded wrist clips: what it would have seen and commanded first.

For each pick attempt in a toss_cycles run directory (cycles.json, toss_cycles.log
and the <clip>_wrist.mp4 recordings), the hover frame is found by tracking the
cube's blob: the approach to the hover makes it grow, the descent takes it out of
the bottom of the image, and the hover is the first steady run in between. The log
gives a second, independent estimate: the recorder grabs a frame about every 0.1 s
from the attempt's confirmation step, and the hover arrival is logged.

On that frame: the cube's pixel error from target_uv; the servo's first commanded
step (Tom's clamped proportional step, per tool axis); and the full correction the
camera model implies (error / 2920 px/m, per axis), in the tool frame and the arm
frame. Outputs an annotated hover frame per attempt, the blob-area track, and
replay.json.

    R=../hitl-pmp-results/tidybot-real
    python scripts/replay_lateral_servo_hover.py \
        --run-dir $R/raw/2026-10-19/16-24-13 \
        --out $R/analysis/2026-10-19-oneshot/replay \
        --reference-frames "$R/raw/2026-10-14/21-5*/wrist_servo/*_00_hover_*_raw.png"

`--reference-frames` (hover frames from another day) gives that day's blob size for
comparison: the gain scales as 1/height, so the same height shows the same size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics
from prpl_tidybot.visual_servo.image_sources import SequenceImageSource
from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownLateralServo,
    WristCube,
    detect_wrist_cube,
)

# The 2026-10-14 probes (onboard controller): ~2920 px/m per tool axis, unmirrored.
_PX_PER_M = 2920.0
_CUBE_SIDE = 0.05
_STEADY_PX = 4.0
_TIME = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]")


def _log_times(log_path: Path) -> list[dict[str, float]]:
    """Per pick attempt, in order: the confirmation step (the clip starts), the
    hover arrival and the grasp arrival, as seconds."""
    attempts: list[dict[str, float]] = []
    for line in log_path.read_text().splitlines():
        match = _TIME.match(line)
        if match is None:
            continue
        t = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        if "Step: Cycle" in line and "Pick attempt" in line:
            attempts.append({"start": t})
        elif attempts and "Move hover" in line and "hover" not in attempts[-1]:
            attempts[-1]["hover"] = t
        elif attempts and "Move grasp" in line and "grasp" not in attempts[-1]:
            attempts[-1]["grasp"] = t
    return attempts


def _clips(cycles: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(clip name, attempt record) for every pick attempt, in order, as
    toss_cycles names them (alternate-camera retries carry a _k suffix)."""
    out = []
    for cycle in cycles["cycles"]:
        counts: dict[int, int] = {}
        for attempt in cycle.get("pick_attempts", []):
            n = attempt["attempt"]
            counts[n] = counts.get(n, 0) + 1
            k = counts[n]
            clip = f"{n}" + (f"_{k}" if k > 1 else "")
            # Since the skill server: the record names its own clip.
            name = attempt.get("wrist_clip", f"cycle_{cycle['cycle']:02d}_pick_{clip}")
            out.append((name, attempt))
    return out


def _read_frames(path: Path) -> list[np.ndarray]:
    capture = cv.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
    capture.release()
    return frames


def _hover_index(cubes: list[WristCube | None], prior: int | None) -> int | None:
    """The middle of the longest run of steady frames (the blob moving at most
    `_STEADY_PX` between frames) with the blob at least half its largest size: the
    arm parked at the hover. The approach to the hover grows the blob; the descent
    takes it out of the bottom of the image (the camera is 5.6 cm off the gripper
    axis), so both are excluded."""
    largest = max((c.area for c in cubes if c is not None), default=0.0)
    near = [c is not None and c.area >= 0.5 * largest for c in cubes]
    runs: list[list[int]] = []
    for i, ok in enumerate(near):
        prev = cubes[i - 1] if i > 0 else None
        cube = cubes[i]
        steady = (
            ok
            and i > 0
            and near[i - 1]
            and prev is not None
            and cube is not None
            and math.hypot(cube.u - prev.u, cube.v - prev.v) <= _STEADY_PX
        )
        if steady and runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        elif steady:
            runs.append([i - 1, i])
    if not runs:
        return None
    # The first hover, not the retreat to it after a miss.
    best = next((r for r in runs if len(r) >= 3), max(runs, key=len))
    index = best[len(best) // 2]
    if prior is not None and abs(index - prior) > 15:
        print(f"  note: blob-track hover frame {index} vs log estimate {prior}")
    return index


def _tool_axes_from_joints(planned_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Tool x and closing (y) axes (arm frame, horizontal) at the logged hover."""
    _, approach, closing = forward_kinematics(
        [math.radians(q) for q in planned_deg], TopDownPickGeometry()
    )
    x_axis = np.cross(closing, approach)[:2]
    return x_axis / np.linalg.norm(x_axis), closing[:2] / np.linalg.norm(closing[:2])


def _first_step(image: np.ndarray, params: LateralServoParams) -> tuple[float, float]:
    """The servo's first commanded step (tool x, y; metres) on this one frame: tool
    axes taken as the arm's, and a move that never arrives, so it stops there."""
    servo = TopDownLateralServo(SequenceImageSource([image]), params)
    result = servo.align(
        (0.0, 0.0, 0.0),
        lambda *_: False,
        lambda _yaw: (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    )
    steps = [t.delta_tool for t in result.trace if t.delta_tool is not None]
    return steps[0] if steps else (0.0, 0.0)


def _annotate(
    image: np.ndarray, cube: WristCube, params: LateralServoParams, lines: list[str]
) -> np.ndarray:
    bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    tu, tv = (int(round(c)) for c in params.target_uv)
    tol = int(params.lateral_tolerance_px)
    cv.rectangle(bgr, (tu - tol, tv - tol), (tu + tol, tv + tol), (0, 0, 255), 1)
    cv.drawMarker(bgr, (tu, tv), (0, 0, 255), cv.MARKER_CROSS, 30, 2)
    centre = (int(round(cube.u)), int(round(cube.v)))
    cv.circle(bgr, centre, 6, (0, 255, 0), -1)
    cv.arrowedLine(bgr, centre, (tu, tv), (0, 255, 255), 2, tipLength=0.08)
    for i, text in enumerate(lines):
        y = 22 + 22 * i
        cv.putText(bgr, text, (8, y), 0, 0.55, (0, 0, 0), 4)
        cv.putText(bgr, text, (8, y), 0, 0.55, (255, 255, 255), 1)
    return bgr


def _area_figure(tracks: dict[str, Any], out: Path) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(
        len(tracks), 1, figsize=(7, 2.2 * len(tracks)), squeeze=False
    )
    for ax, (clip, track) in zip(axes[:, 0], tracks.items()):
        ax.plot(track["areas"], color="#0072B2", linewidth=1.2)
        ax.axvline(track["hover"], color="#999999", linestyle=":", label="hover frame")
        if track["prior"] is not None:
            ax.axvline(
                track["prior"], color="#E69F00", linestyle="--", label="log estimate"
            )
        ax.set_title(f"{clip} ({track['outcome']})", fontsize=9)
        ax.set_ylabel("blob area (px)")
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("frame (~0.1 s)")
    fig.tight_layout()
    fig.savefig(out / "blob_area_tracks.png", dpi=130)


def _main() -> None:  # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-uv", type=float, nargs=2, default=[318.0, 280.0])
    parser.add_argument("--reference-frames", default=None)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    params = LateralServoParams(target_uv=tuple(args.target_uv))

    cycles = json.loads((args.run_dir / "cycles.json").read_text())
    times = _log_times(args.run_dir / "toss_cycles.log")
    rows, tracks = [], {}
    for k, (clip, attempt) in enumerate(_clips(cycles)):
        video = args.run_dir / f"{clip}_wrist.mp4"
        if not video.exists():
            print(f"{clip}: no wrist clip")
            continue
        frames = _read_frames(video)
        cubes = [detect_wrist_cube(f) for f in frames]
        prior = None
        if k < len(times) and "hover" in times[k]:
            prior = int(round(10 * (times[k]["hover"] - times[k]["start"])))
        index = _hover_index(cubes, prior)
        outcome = "held" if attempt.get("held") else "missed"
        if index is None:
            print(f"{clip}: no steady hover frame with the cube in view")
            continue
        cube = cubes[index]
        assert cube is not None
        error = (cube.u - params.target_uv[0], cube.v - params.target_uv[1])
        tool = np.array([-error[0] / _PX_PER_M, -error[1] / _PX_PER_M])
        hover = next(m for m in attempt["moves"] if m["name"] == "hover")
        x_axis, y_axis = _tool_axes_from_joints(hover["planned_deg"])
        arm = tool[0] * x_axis + tool[1] * y_axis
        step = _first_step(frames[index], params)
        row = {
            "clip": clip,
            "outcome": outcome,
            "gripper": attempt.get("gripper"),
            "hover_frame": index,
            "log_estimate_frame": prior,
            "frames": len(frames),
            "cube_uv": [round(cube.u, 1), round(cube.v, 1)],
            "clipped": cube.clipped,
            "area_px": round(cube.area),
            "side_px_per_m": round(math.sqrt(cube.area) / _CUBE_SIDE),
            "error_px": [round(e, 1) for e in error],
            "within_tolerance": max(abs(e) for e in error)
            <= params.lateral_tolerance_px,
            "correction_tool_mm": (tool * 1000).round(1).tolist(),
            "correction_arm_mm": (arm * 1000).round(1).tolist(),
            "correction_mm": round(float(np.linalg.norm(tool)) * 1000, 1),
            "first_step_tool_mm": [round(s * 1000, 1) for s in step],
        }
        rows.append(row)
        tracks[clip] = {
            "areas": [np.nan if c is None else c.area for c in cubes],
            "hover": index,
            "prior": prior,
            "outcome": outcome,
        }
        lines = [
            f"{clip} ({outcome}), hover frame {index}",
            f"cube ({cube.u:.0f}, {cube.v:.0f}) target ({params.target_uv[0]:.0f}, "
            f"{params.target_uv[1]:.0f}): error ({error[0]:+.0f}, {error[1]:+.0f}) px",
            f"correction tool x {tool[0]*1000:+.1f} mm, y {tool[1]*1000:+.1f} mm "
            f"(|{row['correction_mm']}| mm)",
            f"servo first step x {step[0]*1000:+.1f} mm, y {step[1]*1000:+.1f} mm",
        ]
        cv.imwrite(
            str(out / f"{clip}_hover.png"),
            _annotate(frames[index], cube, params, lines),
        )
        cv.imwrite(
            str(out / f"{clip}_hover_raw.png"),
            cv.cvtColor(frames[index], cv.COLOR_RGB2BGR),
        )
        print(json.dumps(row))

    # The held pick's hover is a measured "cube under the pinch" view: each other
    # attempt's offset from it is what that attempt was off by.
    held = next((r for r in rows if r["outcome"] == "held"), None)
    if held is not None:
        for row in rows:
            du = row["cube_uv"][0] - held["cube_uv"][0]
            dv = row["cube_uv"][1] - held["cube_uv"][1]
            row["vs_held_px"] = [round(du, 1), round(dv, 1)]
            row["vs_held_tool_mm"] = [
                round(-du / _PX_PER_M * 1000, 1),
                round(-dv / _PX_PER_M * 1000, 1),
            ]
            print(f"{row['clip']} vs the held pick: {row['vs_held_px']} px")

    reference = []
    if args.reference_frames:
        for path in sorted(glob.glob(args.reference_frames)):
            image = np.asarray(cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB), np.uint8)
            cube = detect_wrist_cube(image)
            if cube is not None and not cube.clipped:
                reference.append(
                    {
                        "frame": path,
                        "area_px": round(cube.area),
                        "side_px_per_m": round(math.sqrt(cube.area) / _CUBE_SIDE),
                    }
                )
    (out / "replay.json").write_text(
        json.dumps({"attempts": rows, "reference_hover_frames": reference}, indent=1)
    )
    if tracks:
        _area_figure(tracks, out)


if __name__ == "__main__":
    _main()
