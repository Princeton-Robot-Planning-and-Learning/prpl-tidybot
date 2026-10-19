"""Tom Silver's align loop for the top-down cube pick (TopDownLateralServo) against the
kinder simulator's wrist camera.

The sim TidyBot hovers 10 cm above where the "ceiling cameras" put the cube; the
cube really is 0-3 cm away in one of eight directions. For each case the servo
aligns from the rendered wrist images, then the sim closes the gripper at the
grasp pose. Reported per case: the pinch-to-cube error before and after, the
servo's looks and moves, and whether the sim's grasp check takes the cube, at the
uncorrected and at the aligned pose.

The sim camera is not the real one (kinder mounts it at its own offset from the end
effector), so the target pixel and the per-axis gain and sign are first calibrated
in the sim, the way target_uv was calibrated from held picks and the gain from the
2026-10-14 probes on the robot: a render with the cube right under the pinch point,
and one each with it 1 cm along tool x and y. The gain keeps Tom's ratio to the
measured one (his 0.0003 m/px is 0.88 of 1/2920 px/m). The calibration's full 2x2
Jacobian is reported, so a sim camera whose axes are not along the tool's shows up.

    python scripts/sim_lateral_servo_demo.py \
        --out ../hitl-pmp-results/tidybot-real/analysis/2026-10-19-oneshot/sim

The sim's grasp check is lenient: it takes a cube touching a 2 x 2 x 7 cm box at the
end effector, i.e. up to ~3.5 cm off. The residual is the stricter measure.
"""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import functools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2 as cv
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
from kinder.envs.kinematic3d.prpl3d import (
    ObjectCentricPrplLab3DEnv,
    PrplLab3DEnvConfig,
)
from pybullet_helpers.geometry import matrix_from_quat

from prpl_tidybot.top_down_pick import TopDownPickGeometry, top_down_joint_targets
from prpl_tidybot.toss_cycles import _servo_step_is_small, _tool_axes
from prpl_tidybot.video import write_mp4
from prpl_tidybot.visual_servo.image_sources import KinderEECameraSource
from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownLateralServo,
    detect_wrist_cube,
)

# Tom's lateral_gain against the 2026-10-14 measured gain (0.0003 m/px * 2920 px/m).
_TOM_GAIN_RATIO = 0.0003 * 2920.0
# The lab's blue foam cube, 5 cm on a side.
_CUBE_RGBA = (0.16, 0.35, 0.9, 1.0)
_CUBE_HALF = 0.025
# The robot faces +y into the floor area in front of the lab.
_BASE = (0.3, -1.2, math.pi / 2)
_ARM_TARGET = (0.55, 0.0)
_CUBE_YAW_REL = 0.0
_STEP_LIMITS = {"max_joint_step": 0.2, "max_wrist_step": 0.9}


class _Sim:
    """One sim scene: the robot parked at _BASE, one cube, joints set directly (the
    sim arm arrives exactly, as the onboard controller does within 0.1 deg)."""

    def __init__(self) -> None:
        config = PrplLab3DEnvConfig(
            block_half_extents=(_CUBE_HALF,) * 3,
            block_rgba=_CUBE_RGBA,
            ee_camera_image_height=480,
        )
        self.env = ObjectCentricPrplLab3DEnv(
            num_cubes=1, config=config, allow_state_access=True
        )
        state, _ = self.env.reset(seed=0)
        self.state = state.copy()
        self.robot = self.state.get_object_from_name("robot")
        self.cube = self.state.get_object_from_name("cube0")
        for name, value in zip(("pos_base_x", "pos_base_y", "pos_base_rot"), _BASE):
            self.state.set(self.robot, name, value)
        self.geom = TopDownPickGeometry()
        self.joints: list[float] = []
        self.hover, self.grasp = top_down_joint_targets(
            *_ARM_TARGET, _CUBE_YAW_REL, self.geom
        )

    def set_joints(self, joints: list[float]) -> None:
        """Command the arm (it arrives)."""
        self.joints = list(joints)
        state = self.state.copy()
        for i, q in enumerate(self.joints):
            state.set(self.robot, f"joint_{i + 1}", float(q))
        self.env.set_state(state)

    def place_cube(self, xy: np.ndarray) -> None:
        """Put the cube on the floor at world `xy`, square to the base."""
        half_yaw = (_BASE[2] + _CUBE_YAW_REL) / 2
        for name, value in (
            ("pose_x", xy[0]),
            ("pose_y", xy[1]),
            ("pose_z", _CUBE_HALF),
            ("pose_qx", 0.0),
            ("pose_qy", 0.0),
            ("pose_qz", math.sin(half_yaw)),
            ("pose_qw", math.cos(half_yaw)),
        ):
            self.state.set(self.cube, name, float(value))
        self.set_joints(self.joints or self.hover)

    def pinch(self) -> tuple[np.ndarray, np.ndarray]:
        """The sim end effector (its grasp point): world position and approach axis."""
        pose = self.env.robot.arm.get_end_effector_pose()
        rot = np.asarray(matrix_from_quat(pose.orientation))
        return np.array(pose.position), rot[:, 2]

    def camera(self) -> KinderEECameraSource:
        """The wrist camera at the current scene with the commanded joints."""
        return KinderEECameraSource(self.env, self.state, lambda: self.joints)

    def to_world(self, arm_vec: np.ndarray) -> np.ndarray:
        """An arm-frame horizontal direction in the world frame."""
        c, s = math.cos(_BASE[2]), math.sin(_BASE[2])
        return np.array(
            [c * arm_vec[0] - s * arm_vec[1], s * arm_vec[0] + c * arm_vec[1]]
        )

    def move(self, _label: str, ax: float, ay: float, yaw: float) -> bool:
        """The toss_cycles hover move: IK at the hover height, the jump check, go."""
        joints, _ = top_down_joint_targets(ax, ay, yaw, self.geom, seed=self.joints)
        if not _servo_step_is_small(self.joints, joints, _STEP_LIMITS):
            return False
        self.set_joints(joints)
        return True

    def grasps_at(self, ax: float, ay: float) -> bool:
        """Descend to the grasp pose at (ax, ay) and close: does the sim take it?"""
        _, grasp = top_down_joint_targets(
            ax, ay, _CUBE_YAW_REL, self.geom, seed=self.joints
        )
        self.set_joints(grasp)
        self.env._grasped_object = None
        action = np.zeros(11, dtype=np.float32)
        action[-1] = -1.0
        self.env.step(action)
        held = self.env._grasped_object == "cube0"
        self.env._grasped_object = None
        return held


def _calibrate(sim: _Sim, tool_axes: Any) -> dict[str, Any]:
    """Target pixel and per-axis gain/sign of the sim camera at the hover."""
    sim.set_joints(sim.hover)
    pinch, approach = sim.pinch()
    x_axis, y_axis = (sim.to_world(a) for a in tool_axes(_CUBE_YAW_REL))
    looks = {}
    for name, shift in (
        ("centre", np.zeros(2)),
        ("x", 0.01 * x_axis),
        ("y", 0.01 * y_axis),
    ):
        sim.place_cube(pinch[:2] + shift)
        image = sim.camera().get_image()
        cube = None if image is None else detect_wrist_cube(image)
        if cube is None:
            raise SystemExit(
                f"Calibration: the sim wrist camera does not see the cube ({name}); "
                "the sim cannot stand in for the robot here."
            )
        looks[name] = np.array([cube.u, cube.v])
    # The cube 1 cm along +x relative to the pinch is the gripper 1 cm along -x.
    jac = -np.column_stack([looks["x"] - looks["centre"], looks["y"] - looks["centre"]])
    jac /= 0.01
    k = np.abs(np.diag(jac))
    return {
        "pinch_world": pinch.tolist(),
        "approach_axis_world": approach.tolist(),
        "target_uv": looks["centre"].tolist(),
        "jacobian_px_per_m": jac.round(1).tolist(),
        "off_diagonal_frac": float(max(abs(jac[0, 1]) / k[1], abs(jac[1, 0]) / k[0])),
        "lateral_sign": np.sign(np.diag(jac)).tolist(),
        "lateral_gain": (_TOM_GAIN_RATIO / k).tolist(),
    }


def _case(
    sim: _Sim,
    params: LateralServoParams,
    offset_world: np.ndarray,
    name: str,
    frames_dir: Path,
) -> dict[str, Any]:
    sim.set_joints(sim.hover)
    estimate = sim.pinch()[0][:2]
    cube_xy = estimate + offset_world
    sim.place_cube(cube_xy)
    held_uncorrected = sim.grasps_at(*_ARM_TARGET)
    sim.set_joints(sim.hover)
    servo = TopDownLateralServo(sim.camera(), params, frames_dir)
    result = servo.align(
        (*_ARM_TARGET, _CUBE_YAW_REL),
        sim.move,
        functools.partial(_tool_axes, sim.geom),
        name,
    )
    after = sim.pinch()[0][:2]
    held = sim.grasps_at(result.ax, result.ay) if result.aligned else False
    errors = [t.error_px for t in result.trace]
    return {
        "case": name,
        "offset_m": offset_world.tolist(),
        "offset_cm": round(float(np.linalg.norm(offset_world)) * 100, 1),
        "aligned": result.aligned,
        "note": result.note,
        "looks": len(result.trace),
        "moves": sum(t.delta_tool is not None for t in result.trace),
        "error_px_before": errors[0] if errors else None,
        "error_px_after": errors[-1] if errors else None,
        "residual_mm_before": round(
            float(np.linalg.norm(estimate - cube_xy)) * 1000, 1
        ),
        "residual_mm_after": round(float(np.linalg.norm(after - cube_xy)) * 1000, 1),
        "sim_grasp_uncorrected": held_uncorrected,
        "sim_grasp_aligned": held,
    }


def _media(frames_dir: Path, name: str, out: Path) -> None:
    """Before/after annotated frames and an mp4 of every look of one case."""
    annotated = sorted(
        f for f in frames_dir.glob(f"{name}_*.png") if not f.stem.endswith("_raw")
    )
    if not annotated:
        return
    first, last = cv.imread(str(annotated[0])), cv.imread(str(annotated[-1]))
    cv.imwrite(str(out / f"{name}_before.png"), first)
    cv.imwrite(str(out / f"{name}_after.png"), last)
    frames = [cv.cvtColor(cv.imread(str(f)), cv.COLOR_BGR2RGB) for f in annotated]
    mp4 = out / f"{name}.mp4"
    write_mp4(frames + [frames[-1]] * 3, mp4, fps=3)
    # OpenCV builds without an H.264 encoder fall back to mp4v, which browsers do
    # not play; re-encode with imageio's ffmpeg.
    h264 = mp4.with_suffix(".h264.mp4")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4)]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264)],
        check=True,
    )
    h264.replace(mp4)


def _figure(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        ax.plot(
            [0, 1],
            [row["residual_mm_before"], row["residual_mm_after"]],
            color="#0072B2",
            alpha=0.5,
            marker="o" if row["aligned"] else "x",
        )
    held = sum(r["sim_grasp_aligned"] for r in rows)
    ax.set_xticks([0, 1], ["planned hover", "after the lateral servo"])
    ax.set_ylabel("pinch-to-cube error (mm)")
    ax.set_title(
        f"Sim wrist camera: {sum(r['aligned'] for r in rows)}/{len(rows)} aligned, "
        f"sim grasp {held}/{len(rows)}"
    )
    fig.tight_layout()
    fig.savefig(out / "residuals.png", dpi=150)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets-cm", type=float, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--directions", type=int, default=8)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    sim = _Sim()
    tool_axes = functools.partial(_tool_axes, sim.geom)
    calibration = _calibrate(sim, tool_axes)
    print(json.dumps(calibration, indent=1))
    if calibration["off_diagonal_frac"] > 0.15:
        print(
            "WARNING: the sim camera's image axes are not along the tool's "
            f"(off-diagonal {calibration['off_diagonal_frac']:.2f}); a per-axis gain "
            "is the wrong model for this camera."
        )
    params = LateralServoParams(
        target_uv=tuple(calibration["target_uv"]),
        lateral_gain=tuple(calibration["lateral_gain"]),
        lateral_sign=tuple(calibration["lateral_sign"]),
    )
    frames_dir = out / "frames"

    rows = []
    for cm in args.offsets_cm:
        directions = 1 if cm == 0 else args.directions
        for d in range(directions):
            angle = 2 * math.pi * d / directions
            offset = cm / 100 * np.array([math.cos(angle), math.sin(angle)])
            name = f"off{cm:g}cm_dir{round(math.degrees(angle)):03d}"
            row = _case(sim, params, offset, name, frames_dir)
            rows.append(row)
            _media(frames_dir, name, out)
            print(
                f"{name}: aligned={row['aligned']} moves={row['moves']} "
                f"error {row['residual_mm_before']} -> {row['residual_mm_after']} mm, "
                f"sim grasp {row['sim_grasp_uncorrected']} -> "
                f"{row['sim_grasp_aligned']} {row['note'] or ''}"
            )
    (out / "results.json").write_text(
        json.dumps({"calibration": calibration, "cases": rows}, indent=1)
    )
    _figure(rows, out)
    aligned = sum(r["aligned"] for r in rows)
    held = sum(r["sim_grasp_aligned"] for r in rows)
    print(f"aligned {aligned}/{len(rows)}; sim grasp after aligning {held}/{len(rows)}")


if __name__ == "__main__":
    _main()
