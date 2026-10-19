"""Place-and-look calibration of the wrist servo's target pixel (``target_uv``).

``target_uv`` is where the cube appears in the wrist camera at the hover when the
pinch point is right above it. The 2026-10-14 value came from compliant-controller
hovers that settled short; under the onboard controller the held pick of 2026-10-19
saw the cube ~145 px lower. This measures it directly, under the onboard controller
the pick uses:

1. The arm goes to the grasp pose in front of the robot, fingers open; the person
   puts the cube on the floor between them, and they close on it.
2. Optionally the cube is centred: the fingers centre it only along their closing
   axis, so it is set down, regrasped with the wrist turned 90 degrees (centring it
   along the pads), set down and regrasped square again.
3. Per sample: the cube is set down at the grasp height, where it was picked up
   from the floor (never dropped: it bounces), the arm lifts straight up to the
   pick's hover and looks (and optionally from higher up), then regrasps. Between
   samples the place point moves by up to ``max_offset``.

The cube is out of the wrist camera's view at the grasp height (the camera is
5.6 cm off the gripper axis), so a set-down cannot be checked by eye there; a hover
look that misses the cube or moved more than ``max_shift_px`` from the first good one
is a bad sample. Every move and gripper action is confirmed first.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    cube_closing_yaw,
    grasp_succeeded,
    pointing_down_joints,
)
from prpl_tidybot.toss_cycles import _AngularMover  # pylint: disable=protected-access
from prpl_tidybot.visual_servo.image_sources import ImageSource
from prpl_tidybot.visual_servo.top_down_cube import WristCube, detect_wrist_cube

PromptFn = Callable[[str], str]

# The wrist camera's optical centre above the pinch point (gen3.xml's camera mount
# plus the 0.12 m pinch offset beyond the flange), for the gain at another height.
_CAMERA_ABOVE_PINCH = 0.123
_CUBE_SIDE = 0.05


@dataclass(frozen=True)
class CalibrationParams:
    """Where (arm frame, metres) and how often to place and look."""

    arm_xy: tuple[float, float] = (0.55, 0.0)
    yaw: float = 0.0
    samples: int = 5
    max_offset: float = 0.01
    seed: int = 0
    center: bool = True
    # Above the grasp height while carrying the cube to the next place point.
    lift: float = 0.02
    # A second look this far above the hover (0: none).
    higher_look: float = 0.05
    # Open in this many gripper moves, as the gripper API has no speed setting.
    open_steps: int = 4
    max_shift_px: float = 15.0
    gripper_timeout_s: float = 4.0


@dataclass
class CalibrationSample:
    """One set-down and its looks."""

    index: int
    place_xy: tuple[float, float]
    hover: dict[str, Any] | None
    higher: dict[str, Any] | None
    good: bool
    note: str | None = None


@dataclass
class CalibrationResult:
    """The samples, the mean target pixel over the good ones, and why it stopped
    early, if it did."""

    samples: list[CalibrationSample] = field(default_factory=list)
    target_uv: tuple[float, float] = (math.nan, math.nan)
    std_uv: tuple[float, float] = (math.nan, math.nan)
    n_good: int = 0
    note: str | None = None
    moves: list[dict[str, Any]] = field(default_factory=list)
    # The gain at the hover implied by the higher looks, if the cube was whole there.
    gain: dict[str, float] | None = None


class _Stop(Exception):
    """End the calibration with this note."""


def calibrate_wrist_target(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    arm: ArmInterface,
    camera: ImageSource | None,
    geom: TopDownPickGeometry,
    params: CalibrationParams,
    prompt: PromptFn,
    out_dir: Path,
    close_position: float,
) -> CalibrationResult:
    """Run the place-and-look calibration; write calibration.json and the looks to
    `out_dir`."""
    if params.samples < 1:
        raise ValueError("need at least one sample")
    if not 0.0 <= params.max_offset <= 0.01:
        raise ValueError("place offsets must be within 1 cm")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(params.seed)
    result = CalibrationResult()
    mover = _AngularMover(
        arm, {"gripper_timeout_s": params.gripper_timeout_s}, geom, result.moves
    )
    grasp_z = geom.floor_z + geom.grasp_height
    hover_z = grasp_z + geom.hover_height
    closing = cube_closing_yaw(params.yaw, geom)
    c, s = math.cos(closing), math.sin(closing)
    x_axis = np.array([-s, c])

    def ask(question: str) -> None:
        if prompt(f"{question} [y/N]: ").strip().lower() not in ("y", "yes"):
            raise _Stop(f"declined: {question}")

    def go(name: str, xy: tuple[float, float], z: float, yaw: float) -> None:
        joints = pointing_down_joints((xy[0], xy[1], z), yaw, geom, mover.joints())
        ask(
            f"{name}: move the pinch point to ({xy[0]:+.3f}, {xy[1]:+.3f}, {z:+.3f}) m, "
            f"fingers at {math.degrees(yaw):+.0f} deg?"
        )
        if not mover.move(name, joints):
            raise _Stop(f"{name} did not arrive")

    def grasp(name: str) -> None:
        ask(f"{name}: close the gripper on the cube?")
        reading = mover.gripper(close_position)
        if not grasp_succeeded(reading, geom):
            ask(f"{name}: the grasp failed (gripper {reading:.2f}); open the gripper?")
            mover.gripper(0.0)
            raise _Stop(f"{name}: grasp failed (gripper {reading:.2f})")

    def set_down(name: str) -> None:
        ask(f"{name}: open the gripper (the cube is on the floor)?")
        start = arm.get_gripper_state()
        for k in range(1, params.open_steps + 1):
            mover.gripper(start * (1 - k / params.open_steps))

    def look(name: str) -> WristCube | None:
        image = None if camera is None else camera.get_image()
        cube = None if image is None else detect_wrist_cube(image)
        if image is not None:
            _save(out_dir, name, image, cube)
        return cube

    xy = (float(params.arm_xy[0]), float(params.arm_xy[1]))
    reference: WristCube | None = None
    mover.begin()
    try:
        ask("open the gripper?")
        mover.gripper(0.0)
        go("hover", xy, hover_z, closing)
        go("grasp", xy, grasp_z, closing)
        _wait_for_cube(prompt, arm, x_axis)
        grasp("grasp")
        if params.center:
            turned = closing + math.pi / 2
            set_down("center_release")
            go("center_up", xy, hover_z, closing)
            go("center_turn", xy, hover_z, turned)
            go("center_down", xy, grasp_z, turned)
            grasp("center_grasp_turned")
            set_down("center_release_turned")
            go("center_up_turned", xy, hover_z, turned)
            go("center_turn_back", xy, hover_z, closing)
            go("center_down_square", xy, grasp_z, closing)
            grasp("center_grasp")
        for i in range(1, params.samples + 1):
            if i > 1 and params.max_offset > 0:
                r = params.max_offset * math.sqrt(rng.uniform())
                a = rng.uniform(0, 2 * math.pi)
                new = (
                    params.arm_xy[0] + r * math.cos(a),
                    params.arm_xy[1] + r * math.sin(a),
                )
                go(f"sample_{i:02d}_lift", xy, grasp_z + params.lift, closing)
                xy = (float(new[0]), float(new[1]))
                go(f"sample_{i:02d}_shift", xy, grasp_z + params.lift, closing)
                go(f"sample_{i:02d}_place", xy, grasp_z, closing)
            set_down(f"sample_{i:02d}_release")
            look(f"sample_{i:02d}_floor")
            go(f"sample_{i:02d}_hover", xy, hover_z, closing)
            cube = look(f"sample_{i:02d}_hover")
            higher = None
            if params.higher_look > 0:
                go(f"sample_{i:02d}_higher", xy, hover_z + params.higher_look, closing)
                higher = look(f"sample_{i:02d}_higher")
            sample = CalibrationSample(
                i, xy, _blob(cube), _blob(higher), cube is not None
            )
            if cube is None:
                sample.note = "cube not seen at the hover"
            elif (
                reference is not None
                and math.hypot(cube.u - reference.u, cube.v - reference.v)
                > params.max_shift_px
            ):
                sample.good = False
                sample.note = (
                    f"cube at ({cube.u:.0f}, {cube.v:.0f}), more than "
                    f"{params.max_shift_px:.0f} px from the first good sample"
                )
            elif reference is None:
                reference = cube
            result.samples.append(sample)
            if i < params.samples:
                go(f"sample_{i:02d}_regrasp", xy, grasp_z, closing)
                grasp(f"sample_{i:02d}_regrasp")
    except _Stop as e:
        result.note = str(e)
    finally:
        mover.end()
    _summarise(result, geom, params)
    (out_dir / "calibration.json").write_text(json.dumps(_as_json(result), indent=1))
    return result


def _wait_for_cube(prompt: PromptFn, arm: ArmInterface, x_axis: NDArray[Any]) -> None:
    """The person puts the cube on the floor between the open fingers."""
    del arm, x_axis
    prompt(
        "Put the cube on the floor between the open fingers, centred along the pads "
        "and square to them, then press Enter: "
    )


def _blob(cube: WristCube | None) -> dict[str, Any] | None:
    if cube is None:
        return None
    return {
        "u": cube.u,
        "v": cube.v,
        "area": cube.area,
        "clipped": cube.clipped,
        "angle_deg": math.degrees(cube.angle),
    }


def _summarise(
    result: CalibrationResult, geom: TopDownPickGeometry, params: CalibrationParams
) -> None:
    good = [s.hover for s in result.samples if s.good and s.hover is not None]
    result.n_good = len(good)
    if not good:
        return
    uv = np.array([[h["u"], h["v"]] for h in good])
    result.target_uv = (float(uv[:, 0].mean()), float(uv[:, 1].mean()))
    std = uv.std(axis=0, ddof=1) if len(good) > 1 else np.zeros(2)
    result.std_uv = (float(std[0]), float(std[1]))
    result.gain = _hover_gain(result, geom, params.higher_look)


def _hover_gain(
    result: CalibrationResult, geom: TopDownPickGeometry, higher: float
) -> dict[str, float] | None:
    """px/m at the hover implied by the whole cube seen from higher up: its side in
    pixels over 5 cm, scaled by the depth ratio (camera to the cube's top face)."""
    sides = [
        math.sqrt(s.higher["area"]) / _CUBE_SIDE
        for s in result.samples
        if s.good and s.higher is not None and not s.higher["clipped"]
    ]
    if not sides or higher <= 0:
        return None
    depth_hover = (
        _CAMERA_ABOVE_PINCH + geom.grasp_height + geom.hover_height - _CUBE_SIDE
    )
    gain_higher = float(np.median(sides))
    return {
        "higher_px_per_m": gain_higher,
        "hover_px_per_m": gain_higher * (depth_hover + higher) / depth_hover,
        "n": float(len(sides)),
    }


def _as_json(result: CalibrationResult) -> dict[str, Any]:
    u, v = result.target_uv
    return {
        "note": result.note,
        "n_good": result.n_good,
        "n_total": len(result.samples),
        "target_uv": [u, v],
        "std_uv": list(result.std_uv),
        "gain": result.gain,
        "yaml": (
            f"target_uv: [{u:.0f}, {v:.0f}]"
            if result.n_good
            else "target_uv: (no good samples)"
        ),
        "samples": [asdict(s) for s in result.samples],
        "moves": result.moves,
    }


def _save(
    out_dir: Path, name: str, image: NDArray[Any], cube: WristCube | None
) -> None:
    bgr = cv.cvtColor(np.asarray(image, np.uint8), cv.COLOR_RGB2BGR)
    cv.imwrite(str(out_dir / f"{name}_raw.png"), bgr)
    if cube is not None:
        cv.circle(bgr, (int(cube.u), int(cube.v)), 6, (0, 255, 0), -1)
        label = f"{name}: cube ({cube.u:.0f}, {cube.v:.0f}) area {cube.area:.0f}" + (
            " clipped" if cube.clipped else ""
        )
    else:
        label = f"{name}: no cube"
    cv.putText(bgr, label, (8, 22), 0, 0.55, (0, 0, 0), 4)
    cv.putText(bgr, label, (8, 22), 0, 0.55, (255, 255, 255), 1)
    cv.imwrite(str(out_dir / f"{name}.png"), bgr)
