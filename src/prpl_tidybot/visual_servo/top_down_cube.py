"""Wrist-camera servo for the top-down cube pick, modelled on Tom Silver's
cylinder servo (:mod:`prpl_tidybot.real_sim.plan_executors.visual_servo`).

The ceiling cameras place the floor cube to a centimetre or two, but where their
views overlap their calibrations disagree by ~2 cm, enough for the 2 cm finger pads
to close past it (2026-10-14, run 7). So at the hover, before descending, the gripper
is aligned over the cube by looking down through the wrist camera.

As in Tom's executor: images come from an :class:`ImageSource`; each correction is a
step proportional to the pixel error, at least ``min_step`` and at most ``max_step``;
the next step is computed only once the arm has arrived; alignment must hold within
``tolerance_px`` for ``confirm_looks`` looks in a row; it gives up after
``max_missed_detections`` looks without the cube, on more than ``travel_limit`` of
sideways travel, or after ``max_moves`` alignment nudges; and every look goes into
the result's trace, with raw and annotated frames written to ``debug_dir``.

What differs is the geometry. Tom's grasp approaches a cylinder sideways, so one
lateral axis with a hand-calibrated ``lateral_gain``/``lateral_sign`` suffices. Here
the camera looks down and the cube can be off in both horizontal directions, so the
pixel error is converted to tool motion through a 2x2 Jacobian (pixel motion of the
cube per metre of gripper motion along the tool's x and closing (y) axes), measured
at the start with two ``probe_step`` moves. It absorbs the camera's rotation,
mirroring and fisheye scale near the target, and needs no per-mount calibration. The
target pixel is where the cube appears when the pinch point is right above it: below
the image centre, since the camera sits 5.6 cm off the gripper axis.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.visual_servo.image_sources import ImageSource

# OpenCV HSV bounds of the blue foam cube in the wrist camera.
_HSV_LOW = (95, 90, 60)
_HSV_HIGH = (130, 255, 255)


@dataclass(frozen=True)
class WristCube:
    """The cube's blob in a wrist image: centroid (px), edge angle in the image (rad,
    in [-pi/4, pi/4), the cube being 90-degree symmetric), area (px), and whether it
    touches the image border (a cut-off blob's centroid is biased)."""

    u: float
    v: float
    angle: float
    area: float
    clipped: bool


def detect_wrist_cube(
    image_rgb: NDArray[np.uint8], min_area: float = 300.0
) -> WristCube | None:
    """The largest cube-coloured blob in an RGB wrist image, or None."""
    hsv = cv.cvtColor(np.asarray(image_rgb, np.uint8), cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, np.array(_HSV_LOW), np.array(_HSV_HIGH))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv.contourArea)
    area = float(cv.contourArea(blob))
    if area < min_area:
        return None
    moments = cv.moments(blob)
    angle_deg = cv.minAreaRect(blob)[2]
    angle = (math.radians(angle_deg) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    x, y, w, h = cv.boundingRect(blob)
    rows, cols = mask.shape
    return WristCube(
        u=moments["m10"] / moments["m00"],
        v=moments["m01"] / moments["m00"],
        angle=angle,
        area=area,
        clipped=x <= 1 or y <= 1 or x + w >= cols - 1 or y + h >= rows - 1,
    )


def jacobian_from_moves(
    pixel_moves: Sequence[Sequence[float]], tool_moves: Sequence[Sequence[float]]
) -> NDArray[np.float64]:
    """The 2x2 Jacobian from two probe moves: the cube's pixel displacement for each
    and how far the gripper really moved (tool x, tool y; metres), which under a
    compliant arm is not what was commanded (run 9)."""
    pixels = np.column_stack([np.asarray(m, float) for m in pixel_moves])
    tools = np.column_stack([np.asarray(m, float) for m in tool_moves])
    return np.asarray(pixels @ np.linalg.inv(tools), float)


def tool_correction(
    jacobian: NDArray[np.float64], cube: WristCube, target_uv: Sequence[float]
) -> NDArray[np.float64]:
    """Gripper motion (tool x, tool y; metres) that brings the cube to `target_uv`."""
    error = np.asarray(target_uv, float) - np.array([cube.u, cube.v])
    return np.asarray(np.linalg.solve(jacobian, error), float)


def edge_direction_in_tool(
    jacobian: NDArray[np.float64], angle: float
) -> NDArray[np.float64]:
    """A cube edge's direction (unit 2-vector, tool x/y) from its image angle. The
    Jacobian maps tool directions to image directions up to sign, which a line
    ignores, so its inverse maps back, absorbing the camera's rotation and mirror."""
    d = np.linalg.solve(jacobian, np.array([math.cos(angle), math.sin(angle)]))
    return np.asarray(d / np.linalg.norm(d), float)


@dataclass(frozen=True)
class TopDownServoParams:
    """Distances in metres, tolerances in pixels / radians."""

    target_uv: tuple[float, float] = (312.0, 320.0)
    probe_step: float = 0.015
    tolerance_px: float = 6.0
    confirm_looks: int = 3
    min_step: float = 0.002
    max_step: float = 0.01
    max_area_change_frac: float = 0.3
    area_history: int = 5
    yaw_tolerance: float = math.radians(4.0)
    max_turn: float = math.radians(10.0)
    align_yaw: bool = True
    max_missed_detections: int = 2
    max_moves: int = 10
    pushed_out_frac: float = 0.8
    travel_limit: float = 0.08


@dataclass(frozen=True)
class TopDownServoTrace:
    """What the servo saw and did on one look."""

    look: int
    label: str
    pinch_xy: tuple[float, float]
    yaw: float
    cube: WristCube | None
    error_px: float | None
    delta_tool: tuple[float, float] | None


@dataclass
class TopDownServoResult:
    """Where the servo left the hover (arm-frame pinch x, y and closing yaw: the
    `top_down_joint_targets` arguments) and whether the cube is aligned under it."""

    ax: float
    ay: float
    yaw: float
    aligned: bool
    note: str | None = None
    jacobian: list[list[float]] | None = None
    trace: list[TopDownServoTrace] = field(default_factory=list)


# (label, ax, ay, yaw) -> whether the arm arrived at the hover there.
MoveHover = Callable[[str, float, float, float], bool]
# () -> where the pinch point really is (arm-frame x, y), from the joint readings.
MeasuredXY = Callable[[], tuple[float, float]]
# yaw -> the tool x and y axes (unit 2-vectors, arm frame) at that closing yaw.
ToolAxes = Callable[[float], tuple[NDArray[np.float64], NDArray[np.float64]]]


class TopDownCubeServo:
    """Align the hovering gripper over the cube with the wrist camera."""

    def __init__(
        self,
        image_source: ImageSource,
        params: TopDownServoParams | None = None,
        debug_dir: str | Path | None = None,
    ) -> None:
        self._images = image_source
        self.params = params or TopDownServoParams()
        self._debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._run = 0

    def align(  # pylint: disable=too-many-locals,too-many-return-statements
        self,
        start: tuple[float, float, float],
        move: MoveHover,
        tool_axes: ToolAxes,
        name: str = "servo",
        measured_xy: MeasuredXY | None = None,
    ) -> TopDownServoResult:
        """Align from the hover at ``start`` = (ax, ay, yaw), where the arm is now.

        With `measured_xy` the probe Jacobian uses how far the gripper really moved;
        without it, the commanded probe step."""
        p = self.params
        self._run += 1
        ax, ay, yaw = start
        result = TopDownServoResult(ax, ay, yaw, False)

        areas: list[float] = []

        def look(label: str) -> WristCube | None:
            image = self._images.get_image()
            cube = None if image is None else detect_wrist_cube(image)
            # As in the cylinder servo, a blob whose size jumps from the recent median
            # is a stray detection, not the cube: a miss, not a step target.
            # A blob cut off by the image border changes size legitimately as it
            # moves, so only whole blobs are checked and remembered.
            if cube is not None and not cube.clipped and areas:
                median = float(np.median(areas[-p.area_history :]))
                if abs(cube.area - median) > p.max_area_change_frac * median:
                    label, cube = f"{label}_outlier", None
            if cube is not None and not cube.clipped:
                areas.append(cube.area)
            error = (
                None
                if cube is None
                else math.hypot(p.target_uv[0] - cube.u, p.target_uv[1] - cube.v)
            )
            entry = TopDownServoTrace(
                len(result.trace), label, (ax, ay), yaw, cube, error, None
            )
            result.trace.append(entry)
            self._dump_debug(image, cube, f"{name}_{len(result.trace) - 1:02d}_{label}")
            return cube

        def fail(note: str) -> TopDownServoResult:
            result.ax, result.ay, result.yaw, result.note = ax, ay, yaw, note
            return result

        first = look("hover")
        if first is None:
            return fail("cube not seen in the wrist camera at the hover")
        x_axis, y_axis = tool_axes(yaw)

        def where() -> NDArray[np.float64]:
            return np.array(measured_xy() if measured_xy is not None else (ax, ay))

        def measure() -> tuple[NDArray[np.float64], bool] | str:
            """Probe along tool x and y from the current hover: the Jacobian, and
            whether any probe saw the cube cut off by the image border (its centroid
            then moves less than the cube, so the gain is biased low)."""
            x_axis, y_axis = tool_axes(yaw)
            pixel_moves, tool_moves, cut = [], [], False
            for label, axis in (("probe_x", x_axis), ("probe_y", y_axis)):
                before, cube_before = where(), result.trace[-1].cube
                if cube_before is None:
                    return f"cube lost before {label}"
                seen = None
                # As the cylinder servo walks a cut-off object into view: if a probe
                # pushes the cube out of (or further off) the image, go the other way.
                for sign in (1.0, -1.0):
                    step = sign * p.probe_step
                    px, py = ax + step * axis[0], ay + step * axis[1]
                    if not move(label, px, py, yaw):
                        return f"{label} move did not arrive"
                    seen = look(label)
                    # A cut-off blob's area wobbles as it slides along the edge
                    # (run 13 re-tried a sideways probe); only a clear shrink is out.
                    pushed_out = seen is None or (
                        seen.clipped
                        and seen.area < p.pushed_out_frac * cube_before.area
                    )
                    if not pushed_out:
                        break
                    if sign > 0 and not move("probe_back", ax, ay, yaw):
                        return "probe return did not arrive"
                if seen is None:
                    return f"cube lost during {label} both ways"
                cut = cut or seen.clipped or cube_before.clipped
                moved = (
                    where() - before
                    if measured_xy is not None
                    else p.probe_step * np.asarray(axis)
                )
                tool_moves.append([float(moved @ x_axis), float(moved @ y_axis)])
                pixel_moves.append([seen.u - cube_before.u, seen.v - cube_before.v])
                if not move("probe_back", ax, ay, yaw):
                    return "probe return did not arrive"
                if look("probe_back") is None:
                    return "cube lost after the probe"
            if min(np.linalg.norm(tool_moves, axis=1)) < 0.3 * p.probe_step:
                return "the probes barely moved the gripper"
            if abs(np.linalg.det(np.array(tool_moves))) < 0.05 * p.probe_step**2:
                return "the probe moves were nearly parallel"
            # Each probe must move the cube by at least a few pixels, or the inverse
            # is all noise (a frozen image, the wrong blob).
            if min(np.linalg.norm(pixel_moves, axis=1)) < 4.0:
                return "the probes barely moved the cube in the image"
            return jacobian_from_moves(pixel_moves, tool_moves), cut

        measured = measure()
        if isinstance(measured, str):
            return fail(measured)
        jacobian, provisional = measured
        result.jacobian = jacobian.round(1).tolist()

        origin = (ax, ay)
        cube = result.trace[-1].cube
        missed = 0 if cube is not None else 1
        aligned_looks = 0
        nudges = 0
        while True:
            if cube is None:
                if missed > p.max_missed_detections:
                    return fail(f"cube lost for {missed} looks while aligning")
                cube = look("retry")
                missed = missed + 1 if cube is None else 0
                continue
            if provisional and not cube.clipped:
                # The cube is now fully in view: measure the gain again, unbiased.
                measured = measure()
                if isinstance(measured, str):
                    return fail(measured)
                jacobian, provisional = measured
                result.jacobian = jacobian.round(1).tolist()
                cube = result.trace[-1].cube
                if cube is None:
                    continue
            delta = tool_correction(jacobian, cube, p.target_uv)
            error_px = math.hypot(p.target_uv[0] - cube.u, p.target_uv[1] - cube.v)
            new_yaw = yaw
            # A cut-off blob's edge angle, or a gain measured on one, can turn the
            # wrist the wrong way (run 12); turn only on a whole cube, a step at most.
            if p.align_yaw and not cube.clipped and not provisional:
                edge = edge_direction_in_tool(jacobian, cube.angle)
                x_axis, y_axis = tool_axes(yaw)
                arm_edge = edge[0] * x_axis + edge[1] * y_axis
                edge_yaw = math.atan2(arm_edge[1], arm_edge[0])
                closing = math.atan2(y_axis[1], y_axis[0])
                turn = (edge_yaw - closing + math.pi / 4) % (math.pi / 2) - math.pi / 4
                if abs(turn) > p.yaw_tolerance:
                    new_yaw = yaw + max(-p.max_turn, min(p.max_turn, turn))
            if error_px <= p.tolerance_px and new_yaw == yaw:
                aligned_looks += 1
                if aligned_looks >= p.confirm_looks:
                    result.ax, result.ay, result.yaw, result.aligned = ax, ay, yaw, True
                    return result
                cube = look("confirm")
                missed = 0 if cube is not None else 1
                continue
            aligned_looks = 0
            if nudges >= p.max_moves:
                return fail(f"not aligned after {p.max_moves} nudges")
            norm = float(np.linalg.norm(delta))
            if norm > p.max_step:
                delta = delta * (p.max_step / norm)
            elif 0 < norm < p.min_step and error_px > p.tolerance_px:
                delta = delta * (p.min_step / norm)
            x_axis, y_axis = tool_axes(yaw)
            nx = ax + delta[0] * x_axis[0] + delta[1] * y_axis[0]
            ny = ay + delta[0] * x_axis[1] + delta[1] * y_axis[1]
            if math.hypot(nx - origin[0], ny - origin[1]) > p.travel_limit:
                return fail(
                    f"aligning would move the gripper more than {p.travel_limit} m"
                )
            result.trace[-1] = TopDownServoTrace(
                result.trace[-1].look,
                result.trace[-1].label,
                result.trace[-1].pinch_xy,
                result.trace[-1].yaw,
                cube,
                error_px,
                (float(delta[0]), float(delta[1])),
            )
            if not move("align", nx, ny, new_yaw):
                return fail("alignment move did not arrive")
            ax, ay, yaw = nx, ny, new_yaw
            nudges += 1
            cube = look("align")
            missed = 0 if cube is not None else 1

    def _dump_debug(
        self, image: NDArray[np.uint8] | None, cube: WristCube | None, name: str
    ) -> None:
        if self._debug_dir is None or image is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv.cvtColor(np.asarray(image, np.uint8), cv.COLOR_RGB2BGR)
        stem = f"{name}_r{self._run}"
        cv.imwrite(str(self._debug_dir / f"{stem}_raw.png"), bgr)
        tu, tv = (int(round(c)) for c in self.params.target_uv)
        cv.drawMarker(bgr, (tu, tv), (0, 0, 255), cv.MARKER_CROSS, 30, 2)
        if cube is not None:
            centre = (int(cube.u), int(cube.v))
            cv.circle(bgr, centre, 6, (0, 255, 0), -1)
            cv.line(bgr, centre, (tu, tv), (0, 255, 255), 2)
            label = f"{name}: cube ({cube.u:.0f}, {cube.v:.0f}) -> target ({tu}, {tv})"
        else:
            label = f"{name}: no cube"
        cv.putText(bgr, label, (8, 22), 0, 0.55, (0, 0, 0), 4)
        cv.putText(bgr, label, (8, 22), 0, 0.55, (255, 255, 255), 1)
        cv.imwrite(str(self._debug_dir / f"{stem}.png"), bgr)
