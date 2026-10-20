"""Wrist-camera servo for the top-down cube pick: the align phase of Tom Silver's
cylinder servo (:mod:`prpl_tidybot.real_sim.plan_executors.visual_servo`).

The ceiling cameras place the floor cube to a centimetre or two, but where their
views overlap their calibrations disagree by ~2 cm, enough for the 2 cm finger pads
to close past it (2026-10-14, run 7). So at the hover, before descending, the gripper
is aligned over the cube by looking down through the wrist camera
(:class:`TopDownLateralServo`).

As in Tom's executor: images come from an :class:`ImageSource`; each correction is a
step proportional to the pixel error, clamped between a minimum and a maximum step;
the next step is computed only once the arm has arrived; alignment must hold within
tolerance for several looks in a row; and every look goes into the result's trace,
with raw and annotated frames written to ``debug_dir``. The target pixel is where the
cube appears when the pinch point is right above it: below the image centre, since
the camera sits 5.6 cm off the gripper axis.
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
    trace: list[TopDownServoTrace] = field(default_factory=list)


# (label, ax, ay, yaw) -> whether the arm arrived at the hover there.
MoveHover = Callable[[str, float, float, float], bool]
# yaw -> the tool x and y axes (unit 2-vectors, arm frame) at that closing yaw.
ToolAxes = Callable[[float], tuple[NDArray[np.float64], NDArray[np.float64]]]


@dataclass(frozen=True)
class LateralServoParams:
    """Tom Silver's align parameters (CylinderVisualServoGapExecutor), with
    `lateral_gain` (metres of tool motion per pixel) and `lateral_sign` given per tool
    axis (x, closing y) instead of for his one lateral axis, and the blob's area in
    place of his edge pair's width.

    Distances in metres, tolerances in pixels; `max_ticks` counts looks.
    """

    target_uv: tuple[float, float] = (318.0, 280.0)
    lateral_gain: tuple[float, float] = (0.0003, 0.0003)
    lateral_sign: tuple[float, float] = (1.0, 1.0)
    lateral_tolerance_px: float = 8.0
    lateral_max_step: float = 0.01
    lateral_min_step: float = 0.005
    max_area_change_frac: float = 0.3
    area_history: int = 5
    align_confirm_ticks: int = 3
    max_missed_detections: int = 10
    max_ticks: int = 60
    lateral_travel_limit: float = 0.08


class TopDownLateralServo:
    """The align phase of Tom Silver's cylinder servo
    (:class:`~prpl_tidybot.real_sim.plan_executors.visual_servo.\
    CylinderVisualServoGapExecutor`), at the top-down pick's hover.

    Each look: detect the cube; a miss counts towards `max_missed_detections`; with
    the pixel error within `lateral_tolerance_px` on both image axes for
    `align_confirm_ticks` looks in a row the gripper is aligned; otherwise each tool
    axis whose image axis is out of tolerance steps proportionally to its error,
    clamped to [`lateral_min_step`, `lateral_max_step`], and the next look is taken
    once the arm has arrived. Stops (not aligned) on a lost cube, a step past
    `lateral_travel_limit` along either tool axis from the start, a step that does
    not arrive, or `max_ticks` looks. The descent after it is the pick's own, as
    Tom's open-loop approach follows his align phase.
    """

    def __init__(
        self,
        image_source: ImageSource,
        params: LateralServoParams | None = None,
        debug_dir: str | Path | None = None,
    ) -> None:
        self.params = params or LateralServoParams()
        if not 0.0 <= self.params.lateral_min_step <= self.params.lateral_max_step:
            raise ValueError("need 0 <= lateral_min_step <= lateral_max_step")
        self._image_source = image_source
        self._debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._run = 0
        self._tick = 0
        self._missed = 0
        self._aligned_ticks = 0
        self._recent_areas: list[float] = []

    def align(  # pylint: disable=too-many-return-statements
        self,
        start: tuple[float, float, float],
        move: MoveHover,
        tool_axes: ToolAxes,
        name: str = "lateral",
    ) -> TopDownServoResult:
        """Align from the hover at ``start`` = (ax, ay, yaw), where the arm is now."""
        p = self.params
        self._reset_run()
        ax, ay, yaw = start
        x_axis, y_axis = tool_axes(yaw)
        result = TopDownServoResult(ax, ay, yaw, False)

        def fail(note: str) -> TopDownServoResult:
            result.ax, result.ay, result.note = ax, ay, note
            return result

        while True:
            self._tick += 1
            if self._tick > p.max_ticks:
                return fail(f"lateral servo gave up after {p.max_ticks} looks")
            image = self._image_source.get_image()
            if image is None:
                return fail("lateral servo got no wrist image")
            cube = self._detect(image)
            label = "align"
            self._dump_debug(image, cube, f"{name}_{self._tick:02d}_{label}")
            if cube is None:
                self._missed += 1
                self._record(result, label, (ax, ay), yaw, None, None)
                if self._missed > p.max_missed_detections:
                    return fail(
                        f"lateral servo lost the cube for {self._missed} consecutive "
                        "looks while aligning"
                    )
                continue
            self._missed = 0
            error = (cube.u - p.target_uv[0], cube.v - p.target_uv[1])

            if max(abs(e) for e in error) <= p.lateral_tolerance_px:
                self._aligned_ticks += 1
                self._record(result, label, (ax, ay), yaw, cube, None)
                if self._aligned_ticks >= p.align_confirm_ticks:
                    result.ax, result.ay, result.aligned = ax, ay, True
                    return result
                continue

            self._aligned_ticks = 0
            delta = (self._lateral_step(0, error[0]), self._lateral_step(1, error[1]))
            nx = ax + delta[0] * x_axis[0] + delta[1] * y_axis[0]
            ny = ay + delta[0] * x_axis[1] + delta[1] * y_axis[1]
            travel = self._check_lateral_travel(start, (nx, ny), x_axis, y_axis)
            if travel is not None:
                return fail(travel)
            self._record(result, label, (ax, ay), yaw, cube, delta)
            if not move("align", nx, ny, yaw):
                return fail("lateral servo step did not arrive (refused or short)")
            ax, ay = nx, ny

    def _detect(self, image: NDArray[np.uint8]) -> WristCube | None:
        """Run the detector and reject an area that jumped from the recent median.

        A clipped blob (cut off by the image border) passes without the jump check and
        leaves the area history alone.
        """
        cube = detect_wrist_cube(image)
        if cube is None or cube.clipped:
            return cube
        if self._recent_areas:
            reference = float(np.median(self._recent_areas))
            change = abs(cube.area - reference) / reference
            if change > self.params.max_area_change_frac:
                return None
        self._recent_areas.append(cube.area)
        del self._recent_areas[: -self.params.area_history]
        return cube

    def _lateral_step(self, axis: int, error: float) -> float:
        """Signed move along tool axis `axis` for a pixel error on its image axis:

        proportional, but at least ``lateral_min_step`` and at most
        ``lateral_max_step``; none for an axis already within tolerance.
        """
        p = self.params
        if abs(error) <= p.lateral_tolerance_px:
            return 0.0
        direction = -p.lateral_sign[axis] * np.sign(error)
        magnitude = min(
            max(p.lateral_gain[axis] * abs(error), p.lateral_min_step),
            p.lateral_max_step,
        )
        return float(direction * magnitude)

    def _check_lateral_travel(
        self,
        start: tuple[float, float, float],
        target: tuple[float, float],
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
    ) -> str | None:
        """Refuse to wander further than lateral_travel_limit from the start along
        either tool axis."""
        moved = np.array(target) - np.array(start[:2])
        for axis_name, axis in (("x", x_axis), ("y", y_axis)):
            travel = abs(float(moved @ axis))
            if travel > self.params.lateral_travel_limit:
                return (
                    f"lateral servo stopped: lateral travel {travel:.3f} m along tool "
                    f"{axis_name} exceeds the {self.params.lateral_travel_limit:.3f} m "
                    "limit; the detection is probably wrong"
                )
        return None

    def _reset_run(self) -> None:
        self._run += 1
        self._tick = 0
        self._missed = 0
        self._aligned_ticks = 0
        self._recent_areas = []

    def _record(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        result: TopDownServoResult,
        label: str,
        pinch_xy: tuple[float, float],
        yaw: float,
        cube: WristCube | None,
        delta: tuple[float, float] | None,
    ) -> None:
        error = (
            None
            if cube is None
            else math.hypot(
                cube.u - self.params.target_uv[0], cube.v - self.params.target_uv[1]
            )
        )
        result.trace.append(
            TopDownServoTrace(
                len(result.trace), label, pinch_xy, yaw, cube, error, delta
            )
        )

    def _dump_debug(
        self, image: NDArray[np.uint8] | None, cube: WristCube | None, name: str
    ) -> None:
        _save_look(self._debug_dir, self.params.target_uv, image, cube, name, self._run)


def _save_look(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    debug_dir: Path | None,
    target_uv: Sequence[float],
    image: NDArray[np.uint8] | None,
    cube: WristCube | None,
    name: str,
    run: int,
) -> None:
    """Write a wrist look, raw and annotated (target cross, cube centre), to `debug_dir`
    if set."""
    if debug_dir is None or image is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv.cvtColor(np.asarray(image, np.uint8), cv.COLOR_RGB2BGR)
    stem = f"{name}_r{run}"
    cv.imwrite(str(debug_dir / f"{stem}_raw.png"), bgr)
    tu, tv = (int(round(c)) for c in target_uv)
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
    cv.imwrite(str(debug_dir / f"{stem}.png"), bgr)
