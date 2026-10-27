"""Offline-testable adaptation of Josh's best-view and compliant-pick geometry.

Reference: prpl-tidybot fe539833, cube_localizer.py and robot_skills.py.
All hardware geometry and motion bounds are explicit calibration inputs. These
helpers neither estimate contact forces nor issue commands to a robot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from prpl_tidybot.sprout.sequence import GraspPose


def _positive(*values: float) -> None:
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected finite positive calibration values")


@dataclass(frozen=True)
class TopDownGeometry:
    """Vertical top-down gripper geometry; map floor is Z=0.

    Clearance is the minimum required fingertip clearance, *after* allowing
    downward tracking/calibration error. Contact height must be measured/modelled
    separately: the minimum permissible height is not a grasp-height estimate.
    """

    floor_z_arm_m: float
    pad_drop_m: float
    minimum_clearance_m: float
    downward_error_budget_m: float

    def __post_init__(self) -> None:
        _positive(self.pad_drop_m, self.minimum_clearance_m)
        if not math.isfinite(self.floor_z_arm_m):
            raise ValueError("Invalid arm-frame floor height")
        if (
            not math.isfinite(self.downward_error_budget_m)
            or self.downward_error_budget_m < 0
        ):
            raise ValueError("Invalid downward error budget")

    @property
    def minimum_pinch_height_m(self) -> float:
        """Return the pinch height needed for the configured floor clearance."""
        return self.pad_drop_m + self.minimum_clearance_m + self.downward_error_budget_m

    def validate_pose(self, pose: GraspPose) -> None:
        """Reject a grasp that violates the calibrated frame or floor clearance."""
        pose.validate()
        if pose.z < self.minimum_pinch_height_m - 1e-9:
            raise ValueError(
                "Grasp violates fingertip floor clearance including error budget"
            )

    def arm_z(self, pose: GraspPose) -> float:
        """Convert map pinch height to the calibrated arm frame."""
        self.validate_pose(pose)
        return self.floor_z_arm_m + pose.z


@dataclass(frozen=True)
class CameraEstimate:
    """One independent, identity-checked estimate; timestamps use local monotonic
    time."""

    camera: int
    grasp: GraspPose
    quality: float
    calibrated_region: bool
    off_nadir_m: float
    captured_at: float
    clear_of_seam: bool = True


@dataclass(frozen=True)
class CameraChoice:
    """Selected camera observation and its cross-view disagreement."""

    estimate: CameraEstimate
    disagreement_m: float
    needs_calibration_check: bool


def select_camera(
    estimates: Sequence[CameraEstimate],
    *,
    now: float,
    max_age_s: float,
    minimum_quality: float,
    disagreement_warning_m: float,
) -> CameraChoice:
    """Never average mismatched views; prefer calibrated, good, overhead views.

    Quality scores must be comparable. A caller must exclude seam-clipped target
    patches and establish left-arm identity before submitting estimates.
    """
    _positive(max_age_s, disagreement_warning_m)
    if not math.isfinite(now) or not 0 <= minimum_quality <= 1:
        raise ValueError("Invalid camera selection parameters")
    valid = []
    seen = set()
    for estimate in estimates:
        if estimate.camera in seen or estimate.camera < 0:
            raise ValueError("Camera estimates must have distinct nonnegative IDs")
        seen.add(estimate.camera)
        estimate.grasp.validate()
        if not all(
            math.isfinite(v)
            for v in (estimate.quality, estimate.off_nadir_m, estimate.captured_at)
        ):
            raise ValueError("Invalid camera estimate")
        if not 0 <= estimate.quality <= 1 or estimate.off_nadir_m < 0:
            raise ValueError("Invalid camera quality or distance")
        if (
            estimate.clear_of_seam
            and 0 <= now - estimate.captured_at <= max_age_s
            and estimate.quality >= minimum_quality
        ):
            valid.append(estimate)
    if not valid:
        raise ValueError("No fresh, unambiguous camera estimate clear of the seam")
    spread = max(
        (
            math.hypot(a.grasp.x - b.grasp.x, a.grasp.y - b.grasp.y)
            for a in valid
            for b in valid
        ),
        default=0.0,
    )
    pool = [e for e in valid if e.calibrated_region] or valid
    best = max(e.quality for e in pool)
    chosen = min(
        (e for e in pool if e.quality >= best - 0.03),
        key=lambda e: (e.off_nadir_m, e.camera),
    )
    return CameraChoice(chosen, spread, spread > disagreement_warning_m)


def corrected_descent_aim(
    goal: GraspPose,
    measured: GraspPose,
    previous_aim: GraspPose,
    geometry: TopDownGeometry,
    max_correction_m: float,
) -> GraspPose:
    """Josh's measured-error correction, with no downward overdrive at contact.

    The returned target still needs path/force/cancellation checks and a finite
    correction-round limit in the hardware backend. This is not force guarding.
    """
    _positive(max_correction_m)
    geometry.validate_pose(goal)
    measured.validate()
    geometry.validate_pose(previous_aim)
    delta = [
        previous_aim.x - measured.x,
        previous_aim.y - measured.y,
        max(0.0, previous_aim.z - measured.z),
    ]
    length = math.sqrt(sum(v * v for v in delta))
    if length > max_correction_m:
        delta = [v * max_correction_m / length for v in delta]
    aim = replace(goal, x=goal.x + delta[0], y=goal.y + delta[1], z=goal.z + delta[2])
    geometry.validate_pose(aim)
    return aim


@dataclass(frozen=True)
class WristObservation:
    """A unique match of the operator-initialized left-forearm patch."""

    uv: tuple[float, float]
    captured_at: float


@dataclass(frozen=True)
class WristCalibration:
    """Pixel-to-map XY response measured at this specific hover and tool yaw."""

    target_uv: tuple[float, float]
    pixel_error_to_map_xy: tuple[tuple[float, float], tuple[float, float]]
    tolerance_px: float
    max_step_m: float
    max_travel_m: float
    max_age_s: float
    timeout_s: float
    stable_frames: int
    max_updates: int
    hover_height_tolerance_m: float
    hover_yaw_tolerance_rad: float

    def __post_init__(self) -> None:
        _positive(
            self.tolerance_px,
            self.max_step_m,
            self.max_travel_m,
            self.max_age_s,
            self.timeout_s,
            self.hover_height_tolerance_m,
            self.hover_yaw_tolerance_rad,
        )
        if (
            len(self.target_uv) != 2
            or len(self.pixel_error_to_map_xy) != 2
            or any(len(row) != 2 for row in self.pixel_error_to_map_xy)
        ):
            raise ValueError("Expected a 2D target and 2x2 pixel-to-map matrix")
        values = (
            *self.target_uv,
            *self.pixel_error_to_map_xy[0],
            *self.pixel_error_to_map_xy[1],
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite wrist calibration")
        a, b = self.pixel_error_to_map_xy
        if abs(a[0] * b[1] - a[1] * b[0]) < 1e-12:
            raise ValueError("Degenerate pixel-to-map calibration")
        if (
            not isinstance(self.stable_frames, int)
            or not isinstance(self.max_updates, int)
            or self.stable_frames < 1
            or self.max_updates < self.stable_frames
        ):
            raise ValueError("Invalid wrist observation budget")


class WristAligner:
    """Bounded XY corrections; call only after measured arrival at each target.

    Fixed hover Z/yaw and a fixed camera are required throughout alignment.
    Observations must be captured after the preceding move settles; a cached
    frame cannot establish convergence. A missing detection is a hard failure.
    """

    def __init__(
        self, calibration: WristCalibration, hover: GraspPose, now: float
    ) -> None:
        hover.validate()
        if not math.isfinite(now):
            raise ValueError("Invalid alignment start time")
        self.calibration, self.hover, self.started = calibration, hover, now
        self.last_capture = -math.inf
        self.last_update = now
        self.updates = self.stable = 0
        self.travel = 0.0

    def update(
        self, observation: WristObservation | None, measured: GraspPose, *, now: float
    ) -> tuple[GraspPose, bool]:
        """Compute a bounded correction from a fresh wrist observation."""
        p = self.calibration
        measured.validate()
        if (
            not math.isfinite(now)
            or now < self.last_update
            or now - self.started > p.timeout_s
            or self.updates >= p.max_updates
        ):
            raise ValueError("Wrist alignment time/update budget exceeded")
        if observation is None:
            raise ValueError("Forearm patch lost or ambiguous")
        if (
            not all(
                math.isfinite(v) for v in (*observation.uv, observation.captured_at)
            )
            or not 0 <= now - observation.captured_at <= p.max_age_s
            or observation.captured_at <= self.last_capture
            or observation.captured_at < self.last_update
        ):
            raise ValueError(
                "Wrist image is stale, repeated, or predates the previous update"
            )
        if (
            abs(measured.z - self.hover.z) > p.hover_height_tolerance_m
            or abs(math.remainder(measured.yaw - self.hover.yaw, 2 * math.pi))
            > p.hover_yaw_tolerance_rad
        ):
            raise ValueError(
                "Hover height/yaw changed; wrist calibration no longer applies"
            )
        self.last_capture, self.last_update = observation.captured_at, now
        self.updates += 1
        error = [observation.uv[i] - p.target_uv[i] for i in range(2)]
        if math.hypot(*error) <= p.tolerance_px:
            self.stable += 1
            return measured, self.stable >= p.stable_frames
        self.stable = 0
        step = [
            sum(row[i] * error[i] for i in range(2)) for row in p.pixel_error_to_map_xy
        ]
        length = math.hypot(*step)
        if length > p.max_step_m:
            step = [v * p.max_step_m / length for v in step]
        distance = math.hypot(*step)
        goal = replace(measured, x=measured.x + step[0], y=measured.y + step[1])
        if (
            self.travel + distance > p.max_travel_m
            or math.hypot(goal.x - self.hover.x, goal.y - self.hover.y) > p.max_travel_m
        ):
            raise ValueError("Wrist alignment travel budget exceeded")
        self.travel += distance
        return goal, False
