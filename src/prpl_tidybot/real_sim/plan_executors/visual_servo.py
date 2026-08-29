"""A visual-servoing gap executor for the last mile of a cylinder grasp.

An alternative to handing the magic Grasp to a human. The planned motion
has already put the gripper at the pre-grasp pose: oriented for the grasp,
a short standoff in front of the cylinder, with both silhouette edges of
the cylinder in the wrist camera's view. From there the executor:

1. **Aligns** laterally: detects the cylinder's edges
   (:mod:`prpl_tidybot.visual_servo.cylinder_edges`), and moves the tool
   along its lateral axis in small steps proportional to the pixel offset
   of the cylinder axis from the image center, until the offset stays
   within ``lateral_tolerance_px`` for ``align_confirm_ticks`` ticks. This
   is the only phase that acts on the camera: it happens at the pre-grasp
   standoff, where both edges are cleanly in view.
2. **Approaches** open loop: moves the tool straight forward along its
   approach axis in ``approach_step`` increments. No lateral correction:
   close to the cylinder the fingers and background clutter enter the band
   the detector looks at and its edge pairs become unreliable, and by then
   the gripper is already facing the cylinder squarely.

   The forward steps are streamed, one every ``approach_ticks_per_step``
   ticks without waiting for the arm to arrive, so the arm's trajectory
   generator blends them into one continuous motion; only the final step
   is waited out before the gripper closes.

   How far to go is estimated from **motion parallax** over the first
   ``range_baseline`` of the approach, so it works for any cylinder
   diameter: the apparent width grows as ``w ∝ 1 / (d0 - Δ)`` with the
   forward displacement Δ, so a line fit of ``1/w`` against Δ gives the
   camera-to-axis distance ``d0`` at the start of the approach without
   knowing the diameter or the focal length. Δ is measured from the
   perceived joints (forward kinematics relative to where the approach
   started), so the streamed commands running ahead of the arm do not
   bias it, and a sample is taken every tick. The total approach is then
   ``d0 - camera_to_grasp_offset``, where the offset (camera optical
   center to the grasp position along the approach axis) is a property of
   the robot, not the object. If too few clean detections come in, or the
   estimate is implausible, the fixed ``approach_distance`` is used
   instead, with a warning.
3. **Closes** the gripper and dwells ``gripper_dwell_ticks``.
4. **Settles** to the SkillCall's predicted post-grasp configuration with
   a :class:`SettleGapExecutor`, the same final phase as the teleop
   hand-off, so the rest of the plan starts where it expects to.

Each step is issued as an absolute joint target from
:class:`~prpl_tidybot.visual_servo.tool_frame.ToolFrameStepper`; the next
step is only computed once the arm has reached the previous target (within
``step_tolerance``) or ``step_timeout_ticks`` have passed, so the loop never
outruns the arm. Every tick's detection and command are appended to
:attr:`trace`, and with ``debug_dir`` set the raw and annotated wrist frames
are written there, so a run can be inspected (and the detector re-tuned on
the raw frames) afterwards without the robot.

Failure modes raise :class:`ExecutionFailure`: no detection for
``max_missed_detections`` consecutive ticks, no image from the source, an
unreachable tool-frame move, a step whose inverse kinematics would jump
any joint by more than ``max_joint_step`` (an IK branch flip near a
singularity), more than ``lateral_travel_limit`` of sideways travel from
where the servo started (a persistent misdetection walking the arm), or
``max_ticks`` elapsed. Each command is at most one ``lateral_max_step`` /
``approach_step`` move, so a misdetection can change the direction of a
step but never its size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np
from kinder_models.structs import SkillCall
from prpl_utils.real_sim import PlanExecutor
from pybullet_helpers.geometry import Pose, matrix_from_quat
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import ArmMotion3DPlanExecutor
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    BaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.real_sim.plan_executors.gap import SettleGapExecutor
from prpl_tidybot.real_sim.plan_executors.kinematic3d import SimAction
from prpl_tidybot.structs import RealAction, TidyBotAction
from prpl_tidybot.visual_servo.cylinder_edges import (
    CylinderEdges,
    EdgeDetectorParams,
    detect_cylinder_edges,
    render_edge_overlay,
)
from prpl_tidybot.visual_servo.image_sources import ImageSource
from prpl_tidybot.visual_servo.tool_frame import (
    ToolFrameStepError,
    ToolFrameStepper,
    tool_delta,
)

_logger = logging.getLogger(__name__)

ALIGN = "align"
APPROACH = "approach"
CLOSE = "close"
SETTLE = "settle"


@dataclass(frozen=True)
class VisualServoTrace:
    """What the executor saw and did on one tick."""

    tick: int
    phase: str
    edges: CylinderEdges | None
    lateral_error_px: float | None
    delta_tool: tuple[float, float, float] | None
    target_joints: tuple[float, ...] | None
    advanced: float


@dataclass
class _Params:
    lateral_gain: float
    lateral_sign: float
    lateral_axis: str
    lateral_tolerance_px: float
    lateral_max_step: float
    lateral_min_step: float
    max_width_change_frac: float
    estimate_range: bool
    range_baseline: float
    camera_to_grasp_offset: float
    min_range_samples: int
    approach_min: float
    approach_max: float
    align_confirm_ticks: int
    approach_axis: str
    approach_distance: float
    approach_step: float
    step_tolerance: float
    step_timeout_ticks: int
    approach_ticks_per_step: int
    gripper_dwell_ticks: int
    gripper_close_position: float
    max_missed_detections: int
    max_ticks: int
    max_joint_step: float
    lateral_travel_limit: float
    detector: EdgeDetectorParams = field(default_factory=EdgeDetectorParams)


class CylinderVisualServoGapExecutor(
    PlanExecutor[SimAction, RealAction, ObjectCentricState]
):
    """Visual-servo the last mile of a cylinder grasp, then settle to the prediction.

    ``lateral_gain`` is metres of tool motion per pixel of lateral error;
    ``lateral_sign`` flips the direction for a camera whose horizontal axis
    runs opposite to the tool's lateral axis (check with the hardware test
    before trusting it on a new mount). Any non-zero lateral correction is
    at least ``lateral_min_step`` and at most ``lateral_max_step``; keep the
    minimum below twice the tolerance in metres so a step taken from just
    outside the tolerance cannot land outside it on the other side.
    During alignment a detection whose apparent width differs from the
    previous accepted one by more than ``max_width_change_frac`` is treated
    as a miss (a stray background edge paired with one cylinder edge).

    With ``estimate_range`` the approach length is fitted from the width
    growth over the first ``range_baseline`` metres (see the module
    docstring); ``camera_to_grasp_offset`` is the robot constant that turns
    the camera-to-axis distance into gripper travel, and the result is
    accepted only within ``approach_min`` .. ``approach_max`` and with at
    least ``min_range_samples`` clean detections. Otherwise, or with
    ``estimate_range=False``, the approach is the fixed ``approach_distance``.
    Distances are in metres, tolerances in pixels / radians, durations in
    ticks.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        image_source: ImageSource,
        base_executor: BaseMotion3DPlanExecutor | None = None,
        arm_executor: ArmMotion3DPlanExecutor | None = None,
        stepper: ToolFrameStepper | None = None,
        detector: EdgeDetectorParams | None = None,
        robot_name: str = "robot",
        lateral_gain: float = 0.0003,
        lateral_sign: float = 1.0,
        lateral_axis: str = "x",
        lateral_tolerance_px: float = 8.0,
        lateral_max_step: float = 0.01,
        lateral_min_step: float = 0.006,
        max_width_change_frac: float = 0.3,
        estimate_range: bool = True,
        range_baseline: float = 0.04,
        camera_to_grasp_offset: float = 0.108,
        min_range_samples: int = 3,
        approach_min: float = 0.04,
        approach_max: float = 0.20,
        align_confirm_ticks: int = 3,
        approach_axis: str = "z",
        approach_distance: float = 0.10,
        approach_step: float = 0.01,
        step_tolerance: float = 0.02,
        step_timeout_ticks: int = 4,
        approach_ticks_per_step: int = 3,
        gripper_dwell_ticks: int = 20,
        gripper_close_position: float = 1.0,
        max_missed_detections: int = 10,
        max_ticks: int = 600,
        max_joint_step: float = 0.2,
        lateral_travel_limit: float = 0.08,
        debug_dir: str | Path | None = None,
    ) -> None:
        if lateral_axis not in ("x", "y", "z") or approach_axis not in ("x", "y", "z"):
            raise ValueError("lateral_axis and approach_axis must be 'x', 'y' or 'z'")
        if approach_step <= 0 or approach_distance < 0:
            raise ValueError("approach_step must be > 0 and approach_distance >= 0")
        if not 0.0 <= lateral_min_step <= lateral_max_step:
            raise ValueError("need 0 <= lateral_min_step <= lateral_max_step")
        self._image_source = image_source
        self._stepper = stepper or ToolFrameStepper()
        self._settle = SettleGapExecutor(
            base_executor=base_executor,
            arm_executor=arm_executor,
            robot_name=robot_name,
        )
        self._robot_name = robot_name
        self._params = _Params(
            lateral_gain=lateral_gain,
            lateral_sign=lateral_sign,
            lateral_axis=lateral_axis,
            lateral_tolerance_px=lateral_tolerance_px,
            lateral_max_step=lateral_max_step,
            lateral_min_step=lateral_min_step,
            max_width_change_frac=max_width_change_frac,
            estimate_range=estimate_range,
            range_baseline=range_baseline,
            camera_to_grasp_offset=camera_to_grasp_offset,
            min_range_samples=min_range_samples,
            approach_min=approach_min,
            approach_max=approach_max,
            align_confirm_ticks=align_confirm_ticks,
            approach_axis=approach_axis,
            approach_distance=approach_distance,
            approach_step=approach_step,
            step_tolerance=step_tolerance,
            step_timeout_ticks=step_timeout_ticks,
            approach_ticks_per_step=approach_ticks_per_step,
            gripper_dwell_ticks=gripper_dwell_ticks,
            gripper_close_position=gripper_close_position,
            max_missed_detections=max_missed_detections,
            max_ticks=max_ticks,
            max_joint_step=max_joint_step,
            lateral_travel_limit=lateral_travel_limit,
            detector=detector or EdgeDetectorParams(),
        )
        self._debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.trace: list[VisualServoTrace] = []
        self._call: SkillCall[ObjectCentricState] | None = None
        self._phase: str = ALIGN
        self._tick: int = 0
        self._missed: int = 0
        self._aligned_ticks: int = 0
        self._advanced: float = 0.0
        self._close_ticks: int = 0
        self._target: list[float] | None = None
        self._ticks_on_target: int = 0
        self._start_pose: Pose | None = None
        self._last_width: float | None = None
        self._range_samples: list[tuple[float, float]] = []
        self._approach_total: float | None = None
        self._approach_start_pose: Pose | None = None
        self._ticks_since_command: int = 0
        self._reset_run()

    # ------------------------------------------------------------------ Public

    @property
    def phase(self) -> str:
        """Current phase: align, approach, close, or settle."""
        return self._phase

    def set_trajectory(
        self, trajectory: Sequence[tuple[ObjectCentricState, SimAction]]
    ) -> None:
        if len(trajectory) != 1 or not isinstance(trajectory[0][1], SkillCall):
            raise ValueError(
                "CylinderVisualServoGapExecutor expects a single (state, SkillCall) "
                f"pair; got {len(trajectory)} pair(s)."
            )
        self._call = trajectory[0][1]
        self._reset_run()

    def step(self, sim_state: ObjectCentricState) -> tuple[RealAction, SimAction]:
        if self._call is None:
            raise RuntimeError(
                "CylinderVisualServoGapExecutor.step called with no trajectory"
            )
        if self._phase == SETTLE:
            return self._settle.step(sim_state)
        self._tick += 1
        if self._tick > self._params.max_ticks:
            raise ExecutionFailure(
                f"Visual servo gave up after {self._tick - 1} ticks in phase "
                f"{self._phase} (advanced {self._advanced:.3f} m of "
                f"{self._params.approach_distance:.3f})."
            )
        perceived = self._perceived_joints(sim_state)
        if self._phase == CLOSE:
            action = self._close_step(sim_state)
        else:
            action = self._servo_step(sim_state, perceived)
        return action, self._call

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._call is None:
            return False
        if self._phase != SETTLE:
            return False
        return self._settle.done(sim_state)

    # ---------------------------------------------------------------- Phases

    def _servo_step(
        self, sim_state: ObjectCentricState, perceived: list[float]
    ) -> TidyBotAction:
        if self._phase == APPROACH:
            return self._approach_step(sim_state, perceived)
        return self._align_step(sim_state, perceived)

    def _align_step(
        self, sim_state: ObjectCentricState, perceived: list[float]
    ) -> TidyBotAction:
        image = self._image_source.get_image()
        if image is None:
            raise ExecutionFailure("Visual servo got no wrist image.")
        edges = self._detect(image)
        error = None if edges is None else edges.lateral_error_px
        self._dump_debug(image, edges)
        if edges is None:
            self._missed += 1
            self._record(edges, error, None)
            if self._missed > self._params.max_missed_detections:
                raise ExecutionFailure(
                    f"Visual servo lost the cylinder for {self._missed} consecutive "
                    "ticks while aligning."
                )
            return self._hold(sim_state, self._target or perceived, gripper=0.0)
        self._missed = 0
        assert error is not None

        if self._waiting_for_arm(perceived):
            self._record(edges, error, None)
            return self._hold(sim_state, self._target or perceived, gripper=0.0)

        if abs(error) <= self._params.lateral_tolerance_px:
            self._aligned_ticks += 1
            self._record(edges, error, None)
            if self._aligned_ticks >= self._params.align_confirm_ticks:
                _logger.info(
                    "Visual servo aligned (|error| <= %.1f px for %d ticks); "
                    "approaching %.3f m open loop.",
                    self._params.lateral_tolerance_px,
                    self._aligned_ticks,
                    self._params.approach_distance,
                )
                self._phase = APPROACH
            return self._hold(sim_state, self._target or perceived, gripper=0.0)

        self._aligned_ticks = 0
        delta = tool_delta(self._params.lateral_axis, self._lateral_step(error))
        return self._issue(sim_state, perceived, delta, edges, error)

    def _approach_step(
        self, sim_state: ObjectCentricState, perceived: list[float]
    ) -> TidyBotAction:
        p = self._params
        if self._approach_start_pose is None:
            self._approach_start_pose = self._stepper.end_effector_pose(perceived)
        travelled = self._forward_travel(perceived)
        edges = self._range_sample(travelled)
        if self._approach_total is None:
            if travelled >= p.range_baseline - 1e-9:
                self._approach_total = self._fit_approach_total()
            elif self._advanced >= p.approach_distance - 1e-9:
                # The commands would overshoot the fixed distance before the
                # arm has covered the baseline; stop here.
                self._approach_total = p.approach_distance
        total = (
            self._approach_total
            if self._approach_total is not None
            else max(p.approach_distance, p.range_baseline)
        )
        remaining = total - self._advanced
        if remaining <= 1e-6:
            # All forward steps are out; wait for the arm on the last one.
            if self._waiting_for_arm(perceived):
                self._record(edges, None, None)
                return self._hold(sim_state, self._target or perceived, gripper=0.0)
            self._record(edges, None, None)
            self._enter_close()
            return self._hold(sim_state, self._target or perceived, gripper=0.0)
        self._ticks_since_command += 1
        if (
            self._target is not None
            and self._ticks_since_command < p.approach_ticks_per_step
        ):
            self._record(edges, None, None)
            return self._hold(sim_state, self._target, gripper=0.0)
        forward = min(p.approach_step, remaining)
        self._advanced += forward
        self._ticks_since_command = 0
        delta = tool_delta(p.approach_axis, forward)
        return self._issue(sim_state, perceived, delta, edges, None)

    def _forward_travel(self, perceived: Sequence[float]) -> float:
        """Perceived displacement along the approach axis since the approach began."""
        assert self._approach_start_pose is not None
        pose = self._stepper.end_effector_pose(perceived)
        moved = np.array(pose.position) - np.array(self._approach_start_pose.position)
        axis = matrix_from_quat(self._approach_start_pose.orientation)[
            :, _axis_index(self._params.approach_axis)
        ]
        return float(moved @ axis)

    def _range_sample(self, travelled: float) -> CylinderEdges | None:
        """Capture a frame during the approach; while the arm is still inside the range
        baseline, keep its (perceived displacement, width) for the parallax fit."""
        p = self._params
        sampling = (
            p.estimate_range
            and self._approach_total is None
            and travelled <= p.range_baseline + 1e-9
        )
        if not sampling and self._debug_dir is None:
            return None
        image = self._image_source.get_image()
        if image is None:
            return None
        edges = self._detect(image)
        self._dump_debug(image, edges)
        if sampling and edges is not None:
            self._range_samples.append((travelled, edges.width_px))
        return edges

    def _fit_approach_total(self) -> float:
        """Camera-to-axis distance from the width samples, turned into gripper travel.

        ``1/w`` is linear in the displacement: ``1/w = (d0 - Δ) / k``. A least-squares
        line through the samples gives ``d0 = -intercept / slope``.
        """
        p = self._params
        fallback = p.approach_distance
        if not p.estimate_range:
            return fallback
        samples = self._range_samples
        if len(samples) < p.min_range_samples:
            _logger.warning(
                "Visual servo range estimate skipped: %d clean detection(s) over the "
                "%.3f m baseline (need %d); using approach_distance %.3f m.",
                len(samples),
                p.range_baseline,
                p.min_range_samples,
                fallback,
            )
            return fallback
        displacement = np.array([d for d, _ in samples])
        inverse_width = np.array([1.0 / w for _, w in samples])
        slope, intercept = np.polyfit(displacement, inverse_width, 1)
        if slope >= 0.0:
            _logger.warning(
                "Visual servo range estimate rejected: the cylinder did not grow "
                "over the baseline (samples %s); using approach_distance %.3f m.",
                [(round(d, 3), round(w, 1)) for d, w in samples],
                fallback,
            )
            return fallback
        camera_to_axis = float(-intercept / slope)
        total = camera_to_axis - p.camera_to_grasp_offset
        if not p.approach_min <= total <= p.approach_max:
            _logger.warning(
                "Visual servo range estimate implausible: camera-to-axis %.3f m -> "
                "approach %.3f m outside [%.3f, %.3f] (samples %s); using "
                "approach_distance %.3f m.",
                camera_to_axis,
                total,
                p.approach_min,
                p.approach_max,
                [(round(d, 3), round(w, 1)) for d, w in samples],
                fallback,
            )
            return fallback
        _logger.info(
            "Visual servo range estimate: camera-to-axis %.3f m from %d samples over "
            "%.3f m -> approach %.3f m (fixed default %.3f m).",
            camera_to_axis,
            len(samples),
            p.range_baseline,
            total,
            fallback,
        )
        return total

    def _detect(self, image: np.ndarray) -> CylinderEdges | None:
        """Run the detector and reject a width that jumped from the last accepted one."""
        edges = detect_cylinder_edges(image, self._params.detector)
        if edges is None:
            return None
        if self._last_width is not None:
            change = abs(edges.width_px - self._last_width) / self._last_width
            if change > self._params.max_width_change_frac:
                _logger.warning(
                    "Visual servo ignored a detection: width %.0f px vs %.0f px "
                    "before (%.0f%% change).",
                    edges.width_px,
                    self._last_width,
                    100 * change,
                )
                return None
        self._last_width = edges.width_px
        return edges

    def _lateral_step(self, error: float) -> float:
        """Signed lateral tool move for a pixel error: proportional, but at least
        ``lateral_min_step`` and at most ``lateral_max_step``. The arm's compliant
        controller does not react to a target a few millimetres away, so smaller
        nudges only stack up and then release in one jump."""
        p = self._params
        direction = -p.lateral_sign * np.sign(error)
        magnitude = min(
            max(p.lateral_gain * abs(error), p.lateral_min_step), p.lateral_max_step
        )
        return float(direction * magnitude)

    def _waiting_for_arm(self, perceived: Sequence[float]) -> bool:
        """Let the arm reach the previous step before issuing the next one."""
        if self._target is None or self._reached(perceived, self._target):
            return False
        self._ticks_on_target += 1
        return self._ticks_on_target < self._params.step_timeout_ticks

    def _issue(
        self,
        sim_state: ObjectCentricState,
        perceived: list[float],
        delta: list[float],
        edges: CylinderEdges | None,
        error: float | None,
    ) -> TidyBotAction:
        """Convert a tool-frame move into a joint target, check it, and command it."""
        origin = self._target or perceived
        if self._start_pose is None:
            self._start_pose = self._stepper.end_effector_pose(origin)
        try:
            target = self._stepper.step(origin, delta)
        except ToolFrameStepError as e:
            raise ExecutionFailure(f"Visual servo step is unreachable: {e}") from e
        self._check_step_is_small(origin, target, delta)
        self._check_lateral_travel(target)
        self._target = target
        self._ticks_on_target = 0
        self._record(edges, error, delta, target)
        return self._hold(sim_state, target, gripper=0.0)

    def _check_step_is_small(
        self, origin: Sequence[float], target: Sequence[float], delta: Sequence[float]
    ) -> None:
        """Refuse an IK solution that jumps a joint by more than max_joint_step."""
        jump = np.abs(_wrap(np.asarray(target) - np.asarray(origin)))
        if float(jump.max()) > self._params.max_joint_step:
            raise ExecutionFailure(
                f"Visual servo refused a step: a {np.round(delta, 4)} m tool move "
                f"maps to a joint change of {np.round(jump, 3)} rad (limit "
                f"{self._params.max_joint_step}); likely an IK branch flip."
            )

    def _check_lateral_travel(self, target: Sequence[float]) -> None:
        """Refuse to wander further than lateral_travel_limit from the start pose."""
        assert self._start_pose is not None
        pose = self._stepper.end_effector_pose(target)
        moved = np.array(pose.position) - np.array(self._start_pose.position)
        lateral_axis = matrix_from_quat(self._start_pose.orientation)[
            :, _axis_index(self._params.lateral_axis)
        ]
        travel = abs(float(moved @ lateral_axis))
        if travel > self._params.lateral_travel_limit:
            raise ExecutionFailure(
                f"Visual servo stopped: lateral travel {travel:.3f} m exceeds the "
                f"{self._params.lateral_travel_limit:.3f} m limit; the detection is "
                "probably wrong."
            )

    def _enter_close(self) -> None:
        _logger.info(
            "Visual servo approach complete (%.3f m); closing the gripper.",
            self._advanced,
        )
        self._phase = CLOSE
        self._close_ticks = 0

    def _close_step(self, sim_state: ObjectCentricState) -> TidyBotAction:
        self._close_ticks += 1
        target = self._target or self._perceived_joints(sim_state)
        if self._close_ticks >= self._params.gripper_dwell_ticks:
            _logger.info("Visual servo grasp closed; settling to the predicted pose.")
            self._phase = SETTLE
            assert self._call is not None
            self._settle.set_trajectory([(sim_state, self._call)])
        self._record(None, None, None)
        return self._hold(
            sim_state, target, gripper=self._params.gripper_close_position
        )

    # ---------------------------------------------------------------- Helpers

    def _reset_run(self) -> None:
        self._phase = ALIGN
        self._tick = 0
        self._missed = 0
        self._aligned_ticks = 0
        self._advanced = 0.0
        self._close_ticks = 0
        self._target = None
        self._ticks_on_target = 0
        self._start_pose = None
        self._last_width = None
        self._range_samples = []
        self._approach_total = None
        self._approach_start_pose = None
        self._ticks_since_command = 0
        self.trace = []

    def _reached(self, perceived: Sequence[float], target: Sequence[float]) -> bool:
        error = np.abs(_wrap(np.asarray(perceived) - np.asarray(target)))
        return bool(np.max(error) <= self._params.step_tolerance)

    def _perceived_joints(self, sim_state: ObjectCentricState) -> list[float]:
        robot = sim_state.get_object_from_name(self._robot_name)
        return [float(sim_state.get(robot, f"joint_{j + 1}")) for j in range(7)]

    def _hold(
        self, sim_state: ObjectCentricState, joints: Sequence[float], gripper: float
    ) -> TidyBotAction:
        robot = sim_state.get_object_from_name(self._robot_name)
        base = SE2(
            x=float(sim_state.get(robot, "pos_base_x")),
            y=float(sim_state.get(robot, "pos_base_y")),
            theta=float(sim_state.get(robot, "pos_base_rot")),
        )
        return TidyBotAction(
            arm_goal=list(joints), base_pose_target_map=base, gripper_goal=gripper
        )

    def _record(
        self,
        edges: CylinderEdges | None,
        error: float | None,
        delta: Sequence[float] | None,
        target: Sequence[float] | None = None,
    ) -> None:
        self.trace.append(
            VisualServoTrace(
                tick=self._tick,
                phase=self._phase,
                edges=edges,
                lateral_error_px=error,
                delta_tool=None if delta is None else (delta[0], delta[1], delta[2]),
                target_joints=None if target is None else tuple(target),
                advanced=self._advanced,
            )
        )

    def _dump_debug(self, image: np.ndarray, edges: CylinderEdges | None) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(
            str(self._debug_dir / f"servo_{self._tick:04d}_raw.png"),
            cv.cvtColor(np.asarray(image, dtype=np.uint8), cv.COLOR_RGB2BGR),
        )
        overlay = render_edge_overlay(
            image,
            edges,
            self._params.detector,
            label=f"t{self._tick:04d} {self._phase}",
        )
        cv.imwrite(
            str(self._debug_dir / f"servo_{self._tick:04d}.png"),
            cv.cvtColor(overlay, cv.COLOR_RGB2BGR),
        )


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _wrap(angles: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angles), np.cos(angles))
