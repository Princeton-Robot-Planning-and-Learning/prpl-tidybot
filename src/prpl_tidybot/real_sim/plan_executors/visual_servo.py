"""A visual-servoing gap executor for the last mile of a cylinder grasp.

An alternative to handing the magic Grasp to a human. The planned motion
has already put the gripper at the pre-grasp pose: oriented for the grasp,
a short standoff in front of the cylinder, with both silhouette edges of
the cylinder in the wrist camera's view. From there the executor:

1. **Aligns** laterally: detects the cylinder's edges
   (:mod:`prpl_tidybot.visual_servo.cylinder_edges`), and moves the tool
   along its lateral axis in small steps proportional to the pixel offset
   of the cylinder axis from the image center, until the offset stays
   within ``lateral_tolerance_px`` for ``align_confirm_ticks`` ticks.
2. **Approaches**: moves the tool forward along its approach axis in
   ``approach_step`` increments until ``approach_distance`` has been
   covered, still correcting laterally along the way.
3. **Closes** the gripper and dwells ``gripper_dwell_ticks``.
4. **Settles** to the SkillCall's predicted post-grasp configuration with
   a :class:`SettleGapExecutor`, the same final phase as the teleop
   hand-off, so the rest of the plan starts where it expects to.

Each step is issued as an absolute joint target from
:class:`~prpl_tidybot.visual_servo.tool_frame.ToolFrameStepper`; the next
step is only computed once the arm has reached the previous target (within
``step_tolerance``) or ``step_timeout_ticks`` have passed, so the loop never
outruns the arm. Every tick's detection and command are appended to
:attr:`trace`, and with ``debug_dir`` set the annotated wrist frames are
written there, so a run can be inspected afterwards without the robot.

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
    align_confirm_ticks: int
    approach_axis: str
    approach_distance: float
    approach_step: float
    step_tolerance: float
    step_timeout_ticks: int
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
    before trusting it on a new mount). Distances are in metres, tolerances
    in pixels / radians, durations in ticks.
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
        align_confirm_ticks: int = 3,
        approach_axis: str = "z",
        approach_distance: float = 0.10,
        approach_step: float = 0.01,
        step_tolerance: float = 0.02,
        step_timeout_ticks: int = 8,
        gripper_dwell_ticks: int = 20,
        gripper_close_position: float = 1.0,
        max_missed_detections: int = 10,
        max_ticks: int = 600,
        max_joint_step: float = 0.2,
        lateral_travel_limit: float = 0.05,
        debug_dir: str | Path | None = None,
    ) -> None:
        if lateral_axis not in ("x", "y", "z") or approach_axis not in ("x", "y", "z"):
            raise ValueError("lateral_axis and approach_axis must be 'x', 'y' or 'z'")
        if approach_step <= 0 or approach_distance < 0:
            raise ValueError("approach_step must be > 0 and approach_distance >= 0")
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
            align_confirm_ticks=align_confirm_ticks,
            approach_axis=approach_axis,
            approach_distance=approach_distance,
            approach_step=approach_step,
            step_tolerance=step_tolerance,
            step_timeout_ticks=step_timeout_ticks,
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
        image = self._image_source.get_image()
        if image is None:
            raise ExecutionFailure("Visual servo got no wrist image.")
        edges = detect_cylinder_edges(image, self._params.detector)
        error = None if edges is None else edges.lateral_error_px
        self._dump_debug(image, edges)
        if edges is None:
            self._missed += 1
            self._record(edges, error, None)
            if self._missed > self._params.max_missed_detections:
                raise ExecutionFailure(
                    f"Visual servo lost the cylinder for {self._missed} consecutive "
                    f"ticks in phase {self._phase}."
                )
            return self._hold(sim_state, self._target or perceived, gripper=0.0)
        self._missed = 0
        assert error is not None

        # Let the arm reach the previous step before issuing the next one.
        if self._target is not None and not self._reached(perceived, self._target):
            self._ticks_on_target += 1
            if self._ticks_on_target < self._params.step_timeout_ticks:
                self._record(edges, error, None)
                return self._hold(sim_state, self._target, gripper=0.0)

        delta = self._next_delta(error)
        if delta is None:
            # Aligned and (if approaching) fully advanced: nothing more to do
            # in this phase.
            self._record(edges, error, None)
            if self._phase == APPROACH:
                self._enter_close()
            return self._hold(sim_state, self._target or perceived, gripper=0.0)

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

    def _next_delta(self, error: float) -> list[float] | None:
        """Tool-frame move for this tick, or None when the current phase has nothing
        left to command."""
        p = self._params
        aligned = abs(error) <= p.lateral_tolerance_px
        lateral = 0.0
        if not aligned:
            lateral = float(
                np.clip(
                    -p.lateral_sign * p.lateral_gain * error,
                    -p.lateral_max_step,
                    p.lateral_max_step,
                )
            )
        if self._phase == ALIGN:
            if aligned:
                self._aligned_ticks += 1
                if self._aligned_ticks >= p.align_confirm_ticks:
                    _logger.info(
                        "Visual servo aligned (|error| <= %.1f px for %d ticks); "
                        "approaching %.3f m.",
                        p.lateral_tolerance_px,
                        self._aligned_ticks,
                        p.approach_distance,
                    )
                    self._phase = APPROACH
                else:
                    return None
            else:
                self._aligned_ticks = 0
                return tool_delta(p.lateral_axis, lateral)
        remaining = p.approach_distance - self._advanced
        if remaining <= 1e-6:
            return None
        forward = min(p.approach_step, remaining)
        self._advanced += forward
        delta = tool_delta(p.approach_axis, forward)
        delta[_axis_index(p.lateral_axis)] += lateral
        return delta

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
