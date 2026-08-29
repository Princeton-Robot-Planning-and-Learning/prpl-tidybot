"""Tests for the cylinder visual-servo gap executor.

Two levels: a synthetic camera whose frame is generated from the tool pose (fast,
deterministic, exercises the state machine and failure modes), and the kinder
simulator's own wrist camera rendered at a real pre-grasp state (slower, exercises the
edge detector on rendered images and the lateral sign convention end to end).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pytest
from kinder.envs.kinematic3d.cylinder_shelf3d import ObjectCentricCylinderShelf3DEnv
from kinder_models.kinematic3d.cylinder_shelf3d.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.structs import SkillCall
from pybullet_helpers.geometry import matrix_from_quat
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    StreamingArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.real_sim.plan_executors.visual_servo import (
    ALIGN,
    APPROACH,
    CLOSE,
    SETTLE,
    CylinderVisualServoGapExecutor,
)
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation
from prpl_tidybot.visual_servo.image_sources import KinderEECameraSource
from prpl_tidybot.visual_servo.tool_frame import ToolFrameStepper

_PRE_GRASP_JOINTS = [0.002, 1.887, 3.142, -1.623, 0.0, 1.678, 1.571]
_IMAGE_W, _IMAGE_H = 640, 480


def _l1(q1: Sequence[float], q2: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.array(q1) - np.array(q2))))


def _state(joints: Sequence[float], gripper: float = 0.0):
    obs = TidyBotObservation(
        arm_conf=list(joints),
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        gripper=gripper,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


class _SyntheticCanCamera:
    """Renders a red bar whose horizontal position follows the tool's lateral offset
    from a virtual can and whose width follows the camera-to-axis distance (pinhole:
    width = focal_px * diameter / distance), using the stepper's forward kinematics on
    the arm's current joints (which the fake arm sets to whatever was last commanded).
    """

    def __init__(
        self,
        stepper: ToolFrameStepper,
        joints_fn,
        can_offset_m: float,
        camera_to_axis_m: float = 0.20,
        diameter_m: float = 0.078,
        focal_px: float = 500.0,
    ):
        self._stepper = stepper
        self._joints_fn = joints_fn
        self._reference = stepper.end_effector_pose(_PRE_GRASP_JOINTS)
        self._can_offset = can_offset_m
        self._camera_to_axis = camera_to_axis_m
        self._diameter = diameter_m
        self._focal = focal_px
        self.frames = 0

    def _displacement(self, axis: int) -> float:
        pose = self._stepper.end_effector_pose(self._joints_fn())
        moved = np.array(pose.position) - np.array(self._reference.position)
        return float(moved @ matrix_from_quat(self._reference.orientation)[:, axis])

    def lateral_offset(self) -> float:
        """Tool-x displacement of the end effector from its reference pose."""
        return self._displacement(0)

    def forward_offset(self) -> float:
        """Tool-z displacement of the end effector from its reference pose."""
        return self._displacement(2)

    def get_image(self):
        """Render the virtual can for the current tool pose."""
        self.frames += 1
        # The can sits `can_offset` along +tool-x; moving the tool +x brings it to
        # the image center. The camera's horizontal axis runs opposite to tool x
        # (a can further along +x appears further left), matching the sim camera.
        distance = max(0.02, self._camera_to_axis - self.forward_offset())
        scale = self._focal / distance  # px per metre at the can
        relative = self._can_offset - self.lateral_offset()
        center = 0.5 * (_IMAGE_W - 1) - relative * scale
        half = 0.5 * self._diameter * scale
        image = np.full((_IMAGE_H, _IMAGE_W, 3), (190, 150, 90), dtype=np.uint8)
        left, right = int(round(center - half)), int(round(center + half))
        image[:, max(0, left) : max(0, right)] = (200, 40, 40)
        return image


class _FakeArm:
    """Tracks the last commanded joints exactly, like FakeArmInterface."""

    def __init__(self, joints: Sequence[float]) -> None:
        self.joints = list(joints)
        self.gripper = 0.0

    def apply(self, action: TidyBotAction) -> None:
        """Adopt the commanded joints and gripper."""
        self.joints = list(action.arm_goal)
        self.gripper = action.gripper_goal


def _run(executor, arm: _FakeArm, max_ticks: int = 400) -> list[TidyBotAction]:
    actions = []
    for _ in range(max_ticks):
        state = _state(arm.joints, arm.gripper)
        if executor.done(state):
            break
        action, sim_action = executor.step(state)
        assert isinstance(sim_action, SkillCall)
        assert isinstance(action, TidyBotAction)
        arm.apply(action)
        actions.append(action)
    return actions


def _make_executor(image_source, stepper, **kwargs) -> CylinderVisualServoGapExecutor:
    return CylinderVisualServoGapExecutor(
        image_source=image_source,
        base_executor=PurePursuitBaseMotion3DPlanExecutor(position_tolerance=1e-3),
        arm_executor=StreamingArmMotion3DPlanExecutor(
            distance_fn=_l1, arrival_tolerance=1e-3
        ),
        stepper=stepper,
        gripper_dwell_ticks=3,
        **kwargs,
    )


def _skill_call(pre_state, predicted) -> SkillCall:
    robot = pre_state.get_object_from_name("robot")
    return SkillCall("Grasp", (robot,), np.zeros(0), predicted)


@pytest.fixture(name="stepper", scope="module")
def _stepper() -> ToolFrameStepper:
    return ToolFrameStepper()


def test_aligns_approaches_closes_and_settles(stepper: ToolFrameStepper):
    """Starting 3 cm off the can laterally, the executor centers it, advances the
    approach distance, closes the gripper, and settles to the predicted joints."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.03)
    executor = _make_executor(
        camera,
        stepper,
        lateral_gain=0.0004,
        estimate_range=False,
        approach_distance=0.06,
        approach_step=0.02,
        gripper_close_position=0.7,
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    # The predicted post-grasp configuration: the carrying pose (gripper closed).
    predicted = _state([0.002, 0.628, 3.142, -2.495, 0.0, 0.291, 1.571], gripper=0.7)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, predicted))])

    actions = _run(executor, arm)

    phases = [entry.phase for entry in executor.trace]
    assert ALIGN in phases and APPROACH in phases and CLOSE in phases
    assert executor.phase == SETTLE
    # Aligned: the last detections before the approach had the can within
    # tolerance of the image center (8 px at 0.5 mm/px is 4 mm).
    align_errors = [
        e.lateral_error_px
        for e in executor.trace
        if e.phase == ALIGN and e.lateral_error_px is not None
    ]
    assert align_errors and max(abs(x) for x in align_errors[-3:]) <= 8.0
    # The approach never acted on the camera.
    assert all(
        e.delta_tool is None or abs(e.delta_tool[0]) < 1e-9
        for e in executor.trace
        if e.phase == APPROACH
    )
    # Approached: the tool advanced the requested distance along its z axis before
    # the settle took over (measured on the last approach-phase target).
    last_approach = next(
        e for e in reversed(executor.trace) if e.phase == APPROACH and e.target_joints
    )
    assert last_approach.target_joints is not None
    reference = stepper.end_effector_pose(_PRE_GRASP_JOINTS)
    approach_axis = matrix_from_quat(reference.orientation)[:, 2]
    final_pose = stepper.end_effector_pose(list(last_approach.target_joints))
    moved = np.array(final_pose.position) - np.array(reference.position)
    assert moved @ approach_axis == pytest.approx(0.06, abs=0.004)
    # The gripper was closed to the configured position during the close phase.
    assert any(a.gripper_goal == pytest.approx(0.7) for a in actions)
    # And the settle drove the arm to the predicted joints.
    predicted_robot = predicted.get_object_from_name("robot")
    predicted_joints = [
        predicted.get(predicted_robot, f"joint_{j + 1}") for j in range(7)
    ]
    assert np.allclose(arm.joints, predicted_joints, atol=1e-3)


def test_already_aligned_skips_lateral_moves(stepper: ToolFrameStepper):
    """With the can centered, no lateral step is ever commanded."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.0)
    executor = _make_executor(
        camera, stepper, approach_distance=0.04, approach_step=0.02
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    _run(executor, arm)
    lateral = [e.delta_tool[0] for e in executor.trace if e.delta_tool is not None]
    assert lateral and all(abs(d) < 1e-9 for d in lateral)


def test_lost_cylinder_raises(stepper: ToolFrameStepper):
    """Frames without a cylinder for more than max_missed_detections ticks fail."""

    class _Blank:
        def get_image(self):
            """A featureless frame."""
            return np.full((_IMAGE_H, _IMAGE_W, 3), 120, dtype=np.uint8)

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    executor = _make_executor(_Blank(), stepper, max_missed_detections=2)
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with pytest.raises(ExecutionFailure, match="lost the cylinder"):
        _run(executor, arm)


def test_missing_image_raises(stepper: ToolFrameStepper):
    """An image source that yields nothing is an execution failure."""

    class _NoImage:
        def get_image(self):
            """No frame available."""
            return None

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    executor = _make_executor(_NoImage(), stepper)
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with pytest.raises(ExecutionFailure, match="no wrist image"):
        _run(executor, arm)


def test_tick_budget_raises(stepper: ToolFrameStepper):
    """A can that never centers (zero gain and no minimum step) exhausts max_ticks."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.05)
    executor = _make_executor(
        camera, stepper, lateral_gain=0.0, lateral_min_step=0.0, max_ticks=15
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with pytest.raises(ExecutionFailure, match="gave up after 15 ticks"):
        _run(executor, arm)


def test_small_corrections_are_raised_to_the_minimum_step(stepper):
    """With a gain that would produce sub-millimetre nudges, every lateral step is at
    least lateral_min_step and the servo still converges without oscillating."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.02)
    executor = _make_executor(
        camera,
        stepper,
        lateral_gain=0.00002,
        lateral_min_step=0.006,
        lateral_tolerance_px=8.0,
        estimate_range=False,
        approach_distance=0.02,
        approach_step=0.02,
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    _run(executor, arm)

    lateral = [e.delta_tool[0] for e in executor.trace if e.delta_tool is not None]
    nonzero = [d for d in lateral if abs(d) > 1e-9]
    assert nonzero and all(abs(d) >= 0.006 - 1e-9 for d in nonzero)
    # 2 cm off with 6 mm steps: three or four steps, no back-and-forth.
    assert len(nonzero) <= 4
    assert executor.phase == SETTLE


def test_minimum_step_validation():
    """lateral_min_step must lie in [0, lateral_max_step]."""
    with pytest.raises(ValueError, match="lateral_min_step"):
        CylinderVisualServoGapExecutor(
            image_source=None,  # type: ignore[arg-type]
            stepper=None,  # type: ignore[arg-type]
            lateral_min_step=0.02,
            lateral_max_step=0.01,
        )


def test_approach_is_open_loop_and_ignores_the_camera(stepper):
    """Once aligned, garbage frames (or none) do not stop or steer the approach."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    real_camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.0)

    class _GoesBlankAfterAlignment:
        def __init__(self):
            self.calls = 0

        def get_image(self):
            """Real frames for the first few captures, then nothing."""
            self.calls += 1
            return real_camera.get_image() if self.calls <= 4 else None

    camera = _GoesBlankAfterAlignment()
    executor = _make_executor(
        camera,
        stepper,
        align_confirm_ticks=2,
        approach_distance=0.04,
        approach_step=0.02,
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    _run(executor, arm)
    assert executor.phase == SETTLE
    forward = [e.delta_tool[2] for e in executor.trace if e.delta_tool is not None]
    assert sum(forward) == pytest.approx(0.04)


def test_width_jump_is_treated_as_a_miss(stepper):
    """A detection whose width differs wildly from the previous one is ignored."""

    class _WidthJumps:
        def __init__(self):
            self.calls = 0

        def get_image(self):
            """A 60 px can, then a 200 px 'can' (a background edge), alternating."""
            self.calls += 1
            image = np.full((_IMAGE_H, _IMAGE_W, 3), (190, 150, 90), dtype=np.uint8)
            half = 30 if self.calls % 2 else 100
            image[:, 160 - half + 40 : 160 + half + 40] = (200, 40, 40)
            return image

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    executor = _make_executor(_WidthJumps(), stepper, max_missed_detections=100)
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    for _ in range(4):
        executor.step(_state(arm.joints))
    widths = [e.edges.width_px for e in executor.trace if e.edges is not None]
    assert widths and all(abs(w - 60) < 4 for w in widths)
    assert any(e.edges is None for e in executor.trace[1:])


@pytest.mark.parametrize("camera_to_axis,diameter", [(0.20, 0.078), (0.26, 0.05)])
def test_range_estimate_sets_the_approach_for_any_diameter(
    stepper, camera_to_axis: float, diameter: float
):
    """The approach length comes from the width growth over the baseline, so cans of
    different diameters at different distances are approached to the same grasp
    standoff, without a known diameter."""
    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(
        stepper,
        lambda: arm.joints,
        can_offset_m=0.0,
        camera_to_axis_m=camera_to_axis,
        diameter_m=diameter,
    )
    executor = _make_executor(
        camera,
        stepper,
        range_baseline=0.04,
        camera_to_grasp_offset=0.108,
        approach_distance=0.10,
        approach_step=0.01,
        approach_max=0.25,
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    _run(executor, arm, max_ticks=800)
    assert executor.phase == SETTLE
    forward = [e.delta_tool[2] for e in executor.trace if e.delta_tool is not None]
    assert sum(forward) == pytest.approx(camera_to_axis - 0.108, abs=0.012)


def test_range_estimate_falls_back_when_samples_are_unusable(stepper, caplog):
    """With the can not growing (a constant-width camera), the fixed distance is used
    and a warning says why."""

    class _ConstantWidth:
        def get_image(self):
            """A 60 px can dead center, whatever the arm does."""
            image = np.full((_IMAGE_H, _IMAGE_W, 3), (190, 150, 90), dtype=np.uint8)
            center = _IMAGE_W // 2
            image[:, center - 30 : center + 30] = (200, 40, 40)
            return image

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    executor = _make_executor(
        _ConstantWidth(),
        stepper,
        range_baseline=0.02,
        approach_distance=0.05,
        approach_step=0.01,
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with caplog.at_level(logging.WARNING):
        _run(executor, arm)
    forward = [e.delta_tool[2] for e in executor.trace if e.delta_tool is not None]
    assert sum(forward) == pytest.approx(0.05)
    assert "did not grow" in caplog.text


def test_persistent_misdetection_stops_at_lateral_travel_limit(stepper):
    """A camera that always reports the can far to one side walks the arm sideways
    one step at a time until the travel limit stops it."""

    class _AlwaysLeft:
        def get_image(self):
            """A can fixed 120 px left of center, whatever the arm does."""
            image = np.full((_IMAGE_H, _IMAGE_W, 3), (190, 150, 90), dtype=np.uint8)
            image[:, 10:70] = (200, 40, 40)
            return image

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    executor = _make_executor(
        _AlwaysLeft(), stepper, lateral_gain=0.001, lateral_travel_limit=0.03
    )
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with pytest.raises(ExecutionFailure, match="lateral travel"):
        _run(executor, arm)
    # Only a few 1 cm steps were ever commanded.
    steps = [e for e in executor.trace if e.delta_tool is not None]
    assert 3 <= len(steps) <= 4
    assert all(abs(e.delta_tool[0]) <= 0.01 + 1e-9 for e in steps)


def test_ik_branch_flip_is_refused(stepper):
    """A stepper that returns a far-away joint solution for a small move is refused."""

    class _FlippingStepper:
        def end_effector_pose(self, joints):
            """Delegate to the real model."""
            return stepper.end_effector_pose(joints)

        def step(self, joints, delta_tool):
            """Return the true solution with joint 5 flipped by pi."""
            target = stepper.step(joints, delta_tool)
            target[4] += np.pi
            return target

    arm = _FakeArm(_PRE_GRASP_JOINTS)
    camera = _SyntheticCanCamera(stepper, lambda: arm.joints, can_offset_m=0.03)
    executor = _make_executor(camera, _FlippingStepper())  # type: ignore[arg-type]
    pre_state = _state(_PRE_GRASP_JOINTS)
    executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
    with pytest.raises(ExecutionFailure, match="IK branch flip"):
        _run(executor, arm)
    assert not any(e.target_joints for e in executor.trace)


def test_validation():
    """Bad axes and a non-SkillCall trajectory are rejected."""
    with pytest.raises(ValueError, match="axis"):
        CylinderVisualServoGapExecutor(
            image_source=None, lateral_axis="w"  # type: ignore[arg-type]
        )
    executor = CylinderVisualServoGapExecutor(
        image_source=None, stepper=None  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="SkillCall"):
        executor.set_trajectory([(_state(_PRE_GRASP_JOINTS), np.zeros(11))])
    with pytest.raises(RuntimeError, match="no trajectory"):
        executor.step(_state(_PRE_GRASP_JOINTS))


# ---------------------------------------------------------------------------
# End to end against the simulator's wrist camera
# ---------------------------------------------------------------------------


def test_servo_on_kinder_wrist_camera(stepper: ToolFrameStepper):
    """From the planned pre-grasp pose with the cylinder shifted 3 cm sideways, the
    executor centers the cylinder in the rendered wrist image and approaches; this
    checks the detector on real renders and the lateral sign convention together."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    try:
        init_state, _ = env.reset(seed=456)
        sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
        controllers = create_lifted_controllers(
            env.action_space, sim  # type: ignore[arg-type]
        )
        robot = init_state.get_object_from_name("robot")
        target = init_state.get_object_from_name("cylinder0")
        stage = controllers["move_to_pre_grasp"].ground((robot, target))
        pre = stage.predict_outcome(  # type: ignore[attr-defined]
            init_state, np.array([0.8, 0.0])
        )
        # Shift the cylinder sideways (base faces +x here, so +y is lateral).
        shifted = pre.copy()
        shifted.set(target, "pose_y", shifted.get(target, "pose_y") + 0.03)

        arm = _FakeArm(list(pre.joint_positions))
        camera = KinderEECameraSource(env, shifted, lambda: arm.joints)
        executor = _make_executor(
            camera,
            stepper,
            lateral_gain=0.0004,
            lateral_tolerance_px=6.0,
            approach_distance=0.04,
            approach_step=0.02,
        )
        pre_state = _state(list(pre.joint_positions))
        executor.set_trajectory([(pre_state, _skill_call(pre_state, pre_state))])
        _run(executor, arm)

        assert executor.phase == SETTLE
        errors = [
            e.lateral_error_px for e in executor.trace if e.lateral_error_px is not None
        ]
        assert abs(errors[0]) > 40, "the shifted cylinder starts well off center"
        align_errors = [
            e.lateral_error_px
            for e in executor.trace
            if e.phase == ALIGN and e.lateral_error_px is not None
        ]
        assert align_errors and max(abs(x) for x in align_errors[-3:]) <= 12.0
        sim.close()
    finally:
        env.close()
