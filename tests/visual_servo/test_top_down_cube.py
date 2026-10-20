"""Tests for the top-down cube wrist servo (prpl_tidybot.visual_servo.top_down_cube)."""

import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownLateralServo,
    detect_wrist_cube,
)

TARGET = (312.0, 320.0)


def _square(image, centre, angle, side=60, colour=(40, 90, 230)):
    box = cv.boxPoints(((centre[0], centre[1]), (side, side), math.degrees(angle)))
    cv.fillPoly(image, [box.astype(np.int32)], colour)
    return image


def _tool_axes(yaw):
    """The top-down tool frame's x and closing (y) axes in the arm frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([-s, c]), np.array([c, s])


class _World:
    """A cube on the floor seen by a wrist camera fixed to the gripper: a cube at
    tool-frame offset r from the pinch point appears at TARGET + K r, with K rotated
    and mirrored relative to the tool axes (as a real mount may be)."""

    def __init__(self, cube_xy, cube_yaw, pinch_xy, yaw, visible=True):
        self.cube = np.array(cube_xy, float)
        self.cube_yaw = cube_yaw
        self.pinch = np.array(pinch_xy, float)
        self.yaw = yaw
        self.visible = visible
        self.k = np.array([[300.0, -1500.0], [-1450.0, -250.0]])
        self.moves = 0

    def move(self, _label, ax, ay, yaw):
        """Put the gripper at the hover over (ax, ay) with closing yaw `yaw`."""
        self.pinch = np.array([ax, ay])
        self.yaw = yaw
        self.moves += 1
        return True

    def get_image(self):
        """The wrist view from where the gripper is now."""
        image = np.full((480, 640, 3), 200, np.uint8)
        if not self.visible:
            return image
        x_axis, y_axis = _tool_axes(self.yaw)
        r = self.cube - self.pinch
        uv = np.array(TARGET) + self.k @ np.array([r @ x_axis, r @ y_axis])
        edge = np.array([math.cos(self.cube_yaw), math.sin(self.cube_yaw)])
        img_edge = self.k @ np.array([edge @ x_axis, edge @ y_axis])
        return _square(image, uv, math.atan2(img_edge[1], img_edge[0]))


def test_detects_the_blue_cube_its_centre_and_edge_angle():
    """Centroid and edge angle of a rotated blue square."""
    image = _square(np.full((480, 640, 3), 200, np.uint8), (300, 350), math.radians(20))
    cube = detect_wrist_cube(image)
    assert cube is not None
    assert (cube.u, cube.v) == pytest.approx((300, 350), abs=1.5)
    assert math.degrees(cube.angle) == pytest.approx(20, abs=2)
    assert not cube.clipped


def test_a_cube_cut_off_by_the_image_border_is_flagged_and_no_cube_is_none():
    """Clipped blobs are flagged; an image without blue has no cube."""
    clipped = detect_wrist_cube(
        _square(np.full((480, 640, 3), 200, np.uint8), (320, 470), 0.0)
    )
    assert clipped is not None and clipped.clipped
    assert detect_wrist_cube(np.full((480, 640, 3), 200, np.uint8)) is None


# Tom Silver's cylinder-servo align loop on two axes. The worlds below put the cube's
# image at TARGET - K r for a cube r (tool frame) from the pinch point: moving the
# gripper +x moves the cube +u, as the 2026-10-14 probes measured (J ~ 2900 I px/m).
_K_ROBOT = 2920.0


def _lateral_world(offset, k=((_K_ROBOT, 0.0), (0.0, _K_ROBOT)), visible=True):
    start = (0.55, 0.0, 0.3)
    world = _World(
        (start[0] + offset[0], start[1] + offset[1]), 0.3, start[:2], start[2], visible
    )
    world.k = -np.array(k)
    return start, world


def _labelled_moves(world):
    labels = []
    move = world.move

    def recording(label, ax, ay, yaw):
        labels.append((label, ax, ay))
        return move(label, ax, ay, yaw)

    return labels, recording


@pytest.mark.parametrize(
    "offset",
    [(0.01, 0.0), (0.0, -0.02), (-0.02, 0.015), (0.021, 0.021), (-0.03, 0.0)],
)
def test_lateral_servo_aligns_over_a_cube_the_ceiling_misplaced(offset):
    """From 1-3 cm off, Tom's loop on both tool axes ends with the cube within
    tolerance on each image axis for align_confirm_ticks looks in a row: the pinch
    point within ~3 mm, no probe moves, the yaw untouched."""
    start, world = _lateral_world(offset)
    labels, move = _labelled_moves(world)
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(start, move, _tool_axes)
    assert result.aligned, result.note
    assert {label for label, _, _ in labels} == {"align"}
    assert result.yaw == start[2] == world.yaw
    residual = np.abs(np.array([result.ax, result.ay]) - world.cube)
    assert residual.max() < 0.0035
    assert [t.label for t in result.trace[-3:]] == ["align"] * 3


def test_lateral_servo_steps_are_clamped_per_axis_like_toms():
    """Every non-zero per-axis step is at least lateral_min_step and at most
    lateral_max_step; an axis already within tolerance does not move."""
    start, world = _lateral_world(tuple(0.03 * _tool_axes(0.3)[0]))
    labels, move = _labelled_moves(world)
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(start, move, _tool_axes)
    assert result.aligned, result.note
    deltas = [t.delta_tool for t in result.trace if t.delta_tool is not None]
    assert deltas
    for dx, dy in deltas:
        for d in (dx, dy):
            assert d == 0.0 or (
                params.lateral_min_step - 1e-12 <= abs(d) <= params.lateral_max_step
            )
    assert all(dy == 0.0 for _, dy in deltas)  # the cube was off along tool x only
    assert len(labels) >= 3  # 3 cm at <= 1 cm per step


def test_lateral_servo_does_not_move_a_cube_already_on_target():
    """Within tolerance from the first look: confirm looks, no move."""
    start, world = _lateral_world((0.0005, -0.0005))
    labels, move = _labelled_moves(world)
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(start, move, _tool_axes)
    assert result.aligned and not labels
    assert len(result.trace) == params.align_confirm_ticks
    assert (result.ax, result.ay) == start[:2]


def test_lateral_servo_follows_a_configured_sign_per_axis():
    """A camera mirrored along tool x needs lateral_sign -1 on x, as Tom's
    lateral_sign flips his one axis."""
    start, world = _lateral_world((0.02, -0.01), k=((-_K_ROBOT, 0.0), (0.0, _K_ROBOT)))
    params = LateralServoParams(target_uv=TARGET, lateral_sign=(-1.0, 1.0))
    result = TopDownLateralServo(world, params).align(start, world.move, _tool_axes)
    assert result.aligned, result.note
    assert math.hypot(result.ax - world.cube[0], result.ay - world.cube[1]) < 0.004


@pytest.mark.parametrize(
    "measured",
    [((2813.1, 162.1), (-90.6, 2959.5)), ((3028.9, 187.6), (-116.2, 2872.7))],
)
def test_lateral_servo_converges_on_the_jacobians_measured_on_the_robot(measured):
    """The probe Jacobians of 2026-10-14 (onboard controller) have off-diagonals
    within 6% of the diagonals: treating the axes independently still converges."""
    start, world = _lateral_world((0.021, -0.021), k=measured)
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(start, world.move, _tool_axes)
    assert result.aligned, result.note
    assert math.hypot(result.ax - world.cube[0], result.ay - world.cube[1]) < 0.004


def test_lateral_servo_fails_after_max_missed_detections_without_moving():
    """No cube for more than max_missed_detections looks in a row: not aligned, no
    move (Tom's ExecutionFailure; here the pick retreats without closing)."""
    start, world = _lateral_world((0.0, 0.0), visible=False)
    labels, move = _labelled_moves(world)
    params = LateralServoParams(target_uv=TARGET, max_missed_detections=4)
    result = TopDownLateralServo(world, params).align(start, move, _tool_axes)
    assert not result.aligned and not labels
    assert "lost the cube" in result.note
    assert len(result.trace) == 5


def test_lateral_servo_refuses_to_travel_past_the_limit():
    """A step that would take the commanded pinch further than lateral_travel_limit
    along a tool axis from the start is not commanded."""
    start, world = _lateral_world((0.0, 0.07), k=((600.0, 0.0), (0.0, 600.0)))
    params = LateralServoParams(target_uv=TARGET, lateral_travel_limit=0.05)
    result = TopDownLateralServo(world, params).align(start, world.move, _tool_axes)
    assert not result.aligned
    assert "lateral travel" in result.note
    assert abs((world.pinch - np.array(start[:2])) @ _tool_axes(start[2])[1]) <= 0.05


def test_lateral_servo_ignores_a_sudden_change_in_blob_area():
    """As Tom's width check: a blob whose area jumps from the recent median is a
    miss, neither a step target nor a new reference."""
    start, world = _lateral_world((0.02, 0.0))
    real_image = world.get_image
    looks = {"n": 0}

    def with_a_stray():
        looks["n"] += 1
        image = real_image()
        if looks["n"] == 2:
            image[:] = 200
            image = _square(image, (100, 100), 0.0, side=160)
        return image

    world.get_image = with_a_stray
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(start, world.move, _tool_axes)
    assert result.trace[1].cube is None and result.trace[1].delta_tool is None
    assert result.aligned, result.note


def test_lateral_servo_stops_when_a_step_does_not_arrive():
    """A refused or unfinished step (an IK branch flip) stops the servo."""
    start, world = _lateral_world((0.02, 0.0))
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params).align(
        start, lambda *_: False, _tool_axes
    )
    assert not result.aligned and "did not arrive" in result.note


def test_lateral_servo_gives_up_after_max_ticks():
    """A cube that never settles in tolerance (here: the camera gain flips sign each
    look) stops after max_ticks looks."""
    start, world = _lateral_world((0.02, 0.0))
    real_image = world.get_image

    def flipping():
        world.k = -world.k
        return real_image()

    world.get_image = flipping
    params = LateralServoParams(target_uv=TARGET, max_ticks=12, lateral_travel_limit=1)
    result = TopDownLateralServo(world, params).align(start, world.move, _tool_axes)
    assert not result.aligned and "12 looks" in result.note
    assert len(result.trace) == 12


def test_lateral_servo_saves_every_look_raw_and_annotated(tmp_path):
    """Each look is on disk, raw and annotated."""
    start, world = _lateral_world((0.015, 0.0))
    params = LateralServoParams(target_uv=TARGET)
    result = TopDownLateralServo(world, params, tmp_path).align(
        start, world.move, _tool_axes, name="c1a1"
    )
    assert len(list(tmp_path.glob("c1a1_*_raw.png"))) == len(result.trace)
    assert len(list(tmp_path.glob("c1a1_*.png"))) == 2 * len(result.trace)


def test_lateral_servo_default_min_step_keeps_toms_rule_at_the_hover():
    """The default minimum step is below twice the tolerance in metres at ~2920 px/m,
    so the servo cannot bounce across the tolerance band."""
    params = LateralServoParams()
    assert params.lateral_min_step < 2 * params.lateral_tolerance_px / _K_ROBOT
