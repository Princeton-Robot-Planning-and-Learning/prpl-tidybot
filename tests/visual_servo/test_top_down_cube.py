"""Tests for the top-down cube wrist servo (prpl_tidybot.visual_servo.top_down_cube)."""

import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownCubeServo,
    TopDownLateralServo,
    TopDownServoParams,
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


@pytest.mark.parametrize(
    "offset,cube_yaw", [((0.02, -0.015), 0.3), ((-0.03, 0.01), -0.6), ((0.0, 0.0), 0.0)]
)
def test_aligns_the_pinch_point_and_fingers_over_a_cube_the_ceiling_misplaced(
    offset, cube_yaw
):
    """Starting 0-3.2 cm off and misturned, the servo ends with the pinch point within
    3 mm of the cube and the fingers square to its faces."""
    start = (0.55, 0.0, 0.1)
    world = _World(
        (start[0] + offset[0], start[1] + offset[1]), cube_yaw, start[:2], start[2]
    )
    servo = TopDownCubeServo(world, TopDownServoParams(target_uv=TARGET))
    result = servo.align(start, world.move, _tool_axes)
    assert result.aligned, result.note
    assert math.hypot(result.ax - world.cube[0], result.ay - world.cube[1]) < 0.003
    turn = (result.yaw - cube_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    assert abs(turn) < math.radians(4)
    assert (result.ax, result.ay, result.yaw) == pytest.approx(
        (*world.pinch, world.yaw)
    )


def test_with_align_yaw_off_only_x_and_y_are_nudged():
    """The nudges are 2-D at the hover height; align_yaw=False keeps the planned
    closing yaw."""
    start = (0.55, 0.0, 0.1)
    world = _World((0.57, -0.01), 0.5, start[:2], start[2])
    params = TopDownServoParams(target_uv=TARGET, align_yaw=False)
    result = TopDownCubeServo(world, params).align(start, world.move, _tool_axes)
    assert result.aligned, result.note
    assert result.yaw == start[2]
    assert math.hypot(result.ax - world.cube[0], result.ay - world.cube[1]) < 0.003


def test_no_cube_in_the_wrist_camera_is_not_aligned_and_does_not_move():
    """Nothing seen at the hover: no probe, no nudge."""
    world = _World((0.55, 0.0), 0.0, (0.55, 0.0), 0.0, visible=False)
    result = TopDownCubeServo(world).align((0.55, 0.0, 0.0), world.move, _tool_axes)
    assert not result.aligned
    assert "not seen" in result.note
    assert world.moves == 0


def test_refuses_to_walk_the_gripper_further_than_the_travel_limit():
    """A cube that would need more than travel_limit of sideways travel (a stray blob,
    a bad Jacobian) stops the servo instead of following it."""
    world = _World((0.55, 0.12), 0.0, (0.55, 0.0), 0.0)
    world.k = world.k * 0.2  # keep the far cube in view
    params = TopDownServoParams(target_uv=TARGET, travel_limit=0.05, max_moves=20)
    result = TopDownCubeServo(world, params).align(
        (0.55, 0.0, 0.0), world.move, _tool_axes
    )
    assert not result.aligned
    assert "more than" in result.note


def test_every_look_is_traced_and_saved_raw_and_annotated(tmp_path):
    """Each look is in the trace and on disk, raw and annotated."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.565, 0.0), 0.0, start[:2], start[2])
    servo = TopDownCubeServo(world, TopDownServoParams(target_uv=TARGET), tmp_path)
    result = servo.align(start, world.move, _tool_axes, name="c1a1")
    assert result.trace and result.trace[0].label == "hover"
    assert len(list(tmp_path.glob("c1a1_*_raw.png"))) == len(result.trace)
    assert len(list(tmp_path.glob("c1a1_*.png"))) == 2 * len(result.trace)


def test_gives_up_after_max_moves_nudges():
    """A cube that keeps not converging (here: the image Jacobian is wrong by 4x after
    the probes) stops after max_moves alignment nudges instead of wandering."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.575, 0.0), 0.0, start[:2], start[2])

    probes = {"n": 0}
    move = world.move

    def counting(label, ax, ay, yaw):
        if label.startswith("probe"):
            probes["n"] += 1
            if probes["n"] == 3:  # after both probes, the camera gain changes
                world.k = world.k * -0.25
        return move(label, ax, ay, yaw)

    params = TopDownServoParams(target_uv=TARGET, max_moves=5, travel_limit=1.0)
    result = TopDownCubeServo(world, params).align(start, counting, _tool_axes)
    assert not result.aligned
    assert "5 nudges" in result.note
    assert sum(t.label == "align" for t in result.trace) == 5


def test_converges_when_the_arm_covers_only_part_of_each_commanded_move():
    """The compliant arm stops short of its targets (run 9: the probes moved it less
    than 1.5 cm, the image gain came out ~2x small and the corrections overshot past
    the travel limit). With the measured pinch position, the Jacobian is built from
    how far the arm really moved, and the nudges converge."""
    start = (0.55, 0.0, 0.1)
    world = _World((0.568, -0.014), 0.1, start[:2], start[2])
    commanded = {"xy": np.array(start[:2])}

    def half_way(_label, ax, ay, yaw):
        target = np.array([ax, ay])
        step = target - commanded["xy"]
        commanded["xy"] = target
        world.pinch = world.pinch + 0.5 * step
        world.yaw = yaw
        return True

    result = TopDownCubeServo(world, TopDownServoParams(target_uv=TARGET)).align(
        start, half_way, _tool_axes, measured_xy=lambda: tuple(world.pinch)
    )
    assert result.aligned, result.note
    assert math.hypot(*(world.pinch - world.cube)) < 0.004


def test_a_sudden_change_in_blob_size_is_a_miss_not_a_target():
    """As in the cylinder servo, a detection whose size jumps from the recent median
    (a stray blue object entering the view) is not stepped towards."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.555, 0.0), 0.0, start[:2], start[2])
    real_image = world.get_image
    looks = {"n": 0}

    def with_a_stray():
        looks["n"] += 1
        image = real_image()
        if looks["n"] == 6:  # the first look after the probes
            image[:] = 200
            image = _square(image, (100, 100), 0.0, side=160)
        return image

    world.get_image = with_a_stray
    result = TopDownCubeServo(world, TopDownServoParams(target_uv=TARGET)).align(
        start, world.move, _tool_axes
    )
    assert any(t.label.endswith("_outlier") for t in result.trace)
    assert result.aligned, result.note


def test_a_cube_cut_off_by_the_image_edge_is_walked_into_view():
    """As in the cylinder servo: the hover can leave the cube half out of the image
    (run 11: v=457 of 480, the ceiling cameras 6 cm off). A probe that pushes it
    further out is retried the other way, a cut-off blob is not an outlier, and the
    nudges bring the cube in and onto the target."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.55, 0.0), 0.0, start[:2], start[2])
    world.k = np.array([[2000.0, 0.0], [0.0, -2000.0]])
    # 7 cm against tool y puts the cube centre at v = 320 + 140 = 460: cut off, and
    # the +y probe pushes it further out (as in run 11).
    _, y_axis = _tool_axes(0.0)
    world.cube = np.array(start[:2]) - 0.07 * y_axis
    result = TopDownCubeServo(
        world, TopDownServoParams(target_uv=TARGET, max_moves=12, travel_limit=0.12)
    ).align(start, world.move, _tool_axes, measured_xy=lambda: tuple(world.pinch))
    assert result.aligned, result.note
    assert math.hypot(*(world.pinch - world.cube)) < 0.004
    assert result.trace[0].cube is not None and result.trace[0].cube.clipped


def test_a_gain_measured_on_a_cut_off_cube_is_remeasured_before_turning_the_wrist():
    """Probes on a cut-off cube under-read the gain (run 12: 838 vs 2753 px/m) and a
    wrist turn computed from it went the wrong way. The servo re-probes once the cube
    is fully in view, turns only then, at most max_turn per nudge, and converges."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.55, 0.0), 0.15, start[:2], start[2])
    world.k = np.array([[2000.0, 0.0], [0.0, -2000.0]])
    _, y_axis = _tool_axes(0.0)
    world.cube = np.array(start[:2]) - 0.07 * y_axis
    turns = []
    move = world.move

    def recording(label, ax, ay, yaw):
        turns.append(abs(yaw - world.yaw))
        return move(label, ax, ay, yaw)

    params = TopDownServoParams(target_uv=TARGET, max_moves=12, travel_limit=0.12)
    result = TopDownCubeServo(world, params).align(
        start, recording, _tool_axes, measured_xy=lambda: tuple(world.pinch)
    )
    assert result.aligned, result.note
    assert math.hypot(*(world.pinch - world.cube)) < 0.004
    assert max(turns) <= params.max_turn + 1e-9
    assert sum(t.label == "probe_x" for t in result.trace) == 2
    turn = (result.yaw - 0.15 + math.pi / 4) % (math.pi / 2) - math.pi / 4
    assert abs(turn) < math.radians(4)


def test_a_sideways_probe_along_the_edge_is_not_mistaken_for_pushing_the_cube_out():
    """Run 13: the x probe slid the cut-off cube along the image edge, its area
    wobbled down a little, and the probe was needlessly retried the other way. Only a
    clear shrink (below pushed_out_frac) counts as pushed out."""
    start = (0.55, 0.0, 0.0)
    world = _World((0.55, 0.0), 0.0, start[:2], start[2])
    world.k = np.array([[2000.0, 0.0], [0.0, -2000.0]])
    _, y_axis = _tool_axes(0.0)
    world.cube = np.array(start[:2]) - 0.07 * y_axis
    real_image = world.get_image
    looks = {"n": 0}

    def wobbling():
        looks["n"] += 1
        image = real_image()
        if looks["n"] == 2:  # the x probe: shave the top rows off the cut-off blob
            rows = np.nonzero((image != 200).any(axis=(1, 2)))[0]
            image[rows.min() : rows.min() + 4] = 200
        return image

    world.get_image = wobbling
    params = TopDownServoParams(target_uv=TARGET, travel_limit=0.12)
    result = TopDownCubeServo(world, params).align(
        start, world.move, _tool_axes, measured_xy=lambda: tuple(world.pinch)
    )
    labels = [t.label for t in result.trace]
    assert labels[: labels.index("probe_y")] == ["hover", "probe_x", "probe_back"]
    assert result.aligned, result.note


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
