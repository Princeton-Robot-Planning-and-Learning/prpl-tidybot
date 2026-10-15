"""Tests for the top-down cube wrist servo (prpl_tidybot.visual_servo.top_down_cube)."""

import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.visual_servo.top_down_cube import (
    TopDownCubeServo,
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
    params = TopDownServoParams(target_uv=TARGET, travel_limit=0.05)
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
