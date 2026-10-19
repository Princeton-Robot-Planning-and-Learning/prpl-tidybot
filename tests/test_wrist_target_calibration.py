"""Tests for the place-and-look calibration of the wrist servo's target pixel
(prpl_tidybot.wrist_target_calibration), on the fake arm with a synthetic wrist
camera and a cube that is held, carried and centred by the closing fingers."""

import json
import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics
from prpl_tidybot.wrist_target_calibration import (
    CalibrationParams,
    calibrate_wrist_target,
)

GEOM = TopDownPickGeometry()
GRASP_Z = GEOM.floor_z + GEOM.grasp_height
HOVER_Z = GRASP_Z + GEOM.hover_height
# Where the cube really appears at the hover when it is under the pinch point.
TRUE_UV = np.array([345.0, 400.0])
K = 2920.0  # px/m at the hover; the cube moves +u for a gripper move +x


class _CubeArm(FakeArmInterface):
    """The fake arm with a cube on the floor: closing at the grasp height on a cube
    within reach holds it (reading 0.49) and centres it along the closing axis; a
    held cube follows the pinch point; opening drops nothing above the floor because
    the test checks where it opens."""

    def __init__(self, cube_xy=None):
        super().__init__()
        self.cube = None if cube_xy is None else np.array(cube_xy, float)
        self.held = False
        self.opens_at_z: list[float] = []
        self.moves: list[float] = []

    def _pinch(self):
        pos, approach, closing = forward_kinematics(self.arm_state, GEOM)
        x_axis = np.cross(closing, approach)[:2]
        return pos, x_axis / np.linalg.norm(x_axis), closing[:2] / np.linalg.norm(
            closing[:2]
        )

    def move_angular(self, goal):
        before = self._pinch()[0][:2]
        out = super().move_angular(goal)
        pos = self._pinch()[0]
        self.moves.append(float(pos[2]))
        if self.held:
            self.cube = self.cube + (pos[:2] - before)
        return out

    def set_gripper_angular(self, position, timeout=4.0):
        super().set_gripper_angular(position, timeout)
        pos, _, y_axis = self._pinch()
        if position < 0.5:
            self.opens_at_z.append(float(pos[2]))
            self.held = False
            self.gripper_state = 0.0
            return 0.0
        near_floor = abs(pos[2] - GRASP_Z) < 0.005
        if self.cube is not None and near_floor and np.linalg.norm(self.cube - pos[:2]) < 0.03:
            # The fingers push the cube onto the pinch point along the closing axis.
            off = self.cube - pos[:2]
            self.cube = self.cube - (off @ y_axis) * y_axis
            self.held = True
            self.gripper_state = 0.49
            return 0.49
        self.gripper_state = 1.0
        return 1.0


class _Camera:
    """The wrist view: the cube at TRUE_UV - K r at the hover (r its tool-frame offset
    from the pinch), scaled for a higher look, out of view near the floor."""

    def __init__(self, arm, hide_on=()):
        self.arm = arm
        self.hide_on = set(hide_on)
        self.hover_looks = 0

    def get_image(self):
        image = np.full((480, 640, 3), 200, np.uint8)
        pos, x_axis, y_axis = self.arm._pinch()  # pylint: disable=protected-access
        if self.arm.cube is None or pos[2] < HOVER_Z - 0.01:
            return image
        if abs(pos[2] - HOVER_Z) < 0.005:
            self.hover_looks += 1
            if self.hover_looks in self.hide_on:
                return image
        depth_hover = 0.215
        scale = depth_hover / (depth_hover + (pos[2] - HOVER_Z))
        r = self.arm.cube - pos[:2]
        centre = np.array([320.0, 240.0])
        uv = centre + scale * (TRUE_UV - centre) - scale * K * np.array(
            [r @ x_axis, r @ y_axis]
        )
        side = 146 * scale
        box = cv.boxPoints(((uv[0], uv[1]), (side, side), 0.0))
        cv.fillPoly(image, [box.astype(np.int32)], (40, 90, 230))
        return image


def _yes(log):
    def prompt(question):
        log.append(question)
        return "y"

    return prompt


def _run(tmp_path, arm, camera, **params):
    log = []
    result = calibrate_wrist_target(
        arm,
        camera,
        GEOM,
        CalibrationParams(**params),
        _yes(log),
        out_dir=tmp_path,
        close_position=1.0,
    )
    return result, log


def _place_under_gripper(arm, offset=(0.0, 0.0)):
    """What the person does at the prompt: the cube on the floor between the pads."""
    pos, _, _ = arm._pinch()  # pylint: disable=protected-access
    arm.cube = pos[:2] + np.array(offset)


def test_calibration_recovers_the_target_pixel_of_a_centred_cube(tmp_path, monkeypatch):
    """Placed 1 cm off along the pads, the cube is centred by a regrasp at 90 degrees,
    and the hover looks give the true target pixel."""
    arm = _CubeArm()
    camera = _Camera(arm)
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_, 0.01 * x_axis),
    )
    result, _ = _run(tmp_path, arm, camera, samples=3, max_offset=0.01)
    assert result.note is None, result.note
    assert result.n_good == 3 and len(result.samples) == 3
    assert np.allclose(result.target_uv, TRUE_UV, atol=2.0)
    assert max(result.std_uv) < 2.0


def test_without_centring_a_cube_off_along_the_pads_biases_the_target(
    tmp_path, monkeypatch
):
    """The closing fingers centre the cube only along the closing axis: without the
    90-degree regrasp a 1 cm placement error along the pads stays in the target."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_, 0.01 * x_axis),
    )
    result, _ = _run(tmp_path, arm, _Camera(arm), samples=2, center=False)
    assert abs(result.target_uv[0] - TRUE_UV[0]) > 20


def test_the_gripper_never_opens_above_the_place_height(tmp_path, monkeypatch):
    """The cube is bouncy: it is set down at the grasp height, where it was picked
    up from the floor, and the gripper only ever opens there (or empty)."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_),
    )
    result, _ = _run(tmp_path, arm, _Camera(arm), samples=3, max_offset=0.01)
    assert result.note is None, result.note
    # The first open is before the cube is placed; every later one is a set-down.
    assert all(abs(z - GRASP_Z) < 0.003 for z in arm.opens_at_z[1:])
    assert len(arm.opens_at_z) >= 1 + 3
    # Hover looks and the higher look, never below the grasp height.
    assert min(arm.moves) > GRASP_Z - 0.003
    assert any(abs(z - HOVER_Z) < 0.003 for z in arm.moves)
    assert any(abs(z - (HOVER_Z + 0.05)) < 0.003 for z in arm.moves)


def test_a_failed_grasp_aborts_without_another_move(tmp_path, monkeypatch):
    """Nothing between the fingers when they close: stop there, gripper opened, no
    further arm motion."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: None,
    )
    result, _ = _run(tmp_path, arm, _Camera(arm), samples=3)
    assert result.note is not None and "grasp" in result.note
    assert result.n_good == 0 and not result.samples
    assert len(arm.moves) == 2  # hover, grasp
    assert arm.gripper_state == 0.0


def test_a_sample_without_the_cube_in_view_is_excluded(tmp_path, monkeypatch):
    """A hover look that misses the cube is recorded as bad and left out of the
    mean."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_),
    )
    result, _ = _run(tmp_path, arm, _Camera(arm, hide_on={2}), samples=3, center=False)
    assert result.n_good == 2 and len(result.samples) == 3
    assert [s.good for s in result.samples] == [True, False, True]
    assert np.allclose(result.target_uv, TRUE_UV, atol=2.0)


def test_every_motion_is_confirmed_and_a_no_stops_it(tmp_path, monkeypatch):
    """Each move and gripper action asks first; answering no stops before it (here
    the third question: open, hover, then the descent is declined)."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_),
    )
    questions = []

    def prompt(question):
        questions.append(question)
        return "y" if len(questions) < 3 else "n"

    result = calibrate_wrist_target(
        arm, _Camera(arm), GEOM, CalibrationParams(samples=2), prompt, tmp_path, 1.0
    )
    assert "declined" in result.note
    assert len(arm.moves) == 1
    assert len(questions) == 3


def test_writes_the_samples_json_and_the_yaml_snippet(tmp_path, monkeypatch):
    """calibration.json has every sample, the mean, the std and the snippet; each
    look is saved raw and annotated."""
    arm = _CubeArm()
    monkeypatch.setattr(
        "prpl_tidybot.wrist_target_calibration._wait_for_cube",
        lambda prompt, arm_, x_axis: _place_under_gripper(arm_),
    )
    result, _ = _run(tmp_path, arm, _Camera(arm), samples=2, center=False)
    data = json.loads((tmp_path / "calibration.json").read_text())
    assert data["n_good"] == 2 and data["n_total"] == 2
    assert len(data["samples"]) == 2
    assert data["yaml"].startswith("target_uv: [")
    assert math.isclose(data["target_uv"][0], result.target_uv[0])
    assert list(tmp_path.glob("sample_01_hover_raw.png"))
    assert list(tmp_path.glob("sample_01_hover.png"))


@pytest.mark.parametrize("bad", [{"samples": 0}, {"max_offset": 0.02}])
def test_rejects_bad_parameters(tmp_path, bad):
    """At least one sample, and place offsets of at most 1 cm."""
    with pytest.raises(ValueError):
        calibrate_wrist_target(
            _CubeArm(), None, GEOM, CalibrationParams(**bad), _yes([]), tmp_path, 1.0
        )
