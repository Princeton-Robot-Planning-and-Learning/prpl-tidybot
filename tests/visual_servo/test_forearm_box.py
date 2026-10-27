# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Local forearm-box fit, map short axis and model-derived pinch geometry."""

import math
from pathlib import Path

import cv2
import kinder
import numpy as np
import pytest

from prpl_tidybot.sprout.pad_geometry import box_grasp_geometry
from prpl_tidybot.sprout.rehearsal import Rehearsal
from prpl_tidybot.visual_servo.forearm_box import box_in_map, fit_box


@pytest.mark.parametrize("angle", [0, 25, 75, 140])
def test_local_rectangle_fit_recovers_short_axis(angle):
    """Verify local rectangle fit recovers short axis."""
    image = np.full((128, 128, 3), 90, np.uint8)
    corners = cv2.boxPoints(((64, 64), (20, 32), angle)).astype(int)
    cv2.fillConvexPoly(image, corners, (180, 180, 180))
    fit = fit_box(image, (64, 64), (31, 27), (29, 29, 70, 70))
    difference = (
        fit["short_axis_image_rad"] - math.radians(angle) + math.pi / 2
    ) % math.pi - math.pi / 2
    assert abs(difference) < math.radians(6)
    assert np.linalg.norm(np.asarray(fit["center_uv"]) - [64, 64]) < 3


@pytest.mark.parametrize("square", [False, True])
def test_weak_or_ambiguous_axis_is_rejected(square):
    """Verify weak or ambiguous axis is rejected."""
    image = np.full((128, 128, 3), 90, np.uint8)
    if square:
        cv2.rectangle(image, (52, 52), (76, 76), (190, 190, 190), -1)
    with pytest.raises(RuntimeError, match="ambiguous"):
        fit_box(image, (64, 64), (31, 27), (29, 29, 70, 70))


def test_projected_short_axis_not_image_yaw():
    """Verify projected short axis not image yaw."""
    image = np.full((128, 128, 3), 90, np.uint8)
    cv2.fillConvexPoly(
        image, cv2.boxPoints(((64, 64), (20, 32), 25)).astype(int), (180, 180, 180)
    )
    box = fit_box(image, (64, 64), (31, 27), (29, 29, 70, 70))
    p = {
        "camera_matrix": [[300, 0, 64], [0, 300, 64], [0, 0, 1]],
        "rvec": [math.pi, 0, 0],
        "tvec": [0, 0, 1],
        "patch_height_m": 0.08,
    }
    mapped = box_in_map(box, p)
    assert mapped["short_axis_yaw_rad"] == pytest.approx(
        -box["short_axis_image_rad"], abs=0.02
    )
    assert 0.05 < mapped["width_m"] < 0.07
    assert mapped["length_m"] > mapped["width_m"]


def model_path():
    """Provide the model path test fixture or callback."""

    return (
        Path(kinder.__file__).parent
        / "envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
    )


def test_eight_cm_top_height_derives_pad_center_and_clearance():
    """Verify eight cm top height derives pad center and clearance."""
    geometry = box_grasp_geometry(model_path(), 0.08)
    assert geometry["grasp_height_m"] == pytest.approx(0.05063539, abs=1e-6)
    assert geometry["predicted_pad_floor_clearance_m"] == pytest.approx(
        0.02125, abs=1e-6
    )
    assert geometry["maximum_open_width_m"] > 0.08


def test_box_preview_does_not_access_sprout_file_or_pose():
    """Verify box preview does not access sprout file or pose."""

    scene = {
        "representation": "forearm_box",
        "body_pose_source": "camera box",
        "forearm_box": {
            "center_xyz": [0.6, 0, 0.08],
            "height_m": 0.08,
            "width_m": 0.06,
            "length_m": 0.09,
            "short_axis_yaw_rad": 0.2,
        },
        "grasp": {"z": 0.05063539},
        "tidybot_base_xyyaw": [0, 0, 0],
        "tidybot_arm_joints_rad": [0, -0.35, 3.14, -2.55, 0, -0.87, 1.57],
    }
    sim = Rehearsal(model_path(), Path("/nonexistent-private-sprout.xml"), scene)
    assert sim.box_mode and sim.limb_joints == []
    assert sim.data.site_xpos[sim.forearm] == pytest.approx([0.6, 0, 0.05063539])


def test_measured_joint_outside_model_is_not_silently_clamped():
    """Verify measured joint outside model is not silently clamped."""

    scene = {
        "representation": "forearm_box",
        "body_pose_source": "synthetic",
        "forearm_box": {
            "center_xyz": [0.6, 0, 0.08],
            "height_m": 0.08,
            "width_m": 0.06,
            "length_m": 0.1,
            "short_axis_yaw_rad": 0,
        },
        "grasp": {"z": 0.05},
        "tidybot_base_xyyaw": [0, 0, 0],
        "tidybot_arm_joints_rad": [0, -0.4, -3.14, -2.6725, 0, -0.8, 1.57],
    }
    sim = Rehearsal(model_path(), None, scene)
    before = sim.data.qpos.copy()
    with pytest.raises(ValueError, match="joint_4=-2.6725"):
        sim.validate_initial_joints()
    assert np.array_equal(before, sim.data.qpos)
