"""Failure-oriented geometry and tracking tests; no hardware imports."""

from dataclasses import replace

import numpy as np
import pytest
from prpl_tidybot.sprout.approach import (
    CameraEstimate,
    TopDownGeometry,
    WristAligner,
    WristCalibration,
    WristObservation,
    corrected_descent_aim,
    select_camera,
)
from prpl_tidybot.sprout.offline_server import SYNTHETIC_GEOMETRY, FakeBackend
from prpl_tidybot.sprout.sequence import ForearmSkill, GraspPose, Recipe
from prpl_tidybot.visual_servo.forearm_template import ForearmPatchTracker, ceiling_view


def test_floor_contact_is_not_the_pinch_height():
    geometry = TopDownGeometry(-0.395, 0.0295, 0.012, 0.010)
    assert geometry.minimum_pinch_height_m == pytest.approx(0.0515)
    assert geometry.arm_z(GraspPose(0.5, 0, 0.0515, 0)) == pytest.approx(-0.3435)
    backend = FakeBackend(0)
    backend.grasp = replace(backend.grasp, z=0.025)
    with pytest.raises(ValueError, match="clearance"):
        ForearmSkill(backend, Recipe(0.1, 0.1), geometry).cycle()
    assert "pregrasp" not in backend.events and "stop" in backend.events


def test_compliance_error_never_overdrives_below_contact():
    goal = GraspPose(0.5, 0, 0.06, 0)
    measured = replace(goal, x=0.46, z=0.09)
    aim = corrected_descent_aim(goal, measured, goal, SYNTHETIC_GEOMETRY, 0.01)
    assert aim.x == pytest.approx(0.51)
    assert aim.z == goal.z  # high measured position must not cause downward push
    raised = corrected_descent_aim(
        goal, replace(goal, z=0.04), goal, SYNTHETIC_GEOMETRY, 0.01
    )
    assert raised.z == pytest.approx(0.07)


def test_camera_disagreement_selects_one_view_without_blending():
    a = CameraEstimate(0, GraspPose(0.5, 0, 0.06, 0), 0.92, True, 0.1, 1)
    b = replace(
        a, camera=1, grasp=replace(a.grasp, x=0.53), quality=0.94, off_nadir_m=0.5
    )
    choice = select_camera(
        [a, b], now=1.1, max_age_s=0.2, minimum_quality=0.8, disagreement_warning_m=0.01
    )
    assert choice.estimate == a
    assert choice.disagreement_m == pytest.approx(0.03)
    assert choice.needs_calibration_check


@pytest.mark.parametrize(
    "changes",
    [
        {"captured_at": 0},
        {"captured_at": 2},
        {"clear_of_seam": False},
        {"quality": 0.2},
    ],
)
def test_bad_camera_estimates_cannot_start_a_grasp(changes):
    a = CameraEstimate(0, GraspPose(0.5, 0, 0.06, 0), 0.9, True, 0.1, 1)
    with pytest.raises(ValueError, match="No fresh"):
        select_camera(
            [replace(a, **changes)],
            now=1.1,
            max_age_s=0.2,
            minimum_quality=0.8,
            disagreement_warning_m=0.01,
        )


def aligner():
    calibration = WristCalibration(
        (100, 100),
        ((0.0003, 0), (0, 0.0003)),
        3,
        0.01,
        0.03,
        0.2,
        3,
        2,
        20,
        0.005,
        0.02,
    )
    pose = GraspPose(0.5, 0, 0.15, 0)
    return WristAligner(calibration, pose, 1), pose


def test_wrist_steps_bounded_and_distinct_frames_required_for_convergence():
    servo, pose = aligner()
    moved, done = servo.update(WristObservation((200, 100), 1.1), pose, now=1.1)
    assert not done and moved.x - pose.x == pytest.approx(0.01)
    assert moved.z == pose.z
    _, done = servo.update(WristObservation((100, 100), 1.2), moved, now=1.2)
    assert not done
    _, done = servo.update(WristObservation((100, 100), 1.3), moved, now=1.3)
    assert done


def test_repeated_frame_does_not_confirm_alignment():
    servo, pose = aligner()
    servo.update(WristObservation((100, 100), 1.1), pose, now=1.1)
    with pytest.raises(ValueError, match="repeated"):
        servo.update(WristObservation((100, 100), 1.1), pose, now=1.2)


def test_lost_target_and_travel_budget_stop_servo():
    servo, pose = aligner()
    with pytest.raises(ValueError, match="lost"):
        servo.update(None, pose, now=1.1)
    with pytest.raises(ValueError, match="travel budget"):
        for i in range(5):
            pose, _ = servo.update(
                WristObservation((200, 100), 1.1 + i * 0.1), pose, now=1.1 + i * 0.1
            )


def test_changed_hover_and_stale_image_rejected():
    servo, pose = aligner()
    with pytest.raises(ValueError, match="calibration"):
        servo.update(WristObservation((100, 100), 1.1), replace(pose, z=0.16), now=1.1)
    with pytest.raises(ValueError, match="stale"):
        servo.update(WristObservation((100, 100), 0.7), pose, now=1.1)


def test_bare_patch_translation_loss_and_duplicate_rejection():
    rng = np.random.default_rng(1)
    image = np.full((100, 140, 3), 150, np.uint8)
    patch = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    image[30:50, 40:60] = patch
    tracker = ForearmPatchTracker(
        image, (40, 30, 20, 20), minimum_quality=0.8, minimum_margin=0.1
    )
    moved = np.full_like(image, 150)
    moved[35:55, 45:65] = patch
    match = tracker.detect(moved, (10, 10, 120, 80))
    assert match is not None and match.center_uv == (55, 45)
    moved[65:85, 90:110] = patch
    assert tracker.detect(moved, (10, 10, 120, 80)) is None
    assert tracker.detect(np.full_like(image, 150), (10, 10, 120, 80)) is None


def test_separate_camera_coordinates_and_seam_mask():
    frame = np.full((1440, 1280, 3), 100, np.uint8)
    upper = ceiling_view(frame, 0)
    lower = ceiling_view(frame, 1)
    assert upper.shape == lower.shape == (720, 1280, 3)
    assert not upper[-8:].any() and not lower[:8].any()
    assert upper[0].all() and lower[-1].all()


def test_featureless_forearm_patch_requires_a_better_initialization():
    with pytest.raises(ValueError, match="texture"):
        ForearmPatchTracker(
            np.full((50, 50, 3), 200, np.uint8),
            (10, 10, 20, 20),
            minimum_quality=0.8,
            minimum_margin=0.1,
        )
