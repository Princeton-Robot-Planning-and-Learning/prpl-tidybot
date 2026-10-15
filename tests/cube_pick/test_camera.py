"""Tests for the wrist RGB-D camera model."""

import numpy as np
import pytest

from prpl_tidybot.cube_pick.camera import (
    KINOVA_WRIST_COLOR,
    KINOVA_WRIST_COLOR_FISHEYE,
    KINOVA_WRIST_DEPTH,
    DepthNoiseModel,
    FisheyeIntrinsics,
    PinholeIntrinsics,
    register_depth_to_color,
)


def test_intrinsics_from_diagonal_fov_reproduce_the_fov():
    """Intrinsics built from a diagonal FOV give back that FOV."""
    intr = PinholeIntrinsics.from_diagonal_fov(
        width=1280, height=720, diagonal_fov_deg=67
    )
    half_diag = np.hypot(1280, 720) / 2
    assert np.degrees(2 * np.arctan(half_diag / intr.fx)) == pytest.approx(67)
    assert intr.fx == intr.fy
    assert intr.cx == pytest.approx(639.5)
    assert intr.vertical_fov_deg < intr.horizontal_fov_deg


def test_kinova_specs_match_the_robot():
    """The Kinova wrist camera specs match what the robot streams."""
    # The robot streams 640x480 colour and 480x270 depth. The reported fisheye is
    # unverified and not modelled.
    assert KINOVA_WRIST_COLOR_FISHEYE is None
    assert (KINOVA_WRIST_COLOR.width, KINOVA_WRIST_COLOR.height) == (640, 480)
    half_diag = np.hypot(640, 480) / 2
    assert np.degrees(
        2 * np.arctan(half_diag / KINOVA_WRIST_COLOR.fx)
    ) == pytest.approx(67)
    assert (KINOVA_WRIST_DEPTH.width, KINOVA_WRIST_DEPTH.height) == (480, 270)
    half_diag = np.hypot(480, 270) / 2
    fov = np.degrees(2 * np.arctan(half_diag / KINOVA_WRIST_DEPTH.fx))
    assert fov == pytest.approx(72)


def test_scaled_intrinsics_keep_the_fov():
    """Scaling the intrinsics keeps the field of view."""
    intr = KINOVA_WRIST_COLOR.scaled(0.5)
    assert (intr.width, intr.height) == (320, 240)
    assert intr.vertical_fov_deg == pytest.approx(KINOVA_WRIST_COLOR.vertical_fov_deg)


def test_project_and_backproject_round_trip():
    """Projecting then back-projecting recovers the points."""
    intr = PinholeIntrinsics.from_diagonal_fov(width=64, height=48, diagonal_fov_deg=70)
    points = np.array([[0.01, -0.02, 0.3], [0.0, 0.0, 0.5]])
    uv, z = intr.project(points)
    back = intr.backproject(uv, z)
    assert np.allclose(back, points)


def test_noise_blacks_out_depth_closer_than_min_range():
    """Depth closer than the minimum range reads zero."""
    depth = np.full((10, 10), 0.10, dtype=np.float32)
    depth[:, 5:] = 0.40
    noisy = DepthNoiseModel(min_range=0.18, seed=0).apply(depth)
    assert np.all(noisy[:, :5] == 0.0)
    valid = noisy[:, 6:][noisy[:, 6:] > 0]
    assert valid.size > 0.8 * noisy[:, 6:].size
    assert np.allclose(valid, 0.40, atol=0.01)


def test_noise_is_deterministic_per_seed():
    """The same seed gives the same noise."""
    depth = np.full((20, 20), 0.5, dtype=np.float32)
    a = DepthNoiseModel(seed=3).apply(depth)
    b = DepthNoiseModel(seed=3).apply(depth)
    assert np.array_equal(a, b)


def test_noise_grows_with_range():
    """Depth noise grows with range."""
    near = DepthNoiseModel(seed=0, dropout=0.0).apply(
        np.full((100, 100), 0.3, np.float32)
    )
    far = DepthNoiseModel(seed=0, dropout=0.0).apply(
        np.full((100, 100), 1.5, np.float32)
    )
    assert np.std(far[far > 0]) > 2 * np.std(near[near > 0])


def test_register_depth_to_color_with_identity_extrinsics_resamples():
    """With identity extrinsics, registration only resamples the depth."""
    depth_intr = PinholeIntrinsics.from_diagonal_fov(
        width=48, height=27, diagonal_fov_deg=72
    )
    color_intr = PinholeIntrinsics.from_diagonal_fov(
        width=64, height=36, diagonal_fov_deg=72
    )
    depth = np.full((27, 48), 0.5, dtype=np.float32)
    out = register_depth_to_color(
        depth=depth,
        depth_intrinsics=depth_intr,
        color_intrinsics=color_intr,
        color_from_depth=np.eye(4),
    )
    assert out.shape == (36, 64)
    assert np.mean(out > 0) > 0.95
    assert np.allclose(out[out > 0], 0.5)


def test_register_depth_to_color_shifts_with_baseline():
    """A baseline between the sensors shifts the registered depth."""
    intr = PinholeIntrinsics.from_diagonal_fov(
        width=100, height=100, diagonal_fov_deg=60
    )
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[50, 50] = 0.5
    color_from_depth = np.eye(4)
    color_from_depth[0, 3] = (
        0.05  # the depth sensor sits 5 cm to the colour camera's right
    )
    out = register_depth_to_color(
        depth=depth,
        depth_intrinsics=intr,
        color_intrinsics=intr,
        color_from_depth=color_from_depth,
        splat=1,
    )
    (v,), (u,) = np.nonzero(out)
    expected_u = intr.cx + intr.fx * 0.05 / 0.5 + (50 - intr.cx)
    assert v == 50
    assert u == round(expected_u)


def test_fisheye_projection_matches_the_equidistant_model():
    """Fisheye projection follows the equidistant model."""
    fish = FisheyeIntrinsics(
        width=640,
        height=480,
        fx=420,
        fy=420,
        cx=319.5,
        cy=239.5,
        distortion=(0.06, 0, 0, 0),
    )
    point = np.array([[0.1, -0.05, 0.3]])
    theta = np.arctan2(np.hypot(0.1, 0.05), 0.3)
    r = 420 * theta * (1 + 0.06 * theta**2)
    uv = fish.project(point)[0]
    assert np.hypot(uv[0] - 319.5, uv[1] - 239.5) == pytest.approx(r, rel=1e-6)


def test_distort_then_undistort_round_trips_a_pinhole_image():
    """Distorting then undistorting restores a pinhole image."""
    fish = FisheyeIntrinsics(
        width=320,
        height=240,
        fx=210,
        fy=210,
        cx=159.5,
        cy=119.5,
        distortion=(0.06, 0, 0, 0),
    )
    source = PinholeIntrinsics(
        width=640, height=480, fx=300, fy=300, cx=319.5, cy=239.5
    )
    target = PinholeIntrinsics(
        width=320, height=240, fx=220, fy=220, cx=159.5, cy=119.5
    )
    # A smooth pattern rendered by the source pinhole camera on a plane at z = 1.
    v, u = np.mgrid[0:480, 0:640]
    x, y = (u - 319.5) / 300, (v - 239.5) / 300
    image = np.uint8(127 + 100 * np.sin(8 * x) * np.cos(8 * y))
    fisheye_image = fish.distort_from_pinhole(image, source)
    restored = fish.undistort(fisheye_image, target)
    v2, u2 = np.mgrid[0:240, 0:320]
    x2, y2 = (u2 - 159.5) / 220, (v2 - 119.5) / 220
    expected = 127 + 100 * np.sin(8 * x2) * np.cos(8 * y2)
    inner = (slice(20, 220), slice(20, 300))
    assert np.mean(np.abs(restored[inner].astype(float) - expected[inner])) < 4.0
