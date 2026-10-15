"""Tests for fitting a known-size cube to depth points."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from prpl_tidybot.cube_pick.cube_fit import (
    CubeFitError,
    fit_cube,
    sample_visible_cube_points,
    yaw_error_mod_90,
)

SIZE = 0.05


def _pose(x, y, z, yaw_deg, tilt_deg=0.0):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "zx", [yaw_deg, tilt_deg], degrees=True
    ).as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose


def test_yaw_error_mod_90():
    assert yaw_error_mod_90(np.radians(89), np.radians(1)) == pytest.approx(
        np.radians(2)
    )
    assert yaw_error_mod_90(np.radians(45), np.radians(-45)) == pytest.approx(0.0)


@pytest.mark.parametrize("yaw", [0.0, 17.0, 44.0, 71.0])
def test_fit_recovers_a_face_down_cube_seen_from_above(yaw):
    truth = _pose(0.4, -0.2, 0.045, yaw)
    camera = np.array([0.42, -0.18, 0.40])
    rng = np.random.default_rng(0)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    points += rng.normal(scale=0.0015, size=points.shape)
    fit = fit_cube(points=points, size=SIZE, viewpoint=camera)
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.004
    assert yaw_error_mod_90(fit.yaw, np.radians(yaw)) < np.radians(4)
    assert fit.up_alignment > 0.99


def test_fit_recovers_an_obliquely_viewed_cube():
    truth = _pose(0.0, 0.0, 0.025, 30.0)
    camera = np.array([-0.25, 0.05, 0.25])
    rng = np.random.default_rng(1)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    points += rng.normal(scale=0.0015, size=points.shape)
    fit = fit_cube(points=points, size=SIZE, viewpoint=camera)
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.004
    assert yaw_error_mod_90(fit.yaw, np.radians(30)) < np.radians(4)


def test_fit_survives_outliers_from_the_bin_floor():
    truth = _pose(0.3, 0.1, 0.045, 10.0)
    camera = np.array([0.3, 0.1, 0.40])
    rng = np.random.default_rng(2)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    floor = np.column_stack(
        [rng.uniform(0.24, 0.36, 150), rng.uniform(0.04, 0.16, 150), np.full(150, 0.02)]
    )
    fit = fit_cube(points=np.vstack([points, floor]), size=SIZE, viewpoint=camera)
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.005


def test_fit_rejects_too_few_points():
    with pytest.raises(CubeFitError):
        fit_cube(points=np.zeros((5, 3)), size=SIZE, viewpoint=np.array([0, 0, 1.0]))


def test_fit_reports_a_tipped_cube():
    truth = _pose(0.0, 0.0, 0.04, 20.0, tilt_deg=35.0)
    camera = np.array([0.05, -0.05, 0.40])
    rng = np.random.default_rng(3)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    fit = fit_cube(points=points, size=SIZE, viewpoint=camera)
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.006
    assert fit.up_alignment < 0.95


def test_fit_does_not_bury_the_cube_under_a_floor_patch_in_its_mask():
    # A floor cube whose mask leaked onto more floor than the cube's own top face:
    # the dominant plane is the floor, and a cube "under" it would explain those
    # points. The known support height rules that out.
    truth = _pose(0.2, 0.0, 0.025, 25.0)
    camera = np.array([0.2, 0.02, 0.38])
    rng = np.random.default_rng(4)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    floor = np.column_stack(
        [rng.uniform(0.16, 0.26, 1500), rng.uniform(-0.05, 0.05, 1500), np.zeros(1500)]
    )
    inside = (np.abs(floor[:, 0] - 0.2) < 0.036) & (np.abs(floor[:, 1]) < 0.036)
    cloud = np.vstack([points, floor[~inside]])
    fit = fit_cube(points=cloud, size=SIZE, viewpoint=camera, support_z=0.0)
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.006


def test_a_container_floor_does_not_reject_a_cube_beside_the_container():
    from prpl_tidybot.cube_pick.cube_fit import SupportSurface

    truth = _pose(0.2, 0.0, 0.025, 10.0)
    camera = np.array([0.2, 0.0, 0.38])
    rng = np.random.default_rng(5)
    points = sample_visible_cube_points(
        cube_pose=truth, size=SIZE, viewpoint=camera, spacing=0.002, rng=rng
    )
    bin_floor = SupportSurface(z=0.02, centre=(0.5, 0.0), half_extents=(0.13, 0.13))
    fit = fit_cube(
        points=points, size=SIZE, viewpoint=camera, support_z=[0.0, bin_floor]
    )
    assert np.linalg.norm(fit.position - truth[:3, 3]) < 0.004
