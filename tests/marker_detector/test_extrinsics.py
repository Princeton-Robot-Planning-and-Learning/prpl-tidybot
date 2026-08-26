"""Tests for marker_detector/extrinsics.py."""

import json

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.marker_detector.extrinsics import (
    FloorProjector,
    save_extrinsics,
    solve_extrinsics,
)

_CAMERA_MATRIX = np.array([[758.0, 0.0, 647.0], [0.0, 759.0, 377.0], [0.0, 0.0, 1.0]])
_CAMERA_POSITION = np.array([0.3, 0.9, 2.33])


def _ground_truth_pose() -> tuple[np.ndarray, np.ndarray]:
    """A downward-looking camera at `_CAMERA_POSITION` with a small tilt."""
    looking_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tilt, _ = cv.Rodrigues(np.array([0.05, -0.03, 0.02]))
    rotation = tilt @ looking_down
    rvec, _ = cv.Rodrigues(rotation)
    tvec = -rotation @ _CAMERA_POSITION
    return rvec.reshape(3), tvec.reshape(3)


def _pixels_of(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    projected, _ = cv.projectPoints(
        points_3d.astype(np.float64), rvec, tvec, _CAMERA_MATRIX, np.zeros(5)
    )
    return projected.reshape(-1, 2)


def test_project_to_heights_inverts_the_camera_projection():
    """Pixels of known 3D points project back to their xy at their height."""
    rvec, tvec = _ground_truth_pose()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, -0.8, 0.0],
            [-1.5, 1.4, 0.378],
            [0.4, 1.7, 0.2],
        ]
    )
    pixels = _pixels_of(points, rvec, tvec)
    projector = FloorProjector(_CAMERA_MATRIX, rvec, tvec)
    recovered = projector.project_to_heights(pixels, points[:, 2])
    assert np.allclose(recovered, points[:, :2], atol=1e-9)


def test_project_to_heights_accepts_scalar_height():
    """A scalar height applies to every pixel."""
    rvec, tvec = _ground_truth_pose()
    points = np.array([[0.5, 0.5, 0.0], [-0.5, 1.0, 0.0]])
    pixels = _pixels_of(points, rvec, tvec)
    projector = FloorProjector(_CAMERA_MATRIX, rvec, tvec)
    recovered = projector.project_to_heights(pixels, 0.0)
    assert np.allclose(recovered, points[:, :2], atol=1e-9)


def test_camera_position_is_recovered_from_extrinsics():
    """FloorProjector exposes the map-frame camera position."""
    rvec, tvec = _ground_truth_pose()
    projector = FloorProjector(_CAMERA_MATRIX, rvec, tvec)
    assert np.allclose(projector.camera_position, _CAMERA_POSITION, atol=1e-12)


def test_solve_extrinsics_recovers_the_true_pose():
    """Noise-free floor correspondences recover the exact camera pose."""
    rvec_true, tvec_true = _ground_truth_pose()
    floor_xy = np.array(
        [[-1.2, 1.2], [1.2, 1.2], [-0.6, 0.0], [0.6, 0.0], [0.0, 1.8], [-1.5, 0.4]]
    )
    points = np.c_[floor_xy, np.zeros(len(floor_xy))]
    pixels = _pixels_of(points, rvec_true, tvec_true)

    rvec, tvec, rms_px = solve_extrinsics(floor_xy, pixels, _CAMERA_MATRIX)

    assert rms_px < 1e-6
    projector = FloorProjector(_CAMERA_MATRIX, rvec, tvec)
    assert np.allclose(projector.camera_position, _CAMERA_POSITION, atol=1e-6)


def test_solve_extrinsics_requires_four_points():
    """Fewer than four correspondences is rejected."""
    with pytest.raises(AssertionError):
        solve_extrinsics(np.zeros((3, 2)), np.zeros((3, 2)), _CAMERA_MATRIX)


def test_save_extrinsics_round_trips_through_json(tmp_path):
    """Saved rvec/tvec and metadata read back exactly."""
    rvec, tvec = _ground_truth_pose()
    path = tmp_path / "cam_extrinsics.json"
    save_extrinsics(path, rvec, tvec, metadata={"reprojection_rms_px": 0.25})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert np.allclose(data["rvec"], rvec)
    assert np.allclose(data["tvec"], tvec)
    assert data["reprojection_rms_px"] == 0.25
