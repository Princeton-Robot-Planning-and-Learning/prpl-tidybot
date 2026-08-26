"""Tests for marker_detector/calibrate_intrinsics.py."""

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.calibrate_intrinsics import (
    calibrate_from_views,
    make_board,
    write_camera_params,
)
from prpl_tidybot.marker_detector.constants import CHARUCO_BOARD_PARAMS

_IMAGE_SIZE = (1280, 720)
_CAMERA_MATRIX = np.array([[758.0, 0.0, 647.0], [0.0, 759.0, 377.0], [0.0, 0.0, 1.0]])
_DIST_COEFFS = np.array([0.09, -0.23, 0.001, -0.001, 0.11])


def test_board_matches_params():
    """The board reflects the configured geometry."""
    board = make_board()
    # pylint: disable=unpacking-non-sequence
    squares_x, squares_y = board.getChessboardSize()
    # pylint: enable=unpacking-non-sequence
    assert (squares_x, squares_y) == (
        CHARUCO_BOARD_PARAMS["squares_x"],
        CHARUCO_BOARD_PARAMS["squares_y"],
    )
    assert CHARUCO_BOARD_PARAMS["marker_length"] < CHARUCO_BOARD_PARAMS["square_length"]


def _synthetic_views(num_views: int) -> tuple[list, list]:
    """Project the board's chessboard corners through a known camera."""
    board = make_board()
    corners_3d = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    num_corners = len(corners_3d)
    all_corners, all_ids = [], []
    for i in range(num_views):
        rvec = np.array([0.1 * (i - 2), 0.08 * ((i % 3) - 1), 0.05 * i])
        tvec = np.array([-0.12 + 0.02 * i, -0.08 + 0.015 * i, 0.5 + 0.05 * i])
        projected, _ = cv.projectPoints(
            corners_3d, rvec, tvec, _CAMERA_MATRIX, _DIST_COEFFS
        )
        all_corners.append(projected.reshape(-1, 1, 2).astype(np.float32))
        all_ids.append(np.arange(num_corners, dtype=np.int32).reshape(-1, 1))
    return all_corners, all_ids


def test_calibrate_from_views_recovers_known_intrinsics():
    """Noise-free synthetic views recover the true camera matrix closely."""
    corners, ids = _synthetic_views(8)
    rms, camera_matrix, dist_coeffs = calibrate_from_views(corners, ids, _IMAGE_SIZE)
    assert rms < 0.1
    assert np.allclose(camera_matrix, _CAMERA_MATRIX, rtol=0.01)
    assert np.allclose(dist_coeffs.reshape(-1)[:2], _DIST_COEFFS[:2], atol=0.02)


def test_write_camera_params_round_trips(tmp_path):
    """The written .yml reads back with the same values."""
    path = tmp_path / "TESTCAM.yml"
    write_camera_params(path, 1280, 720, _CAMERA_MATRIX, _DIST_COEFFS, 0.42)
    fs = cv.FileStorage(str(path), cv.FILE_STORAGE_READ)
    assert int(fs.getNode("image_width").real()) == 1280
    assert int(fs.getNode("image_height").real()) == 720
    assert np.allclose(fs.getNode("camera_matrix").mat(), _CAMERA_MATRIX)
    assert np.allclose(
        fs.getNode("distortion_coefficients").mat().reshape(-1), _DIST_COEFFS
    )
    assert np.isclose(fs.getNode("avg_reprojection_error").real(), 0.42)
    fs.release()
