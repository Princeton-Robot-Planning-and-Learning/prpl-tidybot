"""Tests for marker_detector/calibrate_extrinsics.py."""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.marker_detector.calibrate_extrinsics import (
    average_marker_centers,
    load_marker_positions,
    solve_from_detections,
)

_CAMERA_MATRIX = np.array([[758.0, 0.0, 647.0], [0.0, 759.0, 377.0], [0.0, 0.0, 1.0]])
_CAMERA_POSITION = np.array([0.3, 0.9, 2.33])


def _ground_truth_pose() -> tuple[np.ndarray, np.ndarray]:
    looking_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tilt, _ = cv.Rodrigues(np.array([0.05, -0.03, 0.02]))
    rotation = tilt @ looking_down
    rvec, _ = cv.Rodrigues(rotation)
    tvec = -rotation @ _CAMERA_POSITION
    return rvec.reshape(3), tvec.reshape(3)


def test_average_marker_centers_averages_and_filters():
    """Centres are averaged; markers seen in under half the frames drop out."""
    per_frame = [
        {0: np.array([10.0, 10.0]), 1: np.array([50.0, 50.0])},
        {0: np.array([12.0, 10.0])},
        {0: np.array([11.0, 13.0])},
        {0: np.array([11.0, 11.0])},
    ]
    averaged = average_marker_centers(per_frame)
    assert set(averaged) == {0}  # marker 1: 1/4 frames, below the threshold
    assert np.allclose(averaged[0], [11.0, 11.0])


def test_solve_from_detections_recovers_pose_and_reports_residuals():
    """Noise-free detections yield near-zero residuals and the true height."""
    rvec, tvec = _ground_truth_pose()
    marker_positions = {
        0: (-1.2, 1.2),
        1: (1.2, 1.2),
        2: (-0.6, 0.0),
        3: (0.6, 0.0),
        4: (0.0, 1.8),
    }
    points = np.array([[x, y, 0.0] for x, y in marker_positions.values()])
    projected, _ = cv.projectPoints(points, rvec, tvec, _CAMERA_MATRIX, np.zeros(5))
    detected = dict(zip(marker_positions, projected.reshape(-1, 2)))
    # An unknown marker in view is ignored.
    detected[40] = np.array([100.0, 100.0])

    report = solve_from_detections(detected, marker_positions, _CAMERA_MATRIX)

    assert report["marker_ids"] == sorted(marker_positions)
    assert report["rms_px"] < 1e-6
    assert all(
        abs(dx) < 1e-8 and abs(dy) < 1e-8
        for dx, dy in report["floor_residual_vectors"].values()
    )
    assert report["homography_rms_px"] < 1e-3


def test_solve_from_detections_requires_four_known_markers():
    """Three detected known markers is not enough to solve a pose."""
    detected = {i: np.array([100.0 * i, 100.0]) for i in range(3)}
    positions = {i: (float(i), 0.0) for i in range(3)}
    with pytest.raises(AssertionError):
        solve_from_detections(detected, positions, _CAMERA_MATRIX)


def test_load_marker_positions_parses_yaml(tmp_path):
    """Ids come back as ints and positions as float tuples."""
    path = tmp_path / "markers.yaml"
    path.write_text(
        "marker_positions:\n  0: [-1.2192, 1.2192]\n  7: [0.6096, 0.0]\n",
        encoding="utf-8",
    )
    positions = load_marker_positions(path)
    assert positions == {0: (-1.2192, 1.2192), 7: (0.6096, 0.0)}


def test_load_marker_positions_rejects_reserved_ids(tmp_path):
    """Robot-sticker / target ids cannot be used for calibration markers."""
    path = tmp_path / "markers.yaml"
    path.write_text(
        "marker_positions:\n  23: [0.0, 0.0]\n  0: [0.6096, 0.0]\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        load_marker_positions(path)


def test_example_marker_config_is_valid():
    """The checked-in example config parses and avoids reserved ids."""
    example = (
        Path(__file__).parents[2] / "conf" / "calibration" / "markers_example.yaml"
    )
    positions = load_marker_positions(example)
    assert len(positions) >= 4
