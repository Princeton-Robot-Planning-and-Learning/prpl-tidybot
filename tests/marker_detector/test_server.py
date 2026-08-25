"""Tests for marker_detector/server.py."""

import numpy as np

from prpl_tidybot.marker_detector.server import _height_corrected_corners


def _square(cx: float, cy: float, half: float = 10.0) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )


def test_robot_and_target_markers_get_their_own_ratios():
    """Robot slots scale by robot_ratio, target slots by target_ratio."""
    camera_center = np.array([100.0, 100.0], dtype=np.float32)
    robot = _square(300.0, 100.0)  # robot sticker, slot 0
    target = _square(100.0, 500.0)  # scene target, slot 5
    corners = np.concatenate([robot, target], axis=0)
    slot_indices = np.array([0, 5])

    out = _height_corrected_corners(
        corners,
        slot_indices,
        camera_center,
        num_robot_slots=4,
        robot_ratio=0.8,
        target_ratio=1.0,
    )

    # Robot corners: pulled 20% toward the camera center.
    expected_robot = camera_center + 0.8 * (robot - camera_center)
    assert np.allclose(out[:4], expected_robot)
    # Target corners at ratio 1.0: unchanged.
    assert np.allclose(out[4:], target)


def test_marker_at_camera_center_is_fixed_point():
    """A marker centered on the camera center is unmoved by any ratio."""
    camera_center = np.array([50.0, 50.0], dtype=np.float32)
    marker = _square(50.0, 50.0)
    out = _height_corrected_corners(
        marker,
        np.array([0]),
        camera_center,
        num_robot_slots=4,
        robot_ratio=0.8,
        target_ratio=1.0,
    )
    center = out.mean(axis=0)
    assert np.allclose(center, camera_center)


def test_displacements_scale_uniformly_per_class():
    """Two same-class markers' separation scales by exactly their ratio."""
    camera_center = np.array([0.0, 0.0], dtype=np.float32)
    a = _square(200.0, 0.0)
    b = _square(200.0, 300.0)
    corners = np.concatenate([a, b], axis=0)
    out = _height_corrected_corners(
        corners,
        np.array([6, 7]),  # both scene targets
        camera_center,
        num_robot_slots=4,
        robot_ratio=0.8,
        target_ratio=0.9,
    )
    sep_in = b.mean(axis=0) - a.mean(axis=0)
    sep_out = out[4:].mean(axis=0) - out[:4].mean(axis=0)
    assert np.allclose(sep_out, 0.9 * sep_in)
