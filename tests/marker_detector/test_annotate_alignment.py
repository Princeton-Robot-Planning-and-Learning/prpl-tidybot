"""Tests for marker_detector/annotate_alignment.py."""

import json

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.annotate_alignment import (
    Annotator,
    center_from_relative,
    nearest_corner_index,
    relative_from_center,
)

# An axis-aligned quad in annotation order (image TL, TR, BR, BL).
_CORNERS = [[100, 100], [1100, 100], [1100, 600], [100, 600]]

# A real camera serial whose intrinsics ship in camera_params/.
_SERIAL = "AD8A29AE"


def test_relative_center_round_trips():
    """Center -> relative -> center recovers the original pixel."""
    center = [640, 350]
    relative = relative_from_center(_CORNERS, center)
    assert center_from_relative(_CORNERS, relative) == center


def test_quad_midpoint_is_relative_half():
    """The quad's midpoint has relative coordinates (0.5, 0.5)."""
    relative = relative_from_center(_CORNERS, [600, 350])
    assert np.allclose(relative, [0.5, 0.5], atol=1e-5)


def test_nearest_corner_index():
    """Clicks resolve to the closest corner of the quad."""
    assert nearest_corner_index(_CORNERS, (90, 110)) == 0
    assert nearest_corner_index(_CORNERS, (1090, 120)) == 1
    assert nearest_corner_index(_CORNERS, (1150, 580)) == 2
    assert nearest_corner_index(_CORNERS, (110, 590)) == 3


def test_moving_a_corner_keeps_center_pinned_to_floor(tmp_path):
    """A corner edit preserves the quad-relative center and saves the file."""
    save_path = tmp_path / f"{_SERIAL}.json"
    annotator = Annotator(_SERIAL, save_path)
    relative_before = list(annotator.camera_center_relative)

    annotator.on_mouse(cv.EVENT_LBUTTONDBLCLK, 30, 80)

    assert annotator.camera_corners[0] == [30, 80]
    assert annotator.camera_center_relative == relative_before
    assert annotator.camera_center == center_from_relative(
        annotator.camera_corners, relative_before
    )
    saved = json.loads(save_path.read_text(encoding="utf-8"))
    assert saved["camera_corners"][0] == [30, 80]
    assert saved["camera_center"] == annotator.camera_center


def test_center_mode_sets_center_and_relative(tmp_path):
    """In center mode a double-click moves the center, not the corners."""
    save_path = tmp_path / f"{_SERIAL}.json"
    annotator = Annotator(_SERIAL, save_path, edit_center=True)
    corners_before = [list(c) for c in annotator.camera_corners]

    annotator.on_mouse(cv.EVENT_LBUTTONDBLCLK, 700, 400)

    assert annotator.camera_center == [700, 400]
    assert annotator.camera_corners == corners_before
    assert center_from_relative(
        annotator.camera_corners, annotator.camera_center_relative
    ) == [700, 400]


def test_existing_annotations_are_loaded(tmp_path):
    """An existing JSON seeds the corners and center."""
    save_path = tmp_path / f"{_SERIAL}.json"
    existing = {
        "camera_corners": _CORNERS,
        "camera_center": [640, 350],
        "camera_center_relative": relative_from_center(_CORNERS, [640, 350]),
    }
    save_path.write_text(json.dumps(existing), encoding="utf-8")
    annotator = Annotator(_SERIAL, save_path)
    assert annotator.camera_corners == _CORNERS
    assert annotator.camera_center == [640, 350]


def test_other_click_events_are_ignored(tmp_path):
    """Single clicks and moves do not edit or save anything."""
    save_path = tmp_path / f"{_SERIAL}.json"
    annotator = Annotator(_SERIAL, save_path)
    annotator.on_mouse(cv.EVENT_LBUTTONDOWN, 30, 80)
    annotator.on_mouse(cv.EVENT_MOUSEMOVE, 31, 81)
    assert not save_path.exists()
