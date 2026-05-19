"""Tests for marker_detector/ceiling_image_publisher.py."""

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.ceiling_image_publisher import _make_composite_jpeg


def _decode(jpeg_bytes: bytes) -> np.ndarray:
    return cv.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv.IMREAD_COLOR)


def test_make_composite_jpeg_returns_none_for_empty_input():
    """No frames in → no payload out."""
    assert _make_composite_jpeg([], jpeg_quality=80) is None


def test_make_composite_jpeg_single_frame_round_trips():
    """One frame is encoded as-is (no stacking) and decodes back to the same shape."""
    frame = np.full((40, 60, 3), 50, dtype=np.uint8)
    encoded = _make_composite_jpeg([frame], jpeg_quality=80)
    assert encoded is not None
    decoded = _decode(encoded)
    assert decoded.shape == frame.shape


def test_make_composite_jpeg_vstacks_two_same_width_frames():
    """Two same-width frames stack vertically; decoded composite has summed height."""
    top = np.full((40, 60, 3), 10, dtype=np.uint8)
    bottom = np.full((30, 60, 3), 20, dtype=np.uint8)
    encoded = _make_composite_jpeg([top, bottom], jpeg_quality=95)
    assert encoded is not None
    decoded = _decode(encoded)
    # Vstack puts `top` above `bottom`.
    assert decoded.shape == (70, 60, 3)
    # Top half should be closer to 10, bottom half closer to 20.
    assert decoded[:40].mean() < decoded[40:].mean()


def test_make_composite_jpeg_resizes_to_common_width():
    """Different-width inputs are resized to the max width before vstacking."""
    narrow = np.full((30, 40, 3), 0, dtype=np.uint8)
    wide = np.full((30, 80, 3), 0, dtype=np.uint8)
    encoded = _make_composite_jpeg([narrow, wide], jpeg_quality=95)
    assert encoded is not None
    decoded = _decode(encoded)
    # Narrow gets resized to width 80; its height scales from 30 to 60.
    # Wide stays (30, 80). Composite height = 60 + 30 = 90.
    assert decoded.shape == (90, 80, 3)
