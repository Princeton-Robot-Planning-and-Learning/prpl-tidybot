# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Tests for isolated colored-band localization."""

import numpy as np
import pytest

from prpl_tidybot.visual_servo.forearm_target import detect_band


def detect(image):
    # Synthetic blue only; these are not calibrated robot-camera thresholds.
    """Provide the detect test fixture or callback."""
    return detect_band(image, (110, 100, 100), (130, 255, 255), 100)


def test_band_center():
    """Verify band center."""
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    frame[35:65, 40:100] = (0, 0, 255)
    result = detect(frame)
    assert result is not None
    assert result.center_uv == pytest.approx((69.5, 49.5))
    assert abs(result.long_axis_uv[0]) == pytest.approx(1)


@pytest.mark.parametrize("kind", ["empty", "clipped", "ambiguous"])
def test_reject_unusable_band(kind):
    """Verify reject unusable band."""
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    if kind == "clipped":
        frame[0:30, 40:100] = (0, 0, 255)
    if kind == "ambiguous":
        frame[10:30, 10:40] = (0, 0, 255)
        frame[60:80, 80:110] = (0, 0, 255)
    assert detect(frame) is None
