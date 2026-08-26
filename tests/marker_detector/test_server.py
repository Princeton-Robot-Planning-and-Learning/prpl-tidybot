"""Tests for marker_detector/server.py."""

import numpy as np

from prpl_tidybot.marker_detector.server import _merge_target_estimates


def test_merge_targets_seen_by_both_cameras_are_averaged():
    """Both cameras see marker 23: the estimates average, residual = distance."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {23: (1.0, 2.1)}])
    assert targets == {23: (1.0, 2.05)}
    assert np.isclose(residuals[23], 0.1)


def test_merge_target_seen_by_one_camera_passes_through():
    """A marker seen by a single camera is unchanged and has no residual."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {}])
    assert targets == {23: (1.0, 2.0)}
    assert not residuals


def test_merge_disjoint_targets_across_cameras():
    """Cameras seeing different markers each contribute their own estimate."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {24: (-1.0, 0.5)}])
    assert targets == {23: (1.0, 2.0), 24: (-1.0, 0.5)}
    assert not residuals


def test_merge_no_targets():
    """No detections at all yields empty targets and residuals."""
    assert _merge_target_estimates([{}, {}]) == ({}, {})
    assert _merge_target_estimates([]) == ({}, {})
