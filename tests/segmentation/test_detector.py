"""Tests for `prpl_tidybot.segmentation.detector.SamEdgeDetector`."""

import numpy as np

from prpl_tidybot.segmentation.detector import SamEdgeDetector


class _FakeClient:
    """Canned-response stand-in for SegmentationClient."""

    def __init__(self, instances):
        self.instances = instances
        self.requests = []

    def detect(self, image, prompt):
        """Record the request and return the canned instances."""
        self.requests.append((image.shape, prompt))
        return self.instances

    def close(self):
        """Nothing to release."""


def _instance(left, right, score=0.9, top=100.0, bottom=400.0):
    return {
        "left_x": float(left),
        "right_x": float(right),
        "top_y": top,
        "bottom_y": bottom,
        "score": score,
        "area": int((right - left) * (bottom - top)),
    }


def _detector(instances) -> SamEdgeDetector:
    detector = SamEdgeDetector()
    detector._client = _FakeClient(instances)  # pylint: disable=protected-access
    return detector


_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def test_nearest_centre_plausible_instance_wins():
    """Among plausible instances the one nearest the image centre is chosen
    (the target can is staged roughly ahead of the camera; neighbours sit
    off-centre)."""
    detector = _detector([_instance(500, 620), _instance(280, 480), _instance(60, 160)])
    edges = detector(_IMAGE)
    assert edges is not None
    assert edges.left_x == 280.0
    assert edges.right_x == 480.0
    assert not edges.clipped


def test_low_score_and_wide_instances_filtered():
    """Instances below the score floor or wider than the plausible width (and
    not clipped) are ignored; nothing plausible means no detection."""
    detector = _detector(
        [_instance(200, 400, score=0.1), _instance(50, 590, score=0.9)]
    )
    assert detector(_IMAGE) is None


def test_border_touching_instance_reports_clipped():
    """An instance touching the frame border is selectable (its width is a
    lower bound) and comes back flagged as clipped."""
    detector = _detector([_instance(0, 200)])
    edges = detector(_IMAGE)
    assert edges is not None
    assert edges.clipped_left
    assert not edges.clipped_right


def test_service_failure_returns_none():
    """When the service is unreachable the detector reports no edges, so the
    caller's fallback chain runs."""
    detector = _detector(None)
    assert detector(_IMAGE) is None
