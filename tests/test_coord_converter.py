"""Tests for coord_converter.py."""

import math

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.coord_converter import CoordFrameConverter


def test_identity_calibration_is_identity():
    """When frames A and B coincide at the origin, convert_pose returns the input."""
    converter = CoordFrameConverter(SE2(0, 0, 0), SE2(0, 0, 0))
    pose = SE2(1.5, -0.75, 0.3)
    converted = converter.convert_pose(pose)
    assert np.allclose(converted.A, pose.A)


def test_pure_translation():
    """A pure translation offset between frames shifts converted positions but not
    heading."""
    # Same physical pose: (5, 3, 0) in A, (0, 0, 0) in B — A is just B shifted by (5, 3).
    converter = CoordFrameConverter(SE2(5.0, 3.0, 0.0), SE2(0.0, 0.0, 0.0))
    converted = converter.convert_pose(SE2(7.0, 4.0, 0.1))
    assert converted.x == pytest.approx(2.0)
    assert converted.y == pytest.approx(1.0)
    assert converted.theta() == pytest.approx(0.1)


def test_pure_rotation_about_origin():
    """A pure 90 deg rotation between frames sharing an origin rotates positions and
    headings."""
    converter = CoordFrameConverter(SE2(0.0, 0.0, math.pi / 2), SE2(0.0, 0.0, 0.0))
    converted = converter.convert_pose(SE2(1.0, 0.0, 0.0))
    assert converted.x == pytest.approx(0.0, abs=1e-9)
    assert converted.y == pytest.approx(-1.0)
    assert converted.theta() == pytest.approx(-math.pi / 2)


def test_round_trip_via_inverse_converter():
    """Chaining A→B and B→A converters returns the original pose."""
    pose_a = SE2(2.0, -1.0, 0.7)
    pose_b = SE2(0.3, 0.4, -0.2)
    a_to_b = CoordFrameConverter(pose_a, pose_b)
    b_to_a = CoordFrameConverter(pose_b, pose_a)

    sample = SE2(5.0, 6.0, 0.15)
    round_tripped = b_to_a.convert_pose(a_to_b.convert_pose(sample))
    assert np.allclose(round_tripped.A, sample.A, atol=1e-9)


def test_calibration_point_maps_to_its_partner():
    """The pose pair used to calibrate maps from A→B exactly."""
    pose_a = SE2(2.0, -1.0, 0.7)
    pose_b = SE2(0.3, 0.4, -0.2)
    converter = CoordFrameConverter(pose_a, pose_b)
    assert np.allclose(converter.convert_pose(pose_a).A, pose_b.A, atol=1e-9)


def test_update_replaces_calibration():
    """Update() overwrites prior calibration rather than blending."""
    converter = CoordFrameConverter(SE2(0, 0, 0), SE2(0, 0, 0))
    converter.update(SE2(5.0, 3.0, 0.0), SE2(0.0, 0.0, 0.0))
    converted = converter.convert_pose(SE2(7.0, 4.0, 0.1))
    assert converted.x == pytest.approx(2.0)
    assert converted.y == pytest.approx(1.0)
    assert converted.theta() == pytest.approx(0.1)
