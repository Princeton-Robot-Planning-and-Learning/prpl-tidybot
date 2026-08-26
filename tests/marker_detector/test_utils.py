"""Tests for marker_detector/utils.py."""

import numpy as np
import pytest

from prpl_tidybot.marker_detector.constants import FLOOR_LENGTH, FLOOR_WIDTH
from prpl_tidybot.marker_detector.utils import (
    get_floor_corner_coords,
    load_lab_camera_config,
)


@pytest.mark.parametrize("placement", ["top", "bottom", "top_only"])
def test_floor_corner_coords_shape_and_span(placement):
    """Every placement yields four corners spanning the full floor width."""
    coords = get_floor_corner_coords(placement)
    assert coords.shape == (4, 2)
    assert coords.dtype == np.float32
    assert np.isclose(coords[:, 0].min(), -FLOOR_WIDTH / 2)
    assert np.isclose(coords[:, 0].max(), FLOOR_WIDTH / 2)


def test_top_and_bottom_halves_share_the_midline():
    """The top camera's near edge is the bottom camera's far edge (y = 0)."""
    top = get_floor_corner_coords("top")
    bottom = get_floor_corner_coords("bottom")
    assert np.allclose(top[2:, 1], 0.0)
    assert np.allclose(bottom[:2, 1], 0.0)
    assert np.allclose(top[:2, 1], FLOOR_LENGTH / 2)
    assert np.allclose(bottom[2:, 1], -FLOOR_LENGTH / 2)


def test_load_lab_camera_config_reads_conf():
    """The prpl lab config resolves to two serials and a positive height."""
    serials, height = load_lab_camera_config("prpl")
    assert len(serials) == 2
    assert height > 0
