"""Tests for camera_interface.py."""

import numpy as np

from prpl_tidybot.constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface


def test_fake_camera_interface_defaults():
    """FakeCameraInterface() returns zero images of the expected shape."""
    camera = FakeCameraInterface()
    assert camera.get_wrist_image().shape == WRIST_CAMERA_DIMS
    assert camera.get_base_image().shape == BASE_CAMERA_DIMS
    assert np.all(camera.get_wrist_image() == 0)
    assert np.all(camera.get_base_image() == 0)


def test_fake_camera_interface_swap_images():
    """Overriding wrist_image/base_image is reflected in the getters."""
    camera = FakeCameraInterface()
    camera.wrist_image = np.full(WRIST_CAMERA_DIMS, 255, dtype=np.uint8)
    camera.base_image = np.full(BASE_CAMERA_DIMS, 128, dtype=np.uint8)
    assert np.all(camera.get_wrist_image() == 255)
    assert np.all(camera.get_base_image() == 128)
