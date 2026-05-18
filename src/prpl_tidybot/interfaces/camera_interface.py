"""Camera interface."""

import abc

import numpy as np
from prpl_utils.structs import Image

from prpl_tidybot.constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS


class CameraInterface(abc.ABC):
    """Camera interface."""

    @abc.abstractmethod
    def get_wrist_image(self) -> Image:
        """Get the current wrist image."""

    @abc.abstractmethod
    def get_base_image(self) -> Image:
        """Get the current base image."""


class FakeCameraInterface(CameraInterface):
    """Fake camera interface that returns black images of the expected
    dimensions."""

    def __init__(self) -> None:
        self.wrist_image: Image = np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8)
        self.base_image: Image = np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8)

    def get_wrist_image(self) -> Image:
        return self.wrist_image

    def get_base_image(self) -> Image:
        return self.base_image
