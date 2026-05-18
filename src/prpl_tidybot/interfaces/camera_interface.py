"""Camera interface."""

import abc

import numpy as np
from prpl_utils.structs import Image

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS


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


class RealCameraInterface(CameraInterface):
    """Skeleton real camera interface. Methods raise until the wrist
    (Kinova) and base (Logitech) cameras get wired up."""

    def get_wrist_image(self) -> Image:
        raise NotImplementedError(
            "RealCameraInterface.get_wrist_image: capture a frame from the "
            f"wrist camera (Kinova). Expected shape {WRIST_CAMERA_DIMS}."
        )

    def get_base_image(self) -> Image:
        raise NotImplementedError(
            "RealCameraInterface.get_base_image: capture a frame from the "
            f"base camera (Logitech). Expected shape {BASE_CAMERA_DIMS}."
        )
