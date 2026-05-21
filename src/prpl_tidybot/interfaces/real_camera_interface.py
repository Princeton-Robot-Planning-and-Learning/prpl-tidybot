"""Real camera interface backed by the Kinova wrist camera and Logitech base camera."""

from prpl_utils.structs import Image

from prpl_tidybot.interfaces.camera_interface import CameraInterface
from prpl_tidybot.third_party.cameras import KinovaCamera, LogitechCamera
from prpl_tidybot.third_party.constants import BASE_CAMERA_SERIAL


class RealCameraInterface(CameraInterface):
    """Real camera interface.

    wrist camera: Kinova wrist camera (RTSP via GStreamer)
    base camera: Logitech C930e
    """

    def __init__(self) -> None:
        self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
        self.wrist_camera = KinovaCamera()

    def get_wrist_image(self) -> Image:
        return self.wrist_camera.get_image()

    def get_base_image(self) -> Image:
        return self.base_camera.get_image()

    def close(self) -> None:
        """Release camera resources."""
        self.base_camera.close()
        self.wrist_camera.close()
