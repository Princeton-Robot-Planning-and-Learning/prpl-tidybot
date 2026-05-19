"""JPEG image publisher for the top ceiling camera.

Network-friendly companion to `CameraServer`: `CameraServer` uses shared
memory and only works on the camera host, whereas this publisher serves
JPEG-encoded frames over the same multiprocessing-connection protocol as
`MarkerDetectorServer`, so off-host clients (e.g. the video recorder
running on the NUC) can read frames over TCP.

Runs in-process inside the marker-detector daemon and subscribes to the
top `CameraServer` via the local `CameraClient`. The published payload
is `{"jpeg": <bytes>}` — opaque encoded bytes the client decodes on
receive.
"""

from typing import Any

import cv2 as cv

from prpl_tidybot.marker_detector.camera_client import CameraClient
from prpl_tidybot.marker_detector.constants import (
    CAMERA_SERVER_PORTS,
    CEILING_IMAGE_PORT,
)
from prpl_tidybot.marker_detector.publisher import Publisher


class CeilingImagePublisher(Publisher):
    """Publishes JPEG frames from the top ceiling camera over a TCP socket."""

    def __init__(
        self,
        hostname: str = "localhost",
        port: int = CEILING_IMAGE_PORT,
        camera_port: int = CAMERA_SERVER_PORTS[0],
        jpeg_quality: int = 80,
    ) -> None:
        super().__init__(hostname=hostname, port=port)
        self._camera_client = CameraClient(camera_port)
        self._jpeg_quality = int(jpeg_quality)

    def get_data(self) -> dict[str, Any]:
        image = self._camera_client.get_image()
        success, jpeg = cv.imencode(
            ".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not success:
            return {"jpeg": None}
        return {"jpeg": jpeg.tobytes()}

    def clean_up(self) -> None:
        self._camera_client.close()
