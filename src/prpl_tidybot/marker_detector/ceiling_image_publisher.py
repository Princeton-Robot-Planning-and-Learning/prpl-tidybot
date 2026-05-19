"""JPEG image publisher for the ceiling cameras.

Network-friendly companion to `CameraServer`: `CameraServer` uses shared
memory and only works on the camera host, whereas this publisher serves
JPEG-encoded frames over the same multiprocessing-connection protocol as
`MarkerDetectorServer`, so off-host clients (e.g. the video recorder
running on the NUC) can read frames over TCP.

Runs in-process inside the marker-detector daemon and subscribes to one
or more local `CameraServer`s via `CameraClient`. When more than one
camera is configured the frames are vstacked into a single composite
image before encoding; that keeps the client and renderer single-frame
APIs unchanged. The published payload is `{"jpeg": <bytes>}` — opaque
encoded bytes the client decodes on receive.
"""

from typing import Any, Sequence

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.camera_client import CameraClient
from prpl_tidybot.marker_detector.constants import (
    CAMERA_SERVER_PORTS,
    CEILING_IMAGE_PORT,
)
from prpl_tidybot.marker_detector.publisher import Publisher


def _make_composite_jpeg(
    images: Sequence[np.ndarray], jpeg_quality: int
) -> bytes | None:
    """Vstack frames (resizing to a common width if needed) and JPEG-encode.

    Returns the encoded bytes, or `None` if encoding failed or no images were
    provided. Single-image inputs skip the stacking step.
    """
    if not images:
        return None
    widths = {im.shape[1] for im in images}
    if len(widths) > 1:
        target_w = max(widths)
        images = [
            (
                cv.resize(im, (target_w, int(im.shape[0] * target_w / im.shape[1])))
                if im.shape[1] != target_w
                else im
            )
            for im in images
        ]
    composite = images[0] if len(images) == 1 else np.vstack(list(images))
    success, jpeg = cv.imencode(
        ".jpg", composite, [int(cv.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not success:
        return None
    return jpeg.tobytes()


class CeilingImagePublisher(Publisher):
    """Publishes vstacked JPEG frames from the configured ceiling cameras."""

    def __init__(
        self,
        hostname: str = "localhost",
        port: int = CEILING_IMAGE_PORT,
        camera_ports: Sequence[int] = CAMERA_SERVER_PORTS,
        jpeg_quality: int = 80,
    ) -> None:
        super().__init__(hostname=hostname, port=port)
        self._camera_clients = [CameraClient(p) for p in camera_ports]
        self._jpeg_quality = int(jpeg_quality)

    def get_data(self) -> dict[str, Any]:
        frames = [client.get_image() for client in self._camera_clients]
        return {"jpeg": _make_composite_jpeg(frames, self._jpeg_quality)}

    def clean_up(self) -> None:
        for client in self._camera_clients:
            client.close()
