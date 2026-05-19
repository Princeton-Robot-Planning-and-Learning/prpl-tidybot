"""Render sources for `RealTidyBotEnv.render()`.

`RealTidyBotEnv.render()` defaults to `interface.get_base_image()` (the
on-robot base camera). When that camera isn't a useful viewpoint — e.g.
recording a base-only rollout on the real robot, where the on-robot
camera isn't wired up — pass a `Renderer` to swap in an external view.

The current concrete impl is `CeilingCameraRenderer`, which subscribes
to `CeilingImagePublisher` and yields RGB frames from the top ceiling
camera.
"""

from typing import Protocol

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.ceiling_image_client import CeilingImageClient
from prpl_tidybot.marker_detector.constants import CEILING_IMAGE_PORT
from prpl_tidybot.third_party.constants import SERVER_HOSTNAME


class Renderer(Protocol):
    """Source of a single RGB frame for `RealTidyBotEnv.render()`."""

    def render(self) -> np.ndarray | None:
        """Return the latest RGB frame, or None if no frame is available."""

    def close(self) -> None:
        """Release any underlying resources."""


class CeilingCameraRenderer:
    """Top-camera-backed `Renderer` for off-host recording.

    Subscribes to `CeilingImagePublisher` (running inside the marker-detector
    daemon on the perception PC) via `CeilingImageClient`, decodes each JPEG
    payload into a BGR `ndarray`, and converts to RGB before returning so the
    moviepy recorder sees the colors right.
    """

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = CEILING_IMAGE_PORT,
        poll_timeout_s: float = 1.0,
        client: CeilingImageClient | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else CeilingImageClient(host=host, port=port, poll_timeout_s=poll_timeout_s)
        )

    def render(self) -> np.ndarray | None:
        """Return the latest top-camera frame as RGB, or None if none has arrived."""
        frame_bgr = self._client.get_image()
        if frame_bgr is None:
            return None
        return cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

    def close(self) -> None:
        """Close the underlying client connection."""
        self._client.close()
