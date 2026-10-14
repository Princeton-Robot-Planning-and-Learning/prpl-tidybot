"""Subscriber for `CeilingImagePublisher`.

Decodes the JPEG bytes from each payload into a BGR `ndarray`. Like
`MarkerDetectorClient`, it caches the last successful frame so brief
publisher stalls (or a recorder calling `get_image` between two
publisher refreshes) return the most recent value instead of `None`.
"""

import time
from multiprocessing.connection import Client, Connection

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.constants import CEILING_IMAGE_PORT
from prpl_tidybot.third_party.constants import CONN_AUTHKEY, SERVER_HOSTNAME


class CeilingImageClient:
    """Subscriber for `CeilingImagePublisher`: returns BGR frames as ndarrays."""

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = CEILING_IMAGE_PORT,
        poll_timeout_s: float = 1.0,
    ) -> None:
        self._conn: Connection = Client((host, port), authkey=CONN_AUTHKEY)
        self._poll_timeout_s = poll_timeout_s
        self._conn.send(None)
        self._last: np.ndarray | None = None

    def get_image(self) -> np.ndarray | None:
        """Return the most recent decoded frame, or `None` if none has arrived."""
        if self._conn.poll(timeout=self._poll_timeout_s):
            payload = self._conn.recv()
            self._conn.send(None)
            jpeg = payload.get("jpeg")
            if jpeg:
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                self._last = cv.imdecode(arr, cv.IMREAD_COLOR)
        return self._last

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


class FakeCeilingImageClient:
    """Stand-in for `CeilingImageClient` in fake mode: a constant grey BGR frame.

    Each call waits ``frame_period_s`` to mimic the publisher's cadence, so a
    `CeilingCameraRenderer` poll thread on top of it does not busy-spin.
    """

    def __init__(
        self, height: int = 120, width: int = 160, frame_period_s: float = 0.02
    ) -> None:
        self._frame = np.full((height, width, 3), 128, dtype=np.uint8)
        self._frame_period_s = frame_period_s
        self.closed = False

    def get_image(self) -> np.ndarray | None:
        """Return the constant frame (None once closed)."""
        time.sleep(self._frame_period_s)
        return None if self.closed else self._frame

    def close(self) -> None:
        """Record the close."""
        self.closed = True
