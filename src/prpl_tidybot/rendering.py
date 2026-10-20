"""Render sources for `RealTidyBotEnv.render()`.

`RealTidyBotEnv.render()` defaults to `interface.get_base_image()` (the
on-robot base camera). When that camera isn't a useful viewpoint — e.g.
recording a base-only rollout on the real robot, where the on-robot
camera isn't wired up — pass a `Renderer` to swap in an external view.

The current concrete impl is `CeilingCameraRenderer`, which subscribes
to `CeilingImagePublisher` and yields RGB frames from the top ceiling
camera. The renderer runs its own background poll thread so that
`render()` is a non-blocking cache read — important for callers (the
trajectory recorder) that are on the control hot path and can't afford
to wait the publisher's refresh interval on every call.
"""

import threading
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
    daemon on the perception PC) via `CeilingImageClient`. A background thread
    drains the publisher continuously and stores the latest decoded frame
    (BGR -> RGB) in `self._latest`; `render()` is a non-blocking attribute read
    of that cache.

    Polling on a thread is what lets the recorder call `render()` once per
    inner real-env tick without paying the publisher's per-call blocking
    latency — the same pattern stretched the gap between consecutive base
    commands past the controller's command-timeout in #45.
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
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame_bgr = self._client.get_image()
            except Exception:  # pylint: disable=broad-except
                # Connection broke mid-shutdown (close() raced ahead of the
                # final poll). Exiting the loop cleanly is the right call —
                # `render()` keeps serving the last cached frame.
                break
            if frame_bgr is not None:
                self._latest = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

    def render(self) -> np.ndarray | None:
        """Return the latest cached top-camera frame as RGB, or None if none has
        arrived yet."""
        return self._latest

    def close(self) -> None:
        """Stop the poll thread and close the underlying client."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._client.close()
