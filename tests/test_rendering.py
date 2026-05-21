"""Tests for rendering.py."""

import threading
import time

import numpy as np

from prpl_tidybot.rendering import CeilingCameraRenderer


class _FakeCeilingImageClient:
    """In-memory `CeilingImageClient` stand-in.

    Each `get_image()` returns the next scripted BGR frame (or the last one once the
    script is exhausted), letting tests exercise the renderer without binding a real
    socket. Calls block briefly between returns to mirror the real client's "wait for
    next publisher payload" behaviour, so the poll thread doesn't busy-spin in the test
    process.
    """

    def __init__(
        self, frames: list[np.ndarray | None], inter_call_delay_s: float = 0.005
    ) -> None:
        self._frames = list(frames)
        self._idx = 0
        self._delay = inter_call_delay_s
        self._lock = threading.Lock()
        self.closed = False

    def get_image(self) -> np.ndarray | None:
        """Return the next scripted frame (or the last one if the script is done)."""
        time.sleep(self._delay)
        with self._lock:
            if self.closed:
                return None
            frame = self._frames[min(self._idx, len(self._frames) - 1)]
            self._idx += 1
        return frame

    def close(self) -> None:
        """Record that close() was called so tests can assert on cleanup."""
        with self._lock:
            self.closed = True


def _wait_for_frame(renderer: CeilingCameraRenderer, timeout_s: float = 2.0):
    """Poll renderer.render() until non-None or timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = renderer.render()
        if frame is not None:
            return frame
        time.sleep(0.01)
    return renderer.render()


def test_ceiling_camera_renderer_returns_none_when_no_frame_yet():
    """Before any frame has arrived from the publisher, render() returns None."""
    client = _FakeCeilingImageClient([None])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    try:
        # The bg thread polls the client and stores nothing because every call
        # returns None; render() should keep returning None.
        time.sleep(0.05)
        assert renderer.render() is None
    finally:
        renderer.close()


def test_ceiling_camera_renderer_converts_bgr_to_rgb():
    """The client returns BGR; render() must convert to RGB so moviepy reads colors
    right."""
    # A single pixel where B=10, G=20, R=30 (BGR ordering).
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    client = _FakeCeilingImageClient([bgr])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    try:
        rgb = _wait_for_frame(renderer)
        assert rgb is not None
        # After BGR->RGB, the pixel should read [30, 20, 10].
        assert rgb.shape == bgr.shape
        assert rgb.dtype == bgr.dtype
        assert tuple(rgb[0, 0].tolist()) == (30, 20, 10)
    finally:
        renderer.close()


def test_ceiling_camera_renderer_close_propagates_to_client():
    """Close() stops the poll thread and releases the underlying client."""
    client = _FakeCeilingImageClient([np.zeros((1, 1, 3), dtype=np.uint8)])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    renderer.close()
    assert client.closed
    assert not renderer._thread.is_alive()  # pylint: disable=protected-access
