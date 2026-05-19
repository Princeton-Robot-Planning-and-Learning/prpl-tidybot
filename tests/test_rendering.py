"""Tests for rendering.py."""

import numpy as np

from prpl_tidybot.rendering import CeilingCameraRenderer


class _FakeCeilingImageClient:
    """In-memory `CeilingImageClient` stand-in.

    Each `get_image()` returns the next scripted BGR frame (or the last one once the
    script is exhausted), letting tests exercise the renderer without binding a real
    socket.
    """

    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self._frames = list(frames)
        self._idx = 0
        self.closed = False

    def get_image(self) -> np.ndarray | None:
        """Return the next scripted frame (or the last one if the script is done)."""
        frame = self._frames[min(self._idx, len(self._frames) - 1)]
        self._idx += 1
        return frame

    def close(self) -> None:
        """Record that close() was called so tests can assert on cleanup."""
        self.closed = True


def test_ceiling_camera_renderer_returns_none_when_no_frame_yet():
    """Before any frame has arrived from the publisher, render() returns None."""
    client = _FakeCeilingImageClient([None])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    assert renderer.render() is None


def test_ceiling_camera_renderer_converts_bgr_to_rgb():
    """The client returns BGR; render() must convert to RGB so moviepy reads colors
    right."""
    # A single pixel where B=10, G=20, R=30 (BGR ordering).
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    client = _FakeCeilingImageClient([bgr])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    rgb = renderer.render()
    assert rgb is not None
    # After BGR->RGB, the pixel should read [30, 20, 10].
    assert rgb.shape == bgr.shape
    assert rgb.dtype == bgr.dtype
    assert tuple(rgb[0, 0].tolist()) == (30, 20, 10)


def test_ceiling_camera_renderer_close_propagates_to_client():
    """Close() releases the underlying client."""
    client = _FakeCeilingImageClient([np.zeros((1, 1, 3), dtype=np.uint8)])
    renderer = CeilingCameraRenderer(client=client)  # type: ignore[arg-type]
    renderer.close()
    assert client.closed
