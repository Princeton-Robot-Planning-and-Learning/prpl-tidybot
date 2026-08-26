"""Tests for `prpl_tidybot.video`."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.video import write_mp4


def _uniform_frame(color: int, shape: tuple[int, int] = (16, 24)) -> np.ndarray:
    return np.full((shape[0], shape[1], 3), color, dtype=np.uint8)


def test_write_mp4_produces_decodable_file_with_all_frames(tmp_path: Path):
    """Every input frame round-trips through the encoder."""
    frames = [_uniform_frame(c) for c in (40, 80, 120, 160)]
    out = tmp_path / "clip.mp4"

    write_mp4(frames, out, fps=5)

    assert out.exists() and out.stat().st_size > 0
    cap = cv.VideoCapture(str(out))
    try:
        assert int(cap.get(cv.CAP_PROP_FRAME_COUNT)) == len(frames)
        ok, first = cap.read()
    finally:
        cap.release()
    assert ok
    assert first.shape[:2] == frames[0].shape[:2]


def test_write_mp4_converts_rgb_to_bgr(tmp_path: Path):
    """A pure-red RGB frame must decode back red, not blue — guards the RGB→BGR
    conversion the writer owns."""
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    out = tmp_path / "red.mp4"

    write_mp4([red] * 3, out, fps=5)

    cap = cv.VideoCapture(str(out))
    try:
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    assert ok
    # Decoded frame is BGR: red lives in the last channel. Lossy codec, so
    # compare loosely.
    assert frame_bgr[..., 2].mean() > 200
    assert frame_bgr[..., 0].mean() < 55


def test_write_mp4_resizes_mismatched_frames(tmp_path: Path):
    """Frames that don't match the first frame's shape are resized rather than silently
    dropped by VideoWriter."""
    frames = [_uniform_frame(50, (16, 24)), _uniform_frame(100, (32, 48))]
    out = tmp_path / "mixed.mp4"

    write_mp4(frames, out, fps=5)

    cap = cv.VideoCapture(str(out))
    try:
        assert int(cap.get(cv.CAP_PROP_FRAME_COUNT)) == 2
    finally:
        cap.release()


def test_write_mp4_falls_back_when_preferred_codec_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Builds without an H.264 encoder (avc1 writer fails to open) still write a
    playable file through the mp4v fallback."""
    real_writer = cv.VideoWriter
    attempted: list[int] = []

    class _Unopened:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    class _FailAvc1Factory:
        fourcc = staticmethod(real_writer.fourcc)

        def __call__(self, path: str, fourcc: int, fps: int, size: tuple):
            attempted.append(fourcc)
            if fourcc == real_writer.fourcc(*"avc1"):
                return _Unopened()
            return real_writer(path, fourcc, fps, size)

    import prpl_tidybot.video as video_mod

    monkeypatch.setattr(video_mod.cv, "VideoWriter", _FailAvc1Factory())
    out = tmp_path / "fallback.mp4"

    write_mp4([_uniform_frame(90)] * 3, out, fps=5)

    assert attempted == [real_writer.fourcc(*"avc1"), real_writer.fourcc(*"mp4v")]
    cap = cv.VideoCapture(str(out))
    try:
        assert int(cap.get(cv.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap.release()


def test_write_mp4_rejects_empty_frame_list(tmp_path: Path):
    """No frames is a caller bug — surfaced as ValueError, not a corrupt file."""
    with pytest.raises(ValueError, match="no frames"):
        write_mp4([], tmp_path / "empty.mp4", fps=5)
