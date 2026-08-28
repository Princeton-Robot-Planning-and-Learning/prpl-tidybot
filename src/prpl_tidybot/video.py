"""Mp4 encoding and frame subsampling shared by the plan preview and the
trajectory recorder.

`cv2.VideoWriter` writes the file directly, which is markedly faster for
short clips than going through moviepy's Python-side ffmpeg wrapper —
the encode used to be a visible chunk of the operator's wait for
`preview.mp4` (#92).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np


def write_mp4(frames_rgb: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Encode RGB frames to `out_path` as mp4.

    H.264 (`avc1`) is preferred because browser-based players (including VS
    Code's video preview, used to eyeball previews over VS Code Remote) don't
    decode the MPEG-4 Part 2 video `mp4v` produces; `mp4v` is the fallback
    for OpenCV builds without an H.264 encoder.

    The output size is taken from the first frame; any later frame with a different
    shape is resized to match (VideoWriter silently drops mismatched frames otherwise).
    """
    if not frames_rgb:
        raise ValueError("Cannot write an mp4 with no frames.")
    height, width = frames_rgb[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc in ("avc1", "mp4v"):
        writer = cv.VideoWriter(
            str(out_path), cv.VideoWriter.fourcc(*fourcc), fps, (width, height)
        )
        if writer.isOpened():
            break
        writer.release()
    else:
        raise RuntimeError(
            f"OpenCV could not open a VideoWriter for {out_path} with any "
            "of the attempted codecs (avc1, mp4v)."
        )
    try:
        for frame in frames_rgb:
            if frame.shape[:2] != (height, width):
                frame = cv.resize(frame, (width, height))
            writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    finally:
        writer.release()


def subsample_indices(
    num_frames: int, max_frames: int | None, keep: set[int] | None = None
) -> list[int]:
    """Indices to render: a stride over all frames plus every index in `keep`.

    With ``max_frames=None`` or ``num_frames <= max_frames`` every index is
    returned. Otherwise the stride yields at most ``max_frames`` indices and the
    kept indices (always including the first and last) are merged in, so the
    result may exceed ``max_frames`` by the size of ``keep``.
    """
    if num_frames <= 0:
        return []
    if max_frames is None or num_frames <= max_frames:
        return list(range(num_frames))
    if max_frames < 1:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    stride = math.ceil(num_frames / max_frames)
    must_keep = {0, num_frames - 1} | (keep or set())
    return sorted(set(range(0, num_frames, stride)) | must_keep)
