"""Mp4 encoding shared by the plan preview and the trajectory recorder.

`cv2.VideoWriter` writes the file directly, which is markedly faster for
short clips than going through moviepy's Python-side ffmpeg wrapper —
the encode used to be a visible chunk of the operator's wait for
`preview.mp4` (#92).
"""

from __future__ import annotations

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
