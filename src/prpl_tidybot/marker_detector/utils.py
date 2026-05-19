"""Camera helpers for the marker detector.

Wraps OpenCV's `VideoCapture` to open the Logitech C930e ceiling cameras with
fixed focus / white balance / exposure / gain, and loads the per-serial
intrinsic + floor-alignment files shipped under
`prpl_tidybot.marker_detector.camera_params`.

Adapted from the subset of `yixuanhuang98/tidybot_server/server/utils.py`
that the marker detector actually uses.
"""

import json
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.constants import (
    CAMERA_EXPOSURE,
    CAMERA_FOCUS,
    CAMERA_GAIN,
    CAMERA_TEMPERATURE,
)

CAMERA_PARAMS_DIR = Path(__file__).parent / "camera_params"


def get_video_cap(serial: str, frame_width: int, frame_height: int) -> cv.VideoCapture:
    """Open a Logitech C930e at the given resolution with locked exposure."""
    if sys.platform == "darwin":
        return cv.VideoCapture(0)
    cap = cv.VideoCapture(  # type: ignore[unreachable]
        f"/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0"
    )
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # much better latency

    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv.CAP_PROP_AUTO_WB, 0)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = off, 3 = on

    # Settle the sensor: gain/exposure take effect only after a handful of frames.
    for _ in range(30):
        cap.read()
        cap.set(cv.CAP_PROP_FOCUS, CAMERA_FOCUS)
        cap.set(cv.CAP_PROP_TEMPERATURE, CAMERA_TEMPERATURE)
        cap.set(cv.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
        cap.set(cv.CAP_PROP_GAIN, CAMERA_GAIN)

    assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == frame_width
    assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == frame_height
    assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
    assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
    assert cap.get(cv.CAP_PROP_AUTO_WB) == 0
    assert cap.get(cv.CAP_PROP_AUTO_EXPOSURE) == 1
    assert cap.get(cv.CAP_PROP_FOCUS) == CAMERA_FOCUS
    assert cap.get(cv.CAP_PROP_TEMPERATURE) == CAMERA_TEMPERATURE
    assert cap.get(cv.CAP_PROP_EXPOSURE) == CAMERA_EXPOSURE
    assert cap.get(cv.CAP_PROP_GAIN) == CAMERA_GAIN

    return cap


def get_camera_params(serial: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Load intrinsic camera matrix and distortion coefficients for `serial`."""
    path = CAMERA_PARAMS_DIR / f"{serial}.yml"
    assert path.exists(), f"missing camera params file: {path}"
    fs = cv.FileStorage(str(path), cv.FILE_STORAGE_READ)
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs


def get_camera_alignment_params(serial: str) -> tuple[Any, Any]:
    """Load the floor-corner image annotations used to fit the map homography."""
    path = CAMERA_PARAMS_DIR / f"{serial}.json"
    assert path.exists(), f"missing camera alignment file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels["camera_center"], labels["camera_corners"]
