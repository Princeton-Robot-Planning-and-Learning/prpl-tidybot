"""Offline-testable colored-band detection for a possible forearm target.

This returns image coordinates only. It does NOT identify a bare arm, estimate
depth, validate left/right identity, or command motion. HSV bounds and the grasp
pixel must be calibrated from real frames. A physical band is still optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class BandDetection:
    center_uv: tuple[float, float]
    long_axis_uv: tuple[float, float]
    area_px: float


def detect_band(
    image_rgb: np.ndarray,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
    min_area_px: float,
) -> BandDetection | None:
    """Find exactly one unclipped band-colored component; reject ambiguity.

    The long image axis is not automatically the forearm axis: a narrow cuff
    can run across the forearm. Its relation to the grasp requires calibration.
    Hue intervals that wrap around zero require a separate mask implementation.
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a uint8 RGB image")
    if min(image_rgb.shape[:2]) < 3:
        raise ValueError("Image is too small")
    if not np.isfinite(min_area_px) or min_area_px <= 0:
        raise ValueError("Minimum area must be finite and positive")
    if len(hsv_low) != 3 or len(hsv_high) != 3:
        raise ValueError("HSV bounds must have three components")
    for low, high, maximum in zip(hsv_low, hsv_high, (179, 255, 255)):
        if not 0 <= low <= high <= maximum:
            raise ValueError("Invalid OpenCV HSV interval")
    hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv.contourArea(c) >= min_area_px]
    if len(candidates) != 1:
        return None
    contour = candidates[0]
    x, y, width, height = cv.boundingRect(contour)
    rows, cols = image_rgb.shape[:2]
    if x <= 1 or y <= 1 or x + width >= cols - 1 or y + height >= rows - 1:
        return None
    moments = cv.moments(contour)
    if moments["m00"] <= 0:
        return None
    box = cv.boxPoints(cv.minAreaRect(contour))
    edges = np.roll(box, -1, axis=0) - box
    axis = edges[np.argmax(np.linalg.norm(edges, axis=1))].astype(float)
    axis /= np.linalg.norm(axis)
    return BandDetection(
        (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
        (float(axis[0]), float(axis[1])),
        float(cv.contourArea(contour)),
    )
