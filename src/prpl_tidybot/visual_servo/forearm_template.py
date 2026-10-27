"""Bare-forearm patch matching for the fixed-pose first trial.

An operator selects a distinctive patch of the LEFT forearm separately in each
camera. This is constrained tracking, not anatomical recognition. Templates are
camera/hover-specific and must be reinitialized after pose or scale changes.
No model assets, image files, or robot interfaces are embedded in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class PatchMatch:
    center_uv: tuple[float, float]
    quality: float
    runner_up_quality: float


def _gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image")
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def _rectangle(rect: tuple[int, int, int, int], width: int, height: int) -> None:
    x, y, w, h = rect
    if (
        not all(isinstance(v, int) for v in rect)
        or min(w, h) < 8
        or x < 0
        or y < 0
        or x + w > width
        or y + h > height
    ):
        raise ValueError("Invalid image rectangle (x, y, width, height)")


class ForearmPatchTracker:
    """Match one operator-identified patch within a restricted camera-local ROI.

    Rejects weak/ambiguous matches and patches touching ROI/image borders. Keep
    the ROI inside a single camera, away from the composite seam. The returned
    patch centre is not the grasp centre: calibrate that offset/target pixel.
    """

    def __init__(
        self,
        reference_rgb: np.ndarray,
        patch_xywh: tuple[int, int, int, int],
        *,
        minimum_quality: float,
        minimum_margin: float,
    ) -> None:
        gray = _gray(reference_rgb)
        _rectangle(patch_xywh, gray.shape[1], gray.shape[0])
        if not 0 < minimum_quality <= 1 or not 0 < minimum_margin < 1:
            raise ValueError("Invalid matching thresholds")
        x, y, w, h = patch_xywh
        self.template = gray[y : y + h, x : x + w].copy()
        if float(self.template.std()) < 5:
            raise ValueError(
                "Patch has too little texture; include a forearm edge/detail"
            )
        self.minimum_quality = minimum_quality
        self.minimum_margin = minimum_margin

    def detect(
        self, image_rgb: np.ndarray, search_xywh: tuple[int, int, int, int]
    ) -> PatchMatch | None:
        gray = _gray(image_rgb)
        _rectangle(search_xywh, gray.shape[1], gray.shape[0])
        x, y, w, h = search_xywh
        th, tw = self.template.shape
        if w <= tw + 2 or h <= th + 2:
            raise ValueError("Search region must be larger than the patch")
        scores = cv.matchTemplate(
            gray[y : y + h, x : x + w], self.template, cv.TM_CCOEFF_NORMED
        )
        if not np.isfinite(scores).all():
            return None
        _, best, _, location = cv.minMaxLoc(scores)
        u, v = location
        if (
            best < self.minimum_quality
            or u <= 1
            or v <= 1
            or u + tw >= w - 1
            or v + th >= h - 1
        ):
            return None
        # Suppress only the local peak; another plausible patch causes rejection.
        others = scores.copy()
        rx, ry = max(2, tw // 4), max(2, th // 4)
        others[max(0, v - ry) : v + ry + 1, max(0, u - rx) : u + rx + 1] = -1
        runner_up = float(others.max())
        if best - runner_up < self.minimum_margin:
            return None
        return PatchMatch((x + u + tw / 2, y + v + th / 2), float(best), runner_up)


def ceiling_view(image_rgb: np.ndarray, camera: int, *, seam_px: int = 8) -> np.ndarray:
    """Split the existing two-camera composite, masking its seam edge only.

    Returned pixel coordinates remain camera-local; use that camera's intrinsics
    and extrinsics. Detection ROIs must also avoid the masked edge.
    """
    _gray(image_rgb)
    if camera not in (0, 1) or image_rgb.shape[0] % 2 or seam_px < 1:
        raise ValueError(
            "Expected two equal-height camera views and positive seam mask"
        )
    rows = image_rgb.shape[0] // 2
    if rows <= seam_px:
        raise ValueError("Image too small for seam mask")
    result = image_rgb[camera * rows : (camera + 1) * rows].copy()
    if camera == 0:
        result[-seam_px:] = 0
    else:
        result[:seam_px] = 0
    return result
