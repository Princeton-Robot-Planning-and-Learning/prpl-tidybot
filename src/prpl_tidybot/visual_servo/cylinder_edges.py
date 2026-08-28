"""Find the two vertical edges of a cylinder in a wrist-camera image.

The cylinder is assumed to stand roughly upright in front of the gripper
with both of its silhouette edges in view, which is what the pre-grasp
pose of the cylinder-shelf skills gives. The detector is a column-wise
horizontal-gradient profile: within a horizontal band of the image
(``roi_top`` to ``roi_bottom``, chosen to exclude the gripper fingers that
show at the bottom of the frame) the absolute Sobel-x response is
averaged over rows, smoothed, and its local maxima are the candidate
vertical edges. The pair of peaks with the strongest combined response
whose separation is a plausible cylinder width wins.

Everything is a pure function of the image and :class:`EdgeDetectorParams`,
so it can be run and tuned offline on saved frames
(``scripts/detect_cylinder_edges.py``) and unit-tested on synthetic images.
:func:`render_edge_overlay` draws the band, the profile, the detected edges,
and the lateral error on a copy of the image for that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from prpl_utils.structs import Image


@dataclass(frozen=True)
class EdgeDetectorParams:
    """Tunables for :func:`detect_cylinder_edges`.

    ``roi_top`` / ``roi_bottom`` are fractions of the image height bounding
    the rows that feed the column profile. Widths are fractions of the image
    width. ``min_edge_strength`` is the smallest mean |Sobel-x| response (gray
    levels per pixel, before normalisation; the simulator's soft-shaded
    cylinder measures ~3, a bare floor ~1) the strongest column must reach
    for the frame to count as containing an edge at all; ``min_peak_frac`` is
    the minimum height of an accepted peak relative to that maximum.
    ``peak_min_separation_px`` suppresses secondary maxima within that many
    columns of a stronger one. ``min_contrast`` is the smallest difference in
    mean gray level between the band inside the two edges and the band
    outside them: a cylinder is a silhouette, not just two lines.
    """

    roi_top: float = 0.25
    roi_bottom: float = 0.75
    blur_ksize: int = 5
    profile_smooth_px: int = 9
    min_width_frac: float = 0.06
    max_width_frac: float = 0.7
    min_edge_strength: float = 1.5
    min_peak_frac: float = 0.2
    peak_min_separation_px: int = 6
    min_contrast: float = 10.0


@dataclass(frozen=True)
class CylinderEdges:
    """The two detected vertical edges, in image columns (sub-pixel)."""

    left_x: float
    right_x: float
    image_width: int
    image_height: int
    left_response: float
    right_response: float
    contrast: float

    @property
    def center_x(self) -> float:
        """Column of the cylinder's vertical axis."""
        return 0.5 * (self.left_x + self.right_x)

    @property
    def width_px(self) -> float:
        """Apparent cylinder width in columns."""
        return self.right_x - self.left_x

    @property
    def lateral_error_px(self) -> float:
        """Signed offset of the cylinder axis from the image center: positive when
        the cylinder is to the right of center."""
        return self.center_x - 0.5 * (self.image_width - 1)


def column_edge_profile(
    image: Image, params: EdgeDetectorParams = EdgeDetectorParams()
) -> np.ndarray:
    """Per-column vertical-edge response over the ROI band, normalised to max 1.

    Returns a float array of length ``image.shape[1]``; all zeros if the
    image has no horizontal gradient at all.
    """
    profile, peak = _raw_column_edge_profile(image, params)
    if peak <= 0.0:
        return np.zeros_like(profile)
    return np.asarray(profile, dtype=np.float64) / peak


def _roi_band(image: Image, params: EdgeDetectorParams) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if image.ndim == 3 else image
    height = gray.shape[0]
    top = int(round(params.roi_top * height))
    bottom = max(top + 1, int(round(params.roi_bottom * height)))
    return gray[top:bottom]


def _raw_column_edge_profile(
    image: Image, params: EdgeDetectorParams
) -> tuple[np.ndarray, float]:
    """Un-normalised column profile (mean |Sobel-x| over the band, smoothed) and its
    maximum, in gray levels per pixel (Sobel's 3x3 kernel scale included)."""
    band = _roi_band(image, params)
    if params.blur_ksize > 1:
        band = cv.GaussianBlur(band, (params.blur_ksize, params.blur_ksize), 0)
    gradient_x = (
        np.asarray(cv.Sobel(band, cv.CV_32F, 1, 0, ksize=3), dtype=np.float64) / 8.0
    )
    profile = np.abs(gradient_x).mean(axis=0).astype(np.float64)
    if params.profile_smooth_px > 1:
        kernel = np.ones(params.profile_smooth_px) / params.profile_smooth_px
        profile = np.convolve(profile, kernel, mode="same")
    return profile, float(profile.max())


def detect_cylinder_edges(
    image: Image, params: EdgeDetectorParams = EdgeDetectorParams()
) -> CylinderEdges | None:
    """Locate the cylinder's left and right edges, or return None if no plausible pair
    of vertical edges is found."""
    raw_profile, peak = _raw_column_edge_profile(image, params)
    if peak < params.min_edge_strength:
        return None
    profile = np.asarray(raw_profile, dtype=np.float64) / peak
    peaks = _find_peaks(profile, params)
    if len(peaks) < 2:
        return None
    width = image.shape[1]
    min_width = params.min_width_frac * width
    max_width = params.max_width_frac * width
    band = _roi_band(image, params).astype(np.float32)
    column_means = band.mean(axis=0)
    best_score = -1.0
    best: tuple[int, int, float] | None = None
    for i, left in enumerate(peaks):
        for right in peaks[i + 1 :]:
            separation = right - left
            if separation < min_width or separation > max_width:
                continue
            contrast = _silhouette_contrast(column_means, left, right)
            if contrast < params.min_contrast:
                continue
            score = float(profile[left] + profile[right])
            if score > best_score:
                best_score = score
                best = (left, right, contrast)
    if best is None:
        return None
    left, right, contrast = best
    return CylinderEdges(
        left_x=_refine_peak(profile, left),
        right_x=_refine_peak(profile, right),
        image_width=width,
        image_height=image.shape[0],
        left_response=float(profile[left]),
        right_response=float(profile[right]),
        contrast=contrast,
    )


def render_edge_overlay(
    image: Image,
    edges: CylinderEdges | None,
    params: EdgeDetectorParams = EdgeDetectorParams(),
    label: str = "",
) -> Image:
    """Draw the ROI band, the column profile, the detected edges and the lateral error
    on a copy of `image` (RGB in, RGB out)."""
    out = np.ascontiguousarray(image.copy())
    height, width = out.shape[:2]
    top = int(round(params.roi_top * height))
    bottom = int(round(params.roi_bottom * height))
    cv.rectangle(out, (0, top), (width - 1, bottom), (255, 255, 0), 1)
    # Column profile plotted along the bottom quarter of the frame.
    profile = column_edge_profile(image, params)
    plot_height = max(1, height // 4)
    baseline = height - 1
    points = [(x, int(baseline - profile[x] * (plot_height - 1))) for x in range(width)]
    cv.polylines(out, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 1)
    center_col = int(round(0.5 * (width - 1)))
    cv.line(out, (center_col, 0), (center_col, height - 1), (255, 255, 255), 1)
    if edges is not None:
        for x in (edges.left_x, edges.right_x):
            col = int(round(x))
            cv.line(out, (col, 0), (col, height - 1), (0, 255, 0), 2)
        axis = int(round(edges.center_x))
        cv.line(out, (axis, top), (axis, bottom), (255, 0, 0), 2)
        text = (
            f"err {edges.lateral_error_px:+.1f}px  width {edges.width_px:.0f}px  "
            f"resp {edges.left_response:.2f}/{edges.right_response:.2f}"
        )
    else:
        text = "no cylinder edges"
    if label:
        text = f"{label}  {text}"
    cv.putText(
        out, text, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA
    )
    cv.putText(
        out, text, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA
    )
    return out


def _silhouette_contrast(column_means: np.ndarray, left: int, right: int) -> float:
    """Absolute difference between the mean gray level between the two columns and the
    mean gray level outside them (0 if there is nothing outside)."""
    inside = column_means[left : right + 1]
    outside = np.concatenate([column_means[:left], column_means[right + 1 :]])
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(abs(inside.mean() - outside.mean()))


def _find_peaks(profile: np.ndarray, params: EdgeDetectorParams) -> list[int]:
    """Local maxima above ``min_peak_frac`` with non-maximum suppression."""
    threshold = params.min_peak_frac * float(profile.max()) if profile.size else 0.0
    if threshold <= 0.0:
        return []
    candidates = [
        i
        for i in range(1, len(profile) - 1)
        if profile[i] >= threshold
        and profile[i] > profile[i - 1]
        and profile[i] >= profile[i + 1]
    ]
    candidates.sort(key=lambda i: -profile[i])
    kept: list[int] = []
    for i in candidates:
        if all(abs(i - k) >= params.peak_min_separation_px for k in kept):
            kept.append(i)
    return sorted(kept)


def _refine_peak(profile: np.ndarray, index: int) -> float:
    """Parabolic sub-pixel refinement of a peak location."""
    if index <= 0 or index >= len(profile) - 1:
        return float(index)
    left, center, right = profile[index - 1], profile[index], profile[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)
