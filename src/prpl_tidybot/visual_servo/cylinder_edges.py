"""Find the two silhouette edges of a cylinder in a wrist-camera image.

The cylinder is assumed to stand roughly upright in front of the gripper
with both of its silhouette edges in view, which is what the pre-grasp
pose of the cylinder-shelf skills gives.

The detector segments first and refines with edges second, because the
strongest vertical edges are not always the silhouette: a printed label
(a white nutrition panel next to green print) makes an inner edge that
outscores the cylinder's own outline against a light floor. Within a
horizontal band of the image (``roi_top`` to ``roi_bottom``, chosen to
exclude the gripper fingers at the bottom of the frame) each column is
summarised by its mean gray level and its vertical texture (standard
deviation). The outermost columns of the band model the background; a
column belongs to the object when its mean differs from the background
or it is textured where the background is smooth. The widest contiguous
run of object columns of plausible width is the cylinder, and each of its
two boundaries is then snapped to the strongest horizontal-gradient peak
nearby (sub-pixel), so the result is as precise as an edge detector and
as robust as a segmentation.

Everything is a pure function of the image and :class:`EdgeDetectorParams`,
so it can be run and tuned offline on saved frames
(``scripts/detect_cylinder_edges.py``) and unit-tested on synthetic images.
:func:`render_edge_overlay` draws the band, the gradient profile, the object
columns, the detected edges, and the lateral error on a copy of the image.
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
    the rows that feed the column statistics: low enough to leave out the
    wall and furniture the wrist camera sees above the floor at the
    pre-grasp pose of a tall cylinder (which drown every column in texture
    and can turn a dark strip at the image's edge into the only "object"),
    high enough to leave out the gripper fingers at the bottom corners.
    Widths are fractions of the
    image width. ``background_margin_frac`` is the fraction of columns at
    each side of the band that model the background. A column is object
    when its mean saturation differs from the background's by more than
    ``min_saturation_contrast`` (0-255), or its brightness differs from a
    per-row background interpolated between the margins by more than
    ``min_brightness_contrast`` in most rows, or its vertical standard
    deviation exceeds ``texture_factor`` times the background's (and
    ``min_texture``); the background's texture is the smaller of the margins'
    and the ``background_texture_percentile`` of all columns, so a cylinder
    reaching into a margin, or a grout line crossing it, cannot inflate the
    threshold past the cylinder's own texture. Gaps of
    up to ``gap_px`` object-free columns inside a run are closed. Each run
    boundary is snapped to the strongest gradient peak within
    ``refine_px`` when that peak carries at least ``min_boundary_response`` of
    the strongest gradient in the band (a boundary with no such peak, like a
    white label against a light floor, keeps the segmentation's column); of
    the plausible runs the one nearest the image center wins.
    ``min_edge_strength`` is the smallest mean |Sobel-x|
    response (gray levels per pixel; the simulator's soft-shaded cylinder
    measures ~3, a bare floor ~1) the strongest column must reach for the
    frame to count as containing an edge at all.
    """

    roi_top: float = 0.40
    roi_bottom: float = 0.80
    blur_ksize: int = 5
    profile_smooth_px: int = 9
    min_width_frac: float = 0.06
    max_width_frac: float = 0.7
    background_margin_frac: float = 0.1
    min_saturation_contrast: float = 25.0
    min_brightness_contrast: float = 35.0
    texture_factor: float = 2.5
    min_texture: float = 6.0
    background_texture_percentile: float = 25.0
    gap_px: int = 12
    refine_px: int = 14
    min_boundary_response: float = 0.25
    min_edge_strength: float = 1.5


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
    # The object run reaches the image border on that side: the cylinder is
    # only partly in view, so ``center_x`` and ``width_px`` are lower bounds
    # and the only reliable information is which way it lies.
    clipped_left: bool = False
    clipped_right: bool = False

    @property
    def clipped(self) -> bool:
        """True when the cylinder is cut off by the left or right image border."""
        return self.clipped_left or self.clipped_right

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

    @property
    def servo_error_px(self) -> float:
        """The lateral error to servo on: ``lateral_error_px`` for a fully visible
        cylinder; for one cut off by the image border, whose centre cannot be
        measured, a full half-frame toward the clipped side, which is never within
        tolerance and steps the tool that way at the maximum rate."""
        if self.clipped_left:
            return -0.5 * self.image_width
        if self.clipped_right:
            return 0.5 * self.image_width
        return self.lateral_error_px


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


def _roi_band_rgb(image: Image, params: EdgeDetectorParams) -> np.ndarray:
    rgb = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    height = rgb.shape[0]
    top = int(round(params.roi_top * height))
    bottom = max(top + 1, int(round(params.roi_bottom * height)))
    return np.ascontiguousarray(rgb[top:bottom])


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


def object_columns(
    image: Image, params: EdgeDetectorParams = EdgeDetectorParams()
) -> np.ndarray:
    """Boolean mask over columns: True where the band looks like object rather than
    background (see the module docstring).

    Three cues, any of which marks a column as object, all measured against a
    background model taken from the band's outermost columns:

    * saturation differs from the background's (a coloured object on a grey
      floor, or a grey label on a wooden one) — immune to the fisheye's
      vignetting and to shadows, which change brightness but not colour;
    * texture: the column's vertical standard deviation is well above the
      background's (printed labels on a smooth floor);
    * brightness differs by a large margin from a background interpolated
      per row between the left and right margins, which absorbs the gradual
      edge-to-center brightening of a fisheye.
    """
    band_rgb = _roi_band_rgb(image, params)
    if params.blur_ksize > 1:
        band_rgb = cv.GaussianBlur(band_rgb, (params.blur_ksize, params.blur_ksize), 0)
    hsv = cv.cvtColor(band_rgb, cv.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1].mean(axis=0)
    value = hsv[..., 2]
    width = value.shape[1]
    margin = max(1, int(round(params.background_margin_frac * width)))
    left_cols = np.arange(margin)
    right_cols = np.arange(width - margin, width)

    stds = value.std(axis=0)
    # When the cylinder is only partly in view it occupies one margin, and
    # that margin must not feed the background model. A margin the cylinder
    # covers is far more textured than the floor (a printed label) or
    # differs from the other margin in saturation, so when the two margins
    # disagree, the smoother one alone models the background; otherwise
    # both do, with the brightness interpolated between them.
    left_texture = float(np.median(stds[left_cols]))
    right_texture = float(np.median(stds[right_cols]))
    saturation_gap = abs(
        float(np.median(saturation[left_cols]))
        - float(np.median(saturation[right_cols]))
    )
    textures_disagree = max(left_texture, right_texture) > max(
        params.min_texture, params.texture_factor * min(left_texture, right_texture)
    )
    if textures_disagree or saturation_gap > params.min_saturation_contrast:
        background_cols = left_cols if left_texture <= right_texture else right_cols
        if not textures_disagree:
            background_cols = (
                left_cols
                if float(np.median(saturation[left_cols]))
                <= float(np.median(saturation[right_cols]))
                else right_cols
            )
        background_saturation = float(np.median(saturation[background_cols]))
        background_std = float(np.median(stds[background_cols]))
        row_background = np.median(value[:, background_cols], axis=1, keepdims=True)
        row_background = np.repeat(row_background, width, axis=1)
    else:
        background_saturation = float(
            np.median(np.concatenate([saturation[left_cols], saturation[right_cols]]))
        )
        background_std = float(
            np.median(np.concatenate([stds[left_cols], stds[right_cols]]))
        )
        left_level = np.median(value[:, left_cols], axis=1, keepdims=True)
        right_level = np.median(value[:, right_cols], axis=1, keepdims=True)
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        row_background = left_level + (right_level - left_level) * ramp
    by_saturation = (
        np.abs(saturation - background_saturation) > params.min_saturation_contrast
    )
    # The floor's texture is what the smoothest columns show; a cylinder
    # reaching into a margin (a white can staged at the frame's edge, whose
    # white body has no colour or brightness contrast against a light floor)
    # or a grout line crossing it would otherwise raise the threshold above
    # the cylinder's own texture and hide it.
    background_std = min(
        background_std,
        float(np.percentile(stds, params.background_texture_percentile)),
    )
    texture_threshold = max(params.min_texture, params.texture_factor * background_std)
    by_texture = stds > texture_threshold
    differs = np.abs(value - row_background) > params.min_brightness_contrast
    by_brightness = differs.mean(axis=0) > 0.5

    return _close_gaps(by_saturation | by_texture | by_brightness, params.gap_px)


def detect_cylinder_edges(
    image: Image, params: EdgeDetectorParams = EdgeDetectorParams()
) -> CylinderEdges | None:
    """Locate the cylinder's left and right silhouette edges, or return None if no
    plausible object run is found."""
    raw_profile, peak = _raw_column_edge_profile(image, params)
    if peak < params.min_edge_strength:
        return None
    profile = np.asarray(raw_profile, dtype=np.float64) / peak
    width = image.shape[1]
    min_width = params.min_width_frac * width
    max_width = params.max_width_frac * width
    center = 0.5 * (width - 1)
    candidates: list[tuple[float, int, int, bool, bool]] = []
    for start, stop in _runs(object_columns(image, params)):
        # A run touching a border is a cylinder cut off by the frame: its
        # width is a lower bound, so only the minimum applies, and the
        # clipped side keeps the border column instead of a gradient peak.
        clipped_left = start == 0
        clipped_right = stop == width
        if stop - start < min_width:
            continue
        if not (clipped_left or clipped_right) and stop - start > max_width:
            continue
        left = 0 if clipped_left else _snap_to_peak(profile, start, params)
        right = width - 1 if clipped_right else _snap_to_peak(profile, stop - 1, params)
        if right - left < min_width:
            continue
        candidates.append(
            (
                abs(0.5 * (left + right) - center),
                left,
                right,
                clipped_left,
                clipped_right,
            )
        )
    if not candidates:
        return None
    _, left, right, clipped_left, clipped_right = min(candidates)
    band = _roi_band(image, params).astype(np.float32)
    column_means = band.mean(axis=0)
    return CylinderEdges(
        left_x=float(left) if clipped_left else _refine_peak(profile, left),
        right_x=float(right) if clipped_right else _refine_peak(profile, right),
        image_width=width,
        image_height=image.shape[0],
        left_response=0.0 if clipped_left else float(profile[left]),
        right_response=0.0 if clipped_right else float(profile[right]),
        contrast=_silhouette_contrast(column_means, left, right),
        clipped_left=clipped_left,
        clipped_right=clipped_right,
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
    # Object columns (the segmentation) as a thin magenta strip under the band.
    mask = object_columns(image, params)
    strip_top = min(height - 1, bottom + 2)
    strip_bottom = min(height - 1, bottom + 6)
    for x in np.where(mask)[0]:
        out[strip_top:strip_bottom, x] = (255, 0, 255)
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
        if edges.clipped:
            side = "left" if edges.clipped_left else "right"
            text = f"clipped {side}  {text}"
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


def _close_gaps(mask: np.ndarray, gap_px: int) -> np.ndarray:
    """Fill runs of False shorter than or equal to `gap_px` that sit between Trues."""
    closed = mask.copy()
    if gap_px <= 0:
        return closed
    true_idx = np.where(mask)[0]
    for a, b in zip(true_idx[:-1], true_idx[1:]):
        if 1 < b - a <= gap_px + 1:
            closed[a:b] = True
    return closed


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of the contiguous True runs in `mask`."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _snap_to_peak(profile: np.ndarray, column: int, params: EdgeDetectorParams) -> int:
    """Column of the strongest gradient response within ``refine_px`` of `column`, if
    it carries at least ``min_boundary_response``; otherwise the segmentation boundary
    itself (a white label against a light floor has almost no gradient, but the
    saturation and texture cues still place the boundary within a few pixels)."""
    lo = max(0, column - params.refine_px)
    hi = min(len(profile), column + params.refine_px + 1)
    best = int(lo + np.argmax(profile[lo:hi]))
    if profile[best] < params.min_boundary_response:
        return int(np.clip(column, 0, len(profile) - 1))
    # A maximum on the window's rim is a slope towards a peak outside the
    # window, not an edge here: keep the segmentation boundary.
    on_rim = (best == lo and lo > 0) or (best == hi - 1 and hi < len(profile))
    if on_rim:
        return int(np.clip(column, 0, len(profile) - 1))
    return best


def _refine_peak(profile: np.ndarray, index: int) -> float:
    """Parabolic sub-pixel refinement of a peak location."""
    if index <= 0 or index >= len(profile) - 1:
        return float(index)
    left, center, right = profile[index - 1], profile[index], profile[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return float(index)
    # A true peak's vertex lies within half a pixel of the sample; anything
    # further means the samples are not a peak (a slope), so stay put.
    offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return float(index + offset)
