"""Tests for the cylinder edge detector on synthetic wrist frames."""

import numpy as np
import pytest

from prpl_tidybot.visual_servo.cylinder_edges import (
    EdgeDetectorParams,
    _refine_peak,
    _snap_to_peak,
    column_edge_profile,
    detect_cylinder_edges,
    render_edge_overlay,
)


def _synthetic_frame(
    left: int,
    right: int,
    *,
    width: int = 320,
    height: int = 240,
    fingers: bool = True,
    noise: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """A wood-ish background with a red cylinder spanning columns [left, right) and,
    optionally, two bright gripper fingers along the bottom edge."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.float64)
    # Background: warm tone with faint vertical grain so the profile is not flat.
    image[..., 0] = 190 + 10 * np.sin(np.arange(width) / 7.0)[None, :]
    image[..., 1] = 150 + 8 * np.sin(np.arange(width) / 11.0)[None, :]
    image[..., 2] = 90
    image[:, left:right] = (200, 40, 40)
    if fingers:
        image[int(0.85 * height) :, : int(0.12 * width)] = (230, 230, 230)
        image[int(0.85 * height) :, int(0.88 * width) :] = (230, 230, 230)
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_detects_centered_cylinder():
    """Both edges are found within a pixel or two and the error is ~0."""
    frame = _synthetic_frame(120, 200)
    edges = detect_cylinder_edges(frame)
    assert edges is not None
    assert edges.left_x == pytest.approx(120, abs=2.0)
    assert edges.right_x == pytest.approx(200, abs=2.0)
    assert edges.width_px == pytest.approx(80, abs=3.0)
    assert abs(edges.lateral_error_px) < 2.0


@pytest.mark.parametrize("left,right", [(40, 120), (190, 270), (100, 250)])
def test_detects_offcenter_cylinders_with_correct_sign(left: int, right: int):
    """The lateral error is positive for a cylinder right of center, negative left."""
    frame = _synthetic_frame(left, right)
    edges = detect_cylinder_edges(frame)
    assert edges is not None
    expected = 0.5 * (left + right) - 0.5 * (frame.shape[1] - 1)
    assert edges.lateral_error_px == pytest.approx(expected, abs=2.0)


def test_gripper_fingers_are_ignored():
    """The bright finger blocks at the bottom of the frame have strong vertical edges but
    lie outside the ROI band, so they never win over the cylinder."""
    frame = _synthetic_frame(150, 210, fingers=True)
    edges = detect_cylinder_edges(frame)
    assert edges is not None
    assert edges.left_x == pytest.approx(150, abs=2.0)
    assert edges.right_x == pytest.approx(210, abs=2.0)


def test_returns_none_without_a_cylinder():
    """A frame with only background grain has no pair of strong vertical edges."""
    frame = _synthetic_frame(0, 0, fingers=False)
    assert detect_cylinder_edges(frame) is None


def test_width_bounds_reject_implausible_pairs():
    """A pair of edges narrower than min_width_frac is not accepted."""
    frame = _synthetic_frame(150, 158, fingers=False)
    params = EdgeDetectorParams(min_width_frac=0.1)
    assert detect_cylinder_edges(frame, params) is None


def test_profile_is_normalised_and_peaks_at_edges():
    """The column profile has its maxima at the cylinder edges and max value 1."""
    frame = _synthetic_frame(100, 180, fingers=False, noise=0.0)
    profile = column_edge_profile(frame)
    assert profile.shape == (frame.shape[1],)
    assert profile.max() == pytest.approx(1.0)
    assert profile[100] > 0.8 and profile[180] > 0.8
    assert profile[[20, 60, 140, 220, 300]].max() < 0.3


def test_overlay_keeps_shape_and_handles_no_detection():
    """The overlay is an RGB image of the input shape, with or without edges."""
    frame = _synthetic_frame(120, 200)
    edges = detect_cylinder_edges(frame)
    overlay = render_edge_overlay(frame, edges, label="t0")
    assert overlay.shape == frame.shape and overlay.dtype == np.uint8
    assert not np.array_equal(overlay, frame)
    overlay_none = render_edge_overlay(frame, None)
    assert overlay_none.shape == frame.shape


def test_grayscale_input_is_accepted():
    """A single-channel frame works too."""
    frame = _synthetic_frame(120, 200, fingers=False)
    gray = frame.mean(axis=2).astype(np.uint8)
    edges = detect_cylinder_edges(gray)
    assert edges is not None
    assert edges.center_x == pytest.approx(160, abs=2.0)


@pytest.mark.parametrize(
    "left, right, clipped_left, clipped_right",
    [(0, 70, True, False), (250, 320, False, True)],
)
def test_cylinder_cut_off_by_the_border_is_reported_as_clipped(
    left, right, clipped_left, clipped_right
):
    """A cylinder run touching the image border is detected with the clipped flag on
    that side and the border column as that edge; the cylinder covering one
    background margin must not poison the background model."""
    edges = detect_cylinder_edges(_synthetic_frame(left, right))
    assert edges is not None
    assert (edges.clipped_left, edges.clipped_right) == (clipped_left, clipped_right)
    assert edges.clipped
    if clipped_left:
        assert edges.left_x == 0.0
        assert abs(edges.right_x - right) <= 3
        assert edges.lateral_error_px < 0
    else:
        assert edges.right_x == 319.0
        assert abs(edges.left_x - left) <= 3
        assert edges.lateral_error_px > 0


def test_fully_visible_cylinder_is_not_clipped():
    """The clipped flags are off for a cylinder with both edges inside the frame."""
    edges = detect_cylinder_edges(_synthetic_frame(120, 200))
    assert edges is not None
    assert not edges.clipped


def test_overlay_labels_a_clipped_detection():
    """The overlay text names the clipped side."""
    image = _synthetic_frame(0, 70)
    edges = detect_cylinder_edges(image)
    overlay = render_edge_overlay(image, edges)
    assert overlay.shape == image.shape


def _white_can_frame(left: int, right: int, label_from: int, **kwargs) -> np.ndarray:
    """A pale, unsaturated but textured can body from `left` to `label_from` (a white
    cap with printed rings against a light floor) and a saturated blue label from
    `label_from` to `right`, on the usual background."""
    image = _synthetic_frame(label_from, right, **kwargs).astype(np.float64)
    height = image.shape[0]
    rows = np.arange(height)[:, None]
    body = np.where((rows // 6) % 2 == 0, 205.0, 165.0)
    image[:, left:label_from] = np.stack([body, body, body], axis=-1)[
        :, : label_from - left
    ]
    image[:, label_from:right] = (40, 60, 200)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_white_can_reaching_into_the_margin_is_detected_whole():
    """A white can whose body has no colour or brightness contrast against the
    floor, staged so that it reaches into the left background margin, is still
    found from its silhouette edge: the background's texture is estimated from
    the smoothest columns, not from the margin the can occupies."""
    edges = detect_cylinder_edges(_white_can_frame(20, 210, 150))
    assert edges is not None
    assert abs(edges.left_x - 20) <= 4, edges.left_x
    assert abs(edges.right_x - 210) <= 4, edges.right_x


def test_snap_keeps_the_boundary_when_the_gradient_only_rises_toward_a_far_peak():
    """A window maximum on the window's rim is a slope toward a peak outside the
    window, not an edge here; the segmentation boundary is kept."""
    params = EdgeDetectorParams(refine_px=5, min_boundary_response=0.1)
    profile = np.zeros(100)
    profile[40:61] = np.linspace(0.0, 1.0, 21)  # rising slope; peak at 60
    assert _snap_to_peak(profile, 50, params) == 50
    assert _snap_to_peak(profile, 58, params) == 60


def test_refine_peak_moves_at_most_half_a_pixel():
    """Sub-pixel refinement on a slope (not a peak) stays put instead of jumping
    tens of pixels along the fitted parabola."""
    profile = np.linspace(0.0, 1.0, 50)
    assert abs(_refine_peak(profile, 25) - 25) <= 0.5
    peak = np.array([0.0, 0.5, 1.0, 0.6, 0.0])
    assert 1.5 < _refine_peak(peak, 2) < 2.5
