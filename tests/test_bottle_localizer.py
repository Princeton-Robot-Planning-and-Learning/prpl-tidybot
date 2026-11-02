"""Tests for the ceiling-camera bottle localizer: a bottle lying on the floor, found by
its red label (the clear plastic barely shows from the ceiling) and fitted as a
label-sized cylinder, so its pose is the label's centre and the bottle's axis."""

import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bottle_localizer import (
    BottleModel,
    CeilingBottleSource,
    bottle_mask,
    label_silhouette,
    localize_bottle,
)
from prpl_tidybot.cube_localizer import load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")
H, W = 720, 1280
MODEL = BottleModel()
BOUNDS = (-1.5, 1.7, -1.5, 1.5)


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    """One OpenCV HSV colour as BGR."""
    px = np.array([[hsv]], np.uint8)
    b, g, r = (int(v) for v in cv.cvtColor(px, cv.COLOR_HSV2BGR)[0, 0])
    return b, g, r


# From a real ceiling frame (2026-11-02): floor H ~10 S ~92 V ~177; the vitaminwater
# label H 177-179 and 0-2 (red wraps around 0) S ~245; the blue bin H ~127.
FLOOR_BGR = _hsv_to_bgr((10, 92, 177))
LABEL_BGR = _hsv_to_bgr((178, 245, 200))
LABEL_LOW_HUE_BGR = _hsv_to_bgr((2, 240, 190))
BIN_BGR = _hsv_to_bgr((127, 170, 110))
CARPET_BGR = (38, 38, 40)


def _quad(cam, x0, x1, y0, y1, z=0.0):
    """Image polygon of a floor rectangle."""
    pts = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return cam.project(pts).astype(np.int32)


def _render(cameras, bottles=(), patches=(), seed=0, noise=3.0, colour=LABEL_BGR):
    """A stitched frame: floor, coloured floor patches ((x0, x1, y0, y1), bgr) and
    lying bottles' labels at `bottles` (x, y, yaw)."""
    rng = np.random.default_rng(seed)
    halves = []
    for cam in cameras:
        img = np.empty((H, W, 3), np.float32)
        img[:] = FLOOR_BGR
        for box, bgr in patches:
            cv.fillConvexPoly(img, _quad(cam, *box), bgr, lineType=cv.LINE_AA)
        for x, y, yaw in bottles:
            poly = label_silhouette(cam, x, y, yaw, MODEL).astype(np.int32)
            cv.fillConvexPoly(img, poly, colour, lineType=cv.LINE_AA)
        img += rng.normal(0, noise, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    """The lab's calibrated ceiling cameras."""
    return load_ceiling_cameras(SERIALS)


def _axis_err(a, b):
    """A label looks the same end to end: compare axes modulo 180 degrees."""
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def test_the_model_defaults_to_a_20_oz_bottle():
    """The defaults are the vitaminwater bottle's."""
    assert MODEL.diameter == pytest.approx(0.066)
    assert MODEL.label_length == pytest.approx(0.10)


def test_the_mask_keeps_red_on_both_sides_of_hue_zero_and_nothing_else():
    """Red at hue ~2 and ~178 passes; the floor, the blue bin and a carpet do not."""
    img = np.zeros((40, 200, 3), np.uint8)
    colours = (FLOOR_BGR, BIN_BGR, CARPET_BGR, LABEL_BGR, LABEL_LOW_HUE_BGR)
    for k, bgr in enumerate(colours):
        img[:, 40 * k : 40 * (k + 1)] = bgr
    mask = bottle_mask(img)
    assert mask[:, :120].sum() == 0
    assert mask[5:35, 125:195].all()


@pytest.mark.parametrize(
    "pose", [(0.37, -0.67, 1.18), (-0.8, 0.9, 0.0), (1.2, -1.1, -0.7)]
)
def test_a_lying_bottle_is_localized(cameras, pose):
    """A label rendered at a known pose through the real calibration is recovered."""
    frames = [_render(cameras, bottles=[pose], seed=s) for s in range(3)]
    fit = localize_bottle(frames, cameras, MODEL, BOUNDS, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015
    assert _axis_err(fit.yaw, pose[2]) < math.radians(5)


def test_a_bottle_beside_the_blue_bin_on_the_carpet_is_found(cameras):
    """The blue bin and a dark carpet next to the bottle do not disturb the fit."""
    pose = (0.6, 1.2, 0.4)
    patches = [((-1.2, 1.6, 0.35, 2.0), CARPET_BGR), ((0.1, 0.4, 1.1, 1.4), BIN_BGR)]
    frame = _render(cameras, bottles=[pose], patches=patches)
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015


def test_nothing_red_gives_none(cameras):
    """No label, no bottle."""
    frame = _render(cameras, patches=[((0.1, 0.4, 1.1, 1.4), BIN_BGR)])
    assert localize_bottle([frame], cameras, MODEL, BOUNDS) is None


def test_a_large_red_patch_is_not_a_bottle(cameras):
    """A red region much bigger than a label is rejected."""
    frame = _render(cameras, patches=[((-0.5, 0.1, -0.9, -0.5), LABEL_BGR)])
    assert localize_bottle([frame], cameras, MODEL, BOUNDS) is None


def test_the_robot_footprint_is_excluded(cameras):
    """A label inside the robot's footprint is ignored."""
    frame = _render(cameras, bottles=[(0.5, -0.6, 0.0)])
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, exclude=[(0.5, -0.6, 0.45)])
    assert fit is None


class _Images:
    def __init__(self, frames):
        self._frames = list(frames)

    def get_image(self):
        """The next frame, then the last one forever."""
        return self._frames.pop(0) if len(self._frames) > 1 else self._frames[0]

    def close(self):
        """Nothing to release."""


class _Markers:
    def get_latest(self):
        """The robot, well away from the bottle."""
        return {"poses": {"robot": (-1.0, -1.0, 0.0)}}

    def close(self):
        """Nothing to release."""


def test_ceiling_bottle_source_gives_target_and_yaw(cameras):
    """The source returns the label's centre at axis height and the bottle's axis."""
    pose = (0.37, -0.67, 1.18)
    frames = [_render(cameras, bottles=[pose], seed=s) for s in range(2)]
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images(frames),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
        num_frames=2,
        frame_period_s=0.0,
    )
    x, y, z = source.get_target()
    assert math.hypot(x - pose[0], y - pose[1]) < 0.015
    assert z == pytest.approx(MODEL.diameter / 2)
    assert _axis_err(source.get_yaw(), pose[2]) < math.radians(5)


def test_ceiling_bottle_source_raises_without_a_bottle(cameras):
    """No bottle in view raises rather than inventing a target."""
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images([_render(cameras)]),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
        num_frames=1,
        frame_period_s=0.0,
    )
    with pytest.raises(RuntimeError):
        source.get_target()
