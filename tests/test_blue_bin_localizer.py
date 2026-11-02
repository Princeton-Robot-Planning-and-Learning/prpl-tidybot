"""Tests for the bin localizer on a coloured, tapered bin: the blue deskside recycling
bin found by its hue (not its darkness, which a dark carpet shares) and fitted as a
box whose rim is wider than its base."""

import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bin_localizer import (
    BinModel,
    CeilingBinSource,
    bin_mask,
    bin_silhouette,
    localize_bin,
)
from prpl_tidybot.cube_localizer import load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")
H, W = 720, 1280
# Rubbermaid FG295673BLUE: the rim from the spec sheet, the base narrower.
MODEL = BinModel(
    length=0.365, width=0.26, height=0.384, base_length=0.29, base_width=0.19
)
HUE = (100, 145)
SEARCH = (-1.8, 2.0, -1.8, 1.8)


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    """One OpenCV HSV colour as BGR."""
    px = np.array([[hsv]], np.uint8)
    b, g, r = (int(v) for v in cv.cvtColor(px, cv.COLOR_HSV2BGR)[0, 0])
    return b, g, r


# Sampled from a real ceiling frame (2026-11-02): floor H ~10 S ~92 V ~177; the bin's
# lit walls H ~127 S ~170 V ~110 and its inside V ~55; the bottle label H ~178 S ~245.
# The floor's saturation overlaps the bin's dark side, so saturation alone cannot
# separate them; hue can.
FLOOR_BGR = _hsv_to_bgr((10, 92, 177))
WALL_BGR = _hsv_to_bgr((127, 170, 110))
INSIDE_BGR = _hsv_to_bgr((125, 200, 55))
CARPET_BGR = (38, 38, 40)
LABEL_BGR = _hsv_to_bgr((178, 245, 200))


def _quad(cam, x0, x1, y0, y1):
    """Image polygon of a floor rectangle."""
    pts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float)
    return cam.project(pts).astype(np.int32)


def _render(cameras, bins=(), carpets=(), labels=(), seed=0, noise=3.0):
    """A stitched frame: floor, dark carpets (x0, x1, y0, y1), red bottle labels
    (x0, x1, y0, y1) and blue tapered bins (x, y, yaw)."""
    rng = np.random.default_rng(seed)
    halves = []
    for cam in cameras:
        img = np.empty((H, W, 3), np.float32)
        img[:] = FLOOR_BGR
        for box in carpets:
            cv.fillConvexPoly(img, _quad(cam, *box), CARPET_BGR, lineType=cv.LINE_AA)
        for box in labels:
            cv.fillConvexPoly(img, _quad(cam, *box), LABEL_BGR, lineType=cv.LINE_AA)
        for x, y, yaw in bins:
            outer = bin_silhouette(cam, x, y, yaw, MODEL).astype(np.int32)
            cv.fillConvexPoly(img, outer, WALL_BGR, lineType=cv.LINE_AA)
            rim = BinModel(MODEL.length - 0.02, MODEL.width - 0.02, MODEL.height)
            c, s = math.cos(yaw), math.sin(yaw)
            pts = [
                [x + c * sx * rim.length - s * sy * rim.width,
                 y + s * sx * rim.length + c * sy * rim.width,
                 MODEL.height]
                for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)
            ]  # fmt: skip
            hull = cv.convexHull(cam.project(np.array(pts)).astype(np.float32))
            cv.fillConvexPoly(img, hull.reshape(-1, 2).astype(np.int32), INSIDE_BGR)
        img += rng.normal(0, noise, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    """The lab's calibrated ceiling cameras."""
    return load_ceiling_cameras(SERIALS)


def _yaw_err(a, b):
    """Smallest difference between two bin yaws (a box repeats every 180 degrees)."""
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def test_a_bin_model_defaults_to_straight_walls():
    """Without a base size the base is the rim's size."""
    model = BinModel(length=0.4, width=0.3, height=0.2)
    assert (model.base_length, model.base_width) == (0.4, 0.3)


def test_a_tapered_bin_has_a_smaller_silhouette_than_a_straight_one(cameras):
    """The base, not the rim, sets the silhouette's lower corners."""
    straight = BinModel(length=0.365, width=0.26, height=0.384)
    cam = cameras[0]
    area_tapered = cv.contourArea(
        bin_silhouette(cam, 0.2, 1.6, 0.0, MODEL).astype(np.float32)
    )
    area_straight = cv.contourArea(
        bin_silhouette(cam, 0.2, 1.6, 0.0, straight).astype(np.float32)
    )
    assert area_tapered < area_straight


def test_the_hue_mask_keeps_the_bin_and_drops_floor_carpet_and_label():
    """Only the bin's hue passes: not the warm floor, a dark carpet or a red label."""
    img = np.zeros((40, 160, 3), np.uint8)
    for k, bgr in enumerate((FLOOR_BGR, CARPET_BGR, LABEL_BGR, WALL_BGR)):
        img[:, 40 * k : 40 * (k + 1)] = bgr
    mask = bin_mask(img, hue_range=HUE, min_saturation=50.0)
    assert mask[:, :120].sum() == 0
    assert mask[5:35, 125:155].all()


@pytest.mark.parametrize(
    "pose", [(0.2, 1.65, 0.0), (-0.6, 1.4, 0.5), (0.9, -0.9, -1.0)]
)
def test_a_blue_bin_on_a_dark_carpet_is_localized(cameras, pose):
    """A tapered blue bin standing on a dark carpet is recovered at its pose."""
    carpet = (-1.2, 1.6, 0.35, 2.0) if pose[1] > 0 else (0.3, 1.6, -1.6, -0.3)
    frames = [
        _render(cameras, bins=[pose], carpets=[carpet], labels=[(0.4, 0.5, 0.9, 0.97)],
                seed=s)
        for s in range(3)
    ]  # fmt: skip
    fit = localize_bin(
        frames,
        cameras,
        MODEL,
        search_bounds=SEARCH,
        serials=SERIALS,
        hue_range=HUE,
        min_saturation=50.0,
    )
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.02
    assert _yaw_err(fit.yaw, pose[2]) < math.radians(3)
    assert (fit.base_length, fit.base_width) == (MODEL.base_length, MODEL.base_width)


def test_a_carpet_and_a_red_label_alone_are_not_a_bin(cameras):
    """Dark and red regions are not a bin under the hue mask."""
    frame = _render(
        cameras, carpets=[(-1.2, 1.6, 0.35, 2.0)], labels=[(0.4, 0.5, -0.7, -0.63)]
    )
    assert (
        localize_bin(
            [frame], cameras, MODEL, search_bounds=SEARCH, hue_range=HUE,
            min_saturation=50.0,
        )
        is None
    )  # fmt: skip


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
        """The robot, well away from the bin."""
        return {"poses": {"robot": (-1.0, -1.0, 0.0)}}

    def close(self):
        """Nothing to release."""


def test_ceiling_bin_source_takes_the_hue_and_taper(cameras):
    """The source threads the hue range and the base size through to the fit."""
    pose = (0.2, 1.65, 0.3)
    frames = [
        _render(cameras, bins=[pose], carpets=[(-1.2, 1.6, 0.35, 2.0)], seed=s)
        for s in range(2)
    ]
    source = CeilingBinSource(
        SERIALS,
        image_client=_Images(frames),
        marker_client=_Markers(),
        length=MODEL.length,
        width=MODEL.width,
        height=MODEL.height,
        base_length=MODEL.base_length,
        base_width=MODEL.base_width,
        num_frames=2,
        frame_period_s=0.0,
        hue_range=HUE,
        min_saturation=50.0,
    )
    fit = source.localize()
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.02
