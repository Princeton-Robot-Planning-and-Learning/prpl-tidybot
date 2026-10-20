"""Localize the open cardboard bin on the floor from the ceiling cameras.

The bin is an open box of known size that may stand anywhere on the floor. Seen from
the ceiling, its outer walls and shadowed inside are much darker than the floor
around it, but they share the floor's warm hue (in a real frame the floor read
Lab a*/b* ~139/147 and the cardboard ~140/140-160), so colour alone does not separate
them; brightness does, *relative to the local floor*, which itself ranges from L ~205
in the middle of the room to ~115 in the dim corner by the wall. The mask is therefore
"darker than ``darkness`` times the local floor", the floor being a large median of
the image (much larger than the bin, so the bin cannot drag it down).

Candidates are found where a patch inside the bin's footprint is dark while a ring
around it is mostly not (an isolated dark object of the bin's size: furniture and
large shadows fail the ring, small dark things the patch). Each candidate is then
fitted as the cube is in :mod:`prpl_tidybot.cube_localizer`: the bin rests flat on
the floor, so its pose has three degrees of freedom (x, y, yaw), and its silhouette in
a calibrated camera is the convex hull of its eight outer corners - the rim and the
outer walls; whatever is inside the open top lies within that hull. The IoU between
that silhouette and the mask (averaged over frames, stitch-line rows excluded) is
maximised per camera, and the best single view is kept.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_localizer import (
    CeilingCamera,
    in_calibrated_region,
    load_ceiling_cameras,
)
from prpl_tidybot.real_sim.perceivers.target_source import TargetSource

_logger = logging.getLogger(__name__)

# Unit box corners, centred in x/y, bottom face on z = 0.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)]
)
_SEAM_PX = 8
# The local floor: the median of the L channel over this many full-res pixels
# (computed on an 8x-downsampled image).
_FLOOR_DOWNSAMPLE = 8
_FLOOR_KERNEL = 41
# Candidate grid spacing on the floor (m), and how many candidates to fit.
_GRID_STEP = 0.04
_MAX_CANDIDATES = 10
# A candidate's inner patch must be this dark, and its surrounding ring at most this
# (furniture beside the bin can fill a side of the ring; a large dark region fills
# all of it).
_PATCH_MIN_FILL = 0.8
_RING_MAX_FILL = 0.7
# The fit scores the mask within the silhouette grown by this band (m) on each side,
# so a bin-sized region must also be clear of dark pixels just around it, and dark
# things further away do not pull the fit.
_BAND_M = 0.05
# The most of the dark region under a fit that may lie outside its grown silhouette.
_SPILL_MAX = 0.25
# Pixels of floor kept around the silhouette when fitting.
_WINDOW_MARGIN_PX = 20


@dataclass(frozen=True)
class BinModel:
    """The bin's outer dimensions (m): ``length`` along its own x axis (its yaw),
    ``width`` across, ``height`` to the rim; ``wall`` is the cardboard's thickness."""

    length: float = 0.40
    width: float = 0.32
    height: float = 0.19
    wall: float = 0.005


@dataclass(frozen=True)
class BinFit:  # pylint: disable=too-many-instance-attributes
    """A fitted bin pose (map frame), the model it was fitted with and how well its
    silhouette matched."""

    x: float
    y: float
    # Radians in [-pi/2, pi/2): a box looks the same every 180 degrees.
    yaw: float
    length: float
    width: float
    height: float
    wall: float = 0.005
    iou: float = 1.0
    num_views: int = 0
    # Per-camera fits (camera index, x, y, yaw, iou) and their spread.
    per_camera: tuple[tuple[int, float, float, float, float], ...] = ()
    camera_disagreement_m: float = 0.0
    # The camera whose fit was kept (-1 when none was chosen).
    camera: int = -1

    @property
    def model(self) -> BinModel:
        """The box this fit describes."""
        return BinModel(self.length, self.width, self.height, self.wall)

    def as_dict(self) -> dict[str, float]:
        """The bin as plain floats: the skill server's ``observe()["bin"]``."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "yaw": float(self.yaw),
            "length": float(self.length),
            "width": float(self.width),
            "height": float(self.height),
        }


def cube_in_bin(
    cube_xy: Sequence[float], bin_fit: BinFit, margin: float = 0.05
) -> bool:
    """Whether the floor point ``cube_xy`` (the cube's centre) lies inside the bin's
    inner footprint - inside the walls - shrunk by ``margin`` on every side: a
    scoring region strictly inside the bin (Tossing3D scores a box inside its bin
    too), not the whole footprint."""
    dx, dy = cube_xy[0] - bin_fit.x, cube_xy[1] - bin_fit.y
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    half_l = 0.5 * bin_fit.length - bin_fit.wall - margin
    half_w = 0.5 * bin_fit.width - bin_fit.wall - margin
    return abs(lx) <= half_l and abs(ly) <= half_w


def bin_silhouette(
    camera: CeilingCamera, x: float, y: float, yaw: float, model: BinModel
) -> NDArray[np.float64]:
    """The image polygon (convex hull) of the bin standing on the floor at pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = _CORNERS * [model.length, model.width, model.height]
    world = np.c_[
        x + c * local[:, 0] - s * local[:, 1],
        y + s * local[:, 0] + c * local[:, 1],
        local[:, 2],
    ]
    hull = cv.convexHull(camera.project(world).astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float64)


def bin_mask(
    bgr: NDArray[np.uint8], darkness: float = 0.85, min_saturation: float | None = None
) -> NDArray[np.uint8]:
    """1 where a pixel is darker than ``darkness`` times the local floor, holes (a cube
    in the bin, a white label on it) filled.

    With ``min_saturation`` it is instead where the HSV saturation exceeds it: the
    cardboard, and not the bin's shadow on the floor, which darkness also counts (on
    2026-10-20 the floor read ~80, the shadow ~80-100, the cardboard ~160-200, and the
    darkness fit sat 2 cm off the marker in the bin's centre).

    Pass one camera's image: the local floor is estimated within it.
    """
    if min_saturation is not None:
        saturation: Any = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)[:, :, 1]
        card: Any = (saturation.astype(np.float32) > min_saturation).astype(np.uint8)
        card = cv.morphologyEx(card, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        card = cv.morphologyEx(card, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        outlines, _ = cv.findContours(card, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(card)
        cv.drawContours(solid, outlines, -1, (1,), thickness=cv.FILLED)
        return solid.astype(np.uint8)
    lightness: Any = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)[:, :, 0]
    h, w = lightness.shape
    small = cv.resize(
        lightness,
        (max(1, w // _FLOOR_DOWNSAMPLE), max(1, h // _FLOOR_DOWNSAMPLE)),
        interpolation=cv.INTER_AREA,
    )
    floor: Any = cv.resize(
        cv.medianBlur(small, _FLOOR_KERNEL), (w, h), interpolation=cv.INTER_LINEAR
    )
    mask: Any = (lightness.astype(np.float32) < darkness * floor).astype(np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, (1,), thickness=cv.FILLED)
    return filled.astype(np.uint8)


def _split(frame: NDArray[np.uint8], num_cameras: int) -> list[NDArray[np.uint8]]:
    rows = frame.shape[0] // num_cameras
    return [frame[i * rows : (i + 1) * rows] for i in range(num_cameras)]


def _valid_rows(rows: int, cols: int, index: int, num: int) -> NDArray[np.float32]:
    """1 except at the stitch line(s) of camera ``index``, which are unknown."""
    valid = np.ones((rows, cols), np.float32)
    if index > 0:
        valid[:_SEAM_PX] = 0
    if index < num - 1:
        valid[rows - _SEAM_PX :] = 0
    return valid


def _excluded(x: float, y: float, exclude: Sequence[Sequence[float]]) -> bool:
    return any(abs(x - ex) < half and abs(y - ey) < half for ex, ey, half in exclude)


def _box_fill(
    integral: Any,
    valid_integral: Any,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    half: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mask fill and visible fraction of the axis-aligned squares (u, v, half)."""
    rows, cols = integral.shape[0] - 1, integral.shape[1] - 1
    u0 = np.clip(np.round(u - half), 0, cols).astype(int)
    u1 = np.clip(np.round(u + half), 0, cols).astype(int)
    v0 = np.clip(np.round(v - half), 0, rows).astype(int)
    v1 = np.clip(np.round(v + half), 0, rows).astype(int)

    def total(img: Any) -> NDArray[np.float64]:
        return img[v1, u1] - img[v0, u1] - img[v1, u0] + img[v0, u0]

    area = (2 * half) ** 2
    seen = total(valid_integral)
    fill = np.where(seen > 0, total(integral) / np.maximum(seen, 1), 0.0)
    return fill, seen / area


def find_bin_candidates(
    frame: NDArray[np.uint8],
    cameras: Sequence[CeilingCamera],
    search_bounds: Sequence[float],
    model: BinModel,
    exclude: Sequence[Sequence[float]] = (),
    darkness: float = 0.85,
    min_saturation: float | None = None,
) -> list[tuple[float, float]]:
    """Map-frame (x, y) of each isolated, bin-sized dark region in ``search_bounds``,
    best first: a patch well inside the footprint (whatever the yaw) is dark, and a
    ring just outside the footprint (whatever the yaw) mostly is not."""
    xs = np.arange(search_bounds[0], search_bounds[1] + 1e-9, _GRID_STEP)
    ys = np.arange(search_bounds[2], search_bounds[3] + 1e-9, _GRID_STEP)
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    patch_m = 0.3 * min(model.length, model.width)
    ring_in_m = 0.5 * math.hypot(model.length, model.width) + 0.12
    ring_out_m = ring_in_m + 0.12
    best = np.zeros(gx.shape)
    for i, (cam, img) in enumerate(zip(cameras, _split(frame, len(cameras)))):
        rows, cols = img.shape[:2]
        mask = bin_mask(img, darkness, min_saturation).astype(np.float64)
        valid = _valid_rows(rows, cols, i, len(cameras)).astype(np.float64)
        mask *= valid
        integral = cv.integral(mask)
        valid_integral = cv.integral(valid)
        centre = cam.project(np.c_[gx, gy, np.full(gx.shape, model.height / 2)])
        aside = cam.project(np.c_[gx + 0.1, gy, np.full(gx.shape, model.height / 2)])
        px_per_m = np.linalg.norm(aside - centre, axis=1) / 0.1
        u, v = centre[:, 0], centre[:, 1]
        patch, seen = _box_fill(integral, valid_integral, u, v, patch_m * px_per_m)
        inner, _ = _box_fill(integral, valid_integral, u, v, ring_in_m * px_per_m)
        outer, ring_seen = _box_fill(
            integral, valid_integral, u, v, ring_out_m * px_per_m
        )
        area_in, area_out = (2 * ring_in_m) ** 2, (2 * ring_out_m) ** 2
        ring = (outer * area_out - inner * area_in) / (area_out - area_in)
        ok = (seen > 0.5) & (ring_seen > 0.5) & (patch >= _PATCH_MIN_FILL)
        ok &= ring <= _RING_MAX_FILL
        best = np.maximum(best, np.where(ok, patch - ring, 0.0))
    candidates: list[tuple[float, float]] = []
    for k in np.argsort(-best):
        if best[k] <= 0 or len(candidates) >= _MAX_CANDIDATES:
            break
        x, y = float(gx[k]), float(gy[k])
        if _excluded(x, y, exclude):
            continue
        if all(math.hypot(x - cx, y - cy) > 0.25 for cx, cy in candidates):
            candidates.append((x, y))
    return candidates


def fit_bin_pose(  # pylint: disable=too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    guess_xy: tuple[float, float],
    model: BinModel,
    only: Sequence[int] | None = None,
    darkness: float = 0.85,
    min_saturation: float | None = None,
) -> BinFit | None:
    """3-DoF silhouette fit around ``guess_xy``; masks are averaged over ``frames``
    first.

    ``only`` restricts the fit to those camera indices.
    """
    views = []
    reach = 0.5 * math.hypot(model.length, model.width) + 0.08
    for i, cam in enumerate(cameras):
        if only is not None and i not in only:
            continue
        imgs = [_split(f, len(cameras))[i] for f in frames]
        rows, cols = imgs[0].shape[:2]
        ring = [
            [guess_xy[0] + sx * reach, guess_xy[1] + sy * reach, z]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for z in (0.0, model.height)
        ]
        px = cam.project(np.array(ring))
        u0 = int(max(px[:, 0].min() - _WINDOW_MARGIN_PX, 0))
        v0 = int(max(px[:, 1].min() - _WINDOW_MARGIN_PX, 0))
        u1 = int(min(px[:, 0].max() + _WINDOW_MARGIN_PX, cols))
        v1 = int(min(px[:, 1].max() + _WINDOW_MARGIN_PX, rows))
        if u1 - u0 < 10 or v1 - v0 < 10:
            continue
        prob = np.mean(
            [bin_mask(img, darkness, min_saturation)[v0:v1, u0:u1] for img in imgs],
            axis=0,
        )
        valid = _valid_rows(rows, cols, i, len(cameras))[v0:v1, u0:u1]
        prob = prob * valid
        if prob.sum() > 0:
            views.append((cam, u0, v0, prob, valid))
    if not views:
        return None

    grown = BinModel(
        model.length + 2 * _BAND_M, model.width + 2 * _BAND_M, model.height
    )

    def score(x: float, y: float, yaw: float) -> float:
        """IoU of the silhouette and the mask within the silhouette grown by a band:
        dark things further away (furniture, the wall) neither count nor pull."""
        total = 0.0
        for cam, u0, v0, prob, valid in views:
            sil = np.zeros(prob.shape, np.float32)
            near = np.zeros(prob.shape, np.float32)
            for img, box in ((sil, model), (near, grown)):
                poly = ((bin_silhouette(cam, x, y, yaw, box) - [u0, v0]) * 16).astype(
                    np.int32
                )
                cv.fillConvexPoly(img, poly, (1.0,), lineType=cv.LINE_AA, shift=4)
            sil = sil * valid
            inter = float((sil * prob).sum())
            seen = float((near * valid * prob).sum())
            union = seen + float(sil.sum()) - inter
            total += inter / union if union > 0 else 0.0
        return total / len(views)

    coarse = []
    for yaw in np.deg2rad(np.arange(-90.0, 90.0, 7.5)):
        for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
            for dy in (-0.06, -0.03, 0.0, 0.03, 0.06):
                cand = np.array([guess_xy[0] + dx, guess_xy[1] + dy, yaw])
                coarse.append((score(*cand), cand))
    coarse.sort(key=lambda t: -t[0])
    best_score, best = -1.0, np.zeros(3)
    for start_score, start in coarse[:3]:
        sc, pose = start_score, start.copy()
        step = np.array([0.01, 0.01, math.radians(3.0)])
        while step[0] > 5e-4:  # coordinate descent with shrinking steps
            improved = False
            for k in range(3):
                for sign in (-1.0, 1.0):
                    cand = pose.copy()
                    cand[k] += sign * step[k]
                    cand_score = score(*cand)
                    if cand_score > sc:
                        sc, pose, improved = cand_score, cand, True
            if not improved:
                step = step / 2
        if sc > best_score:
            best_score, best = sc, pose
    yaw = (best[2] + math.pi / 2) % math.pi - math.pi / 2
    return BinFit(
        float(best[0]),
        float(best[1]),
        float(yaw),
        model.length,
        model.width,
        model.height,
        model.wall,
        best_score,
        len(views),
    )


def _off_nadir(camera: CeilingCamera, x: float, y: float) -> float:
    """Horizontal distance from the point under the camera to (x, y)."""
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    centre = -rot.T @ camera.tvec
    return float(math.hypot(x - centre[0], y - centre[1]))


def _isolated(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    frame: NDArray[np.uint8],
    cameras: Sequence[CeilingCamera],
    index: int,
    fit: BinFit,
    model: BinModel,
    darkness: float,
    min_saturation: float | None = None,
) -> bool:
    """Whether the dark region under the fit is the bin alone: at most ``_SPILL_MAX`` of
    the connected dark region containing it lies outside the silhouette grown by twice
    the band.

    The corner of a large dark region (a rug, a shadow) fits a bin's outline on two
    sides but spills far past it; a bin that *touches* dark furniture is merged with it
    and rejected too (move one apart).
    """
    img = _split(frame, len(cameras))[index]
    mask = bin_mask(img, darkness, min_saturation)
    rows, cols = mask.shape
    mask *= _valid_rows(rows, cols, index, len(cameras)).astype(np.uint8)
    _, labels = cv.connectedComponents(mask)
    cam = cameras[index]
    sil = np.zeros(mask.shape, np.uint8)
    cv.fillConvexPoly(
        sil, bin_silhouette(cam, fit.x, fit.y, fit.yaw, model).astype(np.int32), (1,)
    )
    ids = np.unique(labels[(sil > 0) & (mask > 0)])
    ids = ids[ids > 0]
    if ids.size == 0:
        return False
    region = np.isin(labels, ids)
    grown_model = BinModel(
        model.length + 4 * _BAND_M, model.width + 4 * _BAND_M, model.height
    )
    grown = np.zeros(mask.shape, np.uint8)
    cv.fillConvexPoly(
        grown,
        bin_silhouette(cam, fit.x, fit.y, fit.yaw, grown_model).astype(np.int32),
        (1,),
    )
    spill = float((region & (grown == 0)).sum()) / float(region.sum())
    return spill <= _SPILL_MAX


def localize_bin(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    model: BinModel = BinModel(),
    search_bounds: Sequence[float] = (-1.8, 2.0, -1.8, 1.8),
    exclude: Sequence[Sequence[float]] = (),
    min_iou: float = 0.7,
    serials: Sequence[str] | None = None,
    darkness: float = 0.85,
    min_saturation: float | None = None,
) -> BinFit | None:
    """The best-fitting bin whose centre lies in ``search_bounds``, or None if nothing
    fits well enough.

    Each camera is fitted on its own and the best single view kept (the cameras'
    calibrations need not agree); a near tie goes to a camera whose calibrated region
    holds the bin, then to the camera that sees it more directly from above.
    """
    best: BinFit | None = None
    for guess in find_bin_candidates(
        frames[-1], cameras, search_bounds, model, exclude, darkness, min_saturation
    ):
        fit = fit_bin_pose(
            frames[-2:],
            cameras,
            guess,
            model,
            darkness=darkness,
            min_saturation=min_saturation,
        )
        if fit is not None and (best is None or fit.iou > best.iou):
            best = fit
    if best is None or best.iou < min_iou:
        return None
    guess = (best.x, best.y)
    singles = []
    for i in range(len(cameras)):
        fit = fit_bin_pose(
            frames,
            cameras,
            guess,
            model,
            only=[i],
            darkness=darkness,
            min_saturation=min_saturation,
        )
        if fit is not None:
            singles.append((i, fit))
    if not singles:
        return None
    pool = singles
    if serials is not None:
        calibrated = [
            (i, f)
            for i, f in singles
            if in_calibrated_region(cameras[i], i, f.x, f.y, serials)
        ]
        pool = calibrated or singles
    top_iou = max(f.iou for _, f in pool)
    near = [(i, f) for i, f in pool if f.iou >= top_iou - 0.03]
    cam_i, chosen = min(near, key=lambda t: _off_nadir(cameras[t[0]], t[1].x, t[1].y))
    if chosen.iou < min_iou or not _isolated(
        frames[-1], cameras, cam_i, chosen, model, darkness, min_saturation
    ):
        return None
    if not (
        search_bounds[0] <= chosen.x <= search_bounds[1]
        and search_bounds[2] <= chosen.y <= search_bounds[3]
    ):
        return None
    spread = max(
        (math.hypot(a.x - b.x, a.y - b.y) for _, a in singles for _, b in singles),
        default=0.0,
    )
    return BinFit(
        chosen.x,
        chosen.y,
        chosen.yaw,
        model.length,
        model.width,
        model.height,
        model.wall,
        chosen.iou,
        chosen.num_views,
        tuple((i, f.x, f.y, f.yaw, f.iou) for i, f in singles),
        spread,
        cam_i,
    )


class ConstantBinSource(TargetSource):
    """A bin that is wherever the config says (fake mode)."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        length: float = 0.40,
        width: float = 0.32,
        height: float = 0.19,
        wall: float = 0.005,
    ) -> None:
        self.last_fit: BinFit | None = BinFit(
            float(x), float(y), float(yaw), length, width, height, wall
        )

    def localize(self) -> BinFit:
        """The configured bin."""
        assert self.last_fit is not None
        return self.last_fit

    def get_target(self) -> tuple[float, float, float]:
        fit = self.localize()
        return (fit.x, fit.y, fit.height)

    def close(self) -> None:
        """Nothing to release."""


class CeilingBinSource(TargetSource):
    """The bin's pose from the ceiling cameras, as ``CeilingCubeTargetSource`` does for
    the cube.

    Reads ``num_frames`` stitched frames from the ceiling image publisher and the
    robot's map pose from the marker detector (to ignore the robot's own footprint),
    then runs :func:`localize_bin`. ``localize`` raises if no bin is found, so nothing
    is ever aimed at a stale or invented bin, and saves that miss's frames and inputs
    under ``failure_dir`` (``NN_HH-MM-SS_bin``) so it can be replayed. ``get_target`` is
    the centre of the opening, at rim height.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        serials: Sequence[str],
        image_client: Any = None,
        marker_client: Any = None,
        length: float = 0.40,
        width: float = 0.32,
        height: float = 0.19,
        wall: float = 0.005,
        search_bounds: Sequence[float] = (-1.8, 2.0, -1.8, 1.8),
        robot_exclusion_half: float = 0.45,
        num_frames: int = 10,
        frame_period_s: float = 0.12,
        min_iou: float = 0.7,
        darkness: float = 0.85,
        host: str | None = None,
        min_saturation: float | None = None,
        failure_dir: str | Path | None = None,
    ) -> None:
        if image_client is None:
            from prpl_tidybot.marker_detector.ceiling_image_client import (  # pylint: disable=import-outside-toplevel
                CeilingImageClient,
            )

            image_client = CeilingImageClient(host) if host else CeilingImageClient()
        if marker_client is None:
            from prpl_tidybot.marker_detector.client import (  # pylint: disable=import-outside-toplevel
                MarkerDetectorClient,
            )

            marker_client = (
                MarkerDetectorClient(host) if host else MarkerDetectorClient()
            )
        self._images = image_client
        self._markers = marker_client
        self._serials = list(serials)
        self._cameras = load_ceiling_cameras(self._serials)
        self._failure_dir = None if failure_dir is None else Path(failure_dir)
        self._model = BinModel(float(length), float(width), float(height), float(wall))
        self._search = tuple(float(v) for v in search_bounds)
        self._robot_half = float(robot_exclusion_half)
        self._num_frames = int(num_frames)
        self._period = float(frame_period_s)
        self._min_iou = float(min_iou)
        self._darkness = float(darkness)
        self._min_saturation = None if min_saturation is None else float(min_saturation)
        self.last_fit: BinFit | None = None

    def _frames(self) -> list[NDArray[np.uint8]]:
        frames: list[NDArray[np.uint8]] = []
        deadline = time.time() + 10 * max(self._period, 0.1) * self._num_frames
        while len(frames) < self._num_frames and time.time() < deadline:
            img = self._images.get_image()
            if img is not None and (not frames or img is not frames[-1]):
                frames.append(np.array(img, copy=True))
            if self._period:
                time.sleep(self._period)
        if not frames:
            raise RuntimeError("CeilingBinSource: no ceiling image arrived")
        return frames

    def _exclusions(self) -> list[tuple[float, float, float]]:
        poses = self._markers.get_latest().get("poses") or {}
        return [(float(p[0]), float(p[1]), self._robot_half) for p in poses.values()]

    def localize(self) -> BinFit:
        """Run the localizer now and return the fit (raises if there is no bin, and
        saves the frames and inputs under ``failure_dir``)."""
        frames, exclude = self._frames(), self._exclusions()
        fit = localize_bin(
            frames,
            self._cameras,
            self._model,
            search_bounds=self._search,
            exclude=exclude,
            min_iou=self._min_iou,
            serials=self._serials,
            darkness=self._darkness,
            min_saturation=self._min_saturation,
        )
        if fit is None:
            self._save_failure(frames, exclude)
            raise RuntimeError(
                f"CeilingBinSource: no bin found in the search box {self._search}"
            )
        self.last_fit = fit
        return fit

    def _save_failure(
        self,
        frames: Sequence[NDArray[np.uint8]],
        exclude: Sequence[Sequence[float]],
    ) -> None:
        if self._failure_dir is None:
            return
        stamp = time.strftime("%H-%M-%S")
        n = len(list(self._failure_dir.glob("*"))) if self._failure_dir.exists() else 0
        out = self._failure_dir / f"{n:02d}_{stamp}_bin"
        out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv.imwrite(str(out / f"frame_{i:02d}.png"), frame)
        model = self._model
        info = {
            "object": "bin",
            "time": time.time(),
            "search_bounds": list(self._search),
            "exclusions": [list(e) for e in exclude],
            "min_iou": self._min_iou,
            "darkness": self._darkness,
            "min_saturation": self._min_saturation,
            "model": {
                "length": model.length,
                "width": model.width,
                "height": model.height,
                "wall": model.wall,
            },
            "serials": self._serials,
        }
        (out / "localization.json").write_text(json.dumps(info, indent=1))
        _logger.warning("No bin found; saved the frames and inputs to %s.", out)

    def get_target(self) -> tuple[float, float, float]:
        fit = self.localize()
        return (fit.x, fit.y, fit.height)

    def close(self) -> None:
        """Close the image and marker clients."""
        self._images.close()
        self._markers.close()
