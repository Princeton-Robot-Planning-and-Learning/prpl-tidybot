"""One occlusion-aware cube detector for the floor and the bin, from the ceiling
cameras.

The cube is the only cool-coloured thing in the room: the floor, the cardboard bin and
the bin's shadowed inside are all warm (Lab b* above ~140 where they are lit) or near
black. :func:`cube_pixels` is one rule for every background, a b* ceiling that falls
with brightness (:data:`SEGMENTATION`): lit, the cube is pale blue-grey (b* ~110-130
against the floor's ~147); in the bin's shadow it is dim (L ~25-60) and only slightly
blue (b* ~120-127), and the dark cardboard around it reads b* ~127-131 at L below ~20.

A candidate pose (x, y, yaw, flat on the floor or on the bin's floor) is scored by the
IoU of its projected silhouette with the cube mask *on the part the camera can see*: the
bin's four walls (5 mm cardboard slabs on the bin fit, 0.21 m tall) hide whatever lies
behind them from that camera, so the pixels a wall covers are left out of the score, as
are the stitch-line rows. A pose must keep at least ``min_visible`` of its silhouette
visible, so a cube almost wholly behind a wall is a clean miss, never a guess. A wall
hides the cube only when it stands between the two, i.e. when the camera and the cube
are on opposite sides of the wall's plane.

:func:`detect_cube` runs the whole search in two halves, so a look can segment the
frames while the bin detector runs (:meth:`CeilingCubeDetectorSource.look`):
:func:`cube_evidence` (the mask, averaged over the frames, and the candidate blobs;
needs no bin) and :func:`score_candidates` (the occlusion-aware fits, once the bin is
known). The in-bin decision comes last, from the fitted centre against the bin's
inner walls (``cube_in_bin``, no margin): the same "anywhere inside the walls" rule
as scoring.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.bin_localizer import BinFit, cube_in_bin
from prpl_tidybot.cube_localizer import (
    CeilingCamera,
    CeilingCubeTargetSource,
    CubeFit,
    cube_silhouette,
    in_calibrated_region,
)


@dataclass(frozen=True)
class Segmentation:
    """:func:`cube_pixels`' rule: a pixel is the cube when its Lab L* (OpenCV's 0-255
    scale) is at least ``l_min`` and its b* below ``min(b_max, b_dark + slope * (L -
    l_dark))``."""

    l_min: int = 12
    b_max: float = 131.0
    b_dark: float = 128.0
    l_dark: float = 0.0
    slope: float = 0.1


SEGMENTATION = Segmentation()

# Unit cube corners, centred in x/y, bottom face on z = 0.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)]
)
_SEAM_PX = 8
# Fit window half size (px) around a candidate.
_WINDOW = 50
# Blob areas (full-res px) a cube or the visible part of one can have.
_AREA_RANGE = (12, 1600)
# A blob's area relative to the cube's whole silhouette at its spot: at most this
# (shelf edges and furniture are larger), and at least this times ``min_visible``.
_AREA_MAX_RATIO = 2.0
_AREA_MIN_RATIO = 0.45
# Longest over shortest side: a whole cube is square-ish (<= 2), a partly hidden one
# can be a strip.
_MAX_ELONGATION = 3.0
# Pose grid around a candidate: (x, y) offsets (m) and yaws (deg), then coordinate
# descent; as the floor localizer's.
_OFFSETS = np.arange(-0.04, 0.0401, 0.005)
_YAWS = np.deg2rad(np.arange(-45.0, 45.0, 3.0))
# A near tie (IoU) between cameras goes to the one that sees the cube from more
# directly above.
_TIE = 0.03
# Poses scored per batch.
_CHUNK = 8
# The top face's corners (of _CORNERS), in order around it.
_TOP = np.array([1, 3, 7, 5])
# Mask blobs this far (px) beyond a candidate's silhouette are not its.
_NEAR_PX = 6
# Pixels this close (m, at the cube) to a hiding wall's edge are not scored.
_EDGE_MARGIN_M = 0.01

MIN_IOU = 0.6
MIN_VISIBLE = 0.4


@dataclass
class CubeEvidence:
    """What :func:`cube_evidence` extracts from the frames: the per-pixel cube fraction
    (mean of the frames' masks, stitched layout) and the candidate floor points (x,
    y)."""

    prob: NDArray[np.float32]
    candidates: list[tuple[float, float]]
    num_frames: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _lookup(seg: Segmentation) -> NDArray[np.bool_]:
    """The rule as a 256 x 256 (L, b) table."""
    lum = np.arange(256, dtype=np.float64)[:, None]
    limit = np.minimum(seg.b_max, seg.b_dark + seg.slope * (lum - seg.l_dark))
    b = np.arange(256, dtype=np.float64)[None, :]
    return np.asarray((lum >= seg.l_min) & (b < limit), np.bool_)


_TABLES: dict[Segmentation, NDArray[np.bool_]] = {}


def cube_pixels(
    bgr: NDArray[np.uint8], seg: Segmentation = SEGMENTATION
) -> NDArray[np.uint8]:
    """1 where a pixel looks like the cube (lit on the floor or shadowed in the bin),
    holes (the printed digit) filled."""
    table = _TABLES.get(seg)
    if table is None:
        table = _TABLES[seg] = _lookup(seg)
    lab: Any = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    mask: Any = table[lab[:, :, 0], lab[:, :, 2]].astype(np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, (1,), thickness=cv.FILLED)
    return np.asarray(filled, np.uint8)


def cube_evidence(  # pylint: disable=too-many-arguments,too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    side: float,
    exclude: Sequence[Sequence[float]] = (),
    min_visible: float = MIN_VISIBLE,
    seg: Segmentation = SEGMENTATION,
    also: Sequence[Sequence[float]] = (),
) -> CubeEvidence:
    """The cube mask averaged over ``frames`` and every blob in it that could be the
    cube, or the visible part of one, inside ``floor_bounds`` (or any box in
    ``also``: where the bin may stand) and outside the ``exclude`` squares (x, y,
    half side: the robot)."""
    prob = np.zeros(frames[0].shape[:2], np.float32)
    for frame in frames:
        prob += cube_pixels(frame, seg)
    prob /= len(frames)
    num = len(cameras)
    rows = prob.shape[0] // num
    blobs = (prob >= 0.5).astype(np.uint8)
    for i in range(1, num):
        blobs[i * rows - _SEAM_PX : i * rows + _SEAM_PX, :] = 0
    count, labels_any, stats, centroids = cv.connectedComponentsWithStats(blobs)
    labels = np.asarray(labels_any, np.int32)
    candidates: list[tuple[float, float]] = []
    rejected: list[dict[str, Any]] = []
    for k in range(1, count):
        area = int(stats[k, cv.CC_STAT_AREA])
        if not _AREA_RANGE[0] <= area <= _AREA_RANGE[1]:
            continue
        u, v = centroids[k]
        i = min(int(v // rows), num - 1)
        xy = cameras[i].to_floor(u, v - i * rows, z=side / 2)
        if _excluded(xy, exclude) or not any(
            _in_bounds(xy, box) for box in (floor_bounds, *also)
        ):
            continue
        why = _not_cube_like(labels, k, stats[k], cameras[i], xy, side, min_visible)
        if why:
            rejected.append({"xy": [float(xy[0]), float(xy[1])], "why": why})
            continue
        if all(math.hypot(xy[0] - cx, xy[1] - cy) > 2 * side for cx, cy in candidates):
            candidates.append((float(xy[0]), float(xy[1])))
    return CubeEvidence(
        prob, candidates, len(frames), {"rejected_blobs": rejected[:20]}
    )


def _in_bounds(xy: NDArray[np.float64], bounds: Sequence[float]) -> bool:
    return bool(bounds[0] < xy[0] < bounds[1] and bounds[2] < xy[1] < bounds[3])


def _excluded(xy: NDArray[np.float64], exclude: Sequence[Sequence[float]]) -> bool:
    return any(
        abs(xy[0] - ex) < half and abs(xy[1] - ey) < half for ex, ey, half in exclude
    )


def _not_cube_like(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    labels: NDArray[np.int32],
    k: int,
    stat: NDArray[np.int32],
    camera: CeilingCamera,
    xy: NDArray[np.float64],
    side: float,
    min_visible: float,
) -> str:
    """Why blob ``k`` cannot be the cube or the visible part of one, or ''."""
    expected = [
        cv.contourArea(
            cube_silhouette(camera, xy[0], xy[1], yaw, side).astype(np.float32)
        )
        for yaw in (0.0, math.pi / 4)
    ]
    area = float(stat[cv.CC_STAT_AREA])
    if area > _AREA_MAX_RATIO * max(expected):
        return "too large"
    if area < _AREA_MIN_RATIO * min_visible * min(expected):
        return "too small"
    x, y, w, h = (
        int(stat[c])
        for c in (cv.CC_STAT_LEFT, cv.CC_STAT_TOP, cv.CC_STAT_WIDTH, cv.CC_STAT_HEIGHT)
    )
    if x <= 0 or x + w >= labels.shape[1] or y <= 0 or y + h >= labels.shape[0]:
        return "cut by the image border"
    points = cv.findNonZero((labels[y : y + h, x : x + w] == k).astype(np.uint8))
    (_, _), (rw, rh), _ = cv.minAreaRect(points)
    if max(rw, rh) > _MAX_ELONGATION * max(min(rw, rh), 1.0):
        return "elongated"
    return ""


# ------------------------------------------------------------------ occlusion model


def camera_centre(camera: CeilingCamera) -> NDArray[np.float64]:
    """The camera's optical centre, map frame."""
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    return np.asarray(-rot.T @ camera.tvec, np.float64).reshape(3)


def _walls(bin_fit: BinFit) -> list[tuple[int, float, NDArray[np.float64]]]:
    """The bin's four walls: (axis in the bin's frame, the wall's mid-plane offset on
    it, its 8 corners in the map frame)."""
    hl, hw, wall = bin_fit.length / 2, bin_fit.width / 2, bin_fit.wall
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    out = []
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
        if axis == 0:
            lo = np.array([sign * hl - (wall if sign > 0 else -wall), -hw])
            hi = np.array([sign * hl, hw])
        else:
            lo = np.array([-hl, sign * hw - (wall if sign > 0 else -wall)])
            hi = np.array([hl, sign * hw])
        x0, x1 = sorted((lo[0], hi[0]))
        y0, y1 = sorted((lo[1], hi[1]))
        local = np.array(
            [
                [px, py, pz]
                for px in (x0, x1)
                for py in (y0, y1)
                for pz in (0.0, bin_fit.height)
            ]
        )
        world = np.c_[
            bin_fit.x + c * local[:, 0] - s * local[:, 1],
            bin_fit.y + s * local[:, 0] + c * local[:, 1],
            local[:, 2],
        ]
        mid = sign * (hl if axis == 0 else hw) - sign * wall / 2
        out.append((axis, mid, world))
    return out


def _to_bin(bin_fit: BinFit, xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map-frame points (N, 2+) -> the bin's frame (N, 2)."""
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    dx, dy = xy[:, 0] - bin_fit.x, xy[:, 1] - bin_fit.y
    return np.c_[c * dx + s * dy, -s * dx + c * dy]


def occluding_walls(
    camera: CeilingCamera, bin_fit: BinFit, xy: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """(N, 4): whether each wall of the bin stands between the camera and a cube centred
    at each map point ``xy`` (N, 2): they are on opposite sides of its plane."""
    cam_local = _to_bin(bin_fit, camera_centre(camera)[None, :2])[0]
    local = _to_bin(bin_fit, np.asarray(xy, np.float64).reshape(-1, 2))
    walls = _walls(bin_fit)
    out = np.zeros((len(local), len(walls)), np.bool_)
    for j, (axis, mid, _) in enumerate(walls):
        out[:, j] = np.sign(local[:, axis] - mid) != np.sign(cam_local[axis] - mid)
    return out


def occluder_mask(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    camera: CeilingCamera,
    bin_fit: BinFit,
    which: Sequence[bool],
    shape: tuple[int, int],
    u0: int = 0,
    v0: int = 0,
) -> NDArray[np.float32]:
    """The image (``shape``, origin at pixel (u0, v0)) of the walls flagged in ``which``
    (:func:`occluding_walls`), anti-aliased: 1 where they hide the floor."""
    mask = np.zeros(shape, np.float32)
    for flag, (_, _, world) in zip(which, _walls(bin_fit)):
        if not flag:
            continue
        px = camera.project(world).astype(np.float32)
        hull = cv.convexHull(px).reshape(-1, 2).astype(np.float64)
        fixed = ((hull - [u0, v0]) * 16).astype(np.int32)
        cv.fillConvexPoly(mask, fixed, (1.0,), lineType=cv.LINE_AA, shift=4)
    return np.minimum(mask, 1.0)


# ------------------------------------------------------------------------ the fit


def opening_mask(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    camera: CeilingCamera,
    bin_fit: BinFit,
    shape: tuple[int, int],
    u0: int = 0,
    v0: int = 0,
) -> NDArray[np.float32]:
    """The image (``shape``, origin at pixel (u0, v0)) of the bin's opening, the rim's
    inner rectangle: everything a camera sees inside the bin it sees through it."""
    hl = bin_fit.length / 2 - bin_fit.wall
    hw = bin_fit.width / 2 - bin_fit.wall
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    world = np.c_[
        bin_fit.x + c * local[:, 0] - s * local[:, 1],
        bin_fit.y + s * local[:, 0] + c * local[:, 1],
        np.full(4, bin_fit.height),
    ]
    px = camera.project(world)
    mask = np.zeros(shape, np.float32)
    fixed = ((px - [u0, v0]) * 16).astype(np.int32)
    cv.fillConvexPoly(mask, fixed, (1.0,), lineType=cv.LINE_AA, shift=4)
    return mask


# ------------------------------------------------------------------------ the fit


@dataclass
class _View:
    """One camera's window onto a candidate: the averaged mask, the stitch-line validity
    and the bin (for its occluders and its opening)."""

    camera: CeilingCamera
    u0: int
    v0: int
    prob: NDArray[np.float32]
    seam: NDArray[np.float32]
    bin_fit: BinFit | None
    margin_px: int = 0
    # The mask blobs at the candidate (dilated): inside the bin only these are the
    # cube's (the bin's floor carries tape, a marker, paper).
    own: NDArray[np.float32] | None = None
    _cache: dict[
        tuple[bool, tuple[bool, ...]], tuple[NDArray[np.float32], NDArray[np.float32]]
    ] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """The window's (rows, columns)."""
        return (int(self.prob.shape[0]), int(self.prob.shape[1]))

    def scored(
        self, inside: bool, which: tuple[bool, ...]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """(the pixels scored, the mask there) for poses ``inside`` the bin or not, that
        the walls ``which`` hide: not the seam, not behind those walls nor within
        ``margin_px`` of their edge (the bin's fit is good to ~2 cm, so where exactly a
        wall's edge falls is not known); for a pose inside the bin, only the mask seen
        through the bin's opening, and of that only the blobs at the candidate."""
        key = (inside, which)
        out = self._cache.get(key)
        if out is None:
            valid = self.seam
            prob = self.prob
            if self.bin_fit is not None and any(which):
                occ = occluder_mask(
                    self.camera, self.bin_fit, which, self.shape, self.u0, self.v0
                )
                hidden: Any = (occ > 0.02).astype(np.uint8)
                if self.margin_px > 0:
                    size = 2 * self.margin_px + 1
                    hidden = cv.dilate(hidden, np.ones((size, size), np.uint8))
                valid = self.seam * (1.0 - hidden.astype(np.float32))
            if self.bin_fit is not None and inside:
                prob = prob * opening_mask(
                    self.camera, self.bin_fit, self.shape, self.u0, self.v0
                )
                if self.own is not None:
                    prob = prob * self.own
            out = self._cache[key] = (valid, prob * valid)
        return out


def _view(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    prob: NDArray[np.float32],
    cameras: Sequence[CeilingCamera],
    index: int,
    guess: tuple[float, float],
    side: float,
    bin_fit: BinFit | None,
) -> _View | None:
    cam = cameras[index]
    rows = prob.shape[0] // len(cameras)
    sil = cube_silhouette(cam, *guess, 0.0, side)
    u, v = sil.mean(axis=0)
    if not (0 <= u < prob.shape[1] and 0 <= v < rows):
        return None
    u0, v0 = max(int(u) - _WINDOW, 0), max(int(v) - _WINDOW, 0)
    u1, v1 = int(u) + _WINDOW, int(v) + _WINDOW
    tile = prob[index * rows : (index + 1) * rows]
    window = np.ascontiguousarray(tile[v0:v1, u0:u1], np.float32)
    seam = np.ones_like(window)
    if index > 0:
        seam[: max(0, _SEAM_PX - v0), :] = 0
    if index < len(cameras) - 1:
        seam[max(0, rows - _SEAM_PX - v0) :, :] = 0
    window = window * seam
    if window.sum() <= 0:
        return None
    side_px = math.sqrt(max(cv.contourArea(sil.astype(np.float32)), 1.0))
    margin = int(math.ceil(_EDGE_MARGIN_M / side * side_px))
    own = None
    if bin_fit is not None:
        radius = float(np.linalg.norm(sil - [u, v], axis=1).max()) + _NEAR_PX
        _, labels_any, _, _ = cv.connectedComponentsWithStats(
            (window >= 0.5).astype(np.uint8)
        )
        labels = np.asarray(labels_any, np.int32)
        vv, uu = np.mgrid[0 : window.shape[0], 0 : window.shape[1]]
        near = np.hypot(uu - (u - u0), vv - (v - v0)) <= radius
        keep = np.isin(labels, np.unique(labels[near & (labels > 0)]))
        grown: Any = cv.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8))
        own = np.asarray(grown > 0, np.float32)
    return _View(cam, u0, v0, window, seam, bin_fit, margin, own)


def _scores(  # pylint: disable=too-many-locals,too-many-statements
    view: _View,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    yaw: float,
    side: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(IoU on the visible part, visible fraction) of a cube at each (xs[k], ys[k], yaw)
    in ``view``.

    On the floor the whole silhouette is scored, as the floor localizer always did
    (its lit side faces are as pale as its top). Inside the bin the side faces a
    camera sees are in the walls' shadow, too dark to segment, so they count for the
    cube where they are in the mask but not against it where they are not: the
    union is the mask plus the top face, the intersection the mask within the whole
    silhouette, and the visible fraction is the top face's.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    local = _CORNERS * side
    n = len(xs)
    world = np.empty((n, len(local), 3))
    world[:, :, 0] = xs[:, None] + c * local[:, 0] - s * local[:, 1]
    world[:, :, 1] = ys[:, None] + s * local[:, 0] + c * local[:, 1]
    world[:, :, 2] = local[:, 2]
    px = view.camera.project(world.reshape(-1, 3)).astype(np.float32).reshape(n, -1, 2)
    fixed = ((px.astype(np.float64) - [view.u0, view.v0]) * 16).astype(np.int32)
    groups: dict[tuple[bool, tuple[bool, ...]], list[int]] = {}
    if view.bin_fit is None:
        groups[(False, ())] = list(range(n))
    else:
        bin_fit = view.bin_fit
        flags = occluding_walls(view.camera, bin_fit, np.c_[xs, ys])
        for k, row in enumerate(flags):
            inside = cube_in_bin((xs[k], ys[k]), bin_fit, margin=0.0)
            groups.setdefault((inside, tuple(bool(f) for f in row)), []).append(k)
    iou, visible = np.empty(n), np.empty(n)
    sil = np.empty((_CHUNK, *view.prob.shape), np.float32)
    top = np.empty_like(sil)
    for (inside, which), members in groups.items():
        valid, prob = view.scored(inside, which)
        prob_sum = float(prob.sum())
        for start in range(0, len(members), _CHUNK):
            chunk = members[start : start + _CHUNK]
            m = len(chunk)
            sil[:m] = 0.0
            top[:m] = 0.0
            for j, k in enumerate(chunk):
                hull = cv.convexHull(px[k], returnPoints=False).ravel()
                cv.fillConvexPoly(
                    sil[j], fixed[k][hull], (1.0,), lineType=cv.LINE_AA, shift=4
                )
                if inside:
                    cv.fillConvexPoly(
                        top[j], fixed[k][_TOP], (1.0,), lineType=cv.LINE_AA, shift=4
                    )
            idx = np.asarray(chunk)
            inter = (sil[:m] * prob).reshape(m, -1).sum(axis=1)
            if inside:
                face, face_full = top[:m] * valid, top[:m].reshape(m, -1).sum(axis=1)
                face_sum = face.reshape(m, -1).sum(axis=1)
                face_inter = (face * prob).reshape(m, -1).sum(axis=1)
                union = prob_sum + face_sum - face_inter
            else:
                face_full = sil[:m].reshape(m, -1).sum(axis=1)
                face_sum = (sil[:m] * valid).reshape(m, -1).sum(axis=1)
                union = prob_sum + face_sum - inter
            iou[idx] = inter / np.maximum(union, 1e-9)
            visible[idx] = face_sum / np.maximum(face_full, 1e-9)
    return iou, visible


def fit_visible(  # pylint: disable=too-many-locals
    view: _View, guess: tuple[float, float], side: float, min_visible: float
) -> CubeFit | None:
    """The pose around ``guess`` whose visible silhouette best matches ``view``'s mask,
    among poses with at least ``min_visible`` of the cube in sight, or None."""

    def score(x: float, y: float, yaw: float) -> tuple[float, float]:
        iou, vis = _scores(view, np.array([x]), np.array([y]), yaw, side)
        return (float(iou[0]) if vis[0] >= min_visible else -1.0), float(vis[0])

    best_score, best, best_vis = -1.0, np.zeros(3), 0.0
    xs = guess[0] + np.repeat(_OFFSETS, len(_OFFSETS))
    ys = guess[1] + np.tile(_OFFSETS, len(_OFFSETS))
    for yaw in _YAWS:
        iou, vis = _scores(view, xs, ys, float(yaw), side)
        iou = np.where(vis >= min_visible, iou, -1.0)
        k = int(np.argmax(iou))
        if iou[k] > best_score:
            best_score, best = float(iou[k]), np.array([xs[k], ys[k], yaw])
            best_vis = float(vis[k])
    if best_score < 0:
        return None
    step = np.array([0.0025, 0.0025, math.radians(1.5)])
    while step[0] > 2e-4:  # coordinate descent with shrinking steps
        improved = False
        for k in range(3):
            for sign in (-1.0, 1.0):
                cand = best.copy()
                cand[k] += sign * step[k]
                sc, vis_c = score(*cand)
                if sc > best_score:
                    best_score, best, best_vis, improved = sc, cand, vis_c, True
        if not improved:
            step = step / 2
    yaw = (best[2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return CubeFit(
        float(best[0]), float(best[1]), float(yaw), best_score, 1, visible=best_vis
    )


def _off_nadir(camera: CeilingCamera, x: float, y: float) -> float:
    centre = camera_centre(camera)
    return float(math.hypot(x - centre[0], y - centre[1]))


def score_candidates(  # pylint: disable=too-many-arguments,too-many-locals
    evidence: CubeEvidence,
    cameras: Sequence[CeilingCamera],
    side: float,
    bin_fit: BinFit | None = None,
    *,
    floor_bounds: Sequence[float] | None = None,
    min_iou: float = MIN_IOU,
    min_visible: float = MIN_VISIBLE,
    serials: Sequence[str] | None = None,
    bin_box: Sequence[float] | None = None,
    executor: Executor | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> CubeFit | None:
    """The best cube among ``evidence``'s candidates, each fitted in every camera on the
    part of it that camera sees (``bin_fit``'s walls hidden), or None.

    Per candidate the single best view is kept (as the floor localizer's ``best_view``):
    only views fitting at ``min_iou`` or better with ``min_visible`` of the cube in
    sight compete, those whose calibrated region holds the cube first, a near tie going
    to the camera more directly above. The fit's ``in_bin`` is whether its centre lies
    inside the bin's walls (``cube_in_bin``, no margin) or in ``bin_box``. ``executor``
    runs the (candidate, camera) fits in parallel.
    """
    jobs = []
    for guess in evidence.candidates:
        for i in range(len(cameras)):
            view = _view(evidence.prob, cameras, i, guess, side, bin_fit)
            if view is not None:
                jobs.append((guess, i, view))

    def run(job: tuple[tuple[float, float], int, _View]) -> CubeFit | None:
        return fit_visible(job[2], job[0], side, min_visible)

    fits = list(executor.map(run, jobs)) if executor else [run(j) for j in jobs]
    by_guess: dict[tuple[float, float], list[tuple[int, CubeFit]]] = {}
    for (guess, i, _), fit in zip(jobs, fits):
        if fit is not None:
            by_guess.setdefault(guess, []).append((i, fit))
    screened = []
    best: CubeFit | None = None
    for guess in evidence.candidates:
        singles = by_guess.get(guess, [])
        screened.append(
            {
                "guess": list(guess),
                "fits": [
                    {
                        "camera": i,
                        "x": f.x,
                        "y": f.y,
                        "iou": f.iou,
                        "visible": f.visible,
                    }
                    for i, f in singles
                ],
            }
        )
        chosen = _choose(singles, cameras, min_iou, serials)
        if chosen is None:
            continue
        i, fit = chosen
        spread = max(
            (math.hypot(a.x - b.x, a.y - b.y) for _, a in singles for _, b in singles),
            default=0.0,
        )
        inside = bool(
            (bin_fit is not None and cube_in_bin((fit.x, fit.y), bin_fit, margin=0.0))
            or (bin_box is not None and _in_box((fit.x, fit.y), bin_box))
        )
        if (
            not inside
            and floor_bounds is not None
            and not _in_bounds(np.array([fit.x, fit.y]), floor_bounds)
        ):
            continue
        full = CubeFit(
            fit.x,
            fit.y,
            fit.yaw,
            fit.iou,
            1,
            tuple((j, f.x, f.y, f.yaw, f.iou) for j, f in singles),
            spread,
            i,
            visible=fit.visible,
            in_bin=inside,
        )
        if best is None or full.iou > best.iou:
            best = full
    if diagnostics is not None:
        diagnostics["candidates"] = screened
    return best


def _in_box(xy: Sequence[float], box: Sequence[float]) -> bool:
    return bool(box[0] <= xy[0] <= box[1] and box[2] <= xy[1] <= box[3])


def _choose(
    singles: list[tuple[int, CubeFit]],
    cameras: Sequence[CeilingCamera],
    min_iou: float,
    serials: Sequence[str] | None,
) -> tuple[int, CubeFit] | None:
    pool = [(i, f) for i, f in singles if f.iou >= min_iou]
    if not pool:
        return None
    if serials is not None:
        calibrated = [
            (i, f)
            for i, f in pool
            if in_calibrated_region(cameras[i], i, f.x, f.y, serials)
        ]
        pool = calibrated or pool
    top = max(f.iou for _, f in pool)
    near = [(i, f) for i, f in pool if f.iou >= top - _TIE]
    return min(near, key=lambda t: _off_nadir(cameras[t[0]], t[1].x, t[1].y))


def detect_cube(  # pylint: disable=too-many-arguments
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    side: float,
    bin_fit: BinFit | None = None,
    *,
    exclude: Sequence[Sequence[float]] = (),
    min_iou: float = MIN_IOU,
    min_visible: float = MIN_VISIBLE,
    serials: Sequence[str] | None = None,
    bin_box: Sequence[float] | None = None,
    executor: Executor | None = None,
    diagnostics: dict[str, Any] | None = None,
    seg: Segmentation = SEGMENTATION,
) -> CubeFit | None:
    """The cube on the floor or in the bin (:func:`cube_evidence`, then
    :func:`score_candidates`), or None."""
    evidence = cube_evidence(
        frames, cameras, floor_bounds, side, exclude, min_visible, seg
    )
    return score_candidates(
        evidence,
        cameras,
        side,
        bin_fit,
        floor_bounds=floor_bounds,
        min_iou=min_iou,
        min_visible=min_visible,
        serials=serials,
        bin_box=bin_box,
        executor=executor,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class Look:  # pylint: disable=too-many-instance-attributes
    """One look from one shared frame grab: the bin (or None and why) and the cube (or
    None), and where the time went (s)."""

    bin_fit: BinFit | None
    bin_error: str
    cube: CubeFit | None
    num_frames: int
    grab_s: float
    compute_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def look_once(  # pylint: disable=too-many-arguments,too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    side: float,
    *,
    exclude: Sequence[Sequence[float]] = (),
    find_bin: Callable[[Sequence[NDArray[np.uint8]], Any], BinFit] | None = None,
    bin_fit: BinFit | None = None,
    bin_bounds: Sequence[float] | None = None,
    bin_box: Sequence[float] | None = None,
    min_iou: float = MIN_IOU,
    min_visible: float = MIN_VISIBLE,
    serials: Sequence[str] | None = None,
    executor: Executor | None = None,
    grab_s: float = 0.0,
) -> Look:
    """The bin and the cube from the same ``frames``: ``find_bin(frames, exclude)``
    (raises ``RuntimeError`` on a miss; then ``bin_fit`` stands) runs on ``executor``
    while the cube's mask and candidates are extracted, and the occlusion-aware fits
    follow once the bin is known.

    ``bin_bounds`` is where the bin may stand (a cube in it may lie outside
    ``floor_bounds``).
    """
    t0 = time.perf_counter()
    future = None
    if find_bin is not None:
        future = (
            executor.submit(find_bin, frames, exclude) if executor is not None else None
        )
    also = [b for b in (bin_bounds, bin_box) if b is not None]
    evidence = cube_evidence(
        frames, cameras, floor_bounds, side, exclude, min_visible, also=also
    )
    error = ""
    if find_bin is not None:
        try:
            found = future.result() if future is not None else find_bin(frames, exclude)
            bin_fit = found
        except RuntimeError as e:
            error = str(e)
    diagnostics: dict[str, Any] = dict(evidence.diagnostics)
    cube = score_candidates(
        evidence,
        cameras,
        side,
        bin_fit,
        floor_bounds=floor_bounds,
        min_iou=min_iou,
        min_visible=min_visible,
        serials=serials,
        bin_box=bin_box,
        executor=executor,
        diagnostics=diagnostics,
    )
    return Look(
        bin_fit,
        error,
        cube,
        len(frames),
        grab_s,
        time.perf_counter() - t0,
        diagnostics,
    )


class CeilingCubeDetectorSource(CeilingCubeTargetSource):
    """:class:`prpl_tidybot.cube_localizer.CeilingCubeTargetSource` on this detector:
    ``get_target`` runs :func:`detect_cube` (``fusion="fused"`` keeps the floor
    localizer's fused fit), and :meth:`look` is the skill server's camera round: one
    grab of ``look_frames`` frames shared by the bin and the cube, the bin detector
    running on another of ``threads`` threads while the cube's mask and candidates
    are extracted, and the cube fitted on what the bin's walls leave visible."""

    def __init__(
        self,
        serials: Sequence[str],
        *,
        look_frames: int = 10,
        min_visible: float = MIN_VISIBLE,
        threads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(serials, **kwargs)
        self._look_frames = int(look_frames)
        self._min_visible = float(min_visible)
        self._threads = int(threads)
        self._executor: ThreadPoolExecutor | None = None
        self.last_look: Look | None = None

    def _pool(self) -> ThreadPoolExecutor | None:
        if self._executor is None and self._threads > 1:
            self._executor = ThreadPoolExecutor(self._threads)
        return self._executor

    def _fit(
        self,
        frames: Sequence[NDArray[np.uint8]],
        exclude: Sequence[Sequence[float]],
        diagnostics: dict[str, Any],
    ) -> CubeFit | None:
        if self._fusion == "fused":
            return super()._fit(frames, exclude, diagnostics)
        return detect_cube(
            frames,
            self._cameras,
            self._floor_bounds,
            self._side,
            exclude=exclude,
            min_iou=self._min_iou,
            min_visible=self._min_visible,
            serials=self._serials,
            executor=self._pool(),
            diagnostics=diagnostics,
        )

    def look(
        self,
        bin_source: Any = None,
        bin_fit: BinFit | None = None,
        box: Sequence[float] | None = None,
    ) -> Look:
        """One camera round: the bin from ``bin_source.localize_frames`` (``bin_fit``
        when there is none or it misses) and the cube, inside the bin or on the floor,
        from the same frames; its ``in_bin`` is its centre inside the bin's walls (or
        in ``box``, the bin's fixed floor box). A cube miss is saved under
        ``failure_dir`` (with the bin) and retried with fresh frames (and the bin
        already found, if it was); the look's ``cube`` stays None if every try
        misses."""
        finder = getattr(bin_source, "localize_frames", None)
        bounds = getattr(bin_source, "search_bounds", None)
        result: Look | None = None
        for _ in range(1 + self._retries):
            t0 = time.perf_counter()
            frames, exclude = self._frames(self._look_frames), self._exclusions()
            result = look_once(
                frames,
                self._cameras,
                self._floor_bounds,
                self._side,
                exclude=exclude,
                find_bin=finder,
                bin_fit=bin_fit,
                bin_bounds=bounds,
                bin_box=box,
                min_iou=self._min_iou,
                min_visible=self._min_visible,
                serials=self._serials,
                executor=self._pool(),
                grab_s=time.perf_counter() - t0,
            )
            if result.cube is not None:
                break
            # The bin does not move between tries: the retry re-grabs for the cube.
            if result.bin_fit is not None:
                finder, bin_fit = None, result.bin_fit
            self._save_failure(
                frames,
                exclude,
                {
                    **result.diagnostics,
                    "bin": None if result.bin_fit is None else result.bin_fit.as_dict(),
                    "bin_error": result.bin_error,
                    "box": None if box is None else list(box),
                },
            )
        assert result is not None
        self.last_look = result
        if result.cube is not None:
            self.last_fit = result.cube
        return result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        super().close()
