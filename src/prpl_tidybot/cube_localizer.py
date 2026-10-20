"""Localize a known-size cube on the floor from the ceiling cameras, without markers.

The floor is a single warm colour and the cube is pale blue, so a colour test separates
them (Lab b* below the floor's). Each colour blob in the drivable area is a candidate,
and the pose is found by **silhouette fitting a known 3D cube**: the cube rests flat on
the floor, so its pose has three degrees of freedom (x, y, yaw), and its silhouette in a
calibrated camera is the convex hull of its eight projected corners (including whichever
side faces that camera sees). We maximise the IoU between that silhouette and the cube
mask, averaging the mask over several frames *before* fitting so that per-frame noise
cannot flip the yaw, and summing over every camera that sees the cube.

The stitched ceiling image (``CeilingImagePublisher``) is the cameras' undistorted
frames stacked top to bottom in ``camera_serials`` order, so camera ``i`` occupies rows
``[i * h, (i + 1) * h)`` and is projected with its intrinsics and no distortion.
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

from prpl_tidybot.marker_detector.extrinsics import load_extrinsics
from prpl_tidybot.marker_detector.utils import CAMERA_PARAMS_DIR, get_camera_params
from prpl_tidybot.real_sim.perceivers.target_source import TargetSource

_logger = logging.getLogger(__name__)

# Unit cube corners, centred in x/y, bottom face on z = 0.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)]
)
# The floor's Lab b* sits at ~145-150; the cube's at ~110-135.
_B_MAX = 139
_L_MIN = 60
# Blob areas (full-res px) a cube-sized object can have at ceiling height.
_AREA_RANGE = (24, 1600)
# Search window around a candidate (px) and the stitch-line rows to ignore.
_WINDOW = 50
_SEAM_PX = 8
# A candidate blob's area relative to a cube's silhouette at its spot, and its longest
# over shortest side: the cube measured 1.07-1.27x and square on the robot; shelf
# edges, tape and specks were 0.05-0.4x or 2.5x, or strips.
_AREA_RATIO = (0.45, 2.0)
_MAX_ELONGATION = 2.0


@dataclass(frozen=True)
class CeilingCamera:
    """A ceiling camera's undistorted pinhole model: map frame -> pixel."""

    camera_matrix: NDArray[np.float64]
    rvec: NDArray[np.float64]
    tvec: NDArray[np.float64]

    def project(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map-frame points (N, 3) -> pixels (N, 2)."""
        px, _ = cv.projectPoints(
            np.asarray(points, np.float64).reshape(-1, 1, 3),
            self.rvec,
            self.tvec,
            self.camera_matrix,
            np.zeros(5),  # frames are already undistorted
        )
        return np.asarray(px, np.float64).reshape(-1, 2)

    def to_floor(self, u: float, v: float, z: float) -> NDArray[np.float64]:
        """Pixel -> the map-frame (x, y) where its ray meets the plane at height z."""
        rot: Any = cv.Rodrigues(self.rvec)[0]
        centre = -rot.T @ self.tvec
        ray = rot.T @ np.linalg.inv(self.camera_matrix) @ np.array([u, v, 1.0])
        return (centre + (z - centre[2]) / ray[2] * ray)[:2]


@dataclass(frozen=True)
class CubeFit:
    """A fitted cube pose (map frame) and how well its silhouette matched."""

    x: float
    y: float
    # Radians in [-pi/4, pi/4): a cube looks the same every 90 degrees.
    yaw: float
    iou: float
    num_views: int
    # Per-camera fits (camera index, x, y, yaw, iou) and their spread; a large spread
    # means the cameras' extrinsics disagree where the cube is.
    per_camera: tuple[tuple[int, float, float, float, float], ...] = ()
    camera_disagreement_m: float = 0.0
    # The camera whose fit was kept (-1 when fused).
    camera: int = -1


def load_ceiling_cameras(serials: Sequence[str]) -> list[CeilingCamera]:
    """Calibrated models for the ceiling cameras, in stitched-image order."""
    cameras = []
    for serial in serials:
        _, _, camera_matrix, _ = get_camera_params(serial)
        rvec, tvec = load_extrinsics(serial)
        cameras.append(CeilingCamera(np.asarray(camera_matrix, np.float64), rvec, tvec))
    return cameras


def in_calibrated_region(
    camera: CeilingCamera, index: int, x: float, y: float, serials: Sequence[str]
) -> bool:
    """Whether the floor point (x, y) projects inside the camera's calibrated image
    region (``camera_corners`` in ``camera_params/<lab>/<serial>.json``), where its
    extrinsics were fitted. Unknown regions count as calibrated."""
    path = next(CAMERA_PARAMS_DIR.glob(f"*/{serials[index]}.json"), None)
    if path is None:
        return True
    with open(path, encoding="utf-8") as f:
        corners = np.array(json.load(f)["camera_corners"], np.float32)
    u, v = camera.project(np.array([[x, y, 0.0]]))[0]
    return cv.pointPolygonTest(corners, (float(u), float(v)), False) >= 0


def cube_mask(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """1 where a pixel looks like the cube rather than the floor, holes filled.

    The printed digit on each face fails the colour test, so interior holes are filled.
    """
    lab: Any = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    mask: Any = ((lab[:, :, 2] < _B_MAX) & (lab[:, :, 0] > _L_MIN)).astype(np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, (1,), thickness=cv.FILLED)
    return filled.astype(np.uint8)


def cube_silhouette(
    camera: CeilingCamera, x: float, y: float, yaw: float, side: float
) -> NDArray[np.float64]:
    """The image polygon (convex hull) of a `side` cube resting on the floor at pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = _CORNERS * side
    world = np.c_[
        x + c * local[:, 0] - s * local[:, 1],
        y + s * local[:, 0] + c * local[:, 1],
        local[:, 2],
    ]
    hull = cv.convexHull(camera.project(world).astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float64)


def _camera_rows(frame: NDArray[np.uint8], num_cameras: int) -> int:
    return frame.shape[0] // num_cameras


def _in_bounds(xy: NDArray[np.float64], bounds: Sequence[float]) -> bool:
    return bool(bounds[0] < xy[0] < bounds[1] and bounds[2] < xy[1] < bounds[3])


def _excluded(xy: NDArray[np.float64], exclude: Sequence[Sequence[float]]) -> bool:
    return any(
        abs(xy[0] - ex) < half and abs(xy[1] - ey) < half for ex, ey, half in exclude
    )


def _cube_like(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    blob: NDArray[np.bool_],
    stat: NDArray[np.int32],
    camera: CeilingCamera,
    xy: NDArray[np.float64],
    side: float,
    index: int,
    rows: int,
    num_cameras: int,
) -> bool:
    """Whether a blob is the size and shape a cube makes at its spot on the floor.

    Each candidate costs a full pose fit, and the clutter in the lab (shelf edges,
    tape, specks) is never cube-sized and square-ish there. A blob cut by the stitch
    line is only held to the upper size bound, since part of the cube is hidden; one
    cut by the outer image border is rejected (the vignetted border passes the colour
    test in cube-sized pieces, and no camera sees a cube there whole)."""
    expected = [
        cv.contourArea(
            cube_silhouette(camera, xy[0], xy[1], yaw, side).astype(np.float32)
        )
        for yaw in (0.0, math.pi / 4)
    ]
    area = float(stat[cv.CC_STAT_AREA])
    if area > _AREA_RATIO[1] * max(expected):
        return False
    x, y, w, h = (
        int(stat[c])
        for c in (cv.CC_STAT_LEFT, cv.CC_STAT_TOP, cv.CC_STAT_WIDTH, cv.CC_STAT_HEIGHT)
    )
    if x <= 0 or x + w >= blob.shape[1] or y <= 0 or y + h >= blob.shape[0]:
        return False  # cut by the outer border: no camera sees the whole cube
    top, bottom = index * rows, (index + 1) * rows
    seam_cut = (index > 0 and y <= top + _SEAM_PX) or (
        index < num_cameras - 1 and y + h >= bottom - _SEAM_PX
    )
    if seam_cut:
        return True
    if area < _AREA_RATIO[0] * min(expected):
        return False
    points = cv.findNonZero(blob[y : y + h, x : x + w].astype(np.uint8))
    (_, _), (rw, rh), _ = cv.minAreaRect(points)
    return max(rw, rh) <= _MAX_ELONGATION * max(min(rw, rh), 1.0)


def find_cube_candidates(
    frame: NDArray[np.uint8],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    side: float,
    exclude: Sequence[Sequence[float]] = (),
) -> list[tuple[float, float]]:
    """Map-frame (x, y) of every cube-coloured, cube-sized blob in the drivable area."""
    rows = _camera_rows(frame, len(cameras))
    mask = cube_mask(frame)
    for i in range(1, len(cameras)):
        mask[i * rows - _SEAM_PX : i * rows + _SEAM_PX, :] = 0
    num, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, float]] = []
    for k in range(1, num):
        if not _AREA_RANGE[0] <= stats[k, cv.CC_STAT_AREA] <= _AREA_RANGE[1]:
            continue
        u, v = centroids[k]
        i = min(int(v // rows), len(cameras) - 1)
        xy = cameras[i].to_floor(u, v - i * rows, z=side / 2)
        if not _in_bounds(xy, floor_bounds) or _excluded(xy, exclude):
            continue
        if not _cube_like(
            labels == k, stats[k], cameras[i], xy, side, i, rows, len(cameras)
        ):
            continue
        if all(math.hypot(xy[0] - cx, xy[1] - cy) > 2 * side for cx, cy in candidates):
            candidates.append((float(xy[0]), float(xy[1])))
    return candidates


def fit_cube_pose(
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    guess_xy: tuple[float, float],
    side: float,
    only: Sequence[int] | None = None,
) -> CubeFit | None:
    """3-DoF silhouette fit around `guess_xy`; masks are averaged over `frames`
    first. ``only`` restricts the fit to those camera indices."""
    rows = _camera_rows(frames[0], len(cameras))
    views = []
    for i, cam in enumerate(cameras):
        if only is not None and i not in only:
            continue
        u, v = cube_silhouette(cam, *guess_xy, 0.0, side).mean(axis=0)
        if not (0 <= u < frames[0].shape[1] and 0 <= v < rows):
            continue
        u0, v0 = max(int(u) - _WINDOW, 0), max(int(v) - _WINDOW, 0)
        u1, v1 = int(u) + _WINDOW, int(v) + _WINDOW
        prob = np.mean(
            [cube_mask(f[i * rows : (i + 1) * rows][v0:v1, u0:u1]) for f in frames],
            axis=0,
        )
        # Rows at the stitch line are unknown: excluded from the score entirely, so a
        # silhouette crossing them is neither credited nor penalised there.
        valid = np.ones_like(prob)
        if i > 0:  # stitch line at the top of this camera's rows
            valid[: max(0, _SEAM_PX - v0), :] = 0
        if i < len(cameras) - 1:  # ... and at the bottom
            valid[max(0, rows - _SEAM_PX - v0) :, :] = 0
        prob = prob * valid
        if prob.sum() > 0:
            views.append((cam, u0, v0, prob, valid))
    if not views:
        return None

    def score(x: float, y: float, yaw: float) -> float:
        total = 0.0
        for cam, u0, v0, prob, valid in views:
            # Sub-pixel, anti-aliased rasterisation keeps the score smooth in the pose.
            sil = np.zeros(prob.shape, np.float32)
            poly = ((cube_silhouette(cam, x, y, yaw, side) - [u0, v0]) * 16).astype(
                np.int32
            )
            cv.fillConvexPoly(sil, poly, (1.0,), lineType=cv.LINE_AA, shift=4)
            sil = sil * valid
            inter = float((sil * prob).sum())
            total += inter / (float(prob.sum()) + float(sil.sum()) - inter)
        return total / len(views)

    best_score, best = -1.0, np.zeros(3)
    for yaw in np.deg2rad(np.arange(-45.0, 45.0, 3.0)):
        for dx in np.arange(-0.04, 0.0401, 0.005):
            for dy in np.arange(-0.04, 0.0401, 0.005):
                cand = np.array([guess_xy[0] + dx, guess_xy[1] + dy, yaw])
                sc = score(*cand)
                if sc > best_score:
                    best_score, best = sc, cand
    step = np.array([0.0025, 0.0025, math.radians(1.5)])
    while step[0] > 2e-4:  # coordinate descent with shrinking steps
        improved = False
        for k in range(3):
            for sign in (-1.0, 1.0):
                cand = best.copy()
                cand[k] += sign * step[k]
                sc = score(*cand)
                if sc > best_score:
                    best_score, best, improved = sc, cand, True
        if not improved:
            step = step / 2
    yaw = (best[2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return CubeFit(float(best[0]), float(best[1]), float(yaw), best_score, len(views))


def localize_cube(
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    side: float,
    exclude: Sequence[Sequence[float]] = (),
    min_iou: float = 0.6,
    fusion: str = "best_view",
    serials: Sequence[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> CubeFit | None:
    """The best-fitting cube in the drivable area, or None if nothing fits well
    enough.

    ``fusion="best_view"`` (default) fits each camera on its own and keeps the best;
    ``"fused"`` scores every camera that sees the cube together, which is only right
    when their calibrations agree. ``diagnostics``, when given, is filled with every
    candidate and its quick-screen fit.
    """
    screened: list[dict[str, Any]] = []
    if diagnostics is not None:
        diagnostics["candidates"] = screened
    passed: list[CubeFit] = []
    for guess in find_cube_candidates(frames[-1], cameras, floor_bounds, side, exclude):
        # Quick screen on the last frames, each camera on its own: in the overlap the
        # cameras' calibrations disagree by ~3 cm, so a fused score can fail a cube one
        # camera sees perfectly well.
        fits = [
            fit_cube_pose(frames[-3:], cameras, guess, side, only=[i])
            for i in range(len(cameras))
        ]
        fit = max((f for f in fits if f is not None), key=lambda f: f.iou, default=None)
        screened.append({"guess": list(guess), "iou": None if fit is None else fit.iou})
        if fit is not None and fit.iou >= min_iou:
            passed.append(fit)
    # Best screen first, but a winner that fails the full refit (a flicker present
    # only in the last frames) must not hide the cube behind it.
    for screen in sorted(passed, key=lambda f: -f.iou):
        fit = _refine(
            frames, cameras, (screen.x, screen.y), side, min_iou, fusion, serials
        )
        if fit is not None and _in_bounds(np.array([fit.x, fit.y]), floor_bounds):
            return fit
    return None


def _refine(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    guess: tuple[float, float],
    side: float,
    min_iou: float,
    fusion: str,
    serials: Sequence[str] | None,
) -> CubeFit | None:
    """The full fit over every frame around a screened guess, or None if it fails."""
    singles = []
    for i in range(len(cameras)):
        fit = fit_cube_pose(frames, cameras, guess, side, only=[i])
        if fit is not None:
            singles.append((i, fit))
    if not singles:
        return None
    per_camera = tuple((i, f.x, f.y, f.yaw, f.iou) for i, f in singles)
    spread = max(
        (math.hypot(a.x - b.x, a.y - b.y) for _, a in singles for _, b in singles),
        default=0.0,
    )
    cam_i = -1
    if fusion == "fused":
        chosen = fit_cube_pose(frames, cameras, guess, side)
    else:
        # The cameras' calibrations need not agree, so blending their views can match
        # neither image: keep the single best view. A near tie goes to the camera that
        # sees the cube more directly from above (less side face, less perspective).
        # Prefer cameras whose calibrated region contains the cube (their extrinsics
        # were fitted there); among those, the best fit, a near tie going to the
        # camera that sees the cube more directly from above.
        # Only cameras that see the cube well enough compete: a calibrated camera
        # with a poor view must not veto a good one.
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
        top_iou = max(f.iou for _, f in pool)
        near = [(i, f) for i, f in pool if f.iou >= top_iou - 0.03]
        cam_i, chosen = min(
            near, key=lambda t: _off_nadir(cameras[t[0]], t[1].x, t[1].y)
        )
    if chosen is None or chosen.iou < min_iou:
        return None
    return CubeFit(
        chosen.x,
        chosen.y,
        chosen.yaw,
        chosen.iou,
        chosen.num_views,
        per_camera,
        spread,
        cam_i,
    )


def find_cube_in_box(
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    box: Sequence[float],
    min_area: float = 40.0,
) -> tuple[float, float] | None:
    """Where the cube lies inside the floor box ``box`` (x_min, x_max, y_min, y_max),
    e.g. the bin, or None.

    Inside the bin the cube is in shadow and beyond the floor search (run 31), so this
    tests hue only, within the box's image: the largest blue blob of the last frame, in
    the camera that sees it largest, projected to the cube's centre height.
    """
    x0, x1, y0, y1 = (float(v) for v in box)
    corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    frame = frames[-1]
    rows = _camera_rows(frame, len(cameras))
    best_area, best_xy = 0.0, None
    for i, camera in enumerate(cameras):
        tile = frame[i * rows : (i + 1) * rows]
        region = np.zeros(tile.shape[:2], np.uint8)
        cv.fillConvexPoly(region, camera.project(corners).astype(np.int32), (1,))
        hsv = np.asarray(cv.cvtColor(tile, cv.COLOR_BGR2HSV))
        blue = cv.inRange(hsv, np.array([90, 40, 30]), np.array([135, 255, 255]))
        blue = np.asarray(blue, np.uint8) & region
        n, _, stats, centres = cv.connectedComponentsWithStats(blue)
        for j in range(1, n):
            area = float(stats[j, cv.CC_STAT_AREA])
            if area < max(min_area, best_area):
                continue
            x, y = camera.to_floor(float(centres[j][0]), float(centres[j][1]), 0.025)
            if x0 <= x <= x1 and y0 <= y <= y1:
                best_area, best_xy = area, (float(x), float(y))
    return best_xy


def _off_nadir(camera: CeilingCamera, x: float, y: float) -> float:
    """Horizontal distance from the point under the camera to (x, y)."""
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    centre = -rot.T @ camera.tvec
    return float(math.hypot(x - centre[0], y - centre[1]))


class CeilingCubeTargetSource(TargetSource):
    """The floor cube's (x, y, z) and yaw from the ceiling cameras, no markers needed.

    Reads ``num_frames`` stitched full-res frames from the ceiling image publisher and
    the robot's map pose from the marker detector (to ignore the robot's own footprint),
    then runs :func:`localize_cube`. ``get_target`` raises if no cube is found, so a
    rollout never runs on a stale or invented pose.
    """

    def __init__(
        self,
        serials: Sequence[str],
        image_client: Any = None,
        marker_client: Any = None,
        side: float = 0.05,
        floor_bounds: Sequence[float] = (-1.5, 1.7, -1.5, 1.5),
        robot_exclusion_half: float = 0.45,
        num_frames: int = 20,
        frame_period_s: float = 0.12,
        min_iou: float = 0.6,
        host: str | None = None,
        fusion: str = "best_view",
        warn_disagreement_m: float = 0.01,
        failure_dir: str | Path | None = None,
        retries: int = 1,
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
        self._side = float(side)
        self._floor_bounds = tuple(floor_bounds)
        self._robot_half = float(robot_exclusion_half)
        self._num_frames = int(num_frames)
        self._period = float(frame_period_s)
        self._min_iou = float(min_iou)
        self._fusion = fusion
        self._warn_m = float(warn_disagreement_m)
        self._failure_dir = None if failure_dir is None else Path(failure_dir)
        self._retries = int(retries)
        self.last_fit: CubeFit | None = None

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
            raise RuntimeError("CeilingCubeTargetSource: no ceiling image arrived")
        return frames

    def _exclusions(self) -> list[tuple[float, float, float]]:
        poses = self._markers.get_latest().get("poses") or {}
        return [(float(p[0]), float(p[1]), self._robot_half) for p in poses.values()]

    def localize(self) -> CubeFit:
        """Run the localizer now and return the fit (raises if there is no cube).

        A miss is retried with fresh frames; each miss saves its frames and inputs
        under ``failure_dir`` so it can be replayed."""
        fit = None
        for _ in range(1 + self._retries):
            frames, exclude = self._frames(), self._exclusions()
            diagnostics: dict[str, Any] = {}
            fit = localize_cube(
                frames,
                self._cameras,
                self._floor_bounds,
                self._side,
                exclude=exclude,
                min_iou=self._min_iou,
                fusion=self._fusion,
                serials=self._serials,
                diagnostics=diagnostics,
            )
            if fit is not None:
                break
            self._save_failure(frames, exclude, diagnostics)
        if fit is None:
            raise RuntimeError(
                "CeilingCubeTargetSource: no cube found in the drivable area "
                f"{self._floor_bounds}"
            )
        if fit.camera_disagreement_m > self._warn_m:
            _logger.warning(
                "Ceiling cameras disagree by %.1f mm on the cube (%s); check the "
                "extrinsics calibration.",
                1000 * fit.camera_disagreement_m,
                ", ".join(
                    f"cam {i}: ({x:+.3f}, {y:+.3f})" for i, x, y, _, _ in fit.per_camera
                ),
            )
        self.last_fit = fit
        return fit

    def find_in_box(self, box: Sequence[float]) -> tuple[float, float] | None:
        """Where the cube lies inside the floor box ``box`` (the bin), or None (see
        :func:`find_cube_in_box`)."""
        return find_cube_in_box(self._frames()[-3:], self._cameras, box)

    def _save_failure(
        self,
        frames: Sequence[NDArray[np.uint8]],
        exclude: Sequence[Sequence[float]],
        diagnostics: dict[str, Any],
    ) -> None:
        if self._failure_dir is None:
            return
        stamp = time.strftime("%H-%M-%S")
        n = len(list(self._failure_dir.glob("*"))) if self._failure_dir.exists() else 0
        out = self._failure_dir / f"{n:02d}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv.imwrite(str(out / f"frame_{i:02d}.png"), frame)
        info = {
            "time": time.time(),
            "floor_bounds": list(self._floor_bounds),
            "exclusions": [list(e) for e in exclude],
            "min_iou": self._min_iou,
            "side": self._side,
            "serials": self._serials,
            **diagnostics,
        }
        (out / "localization.json").write_text(json.dumps(info, indent=1))
        _logger.warning("No cube found; saved the frames and inputs to %s.", out)

    def get_target(self) -> tuple[float, float, float]:
        fit = self.localize()
        return (fit.x, fit.y, self._side / 2)

    def get_yaw(self) -> float:
        if self.last_fit is None:
            self.localize()
        assert self.last_fit is not None
        return self.last_fit.yaw

    def alternates(self) -> list[tuple[float, float, float]]:
        """The other cameras' estimates of the cube from the last localization, most
        confident first, where they differ from the chosen one by over 1 cm (the
        cameras disagree by ~3 cm in their overlap)."""
        fit = self.last_fit
        if fit is None:
            return []
        others = sorted(
            (p for p in fit.per_camera if p[0] != fit.camera), key=lambda p: -p[4]
        )
        return [
            (x, y, yaw)
            for _, x, y, yaw, _ in others
            if math.hypot(x - fit.x, y - fit.y) > 0.01
        ]

    def close(self) -> None:
        self._images.close()
        self._markers.close()
