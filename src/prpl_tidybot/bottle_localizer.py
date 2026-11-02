"""Localize a bottle lying on the floor from the ceiling cameras, by its label.

The bottle is clear plastic, which from the ceiling is barely distinguishable from the
floor, but its label is a saturated red band (2026-11-02, a 20 oz vitaminwater: hue
177-179 and 0-2, saturation ~245, against a floor at hue ~10 and saturation ~92). So
the label is what is found and fitted: a cylinder of the bottle's diameter and the
label's length lying on the floor, whose pose has three degrees of freedom (x, y and
the axis's yaw, which repeats every 180 degrees). Its silhouette in a calibrated camera
is approximated by the convex hull of the bounding box's eight corners, as for the
cube and the bin; the fit maximises its IoU with the label mask, averaged over frames.

The fitted point is the label's centre, which is where a grasp across the bottle
should close. A standing bottle is not handled: from above its label is a thin ring.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
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

# Unit box corners, centred in x/y, bottom face on z = 0.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)]
)
_SEAM_PX = 8
# A blob's area relative to the label's silhouette at its spot (either yaw).
_AREA_RATIO = (0.4, 2.0)
# Pixels of floor kept around the label's silhouette when fitting.
_WINDOW_PX = 40
# Two candidates closer than this (m) are the same bottle.
_SAME_M = 0.1


@dataclass(frozen=True)
class BottleModel:
    """The bottle and its label (m), and the label's colour (OpenCV HSV).

    The defaults are a 20 oz vitaminwater: ~0.20 m long and 0.066 m across (retail
    listing), the label measured 0.10 m long in the calibrated ceiling camera. Red
    wraps around hue 0, so ``hue_ranges`` holds one band on each side of it.
    """

    length: float = 0.20
    diameter: float = 0.066
    label_length: float = 0.10
    hue_ranges: tuple[tuple[int, int], ...] = ((0, 8), (170, 180))
    min_saturation: int = 120
    min_value: int = 60


@dataclass(frozen=True)
class BottleFit:
    """A fitted label pose (map frame) and how well its silhouette matched."""

    x: float
    y: float
    # The bottle's axis, radians in [-pi/2, pi/2): a label looks the same end to end.
    yaw: float
    iou: float
    # Per-camera fits (camera index, x, y, yaw, iou) and their spread.
    per_camera: tuple[tuple[int, float, float, float, float], ...] = ()
    camera_disagreement_m: float = 0.0
    # The camera whose fit was kept (-1 when none was chosen).
    camera: int = -1


def bottle_mask(
    bgr: NDArray[np.uint8], model: BottleModel = BottleModel()
) -> NDArray[np.uint8]:
    """1 where a pixel has the label's colour, holes (white print) filled."""
    hsv: Any = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    mask: Any = np.zeros(hue.shape, bool)
    for lo, hi in model.hue_ranges:
        mask |= (hue >= lo) & (hue <= hi)
    mask &= (hsv[:, :, 1] >= model.min_saturation) & (hsv[:, :, 2] >= model.min_value)
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3)))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, (1,), thickness=cv.FILLED)
    return filled.astype(np.uint8)


def label_silhouette(
    camera: CeilingCamera, x: float, y: float, yaw: float, model: BottleModel
) -> NDArray[np.float64]:
    """The image polygon (convex hull) of the label of a bottle lying at pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = _CORNERS * [model.label_length, model.diameter, model.diameter]
    world = np.c_[
        x + c * local[:, 0] - s * local[:, 1],
        y + s * local[:, 0] + c * local[:, 1],
        local[:, 2],
    ]
    hull = cv.convexHull(camera.project(world).astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float64)


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


def find_bottle_candidates(
    frame: NDArray[np.uint8],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    model: BottleModel,
    exclude: Sequence[Sequence[float]] = (),
) -> list[tuple[float, float]]:
    """Map-frame (x, y) of every label-coloured, label-sized blob in ``floor_bounds``,
    largest first."""
    found: list[tuple[float, float, float]] = []
    for i, (cam, img) in enumerate(zip(cameras, _split(frame, len(cameras)))):
        rows, cols = img.shape[:2]
        mask = bottle_mask(img, model)
        mask *= _valid_rows(rows, cols, i, len(cameras)).astype(np.uint8)
        num, _, stats, centroids = cv.connectedComponentsWithStats(mask)
        for k in range(1, num):
            u, v = centroids[k]
            x, y = cam.to_floor(float(u), float(v), model.diameter / 2)
            if not (
                floor_bounds[0] <= x <= floor_bounds[1]
                and floor_bounds[2] <= y <= floor_bounds[3]
            ) or _excluded(x, y, exclude):
                continue
            expected = [
                cv.contourArea(
                    label_silhouette(cam, x, y, yaw, model).astype(np.float32)
                )
                for yaw in (0.0, math.pi / 2)
            ]
            area = float(stats[k, cv.CC_STAT_AREA])
            if _AREA_RATIO[0] * min(expected) <= area <= _AREA_RATIO[1] * max(expected):
                found.append((area, float(x), float(y)))
    candidates: list[tuple[float, float]] = []
    for _, x, y in sorted(found, reverse=True):
        if all(math.hypot(x - cx, y - cy) > _SAME_M for cx, cy in candidates):
            candidates.append((x, y))
    return candidates


def fit_bottle_pose(  # pylint: disable=too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    camera: CeilingCamera,
    index: int,
    num_cameras: int,
    guess_xy: tuple[float, float],
    model: BottleModel,
) -> BottleFit | None:
    """3-DoF silhouette fit of the label in one camera around ``guess_xy``; masks are
    averaged over ``frames`` first."""
    imgs = [_split(f, num_cameras)[index] for f in frames]
    rows, cols = imgs[0].shape[:2]
    u, v = camera.project(np.array([[*guess_xy, model.diameter / 2]]))[0]
    if not (0 <= u < cols and 0 <= v < rows):
        return None
    u0, v0 = max(int(u) - _WINDOW_PX, 0), max(int(v) - _WINDOW_PX, 0)
    u1, v1 = min(int(u) + _WINDOW_PX, cols), min(int(v) + _WINDOW_PX, rows)
    prob = np.mean([bottle_mask(img, model)[v0:v1, u0:u1] for img in imgs], axis=0)
    valid = _valid_rows(rows, cols, index, num_cameras)[v0:v1, u0:u1]
    prob = prob * valid
    if prob.sum() == 0:
        return None

    def score(x: float, y: float, yaw: float) -> float:
        sil = np.zeros(prob.shape, np.float32)
        poly = ((label_silhouette(camera, x, y, yaw, model) - [u0, v0]) * 16).astype(
            np.int32
        )
        cv.fillConvexPoly(sil, poly, (1.0,), lineType=cv.LINE_AA, shift=4)
        sil = sil * valid
        inter = float((sil * prob).sum())
        union = float(prob.sum()) + float(sil.sum()) - inter
        return inter / union if union > 0 else 0.0

    best_score, best = -1.0, np.zeros(3)
    for yaw in np.deg2rad(np.arange(-90.0, 90.0, 7.5)):
        for dx in np.arange(-0.03, 0.0301, 0.01):
            for dy in np.arange(-0.03, 0.0301, 0.01):
                cand = np.array([guess_xy[0] + dx, guess_xy[1] + dy, yaw])
                sc = score(*cand)
                if sc > best_score:
                    best_score, best = sc, cand
    step = np.array([0.004, 0.004, math.radians(3.0)])
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
    yaw = (best[2] + math.pi / 2) % math.pi - math.pi / 2
    return BottleFit(float(best[0]), float(best[1]), float(yaw), best_score)


def _off_nadir(camera: CeilingCamera, x: float, y: float) -> float:
    """Horizontal distance from the point under the camera to (x, y)."""
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    centre = -rot.T @ camera.tvec
    return float(math.hypot(x - centre[0], y - centre[1]))


def localize_bottle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    model: BottleModel,
    floor_bounds: Sequence[float],
    exclude: Sequence[Sequence[float]] = (),
    min_iou: float = 0.6,
    serials: Sequence[str] | None = None,
) -> BottleFit | None:
    """The best-fitting bottle label in ``floor_bounds``, or None if nothing fits well
    enough.

    Each camera is fitted on its own (their calibrations disagree by a few cm where
    they overlap) and, as for the bin, a near tie goes to a camera whose calibrated
    region holds the label, then to the camera that sees it more directly from above.
    """
    best: BottleFit | None = None
    for guess in find_bottle_candidates(
        frames[-1], cameras, floor_bounds, model, exclude
    ):
        singles = [
            (i, fit)
            for i, cam in enumerate(cameras)
            if (fit := fit_bottle_pose(frames, cam, i, len(cameras), guess, model))
            is not None
            and fit.iou >= min_iou
        ]
        if not singles:
            continue
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
        cam_i, chosen = min(
            near, key=lambda t: _off_nadir(cameras[t[0]], t[1].x, t[1].y)
        )
        if not (
            floor_bounds[0] <= chosen.x <= floor_bounds[1]
            and floor_bounds[2] <= chosen.y <= floor_bounds[3]
        ) or _excluded(chosen.x, chosen.y, exclude):
            continue
        spread = max(
            (math.hypot(a.x - b.x, a.y - b.y) for _, a in singles for _, b in singles),
            default=0.0,
        )
        fit = BottleFit(
            chosen.x,
            chosen.y,
            chosen.yaw,
            chosen.iou,
            tuple((i, f.x, f.y, f.yaw, f.iou) for i, f in singles),
            spread,
            cam_i,
        )
        if best is None or fit.iou > best.iou:
            best = fit
    return best


class CeilingBottleSource(TargetSource):
    """The bottle's label from the ceiling cameras, as ``CeilingCubeTargetSource`` is
    the cube.

    Reads ``num_frames`` stitched frames and the robot's map pose (to ignore the robot's
    own footprint), then runs :func:`localize_bottle`; raises if no bottle is found.
    ``get_target`` is the label's centre at the bottle's axis height, ``get_yaw`` the
    bottle's axis.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        serials: Sequence[str],
        image_client: Any = None,
        marker_client: Any = None,
        floor_bounds: Sequence[float] = (-1.5, 1.7, -1.5, 1.5),
        length: float = 0.20,
        diameter: float = 0.066,
        label_length: float = 0.10,
        robot_exclusion_half: float = 0.45,
        num_frames: int = 10,
        frame_period_s: float = 0.12,
        min_iou: float = 0.6,
        host: str | None = None,
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
        self._model = BottleModel(float(length), float(diameter), float(label_length))
        self._bounds = tuple(float(v) for v in floor_bounds)
        self._robot_half = float(robot_exclusion_half)
        self._num_frames = int(num_frames)
        self._period = float(frame_period_s)
        self._min_iou = float(min_iou)
        self.last_fit: BottleFit | None = None

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
            raise RuntimeError("CeilingBottleSource: no ceiling image arrived")
        return frames

    def _exclusions(self) -> list[tuple[float, float, float]]:
        poses = self._markers.get_latest().get("poses") or {}
        return [(float(p[0]), float(p[1]), self._robot_half) for p in poses.values()]

    def localize(self) -> BottleFit:
        """Run the localizer now and return the fit (raises if there is no bottle)."""
        fit = localize_bottle(
            self._frames(),
            self._cameras,
            self._model,
            self._bounds,
            exclude=self._exclusions(),
            min_iou=self._min_iou,
            serials=self._serials,
        )
        if fit is None:
            raise RuntimeError(
                f"CeilingBottleSource: no bottle found in the floor box {self._bounds}"
            )
        self.last_fit = fit
        return fit

    def get_target(self) -> tuple[float, float, float]:
        fit = self.localize()
        return (fit.x, fit.y, self._model.diameter / 2)

    def get_yaw(self) -> float:
        """The bottle's axis from the last fit (localizes first if there is none)."""
        fit = self.last_fit if self.last_fit is not None else self.localize()
        return fit.yaw

    def alternates(self) -> list[tuple[float, float, float]]:
        """The other cameras' estimates of the bottle from the last localization, most
        confident first, where they differ from the chosen one by over 1 cm: what the
        pick retries at after a miss, as for the cube."""
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
        """Close the image and marker clients."""
        self._images.close()
        self._markers.close()
