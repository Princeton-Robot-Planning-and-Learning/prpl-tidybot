"""The batched pose grid in :func:`fit_cube_pose` returns exactly what the one-pose- at-
a-time search it replaced (10091e0, copied below as the reference) returned: the same
pose, bit for bit, and the same IoU."""

import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.cube_localizer import (
    _SEAM_PX,
    _WINDOW,
    CubeFit,
    _camera_rows,
    cube_mask,
    cube_silhouette,
    find_cube_candidates,
    fit_cube_pose,
    load_ceiling_cameras,
)

SERIALS = ("AD8A29AE", "5A5C29AE")
SIDE = 0.05
H, W = 720, 1280
DATA = Path(__file__).resolve().parent / "data" / "bin_rim"
FLOOR_BOUNDS = (-1.846, 2.046, -1.846, 1.846)


def _reference_fit(frames, cameras, guess_xy, side, only=None):
    """``fit_cube_pose`` as of 10091e0: one ``score`` call per grid pose."""
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
        valid = np.ones_like(prob)
        if i > 0:
            valid[: max(0, _SEAM_PX - v0), :] = 0
        if i < len(cameras) - 1:
            valid[max(0, rows - _SEAM_PX - v0) :, :] = 0
        prob = prob * valid
        if prob.sum() > 0:
            views.append((cam, u0, v0, prob, valid))
    if not views:
        return None

    def score(x, y, yaw):
        total = 0.0
        for cam, u0, v0, prob, valid in views:
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
    while step[0] > 2e-4:
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


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    return load_ceiling_cameras(SERIALS)


def _lab_to_bgr(lab):
    px = np.array([[lab]], np.uint8)
    return tuple(int(v) for v in cv.cvtColor(px, cv.COLOR_LAB2BGR)[0, 0])


def _render(cameras, cubes, seed):
    floor = _lab_to_bgr((206, 138, 148))
    cube = _lab_to_bgr((185, 135, 126))
    rng = np.random.default_rng(seed)
    halves = []
    for cam in cameras:
        img = np.full((H, W, 3), floor, np.float32)
        for x, y, yaw in cubes:
            poly = cube_silhouette(cam, x, y, yaw, SIDE).astype(np.int32)
            cv.fillConvexPoly(img, poly, cube, lineType=cv.LINE_AA)
        img += rng.normal(0, 3.0, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


def _assert_same(frames, cameras, guess):
    for only in ([0], [1], None):
        new = fit_cube_pose(frames, cameras, guess, SIDE, only=only)
        old = _reference_fit(frames, cameras, guess, SIDE, only=only)
        assert new == old, (guess, only, new, old)


@pytest.mark.parametrize(
    "pose, guess",
    [
        ((-1.09, 1.07, math.radians(-19.5)), (-1.08, 1.06)),
        ((0.78, 1.15, 0.0), (0.79, 1.13)),
        ((-0.8, -1.0, math.radians(30)), (-0.8, -1.0)),
        ((-0.8, -0.01, math.radians(10)), (-0.79, -0.02)),  # on the stitch line
        ((-0.184, 0.13, 0.1), (-0.18, 0.14)),
    ],
)
def test_the_batched_fit_equals_the_one_pose_at_a_time_fit(cameras, pose, guess):
    """Rendered cubes (noise, both cameras, the stitch line): identical fits."""
    frames = [_render(cameras, [pose], seed=s) for s in range(3)]
    _assert_same(frames, cameras, guess)


@pytest.mark.parametrize("name", ["close_18-10-36_08", "far_shelf_10-37-51_00"])
def test_the_batched_fit_equals_the_old_fit_on_real_frames(cameras, name):
    """Saved real ceiling frames: every candidate blob, and a guess on the floor with
    nothing there, fit identically."""
    frame = cv.imread(str(DATA / f"{name}.jpg"))
    assert frame is not None
    guesses = find_cube_candidates(frame, cameras, FLOOR_BOUNDS, SIDE)
    for guess in [*guesses, (0.1, -0.5)]:
        _assert_same([frame], cameras, guess)
