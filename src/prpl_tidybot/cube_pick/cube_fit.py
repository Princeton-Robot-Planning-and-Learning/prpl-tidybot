"""Fit a cube of known size to the depth points inside its mask.

This is the geometric half of the "SAM 2 mask + depth fit" perception backend: given
the world-frame points of the pixels a segmenter assigned to the cube, find the cube's
6-DOF pose. Two stages:

1. **Initialise from the dominant face.** RANSAC finds the largest plane among the
   points nearest their median (so a few bin-floor or wall pixels leaking through the
   mask cannot win); the face's in-plane orientation comes from the minimum-area
   rectangle around its inliers. The cube centre is half an edge behind that face.
2. **Refine all six degrees of freedom** by robust least squares on each point's
   distance to the cube's surface. Points on two or three visible faces pin the pose
   down fully; points on one face still pin it to within that face's visible extent.

A cube is 24-fold symmetric, so the reported yaw is the heading of the cube's
horizontal edges modulo 90 degrees, and `up_alignment` says how close to face-down the
cube rests (1.0 = flat on a face).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]

MIN_POINTS = 30


@dataclass(frozen=True)
class SupportSurface:
    """A horizontal surface the cube may rest on, from the robot's map: the floor (no
    footprint: everywhere) or a container's floor (an oriented rectangle)."""

    z: float
    centre: tuple[float, float] | None = None
    half_extents: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0

    def contains(self, xy: FloatArray) -> NDArray[np.bool_]:
        """Which of the (N, 2) points lie over this surface."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if self.centre is None:
            return np.ones(len(xy), bool)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = xy - np.asarray(self.centre)
        local_x = c * d[:, 0] + s * d[:, 1]
        local_y = -s * d[:, 0] + c * d[:, 1]
        return (np.abs(local_x) <= self.half_extents[0]) & (
            np.abs(local_y) <= self.half_extents[1]
        )


def as_supports(
    supports: float | SupportSurface | Iterable[float | SupportSurface] | None,
) -> list[SupportSurface]:
    """Normalise floats (global heights) and SupportSurfaces into SupportSurfaces."""
    if supports is None:
        return []
    if isinstance(supports, (int, float, SupportSurface)):
        supports = [supports]
    return [
        s if isinstance(s, SupportSurface) else SupportSurface(float(s))
        for s in supports
    ]


def height_above_supports(
    points: FloatArray, supports: Sequence[SupportSurface]
) -> FloatArray:
    """Each point's height above the highest support under it that it is not below."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    above = np.full(len(points), np.inf)
    for support in supports:
        rise = points[:, 2] - support.z
        ok = support.contains(points[:, :2]) & (rise > -0.005)
        above = np.where(ok & (rise < above), rise, above)
    return above


class CubeFitError(RuntimeError):
    """The points do not support a cube fit."""


@dataclass(frozen=True)
class CubeFit:
    """A fitted cube: centre, orientation and quality figures."""

    position: FloatArray
    rotation: FloatArray
    yaw: float
    up_alignment: float
    rms_residual: float
    inlier_fraction: float
    num_points: int

    @property
    def pose(self) -> FloatArray:
        """4x4 world-from-cube transform."""
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = self.position
        return pose


def yaw_error_mod_90(a: float, b: float) -> float:
    """Absolute difference of two yaws, modulo the cube's 90-degree symmetry."""
    diff = (a - b) % (np.pi / 2)
    return float(min(diff, np.pi / 2 - diff))


def cube_yaw_and_up(
    rotation: FloatArray, up: FloatArray = np.array([0.0, 0.0, 1.0])
) -> tuple[float, float]:
    """Yaw (mod 90 deg, in [-45, 45)) of the cube's horizontal edges, and how aligned
    its most vertical axis is with `up`."""
    alignment = np.abs(rotation.T @ up)
    vertical = int(np.argmax(alignment))
    horizontal = rotation[:, (vertical + 1) % 3]
    yaw = float(np.arctan2(horizontal[1], horizontal[0]))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return yaw, float(alignment[vertical])


def fit_cube(
    *,
    points: FloatArray,
    size: float,
    viewpoint: FloatArray,
    up: FloatArray = np.array([0.0, 0.0, 1.0]),
    seed: int = 0,
    support_z: float | SupportSurface | Iterable[float | SupportSurface] | None = None,
) -> CubeFit:
    """Fit a cube of edge `size` to world-frame `points` seen from `viewpoint`.

    `support_z` gives the heights of known horizontal support surfaces (the floor, a
    container's floor), from the robot's map: points lying on one are dropped as mask
    leakage, and a fit whose centre sinks below the highest support under it is rejected
    (a cube "under" the floor explains floor points as well as a real one).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    supports = as_supports(support_z)
    for support in supports:
        on = support.contains(points[:, :2]) & (
            np.abs(points[:, 2] - support.z) <= 0.004
        )
        points = points[~on]
    if len(points) < MIN_POINTS:
        raise CubeFitError(f"only {len(points)} depth points on the cube")
    # Keep the points that can belong to one cube around the robust centre.
    centre = np.median(points, axis=0)
    near = np.linalg.norm(points - centre, axis=1) < size * 1.2
    cluster = points[near]
    if len(cluster) < MIN_POINTS:
        raise CubeFitError(f"only {len(cluster)} depth points cluster like a cube")

    init = _initial_pose(cluster, size, viewpoint, np.random.default_rng(seed))
    rotation, position = _refine(cluster, size, init)
    median_z = float(np.median(cluster[:, 2]))
    below = [
        sup.z
        for sup in supports
        if sup.z < median_z and sup.contains(position[None, :2])[0]
    ]
    if below and position[2] < max(below) + size / 2 - 0.01:
        raise CubeFitError(
            f"fit centre z={position[2]:.3f} sinks below the support "
            f"at z={max(below):.3f}"
        )
    residuals = np.abs(_surface_distance(cluster, rotation, position, size))
    inliers = residuals < 0.004
    yaw, alignment = cube_yaw_and_up(rotation, up)
    return CubeFit(
        position=position,
        rotation=rotation,
        yaw=yaw,
        up_alignment=alignment,
        rms_residual=(
            float(np.sqrt(np.mean(residuals[inliers] ** 2)))
            if inliers.any()
            else float("inf")
        ),
        inlier_fraction=float(inliers.mean()),
        num_points=int(len(cluster)),
    )


def _initial_pose(
    points: FloatArray, size: float, viewpoint: FloatArray, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    best_inliers: NDArray[np.bool_] | None = None
    best_normal = np.array([0.0, 0.0, 1.0])
    threshold = max(0.0025, size * 0.05)
    for _ in range(200):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((points - sample[0]) @ normal) < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_normal = inliers, normal
    assert best_inliers is not None
    face = points[best_inliers]
    # Least-squares plane through the inliers, normal facing the camera.
    face_centre = face.mean(axis=0)
    _, _, vt = np.linalg.svd(face - face_centre)
    normal = vt[2]
    if normal @ (viewpoint - face_centre) < 0:
        normal = -normal
    if normal @ best_normal < 0:
        best_normal = -best_normal
    e1 = vt[0] - (vt[0] @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    planar = np.stack([(face - face_centre) @ e1, (face - face_centre) @ e2], axis=1)
    (rx, ry), _, angle = cv.minAreaRect(planar.astype(np.float32))
    theta = np.radians(angle)
    u = np.cos(theta) * e1 + np.sin(theta) * e2
    v = np.cross(normal, u)
    rect_centre = face_centre + rx * e1 + ry * e2
    rotation = np.stack([u, v, normal], axis=1)
    position = rect_centre - normal * size / 2
    return rotation, position


def _refine(
    points: FloatArray, size: float, init: tuple[FloatArray, FloatArray]
) -> tuple[FloatArray, FloatArray]:
    rotation0, position0 = init

    def residual(params: FloatArray) -> FloatArray:
        rotation = Rotation.from_rotvec(params[:3]).as_matrix() @ rotation0
        return _surface_distance(points, rotation, position0 + params[3:], size)

    result = least_squares(
        residual,
        np.zeros(6),
        loss="soft_l1",
        f_scale=0.002,
        x_scale=[0.1] * 3 + [0.01] * 3,
    )
    rotation = Rotation.from_rotvec(result.x[:3]).as_matrix() @ rotation0
    return rotation, position0 + result.x[3:]


def _surface_distance(
    points: FloatArray, rotation: FloatArray, position: FloatArray, size: float
) -> FloatArray:
    local = (points - position) @ rotation
    q = np.abs(local) - size / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside  # type: ignore[no-any-return]


def sample_visible_cube_points(
    *,
    cube_pose: FloatArray,
    size: float,
    viewpoint: FloatArray,
    spacing: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Points on the faces of a cube that face `viewpoint` (for synthetic tests)."""
    half = size / 2
    ticks = np.arange(-half + spacing / 2, half, spacing)
    grid_a, grid_b = np.meshgrid(ticks, ticks)
    grid_a, grid_b = grid_a.ravel(), grid_b.ravel()
    out = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            local = np.zeros((len(grid_a), 3))
            others = [i for i in range(3) if i != axis]
            local[:, axis] = sign * half
            local[:, others[0]] = grid_a
            local[:, others[1]] = grid_b
            normal = cube_pose[:3, :3][:, axis] * sign
            face_centre = cube_pose[:3, 3] + normal * half
            if normal @ (viewpoint - face_centre) <= 0:
                continue
            out.append(local @ cube_pose[:3, :3].T + cube_pose[:3, 3])
    points = np.vstack(out)
    return points[rng.permutation(len(points))]
