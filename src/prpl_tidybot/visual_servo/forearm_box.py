"""Fit an edge-supported forearm cross-section around the selected tape patch.

This is a local rectangular approximation of the taped housing, not general
anatomical recognition. Bounds scale from the existing tape/forearm template.
Weak edges, square/ambiguous axes, and ROI-clipped rectangles are rejected.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def project_pixels(pixels, projection: dict, height: float) -> np.ndarray:
    """Intersect calibrated camera rays with a horizontal map-height plane."""
    rotation, _ = cv2.Rodrigues(np.asarray(projection["rvec"], dtype=float))
    origin = -rotation.T @ np.asarray(projection["tvec"], dtype=float).reshape(3)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays = (
        rotation.T
        @ np.linalg.solve(
            np.asarray(projection["camera_matrix"]),
            np.c_[pixels, np.ones(len(pixels))].T,
        )
    ).T
    if np.any(np.abs(rays[:, 2]) < 1e-8):
        raise ValueError("Box rays parallel to height plane")
    distances = (height - origin[2]) / rays[:, 2]
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("Box plane behind camera")
    return origin + distances[:, None] * rays


def _score(parameters: np.ndarray, gx, gy) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, length, angle = parameters.T
    angle = np.deg2rad(angle)
    u, v = np.c_[np.cos(angle), np.sin(angle)], np.c_[-np.sin(angle), np.cos(angle)]
    center, t = np.c_[x, y], np.linspace(-0.42, 0.42, 14)
    edges = []
    for axis, other, extent, span in ((u, v, width, length), (v, u, length, width)):
        for sign in (-1, 1):
            points = (
                center[:, None, :] + sign * extent[:, None, None] / 2 * axis[:, None, :]
            )
            points = points + t[None, :, None] * span[:, None, None] * other[:, None, :]
            values = []
            # OpenCV remap dimensions must stay below 32767.
            for start in range(0, len(parameters), 10000):
                p = points[start : start + 10000].astype(np.float32)
                dx = cv2.remap(gx, p[:, :, 0], p[:, :, 1], cv2.INTER_LINEAR)
                dy = cv2.remap(gy, p[:, :, 0], p[:, :, 1], cv2.INTER_LINEAR)
                normal = axis[start : start + 10000]
                values.append(
                    np.abs(dx * normal[:, 0, None] + dy * normal[:, 1, None]).mean(1)
                )
            edges.append(np.concatenate(values))
    support = np.array(edges).T
    return 0.5 * support.mean(1) + 0.5 * support.min(1), support


def fit_box(image_rgb: np.ndarray, tape_uv, patch_size, search_xywh) -> dict:
    """Fit all four edges, anchoring identity to the already matched tape."""
    x, y, rw, rh = map(int, search_xywh)
    image = image_rgb[y : y + rh, x : x + rw]
    if image.shape[:2] != (rh, rw):
        raise ValueError("Forearm box search region outside image")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).astype(np.float32) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).astype(np.float32) / 8
    cx, cy = np.asarray(tape_uv) - [x, y]
    small, large = min(patch_size), max(patch_size)
    widths = range(max(10, round(small * 0.5)), round(small * 1.1) + 1, 2)
    lengths = range(max(14, round(small * 0.95)), round(large * 1.6) + 1, 2)
    parameters = np.array(
        [
            (a, b, w, length, angle)
            for a in np.arange(cx - 6, cx + 7, 2)
            for b in np.arange(cy - 6, cy + 7, 2)
            for w in widths
            for length in lengths
            if length >= 1.2 * w
            for angle in range(0, 180, 5)
        ],
        dtype=np.float32,
    )
    score, support = _score(parameters, gx, gy)
    best = int(np.argmax(score))
    a, b, width, length, angle = parameters[best]
    # A materially different axis should not explain the same edges equally well.
    difference = np.abs((parameters[:, 4] - angle + 90) % 180 - 90)
    alternate = float(score[difference >= 30].max())
    if support[best].min() < 5 or score[best] < 10 or score[best] - alternate < 1.5:
        raise RuntimeError(
            "Forearm box edges/short axis ambiguous; capture a clearer view"
        )
    refined = np.array(
        [
            (a + da, b + db, width + dw, length + dl, angle + dt)
            for da in (-1, -0.5, 0, 0.5, 1)
            for db in (-1, -0.5, 0, 0.5, 1)
            for dw in (-1, 0, 1)
            for dl in (-1, 0, 1)
            if length + dl >= 1.2 * (width + dw)
            for dt in range(-4, 5)
        ],
        dtype=np.float32,
    )
    fine_score, fine_support = _score(refined, gx, gy)
    best = int(np.argmax(fine_score))
    a, b, width, length, angle = map(float, refined[best])
    # OpenCV accepts floating dimensions despite the integer-size stub.
    corners = cv2.boxPoints(
        ((a, b), (width, length), angle)  # type: ignore[arg-type]
    ).astype(np.float32)
    if (
        np.any(corners < 2)
        or np.any(corners[:, 0] > rw - 3)
        or np.any(corners[:, 1] > rh - 3)
        or cv2.pointPolygonTest(corners, (float(cx), float(cy)), False) < 0
    ):
        raise RuntimeError("Forearm box clipped or does not contain selected tape")
    return {
        "center_uv": [a + x, b + y],
        "corners_uv": (corners + [x, y]).tolist(),
        "size_px": [width, length],
        "short_axis_image_rad": math.radians(angle),
        "edge_support": fine_support[best].tolist(),
        "score": float(fine_score[best]),
        "axis_score_margin": float(score.max() - alternate),
        "method": "tape-anchored four-edge rectangle; fixed local housing approximation",
    }


def box_in_map(box: dict, projection: dict) -> dict:
    """Project edges before choosing the short map axis (perspective matters)."""
    corners = project_pixels(
        box["corners_uv"], projection, projection["patch_height_m"]
    )
    vectors = np.roll(corners[:, :2], -1, axis=0) - corners[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    index = int(np.argmin(lengths[:2]))
    short, long = (
        float(lengths[index::2].mean()),
        float(lengths[(1 - index) :: 2].mean()),
    )
    if long < 1.15 * short:
        raise RuntimeError(
            "Projected box is too square to choose a reliable short axis"
        )
    yaw = (
        math.atan2(vectors[index, 1], vectors[index, 0]) + math.pi / 2
    ) % math.pi - math.pi / 2
    center = project_pixels(
        [box["center_uv"]], projection, projection["patch_height_m"]
    )[0]
    return {
        **box,
        "center_xyz": center.tolist(),
        "corners_xyz": corners.tolist(),
        "width_m": short,
        "length_m": long,
        "height_m": projection["patch_height_m"],
        "short_axis_yaw_rad": yaw,
    }
