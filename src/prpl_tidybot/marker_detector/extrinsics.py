"""Per-camera extrinsics: solve, store, and use full 6-DoF camera poses.

Each ceiling camera is calibrated in two stages: intrinsics from a ChArUco
board (`camera_params/<lab>/<serial>.yml`) and extrinsics — the camera's
pose in the map frame — stored alongside as
`camera_params/<lab>/<serial>_extrinsics.json`. With both, an undistorted
pixel back-projects to a ray in the map frame, and intersecting that ray
with a horizontal plane gives the world position of anything whose height
is known (floor targets, robot stickers). This replaces the
floor-homography-plus-nadir-scaling approximation: no hand-measured camera
height or annotated camera center is involved.

Extrinsics follow the OpenCV convention: `(rvec, tvec)` map a map-frame
point `p` to the camera frame via `R @ p + t`. Produce the files with
`scripts/calibrate_camera_extrinsics.py`.
"""

import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.utils import CAMERA_PARAMS_DIR

# Distortion is handled upstream (CameraServer publishes undistorted
# frames), so all OpenCV calls here use zero distortion.
_NO_DISTORTION = np.zeros(5)


def extrinsics_path(serial: str) -> Path:
    """Return the (unique) extrinsics file for `serial`, or where it belongs.

    If no extrinsics file exists yet, the path is placed next to the
    camera's intrinsics file.
    """
    candidates = list(CAMERA_PARAMS_DIR.glob(f"*/{serial}_extrinsics.json"))
    if candidates:
        assert len(candidates) == 1, f"multiple extrinsics files for {serial}"
        return candidates[0]
    intrinsics = list(CAMERA_PARAMS_DIR.glob(f"*/{serial}.yml"))
    assert (
        len(intrinsics) == 1
    ), f"expected one intrinsics file for {serial}, found: {intrinsics}"
    return intrinsics[0].parent / f"{serial}_extrinsics.json"


def load_extrinsics(serial: str) -> tuple[np.ndarray, np.ndarray]:
    """Load `(rvec, tvec)` (map frame -> camera frame) for `serial`."""
    path = extrinsics_path(serial)
    assert path.exists(), (
        f"no extrinsics file for camera {serial}; run "
        "scripts/calibrate_camera_extrinsics.py first"
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.array(data["rvec"], dtype=np.float64),
        np.array(data["tvec"], dtype=np.float64),
    )


def save_extrinsics(
    path: Path, rvec: np.ndarray, tvec: np.ndarray, metadata: dict[str, Any]
) -> None:
    """Write an extrinsics file; `metadata` records calibration provenance."""
    data = {"rvec": rvec.tolist(), "tvec": tvec.tolist(), **metadata}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def solve_extrinsics(
    floor_xy: np.ndarray, pixels: np.ndarray, camera_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the camera pose from floor-plane correspondences.

    `floor_xy` holds map-frame positions of points on the floor (z = 0) and
    `pixels` the matching undistorted pixel coordinates; at least four
    non-collinear points are required. Returns `(rvec, tvec, rms_px)` where
    `rms_px` is the root-mean-square reprojection error in pixels — with
    real data it measures how consistent the correspondences and the
    intrinsics are with any rigid camera pose.
    """
    floor_xy = np.asarray(floor_xy, dtype=np.float64)
    object_points = np.c_[floor_xy, np.zeros(len(floor_xy))]
    image_points = np.asarray(pixels, dtype=np.float64)
    assert len(object_points) >= 4, "need at least four correspondences"
    ok, rvec, tvec = cv.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        _NO_DISTORTION,
        flags=cv.SOLVEPNP_IPPE,
    )
    assert ok, "solvePnP failed"
    rvec, tvec = cv.solvePnPRefineLM(
        object_points, image_points, camera_matrix, _NO_DISTORTION, rvec, tvec
    )
    projected, _ = cv.projectPoints(
        object_points, rvec, tvec, camera_matrix, _NO_DISTORTION
    )
    errors = np.asarray(projected, dtype=np.float64).reshape(-1, 2) - image_points
    rms_px = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    return rvec.reshape(3), tvec.reshape(3), rms_px


class FloorProjector:
    """Projects undistorted pixels through a calibrated camera onto
    horizontal planes in the map frame."""

    def __init__(
        self, camera_matrix: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> None:
        self._camera_matrix_inv = np.linalg.inv(
            np.asarray(camera_matrix, dtype=np.float64)
        )
        rotation_raw, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64))
        rotation = np.asarray(rotation_raw, dtype=np.float64)
        self._rot_camera_to_map = rotation.T
        self.camera_position: np.ndarray = -rotation.T @ np.asarray(
            tvec, dtype=np.float64
        ).reshape(3)

    def project_to_heights(
        self, pixels: np.ndarray, heights: np.ndarray | float
    ) -> np.ndarray:
        """Map `(N, 2)` pixels to map-frame `(N, 2)` xy at the given heights.

        `heights` is a scalar or an `(N,)` array giving each pixel's world
        height (the z of the horizontal plane its ray is intersected with).
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        heights = np.broadcast_to(
            np.asarray(heights, dtype=np.float64), pixels.shape[:1]
        )
        homogeneous = np.c_[pixels, np.ones(len(pixels))]
        rays = homogeneous @ self._camera_matrix_inv.T @ self._rot_camera_to_map.T
        scale = (heights - self.camera_position[2]) / rays[:, 2]
        return self.camera_position[:2] + scale[:, None] * rays[:, :2]
