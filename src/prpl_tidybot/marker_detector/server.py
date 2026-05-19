"""Marker-detector server.

Spawns one camera-server process per ceiling camera, then runs an in-process
detector that fuses ArUco detections from each into a `{robot_idx: (x, y, theta)}`
map-frame pose dict. Clients consume the dict over the publisher socket on
`MARKER_DETECTOR_PORT` — see `RealBaseInterface.get_map_base_state`.

Adapted from `yixuanhuang98/tidybot_server/server/marker_detector_server.py`.
"""

import argparse
import math
import time
from multiprocessing import Process
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.camera_client import CameraClient
from prpl_tidybot.marker_detector.camera_server import CameraServer
from prpl_tidybot.marker_detector.constants import (
    CAMERA_HEIGHT,
    CAMERA_SERIALS,
    CAMERA_SERVER_PORTS,
    FLOOR_LENGTH,
    FLOOR_WIDTH,
    MARKER_DETECTOR_PORT,
    MARKER_DICT_ID,
    MARKER_IDS,
    MARKER_PARAMS,
    ROBOT_DIAG,
    ROBOT_HEIGHT,
)
from prpl_tidybot.marker_detector.publisher import Publisher
from prpl_tidybot.marker_detector.utils import get_camera_alignment_params


def _get_angle_offsets() -> dict[tuple[int, int], float]:
    """Precompute the geometric angle between every ordered pair of sticker corners."""
    corners = [(0, 1), (1, 1), (1, 0), (0, 0)]
    offsets: dict[tuple[int, int], float] = {}
    for i, corner1 in enumerate(corners):
        for j, corner2 in enumerate(corners):
            if i != j:
                offsets[(i, j)] = -math.atan2(
                    corner2[1] - corner1[1], corner2[0] - corner1[0]
                )
    return offsets


class Detector:
    """Per-camera ArUco detector that emits map-frame robot poses."""

    def __init__(
        self, placement: str, serial: str, port: int, inverse_heading: bool
    ) -> None:
        assert placement in {"top", "bottom", "top_only"}
        self.placement = placement

        self.camera_center, self.camera_corners = get_camera_alignment_params(serial)
        self.camera_client = CameraClient(port)

        cv.setNumThreads(4)  # tuned for 12 CPUs
        # Legacy OpenCV 4.6 aruco API; removed in 4.7+. The aliased name lets
        # mypy treat these calls as Any until we port to the new API.
        aruco: Any = cv.aruco
        self.marker_dict = aruco.Dictionary_get(MARKER_DICT_ID)
        self.marker_dict.bytesList = self.marker_dict.bytesList[MARKER_IDS]

        # Tightened to reduce false positives.
        self.detector_params = aruco.DetectorParameters_create()
        self.detector_params.minCornerDistanceRate = 0.2  # require fronto-parallel
        self.detector_params.adaptiveThreshWinSizeMin = 23  # all markers same size

        self.transformation_matrix = self._compute_transformation_matrix(
            np.array(self.camera_corners, dtype=np.float32)
        )
        self.height_ratio = (CAMERA_HEIGHT - ROBOT_HEIGHT) / CAMERA_HEIGHT
        self.angle_offsets = _get_angle_offsets()
        self.position_offset = ROBOT_DIAG / 2 - MARKER_PARAMS[
            "sticker_length"
        ] / math.sqrt(2)

        self.inverse_heading = inverse_heading

    def _compute_transformation_matrix(self, src_points: np.ndarray) -> np.ndarray:
        if self.placement == "top":
            dst_points = np.array(
                [
                    [-(FLOOR_WIDTH / 2), FLOOR_LENGTH / 2],
                    [FLOOR_WIDTH / 2, FLOOR_LENGTH / 2],
                    [FLOOR_WIDTH / 2, 0],
                    [-(FLOOR_WIDTH / 2), 0],
                ],
                dtype=np.float32,
            )
        elif self.placement == "bottom":
            dst_points = np.array(
                [
                    [-(FLOOR_WIDTH / 2), 0],
                    [FLOOR_WIDTH / 2, 0],
                    [FLOOR_WIDTH / 2, -(FLOOR_LENGTH / 2)],
                    [-(FLOOR_WIDTH / 2), -(FLOOR_LENGTH / 2)],
                ],
                dtype=np.float32,
            )
        else:  # 'top_only'
            dst_points = np.array(
                [
                    [-(FLOOR_WIDTH / 2), FLOOR_LENGTH / 4],
                    [FLOOR_WIDTH / 2, FLOOR_LENGTH / 4],
                    [FLOOR_WIDTH / 2, -(FLOOR_LENGTH / 4)],
                    [-(FLOOR_WIDTH / 2), -(FLOOR_LENGTH / 4)],
                ],
                dtype=np.float32,
            )

        return cv.getPerspectiveTransform(src_points, dst_points).astype(np.float32)

    def get_poses_from_markers(
        self, corners: Any, indices: Any, debug: bool = False
    ) -> dict:
        """Fuse detected marker corners into per-robot (x, y, theta) poses."""
        data: dict[str, Any] = {"poses": {}, "single_marker_robots": set()}
        if indices is None:
            return data

        # Convert marker corners from pixel to real-world coordinates.
        corners = np.concatenate(corners, axis=1).squeeze(0)
        camera_center = np.array(self.camera_center, dtype=np.float32)
        corners = camera_center + self.height_ratio * (corners - camera_center)
        corners = np.c_[corners, np.ones(corners.shape[0], dtype=np.float32)]
        corners = corners @ self.transformation_matrix.T
        corners = (corners[:, :2] / corners[:, 2:]).reshape(-1, 4, 2)

        centers = corners.mean(axis=1)

        # Per-marker headings, dealing with wraparound by comparing the std
        # of two unwrappings and keeping the more consistent one.
        diffs = (corners - centers.reshape(-1, 1, 2)).reshape(-1, 2)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0]).reshape(-1, 4) + np.radians(
            [-135, -45, 45, 135], dtype=np.float32
        )
        angles1 = np.mod(angles + math.pi, 2 * math.pi) - math.pi
        angles2 = np.mod(angles, 2 * math.pi)
        headings = np.where(
            angles1.std(axis=1) < angles2.std(axis=1),
            angles1.mean(axis=1),
            np.mod(angles2.mean(axis=1) + math.pi, 2 * math.pi) - math.pi,
        )

        positions = centers.copy()
        indices = indices.squeeze(1)
        robot_indices = np.floor_divide(indices, 4)
        for robot_idx in np.unique(robot_indices):
            robot_idx = robot_idx.item()
            robot_mask = robot_indices == robot_idx
            indices_robot = np.mod(indices[robot_mask], 4)
            centers_robot = centers[robot_mask]
            positions_robot = centers_robot.copy()

            single_marker = robot_mask.sum() == 1
            if single_marker:
                heading = headings[robot_mask].item()
            else:
                # Pairwise heading estimates between this robot's visible markers.
                headings_robot = []
                for i, idx1 in enumerate(indices_robot):
                    for j, idx2 in enumerate(indices_robot):
                        if j <= i:
                            continue
                        if idx1 == idx2:  # false positive: same sticker twice
                            continue
                        dx = centers_robot[j][0] - centers_robot[i][0]
                        dy = centers_robot[j][1] - centers_robot[i][1]
                        heading = math.atan2(dy, dx) + self.angle_offsets[(idx1, idx2)]
                        heading = (heading + math.pi) % (2 * math.pi) - math.pi
                        headings_robot.append(heading)
                if len(headings_robot) == 0:  # all-false-positive degenerate case
                    continue
                heading = float(np.array(headings_robot, dtype=np.float32).mean())

            # Project each marker's center to the robot center using its corner offset.
            angles_robot = (
                heading
                + np.radians([-45, -135, 135, 45], dtype=np.float32)[indices_robot]
            )
            positions_robot[:, 0] += self.position_offset * np.cos(angles_robot)
            positions_robot[:, 1] += self.position_offset * np.sin(angles_robot)
            position = positions_robot.mean(axis=0)
            positions[robot_mask] = positions_robot

            if self.inverse_heading:
                heading = (heading + math.pi) % (2 * math.pi)
                if heading > math.pi:
                    heading -= 2 * math.pi

            data["poses"][robot_idx] = (position[0], position[1], heading)
            if single_marker:
                data["single_marker_robots"].add(robot_idx)

        if debug:
            data["debug_data"] = list(
                zip(indices.tolist(), centers.tolist(), positions.tolist())
            )

        return data

    def get_poses(self, debug: bool = False) -> dict:
        """Grab the latest camera frame, detect markers, return fused poses."""
        image = self.camera_client.get_image()

        aruco: Any = cv.aruco
        corners, indices, _ = aruco.detectMarkers(
            image, self.marker_dict, parameters=self.detector_params
        )

        if debug:
            image_copy = image.copy()  # 0.2 ms
            if indices is not None:
                aruco.drawDetectedMarkers(image_copy, corners, indices)
            cv.imshow(f"Detections ({self.placement})", image_copy)  # 0.3 ms

        return self.get_poses_from_markers(corners, indices, debug=debug)


class MarkerDetectorServer(Publisher):
    """Fuses per-camera detectors and publishes `{robot_idx: (x, y, theta)}`."""

    def __init__(
        self,
        hostname: str = "localhost",
        port: int = MARKER_DETECTOR_PORT,
        top_only: bool = False,
        debug: bool = False,
        inverse_heading: bool = False,
    ) -> None:
        super().__init__(hostname=hostname, port=port)
        self.debug = debug
        if top_only:
            self.detectors = [
                Detector(
                    "top_only",
                    CAMERA_SERIALS[0],
                    CAMERA_SERVER_PORTS[0],
                    inverse_heading=inverse_heading,
                )
            ]
        else:
            self.detectors = [
                Detector(
                    "top",
                    CAMERA_SERIALS[0],
                    CAMERA_SERVER_PORTS[0],
                    inverse_heading=inverse_heading,
                ),
                Detector(
                    "bottom",
                    CAMERA_SERIALS[1],
                    CAMERA_SERVER_PORTS[1],
                    inverse_heading=inverse_heading,
                ),
            ]

    def get_data(self) -> dict:
        data: dict[str, Any] = {"poses": {}}
        if self.debug:
            data["debug_data"] = []
        for detector in self.detectors:
            new_data = detector.get_poses(debug=self.debug)
            for robot_idx, pose in new_data["poses"].items():
                if (
                    robot_idx in data["poses"]
                    and robot_idx in new_data["single_marker_robots"]
                ):
                    # Single-marker estimates are noisier; let the multi-marker
                    # pose from the other camera win.
                    continue
                # Bottom detector takes precedence by virtue of being second.
                data["poses"][robot_idx] = pose
            if "debug_data" in new_data:
                data["debug_data"].extend(new_data["debug_data"])
        if self.debug:
            cv.waitKey(1)
        return data

    def clean_up(self) -> None:
        for detector in self.detectors:
            detector.camera_client.close()
        cv.destroyAllWindows()


def _start_camera_server(serial: str, port: int) -> None:
    CameraServer(serial, port=port).run()


def main(
    host: str = "0.0.0.0",
    top_only: bool = False,
    debug: bool = False,
    inverse_heading: bool = False,
) -> None:
    """Spawn the camera-server subprocesses, then run the detector publisher.

    `host` controls only the marker-detector publisher socket. The camera
    servers stay on localhost since they're consumed in-process.
    """
    for serial, port in [
        (CAMERA_SERIALS[0], CAMERA_SERVER_PORTS[0]),
        (CAMERA_SERIALS[1], CAMERA_SERVER_PORTS[1]),
    ]:
        Process(target=_start_camera_server, args=(serial, port), daemon=True).start()
        if top_only:
            break

    time.sleep(1.5)  # let camera servers bind their sockets

    MarkerDetectorServer(
        hostname=host,
        top_only=top_only,
        debug=debug,
        inverse_heading=inverse_heading,
    ).run()


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...server` and `scripts/`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for the marker-detector socket (default: 0.0.0.0).",
    )
    parser.add_argument("--top-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--inverse-heading",
        action="store_true",
        help="Rotate every reported heading by 180 degrees.",
    )
    args = parser.parse_args()
    main(
        host=args.host,
        top_only=args.top_only,
        debug=args.debug,
        inverse_heading=args.inverse_heading,
    )


if __name__ == "__main__":
    cli()
