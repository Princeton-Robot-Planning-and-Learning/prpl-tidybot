"""Marker-detector server.

Spawns one camera-server process per ceiling camera, then runs an in-process
detector that fuses ArUco detections from each into a payload of the shape
`{"poses": {robot_idx: (x, y, theta)}, "targets": {aruco_id: (x, y)}}`.
Robot stickers (`MARKER_IDS`) fuse into a single robot pose; standalone scene
markers (`TARGET_MARKER_IDS`) are reported individually at their world-frame
centre. Clients consume the payload over the publisher socket on
`MARKER_DETECTOR_PORT` — see `MarkerDetectorClient`.
"""

import argparse
import math
import time
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Any

import yaml

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector.camera_client import CameraClient
from prpl_tidybot.marker_detector.camera_server import CameraServer
from prpl_tidybot.marker_detector.ceiling_image_publisher import CeilingImagePublisher
from prpl_tidybot.marker_detector.constants import (
    CAMERA_HEIGHT,
    CAMERA_SERIALS,
    CAMERA_SERVER_PORTS,
    DETECTED_MARKER_IDS,
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
        self,
        placement: str,
        serial: str,
        port: int,
        inverse_heading: bool,
        camera_height: float = CAMERA_HEIGHT,
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
        self.marker_dict.bytesList = self.marker_dict.bytesList[
            list(DETECTED_MARKER_IDS)
        ]
        # detectMarkers returns slot indices into the sliced bytesList; this
        # array maps slot -> actual ArUco ID so we can split robot stickers
        # from scene targets after detection.
        self._slot_to_aruco_id = np.array(DETECTED_MARKER_IDS, dtype=np.int32)
        self._num_robot_slots = len(MARKER_IDS)

        # Tightened to reduce false positives.
        self.detector_params = aruco.DetectorParameters_create()
        self.detector_params.minCornerDistanceRate = 0.2  # require fronto-parallel
        self.detector_params.adaptiveThreshWinSizeMin = 23  # all markers same size

        self.transformation_matrix = self._compute_transformation_matrix(
            np.array(self.camera_corners, dtype=np.float32)
        )
        self.height_ratio = (camera_height - ROBOT_HEIGHT) / camera_height
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
        """Split detections into robot-sticker fusions and standalone scene targets."""
        data: dict[str, Any] = {
            "poses": {},
            "targets": {},
            "single_marker_robots": set(),
        }
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
        indices = indices.squeeze(1)
        positions = centers.copy()

        # Split detections into robot-sticker slots and scene-target slots.
        robot_mask = indices < self._num_robot_slots

        # Standalone targets: report each detected target marker's centre as-is.
        for slot_idx, center in zip(indices[~robot_mask], centers[~robot_mask]):
            aruco_id = int(self._slot_to_aruco_id[slot_idx])
            data["targets"][aruco_id] = (float(center[0]), float(center[1]))

        if robot_mask.any():
            robot_corners = corners[robot_mask]
            sticker_indices = indices[robot_mask]
            sticker_centers = centers[robot_mask]
            positions_robot = sticker_centers.copy()

            # Per-marker headings, dealing with wraparound by comparing the std
            # of two unwrappings and keeping the more consistent one.
            diffs = (robot_corners - sticker_centers.reshape(-1, 1, 2)).reshape(-1, 2)
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

            robot_idx = 0  # the only robot in the scene
            single_marker = sticker_indices.size == 1
            heading: float | None
            if single_marker:
                heading = float(headings.item())
            else:
                # Pairwise heading estimates between this robot's visible markers.
                headings_pairs: list[float] = []
                for i, idx1 in enumerate(sticker_indices):
                    for j, idx2 in enumerate(sticker_indices):
                        if j <= i or idx1 == idx2:
                            continue
                        dx = sticker_centers[j][0] - sticker_centers[i][0]
                        dy = sticker_centers[j][1] - sticker_centers[i][1]
                        h = math.atan2(dy, dx) + self.angle_offsets[(idx1, idx2)]
                        h = (h + math.pi) % (2 * math.pi) - math.pi
                        headings_pairs.append(h)
                heading = (
                    float(np.array(headings_pairs, dtype=np.float32).mean())
                    if headings_pairs
                    else None
                )

            if heading is not None:
                # Project each marker's center to the robot center via its corner offset.
                angles_robot = (
                    heading
                    + np.radians([-45, -135, 135, 45], dtype=np.float32)[
                        sticker_indices
                    ]
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
    """Fuses per-camera detectors and publishes
    `{"poses": {robot_idx: (x, y, theta)}, "targets": {aruco_id: (x, y)}}`.
    """

    def __init__(
        self,
        hostname: str = "localhost",
        port: int = MARKER_DETECTOR_PORT,
        top_only: bool = False,
        debug: bool = False,
        inverse_heading: bool = True,
        camera_serials: list[str] = CAMERA_SERIALS,
        camera_height: float = CAMERA_HEIGHT,
    ) -> None:
        super().__init__(hostname=hostname, port=port)
        self.debug = debug
        if top_only:
            self.detectors = [
                Detector(
                    "top_only",
                    camera_serials[0],
                    CAMERA_SERVER_PORTS[0],
                    inverse_heading=inverse_heading,
                    camera_height=camera_height,
                )
            ]
        else:
            self.detectors = [
                Detector(
                    "top",
                    camera_serials[0],
                    CAMERA_SERVER_PORTS[0],
                    inverse_heading=inverse_heading,
                    camera_height=camera_height,
                ),
                Detector(
                    "bottom",
                    camera_serials[1],
                    CAMERA_SERVER_PORTS[1],
                    inverse_heading=inverse_heading,
                    camera_height=camera_height,
                ),
            ]

    def get_data(self) -> dict:
        data: dict[str, Any] = {"poses": {}, "targets": {}}
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
            # Targets: bottom detector wins by virtue of being second (matches
            # the robot-pose precedence).
            data["targets"].update(new_data["targets"])
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


def _load_lab_camera_config(lab: str) -> tuple[list[str], float]:
    conf_path = (
        Path(__file__).parent.parent.parent.parent / "conf" / "lab" / f"{lab}.yaml"
    )
    with open(conf_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["camera_serials"], cfg["camera_height"]


def main(
    host: str = "0.0.0.0",
    top_only: bool = False,
    debug: bool = False,
    inverse_heading: bool = True,
    lab: str | None = None,
) -> None:
    """Spawn the camera-server subprocesses, then run the detector publisher.

    `host` controls only the marker-detector publisher socket. The camera
    servers stay on localhost since they're consumed in-process.
    Pass `lab` (e.g. 'prpl') to override camera serials and height from
    conf/lab/<lab>.yaml; omit to use whatever PRPL_LAB resolves at import time.
    """
    if lab is not None:
        camera_serials, camera_height = _load_lab_camera_config(lab)
    else:
        camera_serials, camera_height = CAMERA_SERIALS, CAMERA_HEIGHT

    for serial, port in [
        (camera_serials[0], CAMERA_SERVER_PORTS[0]),
        (camera_serials[1], CAMERA_SERVER_PORTS[1]),
    ]:
        Process(target=_start_camera_server, args=(serial, port), daemon=True).start()
        if top_only:
            break

    time.sleep(1.5)  # let camera servers bind their sockets

    # JPEG frame publisher for off-host renderers. Same process as the marker
    # detector, but a separate socket so robot-pose subscribers don't pay the
    # image bytes on every poll. Runs in a daemon thread so the detector loop
    # below stays the main control path. Honours `top_only` so the publisher
    # only subscribes to camera servers that were actually started.
    ceiling_camera_ports = CAMERA_SERVER_PORTS[:1] if top_only else CAMERA_SERVER_PORTS
    ceiling_publisher = CeilingImagePublisher(
        hostname=host, camera_ports=ceiling_camera_ports
    )
    Thread(target=ceiling_publisher.run, daemon=True).start()

    MarkerDetectorServer(
        hostname=host,
        top_only=top_only,
        debug=debug,
        inverse_heading=inverse_heading,
        camera_serials=camera_serials,
        camera_height=camera_height,
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rotate every reported heading by 180 degrees. On by default to "
            "match the ground-truth robot heading; pass --no-inverse-heading "
            "to disable."
        ),
    )
    parser.add_argument(
        "--lab",
        default=None,
        help="Lab name (e.g. 'prpl') to load camera serials and height from conf/lab/<lab>.yaml.",
    )
    args = parser.parse_args()
    main(
        host=args.host,
        top_only=args.top_only,
        debug=args.debug,
        inverse_heading=args.inverse_heading,
        lab=args.lab,
    )


if __name__ == "__main__":
    cli()
