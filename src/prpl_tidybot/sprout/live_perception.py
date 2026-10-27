"""Fresh, immutable ceiling images and chassis poses for the live forearm skill."""

from __future__ import annotations

import threading
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prpl_tidybot.sprout.live_config import calibration_digest
from prpl_tidybot.sprout.sequence import GraspPose
from prpl_tidybot.visual_servo.forearm_template import ForearmPatchTracker, ceiling_view


class LatestPublisher:
    """Network IO stays off the control thread; stale/blocking sources fail closed."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._payload: dict[str, Any] | None = None
        self._connection: Any = None
        self.error: str | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            with Client(
                (self.host, self.port), authkey=b"secret password"
            ) as connection:
                self._connection = connection
                while not self._closed.is_set():
                    connection.send(None)
                    if not connection.poll(2):
                        raise TimeoutError("Publisher timed out")
                    payload = connection.recv()
                    with self._lock:
                        self._payload = payload
        except (OSError, EOFError, TimeoutError) as error:
            self.error = str(error)

    def get(self) -> dict[str, Any]:
        with self._lock:
            if self.error or self._payload is None:
                raise RuntimeError(self.error or "Waiting for live publisher")
            return self._payload

    def close(self) -> None:
        self._closed.set()
        if self._connection is not None:
            self._connection.close()


def fresh_time(timestamp: Any, maximum_age: float) -> float:
    if (
        timestamp is None
        or not np.isfinite(timestamp)
        or not 0 <= time.time() - timestamp <= maximum_age
    ):
        raise RuntimeError(
            "Missing/stale camera acquisition timestamp; update perception servers and synchronize clocks"
        )
    return float(timestamp)


class LivePerception:
    """Same operator-selected tape template and calibrated projection as preview."""

    def __init__(
        self, config: dict, reference: Path, *, source_factory=LatestPublisher
    ) -> None:
        self.config = config
        detection = config["detection"]
        reference_bgr = cv2.imread(str(reference))
        if reference_bgr is None:
            raise ValueError("Missing reviewed reference image")
        self.reference_composite = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        view = ceiling_view(
            cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB), detection["camera_index"]
        )
        self.tracker = ForearmPatchTracker(
            view,
            tuple(detection["patch_xywh"]),
            minimum_quality=detection["minimum_quality"],
            minimum_margin=detection["minimum_margin"],
        )
        self.images = source_factory(config["perception_host"], 6003)
        self.markers = source_factory(config["perception_host"], 6002)
        self.last_capture = -float("inf")

    def image(self) -> tuple[np.ndarray, float]:
        data = self.images.get()
        index = self.config["detection"]["camera_index"]
        cameras = data.get("cameras", [])
        if len(cameras) != 2:
            raise RuntimeError(
                "Ceiling publisher lacks per-camera metadata; update the perception PC"
            )
        camera = cameras[index]
        timestamp = fresh_time(
            camera.get("captured_at_unix"), self.config["servo"]["camera_max_age_s"]
        )
        if (
            camera.get("serial")
            != self.config["detection"]["camera_calibration"]["serial"]
        ):
            raise RuntimeError("Camera serial/order differs from calibration")
        if camera.get("calibration_sha256") != calibration_digest(
            self.config["detection"]["projection"]
        ):
            raise RuntimeError(
                "Perception-PC camera calibration differs from reviewed calibration"
            )
        frame = cv2.imdecode(np.frombuffer(data["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Invalid live camera frame")
        calibration = self.config["detection"]["camera_calibration"]
        if frame.shape != (
            2 * calibration["image_height"],
            calibration["image_width"],
            3,
        ):
            raise RuntimeError(
                "Camera dimensions differ from calibrated two-view composite"
            )
        return ceiling_view(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index), timestamp

    def body(self) -> dict | None:
        config = self.config["detection"].get("body_obstacle", {})
        if not config.get("enabled"):
            return None
        from prpl_tidybot.sprout.body_obstacle import check_contained, detect_body

        payload = self.images.get()
        cameras = payload.get("cameras", [])
        if len(cameras) != 2:
            raise RuntimeError("Body tracking needs both timestamped camera views")
        for view in config["views"]:
            camera = cameras[view["camera_index"]]
            fresh_time(
                camera.get("captured_at_unix"), self.config["servo"]["camera_max_age_s"]
            )
            if camera.get("serial") != view["serial"] or camera.get(
                "calibration_sha256"
            ) != calibration_digest(view["projection"]):
                raise RuntimeError("Body camera calibration/serial mismatch")
        frame = cv2.imdecode(np.frombuffer(payload["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != self.reference_composite.shape:
            raise RuntimeError("Invalid body composite")
        body = detect_body(
            self.reference_composite, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), config
        )
        check_contained(body, config["reviewed_box"])
        body["captured_at_unix"] = min(camera["captured_at_unix"] for camera in cameras)
        return body

    def base(self) -> tuple[np.ndarray, float]:
        data = self.markers.get()
        robot = self.config.get("robot_id", 0)
        if robot not in data.get("poses", {}):
            raise RuntimeError("TidyBot chassis markers are missing")
        timestamp = fresh_time(
            data.get("pose_captured_at_unix", {}).get(robot),
            self.config["servo"]["camera_max_age_s"],
        )
        pose = np.asarray(data["poses"][robot], dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise RuntimeError("Invalid chassis map pose")
        return pose, timestamp

    def target(self, *, new_frame: bool = False) -> tuple[GraspPose, float]:
        view, timestamp = self.image()
        if new_frame and timestamp <= self.last_capture:
            raise RuntimeError("Waiting for a new camera frame")
        d = self.config["detection"]
        match = self.tracker.detect(view, tuple(d["search_xywh"]))
        if match is None:
            raise RuntimeError("Forearm white tape lost or ambiguous")
        p = d["projection"]
        rotation, _ = cv2.Rodrigues(np.asarray(p["rvec"], dtype=float))
        origin = -rotation.T @ np.asarray(p["tvec"], dtype=float).reshape(3)
        ray = rotation.T @ np.linalg.solve(
            np.asarray(p["camera_matrix"]), [*match.center_uv, 1]
        )
        if abs(ray[2]) < 1e-8:
            raise RuntimeError("Invalid target projection")
        distance = (p["patch_height_m"] - origin[2]) / ray[2]
        if distance <= 0:
            raise RuntimeError("Target plane is behind camera")
        xyz = origin + distance * ray
        xyz[:2] += p["patch_to_grasp_xy_m"]
        if d.get("auto_box", {}).get("enabled"):
            from prpl_tidybot.visual_servo.forearm_box import box_in_map, fit_box

            box = box_in_map(
                fit_box(view, match.center_uv, d["patch_xywh"][2:], d["search_xywh"]), p
            )
            # Same target identity and reviewed orientation; do not silently
            # rotate or change size beyond the approved local-box envelope.
            yaw_delta = abs(
                (box["short_axis_yaw_rad"] - p["grasp_yaw_rad"] + np.pi / 2) % np.pi
                - np.pi / 2
            )
            if yaw_delta > d["auto_box"]["maximum_yaw_change_rad"]:
                raise RuntimeError("Forearm short axis changed; preview again")
            old = d["auto_box"]["last_box"]
            if any(
                abs(box[key] / old[key] - 1) > 0.2 for key in ("width_m", "length_m")
            ):
                raise RuntimeError("Forearm box dimensions changed; preview again")
            xyz[:2] = box["center_xyz"][:2]
        pose = GraspPose(
            float(xyz[0]), float(xyz[1]), p["grasp_height_m"], p["grasp_yaw_rad"]
        )
        pose.validate()
        self.last_capture = timestamp
        return pose, timestamp

    def close(self) -> None:
        self.images.close()
        self.markers.close()
