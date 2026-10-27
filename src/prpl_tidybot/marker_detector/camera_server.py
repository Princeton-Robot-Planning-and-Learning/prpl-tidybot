"""Camera server: opens one Logitech C930e, undistorts, publishes via shm.

Each `CameraServer` exposes a tiny RPC on its port that hands out the name +
shape + dtype of a POSIX shared-memory block containing the most recent
undistorted frame. The marker detector reads frames in-process via
`CameraClient`, which mmaps that block.

Adapted from `yixuanhuang98/tidybot_server/server/camera_server.py`.
"""

import time
from multiprocessing import shared_memory
from queue import Queue
from threading import Thread
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector import utils
from prpl_tidybot.marker_detector.constants import (
    CAMERA_EXPOSURE,
    CAMERA_FOCUS,
    CAMERA_GAIN,
    CAMERA_TEMPERATURE,
)
from prpl_tidybot.marker_detector.extrinsics import load_extrinsics
from prpl_tidybot.marker_detector.publisher import Publisher
from prpl_tidybot.sprout.live_config import calibration_digest


class CameraServer(Publisher):
    """Publisher that streams undistorted C930e frames through shared memory."""

    def __init__(self, serial: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(
            serial
        )
        self.cap = utils.get_video_cap(serial, image_width, image_height)  # ~1540 ms
        self.last_read_time = time.time()
        self.queue: Queue = Queue(maxsize=1)
        self.serial = serial

        self.calibration_sha256: str | None
        try:
            rvec, tvec = load_extrinsics(serial)
            self.calibration_sha256 = calibration_digest(
                {
                    "camera_matrix": camera_matrix.tolist(),
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                }
            )
        except FileNotFoundError:
            # Calibration tools must be able to start an uncalibrated camera.
            self.calibration_sha256 = None
        self.sequence = 0

        # cv2 4.6 stubs reject `None` for the rectification matrix, but the
        # underlying C++ accepts it as "identity rectification" — keep it.
        init_undistort: Any = cv.initUndistortRectifyMap
        self.map_x, self.map_y = init_undistort(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (image_width, image_height),
            cv.CV_32FC1,
        )

        image = self.get_image()
        self.shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        self.image_shm: np.ndarray = np.ndarray(
            image.shape, dtype=image.dtype, buffer=self.shm.buf
        )
        Thread(target=self.camera_worker, daemon=True).start()

    def get_image(self) -> np.ndarray:
        """Block on the camera, return one undistorted frame."""
        image = None
        while image is None:
            _, image = self.cap.read()  # 5–17 ms
        self.frame_acquired_at = time.time()
        image = cv.remap(image, self.map_x, self.map_y, cv.INTER_LINEAR)  # 2–5 ms
        return image

    def camera_worker(self) -> None:
        """Background pump: keep a single-slot queue full of fresh frames."""
        while True:
            if self.queue.empty():
                image = self.get_image()
                self.queue.put((self.frame_acquired_at, image))
                # Detect silent driver resets that would un-fix our settings.
                assert self.cap.get(cv.CAP_PROP_FOCUS) == CAMERA_FOCUS
                assert self.cap.get(cv.CAP_PROP_TEMPERATURE) == CAMERA_TEMPERATURE
                assert self.cap.get(cv.CAP_PROP_EXPOSURE) == CAMERA_EXPOSURE
                assert self.cap.get(cv.CAP_PROP_GAIN) == CAMERA_GAIN
            time.sleep(0.0001)

    def get_data(self) -> dict:
        # Reading too quickly causes the V4L pipeline to spike in latency.
        while time.time() - self.last_read_time < 0.0333:  # 30 fps
            time.sleep(0.0001)

        capture_time, image = self.queue.get()
        if time.time() - capture_time > 0.1:  # stale frame
            self.queue.get()  # flush camera buffer
            capture_time, image = self.queue.get()
        self.last_read_time = time.time()
        np.copyto(self.image_shm, image)  # 0.2 ms
        self.sequence += 1
        ok, jpeg = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Camera frame encoding failed")
        # JPEG and timestamp describe the SAME immutable frame. Legacy shared
        # memory readers remain supported, but cannot guarantee that pairing.
        return {
            "name": self.shm.name,
            "shape": image.shape,
            "dtype": image.dtype,
            "jpeg": jpeg.tobytes(),
            "captured_at_unix": capture_time,
            "sequence": self.sequence,
            "serial": self.serial,
            "calibration_sha256": self.calibration_sha256,
        }

    def clean_up(self) -> None:
        self.cap.release()
        self.shm.close()
        self.shm.unlink()
