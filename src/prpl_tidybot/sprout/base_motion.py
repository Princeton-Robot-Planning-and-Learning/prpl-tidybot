# Hardware and optional model imports stay lazy for offline use.
# pylint: disable=import-outside-toplevel
"""Approved NUC-local base transit; existing Vehicle velocity/watchdog controller.

Holds the arm in its existing onboard position mode during transit, checks
measured joints and fresh camera/body state, stops base before arm manipulation.
No automatic homing/retraction. No hardware imports until execute().
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from prpl_tidybot.sprout.base_planning import segment_free, validate_base_plan
from prpl_tidybot.sprout.live_perception import fresh_time


def validate_navigation(config: dict) -> None:
    """Validate commissioned navigation bounds before starting base control."""
    if config.get("verified") is not True:
        raise ValueError(
            "Base route/operating area, transit posture "
            "and physical stop need lab verification"
        )
    for key in (
        "speed_m_s",
        "yaw_speed_rad_s",
        "acceleration_m_s2",
        "yaw_acceleration_rad_s2",
        "position_tolerance_m",
        "yaw_tolerance_rad",
        "path_tolerance_m",
        "timeout_s",
        "period_s",
        "arm_tolerance_rad",
        "camera_max_age_s",
    ):
        value = config.get(key)
        if (
            isinstance(value, bool)
            or value is None
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Fill positive navigation." + key)
    # Existing low-level command watchdog is 2.5 policy periods (0.25s).
    if config["period_s"] > 0.05:
        raise ValueError("Navigation period must be <= 0.05 s")


class BaseMotion:
    """Execute an approved base route with live scene and arm guards."""

    def __init__(
        self, config, perception, arm_reader, stop_event, *, vehicle_factory=None
    ) -> None:
        self.config, self.perception, self.arm_reader, self.stop_event = (
            config,
            perception,
            arm_reader,
            stop_event,
        )
        self.factory = vehicle_factory
        self.vehicle = None
        self.lock = threading.Lock()

    def stop(self):
        """Latch the stop event and stop the active base controller."""
        self.stop_event.set()
        # Don't join from a signal handler; caller is the stop-watchdog thread.
        with self.lock:
            if self.vehicle is not None and self.vehicle.control_loop_running:
                self.vehicle.stop_control()

    def execute(self, plan, start_joints):
        """Follow the approved route while checking measured arm and scene state."""
        validate_navigation(self.config)
        validate_base_plan(plan)
        p = self.config
        if self.stop_event.is_set():
            raise RuntimeError("Base stop latched")
        if self.factory is None:
            from prpl_tidybot.third_party.base_controller import Vehicle

            self.factory = Vehicle
        vehicle = self.factory(
            max_vel=(p["speed_m_s"], p["speed_m_s"], p["yaw_speed_rad_s"]),
            max_accel=(
                p["acceleration_m_s2"],
                p["acceleration_m_s2"],
                p["yaw_acceleration_rad_s2"],
            ),
        )
        with self.lock:
            self.vehicle = vehicle
            if self.stop_event.is_set():
                raise RuntimeError("Base stop latched during startup")
            vehicle.start_control()
        deadline = time.monotonic() + p["timeout_s"]
        last_arm_check = 0.0
        try:
            for a, b in zip(plan["path_xyyaw"], plan["path_xyyaw"][1:]):
                stable = 0
                while True:
                    if self.stop_event.is_set() or not vehicle.control_loop_running:
                        raise RuntimeError("Base motion stopped")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Base route timed out")

                    pose, stamp = self.perception.base()
                    fresh_time(stamp, p["camera_max_age_s"])
                    body = (
                        self.perception.body()
                    )  # both calibrated cameras, containment
                    if body is None:
                        raise RuntimeError(
                            "Live body obstacle required for base transit"
                        )
                    fresh_time(body.get("captured_at_unix"), p["camera_max_age_s"])
                    if time.monotonic() - last_arm_check > 0.5:
                        # No commands while blocking on feedback; Vehicle's command-loss
                        # watchdog neutralizes motors if the read stalls.
                        joints = np.asarray(self.arm_reader()["joints_rad"])
                        delta = (joints - start_joints + np.pi) % (2 * np.pi) - np.pi
                        if np.max(np.abs(delta)) > p["arm_tolerance_rad"]:
                            raise RuntimeError(
                                "Arm moved outside reviewed transit posture"
                            )
                        last_arm_check = time.monotonic()
                    av, bv = np.asarray(a)[:2], np.asarray(b)[:2]
                    direction = bv - av
                    f = np.clip(
                        (pose[:2] - av) @ direction / max(direction @ direction, 1e-12),
                        0,
                        1,
                    )
                    if (
                        np.linalg.norm(pose[:2] - (av + f * direction))
                        > p["path_tolerance_m"]
                    ):
                        raise RuntimeError("Base left reviewed route corridor")
                    # Includes stopping-distance reserve, tracking tolerance,
                    # and latency.
                    reserve = (
                        p["path_tolerance_m"]
                        + p["speed_m_s"] ** 2 / (2 * p["acceleration_m_s2"])
                        + p["speed_m_s"] * (p["camera_max_age_s"] + 0.3)
                    )
                    if not segment_free(
                        pose,
                        b,
                        plan["obstacles"],
                        plan["radius_m"] + reserve,
                        plan["bounds_xy"],
                    ):
                        raise RuntimeError(
                            "Live base stopping envelope intersects keep-out"
                        )
                    error = bv - pose[:2]
                    angle = math.remainder(b[2] - pose[2], 2 * math.pi)
                    arrived = (
                        np.linalg.norm(error) < p["position_tolerance_m"]
                        and abs(angle) < p["yaw_tolerance_rad"]
                    )
                    still = np.linalg.norm(np.asarray(vehicle.dx)) < 0.01
                    stable = stable + 1 if arrived and still else 0
                    if stable >= 5:
                        break
                    velocity = error * 1.0
                    velocity *= min(
                        1.0, p["speed_m_s"] / max(np.linalg.norm(velocity), 1e-9)
                    )
                    if arrived:
                        velocity[:] = 0
                        angle = 0
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    local = np.array([[c, s], [-s, c]]) @ velocity
                    with self.lock:
                        if self.stop_event.is_set():
                            raise RuntimeError("Base stopped")
                        vehicle.set_target_velocity(
                            np.r_[
                                local,
                                np.clip(
                                    angle, -p["yaw_speed_rad_s"], p["yaw_speed_rad_s"]
                                ),
                            ],
                            frame="local",
                        )
                    time.sleep(p["period_s"])
            print(
                "REAL base arrived and stopped; rechecking arm grasp plan", flush=True
            )
        finally:
            with self.lock:
                if vehicle.control_loop_running:
                    vehicle.stop_control()
                self.vehicle = None
