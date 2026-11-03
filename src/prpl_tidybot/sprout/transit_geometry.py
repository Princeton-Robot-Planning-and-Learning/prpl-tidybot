"""Shared PRPL URDF checks for reviewed and live base transit.

TidyBot uses existing real_sim PyBullet collision meshes and primitives.
Sprout/environment boxes are inflated for clearance and stopping uncertainty.
Old circle/robot-box reviews must be regenerated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prpl_tidybot.sprout.live_config import model_digest
from prpl_tidybot.sprout.urdf_collision import PRPLCollisionScene, urdf_digest

DEFAULT_NAVIGATION = {
    "speed_m_s": 0.02,
    "yaw_speed_rad_s": 0.05,
    "acceleration_m_s2": 0.04,
    "yaw_acceleration_rad_s2": 0.075,
    "path_tolerance_m": 0.015,
    "yaw_tolerance_rad": 0.01,
    "arm_tolerance_rad": 0.01,
    "camera_max_age_s": 0.25,
}
SCHEMA = "prpl_urdf_v1"


def stopping_reserve(config: dict, radius: float) -> float:
    """Bound XY tracking, latency and translational/angular braking travel."""
    for key in DEFAULT_NAVIGATION:
        value = config.get(key)
        if (
            isinstance(value, bool)
            or value is None
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Fill positive navigation." + key)
    latency = config["camera_max_age_s"] + 0.3
    return float(
        config["path_tolerance_m"]
        + config["speed_m_s"] ** 2 / (2 * config["acceleration_m_s2"])
        + config["speed_m_s"] * latency
        + radius
        * (
            config["yaw_tolerance_rad"]
            + config["yaw_speed_rad_s"] ** 2 / (2 * config["yaw_acceleration_rad_s2"])
            + config["yaw_speed_rad_s"] * latency
        )
    )


def sweep_radius(joints: list[float]) -> float:
    """Bound point travel for yaw sampling, not an obstacle collision circle."""
    radius = 0.0
    with PRPLCollisionScene() as scene:
        for finger in scene.fingers:
            scene.set([0, 0, 0], joints, finger)
            for low, high in scene.aabbs():
                radius = max(
                    radius,
                    float(
                        np.linalg.norm(np.maximum(np.abs(low[:2]), np.abs(high[:2])))
                    ),
                )
    return radius


def build_geometry(
    model: Path, joints: list[float], clearance: float, navigation: dict
) -> dict:
    """Bind the public URDF model and reviewed posture, without robot box proxies."""
    radius = sweep_radius(joints)
    result = {
        "schema": SCHEMA,
        "urdf_sha256": urdf_digest(),
        "model_sha256": model_digest(model),
        "joints_rad": list(joints),
        "arm_tolerance_rad": navigation["arm_tolerance_rad"],
        "clearance_m": clearance,
        "sweep_radius_m": radius,
        "stopping_reserve_m": stopping_reserve(navigation, radius),
    }
    validate_geometry(result)
    return result


def validate_geometry(geometry: dict) -> None:
    """Reject obsolete reviews and malformed posture or uncertainty metadata."""
    if geometry.get("schema") != SCHEMA:
        raise ValueError("Regenerate preview: PRPL URDF collision review required")
    if geometry.get("urdf_sha256") != urdf_digest():
        raise ValueError("PRPL URDF/assets differ from the reviewed model")
    for name in (
        "clearance_m",
        "arm_tolerance_rad",
        "stopping_reserve_m",
        "sweep_radius_m",
    ):
        value = geometry.get(name)
        if (
            isinstance(value, bool)
            or value is None
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid transit " + name)
    if geometry["clearance_m"] < 0.04:
        raise ValueError("Transit clearance must be at least 0.04 m")
    joints = np.asarray(geometry.get("joints_rad"), float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("Invalid transit joints")


def validate_model_geometry(
    geometry: dict, model: Path, joints: list[float], navigation: dict
) -> None:
    """Verify actual URDF assets, posture and motion budget before execution."""
    validate_geometry(geometry)
    if geometry.get("model_sha256") != model_digest(model) or not np.allclose(
        geometry["joints_rad"], joints, atol=1e-8, rtol=0
    ):
        raise ValueError("Transit model/posture differs from review")
    if geometry["sweep_radius_m"] + 1e-8 < sweep_radius(joints):
        raise ValueError("Transit sweep budget does not bound URDF geometry")
    validate_budget(geometry, navigation)


def validate_budget(geometry: dict, navigation: dict) -> None:
    """Require live uncertainty to fit the reviewed motion-distance budget."""
    reserve = stopping_reserve(navigation, geometry["sweep_radius_m"])
    if (
        navigation["arm_tolerance_rad"] > geometry["arm_tolerance_rad"]
        or reserve > geometry["stopping_reserve_m"] + 1e-9
    ):
        raise ValueError(
            "Navigation exceeds reviewed transit budget; regenerate preview"
        )


class TransitChecker:
    """Use PRPL's actual URDF collision shapes at sampled map/joint poses."""

    def __init__(self, geometry: dict, obstacles: list[dict], bounds: list) -> None:
        validate_geometry(geometry)
        self.geometry, self.obstacles = geometry, obstacles
        self.bounds = np.asarray(bounds, float)
        self.radius = geometry["sweep_radius_m"]
        self.joints = geometry["joints_rad"]
        self.margin = geometry["clearance_m"]
        self.reserve = geometry["stopping_reserve_m"]
        self.scene = PRPLCollisionScene(bounds)
        self._bodies: dict[float, set[int]] = {}

    def __enter__(self) -> TransitChecker:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Release the isolated PyBullet client."""
        self.scene.close()

    def bodies(self, extra: float = 0.0) -> set[int]:
        """Inflate only environment boxes, never replace robot collision shapes."""
        key = math.ceil(extra / 0.005) * 0.005
        if key not in self._bodies:
            padding = np.array([self.margin + self.reserve + key] * 2 + [self.margin])
            self._bodies[key] = {
                self.scene.box(box["center"], np.asarray(box["half_size"]) + padding)
                for box in self.obstacles
            }
        return self._bodies[key]

    def free(self, pose, extra: float = 0.0) -> bool:
        """Check the same collision meshes used by the shelf real_sim planner."""
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return False
        for finger in self.scene.fingers:
            self.scene.set(pose, self.joints, finger)
            if self.scene.collision(self.bodies(extra)):
                return False
            for low, high in self.scene.aabbs():
                pad = self.margin + self.reserve + extra
                if np.any(np.asarray(low)[:2] - pad < self.bounds[0]) or np.any(
                    np.asarray(high)[:2] + pad > self.bounds[1]
                ):
                    return False
        return True

    def segment(self, start, end, resolution: float = 0.02) -> bool:
        """Check interpolated translation/yaw with a point-travel sampling margin."""
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Invalid transit sweep resolution")
        start, end = np.asarray(start, float), np.asarray(end, float)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.all(np.isfinite([start, end]))
        ):
            return False
        delta = end - start
        delta[2] = math.remainder(delta[2], 2 * math.pi)
        travel = float(np.linalg.norm(delta[:2])) + self.radius * abs(delta[2])
        count = max(1, math.ceil(travel / resolution))
        return all(
            self.free(start + f * delta, travel / (2 * count))
            for f in np.linspace(0, 1, count + 1)
        )

    def route(self, start, goal) -> list[list[float]]:
        """Plan through the existing PRPL mobile-base BiRRT, then revalidate."""
        if not self.free(start) or not self.free(goal):
            raise ValueError("Base start/goal intersects URDF keep-out/boundary")
        self.scene.set(start, self.joints, 0.0)
        path = self.scene.base_path(start, goal, self.bodies(0.01))
        if not all(self.segment(a, b) for a, b in zip(path, path[1:])):
            raise ValueError("PRPL base path fails live stopping/swept clearance")
        return path
