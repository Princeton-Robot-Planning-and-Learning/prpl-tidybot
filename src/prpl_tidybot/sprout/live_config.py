"""Explicit physical commissioning inputs for the supervised forearm cycle."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path
from typing import Any

from prpl_tidybot.sprout.approach import TopDownGeometry
from prpl_tidybot.sprout.compliant_arm import HardwareLimits
from prpl_tidybot.sprout.compliant_backend import MotionLimits

CHECKS = (
    "model_mount_and_tool",
    "camera_calibration",
    "finger_floor_clearance",
    "forearm_contact_and_passive_mobility",
    "obstacle_and_held_limb_bounds",
    "physical_stop",
)
SERVO_KEYS = (
    "tolerance_m",
    "max_step_m",
    "max_travel_m",
    "timeout_s",
    "stable_frames",
    "max_updates",
    "camera_max_age_s",
    "base_tolerance_m",
    "base_tolerance_rad",
    "path_resolution_rad",
    "max_joint_step_rad",
    "orientation_speed_rad_s",
    "orientation_tolerance_rad",
    "confirmation_timeout_s",
    "gripper_stall_samples",
    "gripper_stall_epsilon",
    "grasp_reading_min",
    "grasp_reading_max",
    "collision_margin_m",
    "startup_timeout_s",
)


def model_files(path: Path) -> dict[str, Path]:
    """Exact public TidyBot XML/mesh dependencies, including sibling assets."""
    path = path.resolve()
    if path.name != "tidybot.xml" or path.parent.name != "stanford_tidybot":
        raise ValueError("Expected the stanford_tidybot/tidybot.xml model")
    root = ET.parse(path).getroot()
    if root.findall(".//include"):
        raise ValueError("Included XML needs explicit dependency handling")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    files = {"stanford_tidybot/tidybot.xml": path}
    for asset in root.findall("asset/*"):
        name = asset.get("file")
        if name is None:
            continue
        if asset.tag != "mesh":
            raise ValueError("Unsupported file-backed model asset")
        dependency = (path.parent / meshdir / name).resolve()
        relative = dependency.relative_to(path.parent.parent)
        if relative.parts[0] != "assets" or relative.parts[1] not in (
            "base",
            "gen3",
            "2f85",
        ):
            raise ValueError("Asset is outside public TidyBot mesh directories")
        files[str(relative)] = dependency
    return files


def model_digest(path: Path) -> str:
    """Fingerprint XML and referenced meshes; NUC must use the reviewed model."""
    digest = hashlib.sha256()
    for name, child in sorted(model_files(path).items()):
        digest.update(name.encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def calibration_digest(projection: dict[str, Any]) -> str:
    """Hash the camera intrinsics and extrinsics used for projection."""
    data = {key: projection[key] for key in ("camera_matrix", "rvec", "tvec")}
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_config(config: dict[str, Any], *, commissioned: bool = True) -> None:
    """Reject omissions, non-finite bounds, or unverified physical commissioning."""
    if config.get("schema") != "sprout_live_v1":
        raise ValueError("Expected sprout_live_v1 configuration")
    if config.get("arm_host", "192.168.1.10") != "192.168.1.10":
        raise ValueError("Existing Kinova driver is configured for 192.168.1.10")
    missing = []
    for section, keys in (
        ("geometry", [field.name for field in fields(TopDownGeometry)]),
        ("hardware", [field.name for field in fields(HardwareLimits)]),
        ("motion", [field.name for field in fields(MotionLimits)]),
        ("servo", SERVO_KEYS),
    ):
        missing += [
            f"{section}.{key}"
            for key in keys
            if config.get(section, {}).get(key) is None
        ]
    if missing:
        raise ValueError("Fill live.json: " + ", ".join(missing))
    TopDownGeometry(**config["geometry"])
    hardware = HardwareLimits(**config["hardware"])
    motion = MotionLimits(**config["motion"])
    servo = config["servo"]
    for key in SERVO_KEYS:
        value = servo[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid servo.{key}")
    for key in ("stable_frames", "max_updates", "gripper_stall_samples"):
        if (
            not isinstance(servo[key], int)
            or isinstance(servo[key], bool)
            or servo[key] < 2
        ):
            raise ValueError(f"servo.{key} must be an integer >= 2")
    if (
        not 0
        < servo["grasp_reading_min"]
        < servo["grasp_reading_max"]
        < motion.close_position
        <= 1
    ):
        raise ValueError("Grasp reading interval must be below commanded closure")
    if (
        servo["max_step_m"] > servo["max_travel_m"]
        or servo["path_resolution_rad"] > servo["max_joint_step_rad"]
    ):
        raise ValueError("Invalid servo travel/path bounds")
    if hardware.command_timeout_s <= motion.period_s * 2:
        raise ValueError("Command watchdog must allow at least two control periods")
    if not config.get("obstacles"):
        raise ValueError(
            "Provide conservative keep-out boxes for Sprout's torso/other limbs "
            "and surroundings"
        )
    for box in [*config["obstacles"], config.get("held_forearm", {})]:
        for key in ("center", "half_size"):
            values = box.get(key)
            if (
                values is None
                or len(values) != 3
                or not all(math.isfinite(v) for v in values)
            ):
                raise ValueError(f"Invalid collision box {key}")
        if any(v <= 0 for v in box["half_size"]):
            raise ValueError("Collision half-sizes must be positive")
    if config.get("confirmation_mode") != "operator":
        raise ValueError(
            "This first-trial runtime requires explicit "
            "operator grasp/support confirmations"
        )
    for key in ("grasp_xyz", "base_xyyaw", "arm_joints_rad"):
        value = config.get("revalidation_tolerances", {}).get(key)
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Fill positive revalidation_tolerances.{key}")
    if config["revalidation_tolerances"]["grasp_xyz"] > servo["max_travel_m"]:
        raise ValueError("Target revalidation exceeds allowed visual correction")
    if commissioned:
        absent = [
            key for key in CHECKS if config.get("verified", {}).get(key) is not True
        ]
        if config.get("execution_enabled") is not True or absent:
            raise ValueError(
                "Physical commissioning incomplete: "
                + ", ".join(absent or ["execution_enabled"])
            )
