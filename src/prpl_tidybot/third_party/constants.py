"""Global constants."""

# mypy: ignore-errors
# pylint: disable=all

import os
from pathlib import Path

import numpy as np
import yaml

_lab_name = os.environ.get("PRPL_LAB", "prpl")
_lab_conf_path = (
    Path(__file__).parents[3] / "conf" / "lab" / f"{_lab_name}.yaml"
)
with open(_lab_conf_path, "r") as _f:
    _lab_conf = yaml.safe_load(_f)

RETRACT_ARM_CONF = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
BASE_CAMERA_DIMS = (360, 640, 3)
WRIST_CAMERA_DIMS = (480, 640, 3)

################################################################################
# Mobile base

# Vehicle center to steer axis (m)
h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array(
    [-1.0, 1.0, 1.0, -1.0]
)  # Kinova / Franka

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [1988.0 / 4096, 491.0 / 4096, 1266.0 / 4096, 822.0 / 4096]

################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = "localhost"
BASE_RPC_PORT = 50000
ARM_RPC_HOST = "localhost"
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b"secret password"

# Cameras
BASE_CAMERA_SERIAL = "7DEAE8DE"

# Policy
POLICY_SERVER_HOST = "localhost"
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84

################################################################################
# perception PC — loaded from conf/lab/<PRPL_LAB>.yaml (default: "default")

SERVER_HOSTNAME = _lab_conf["marker_detector_host"]
ROBOT_HOSTNAME_PREFIX = _lab_conf["nuc_ip"]
CONN_AUTHKEY = b"secret password"  # shared authentication key

################################################################################
# Arm

MOUNTING_OFFSET = 0.12  # for new kinova mounting offset
HEIGHT_OFFSET = -0.288 - 0.06  # maximum: -0.288 - 0.077, for new kinova height offset

# Arm-dependent heading compensation (set to 0 if unsure)
ARM_HEADING_COMPENSATION = {
    0: -0.7,  # Robot 1 (asset tag: none)
    1: 0.2,  # Robot 2 (asset tag: 000007 402760)
    2: 0.7,  # Robot 3 (asset tag: 000007 402746)
}

################################################################################
# Camera

CAMERA_SERIALS = {
    0: "7DEAE8DE",  # Robot 1
    1: "44251E9E",  # Robot 2
    2: "7E841E9E",  # Robot 3
}
CAMERA_FOCUS = 0
CAMERA_TEMPERATURE = 3900
CAMERA_EXPOSURE = 156
CAMERA_GAIN = 10
