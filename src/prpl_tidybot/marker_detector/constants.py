"""Constants for the marker-detector pipeline.

Connection constants shared with the rest of the codebase (`CONN_AUTHKEY`,
`SERVER_HOSTNAME`) live in `prpl_tidybot.third_party.constants`.
"""

import os
from pathlib import Path

import cv2 as cv
import yaml

_lab_name = os.environ.get("PRPL_LAB", "default")
_lab_conf_path = (
    Path(__file__).parent.parent.parent.parent / "conf" / "lab" / f"{_lab_name}.yaml"
)
with open(_lab_conf_path, "r") as _f:
    _lab_conf = yaml.safe_load(_f)

# Network ports for the publisher sockets.
MARKER_DETECTOR_PORT = 6002
CAMERA_SERVER_PORTS = (6000, 6001)
# JPEG-encoded frames from the top ceiling camera, for off-host renderers
# (e.g. the video recorder on the NUC).
CEILING_IMAGE_PORT = 6003

# ArUco markers used to label the robot. Four stickers, one on each top corner of
# the chassis, in the order assumed by the multi-marker pose fusion (top-left,
# top-right, bottom-right, bottom-left of the sticker quad).
MARKER_PARAMS = {
    "marker_length": 0.09,  # 90 mm
    "sticker_length": 0.12,  # 120 mm
}
MARKER_DICT_ID = cv.aruco.DICT_4X4_50
MARKER_IDS = (10, 13, 17, 21)

# ArUco markers placed in the scene as task targets (currently a single point
# target used by base_motion3d). Disjoint from MARKER_IDS so detector slots map
# cleanly to "robot sticker" vs "target".
TARGET_MARKER_IDS = (23,)

# Flat ordered list of every ID the detector should be configured to recognise.
# Robot stickers occupy slots 0..len(MARKER_IDS)-1; targets follow.
DETECTED_MARKER_IDS = MARKER_IDS + TARGET_MARKER_IDS

# Ceiling cameras. Order is (top, bottom); top precedes except for single-marker
# pose estimates where bottom wins. Different physical cameras than the wrist
# cameras in `third_party.constants.CAMERA_SERIALS`.
# Loaded from conf/lab/<PRPL_LAB>.yaml (default: "default").
CAMERA_SERIALS: list[str] = _lab_conf["camera_serials"]
CAMERA_FOCUS = 0
CAMERA_TEMPERATURE = 3900
CAMERA_EXPOSURE = 77  # 77 is best, 156 is slightly worse, 312 gives motion blur
CAMERA_GAIN = 50  # Increments of 10
CAMERA_HEIGHT: float = _lab_conf["camera_height"]

# Floor extents in the map frame. Origin is the floor center.
NUM_FLOOR_TILES_X = 6
NUM_FLOOR_TILES_Y = 6
FLOOR_TILE_SIZE = 24 * 0.0254  # 2 ft
FLOOR_LENGTH = NUM_FLOOR_TILES_Y * FLOOR_TILE_SIZE
FLOOR_WIDTH = NUM_FLOOR_TILES_X * FLOOR_TILE_SIZE

# Robot geometry used by the marker-center → robot-center correction.
ROBOT_HEIGHT = 0.378  # m
ROBOT_DIAG = 0.665  # m
