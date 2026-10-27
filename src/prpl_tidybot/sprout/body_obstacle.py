"""Conservative two-camera body envelope for a reference supine pose.

Tracks distinctive torso/feet templates, translates enclosing pixel rectangles,
then unions their projections at floor and maximum body height. This is a
reference-pose tracker, not anatomical segmentation. It rejects pose/visibility
changes rather than shrinking a box around only the visible torso.
"""

from __future__ import annotations

import cv2
import numpy as np

from prpl_tidybot.visual_servo.forearm_box import project_pixels
from prpl_tidybot.visual_servo.forearm_template import ForearmPatchTracker, ceiling_view


def detect_body(reference: np.ndarray, image: np.ndarray, config: dict) -> dict:
    """Track both reference views and project their enclosing body envelope."""
    points: list[np.ndarray] = []
    matches = []
    maximum_height = float(config["maximum_height_m"])
    margin = float(config["margin_m"])
    if not 0 < maximum_height < 1 or not 0 < margin < 1:
        raise ValueError("Invalid body height envelope/margin")
    if {v["camera_index"] for v in config["views"]} != {0, 1}:
        raise ValueError("Body envelope requires both calibrated ceiling views")
    for view in config["views"]:
        index = view["camera_index"]
        tracker = ForearmPatchTracker(
            ceiling_view(reference, index),
            tuple(view["patch_xywh"]),
            minimum_quality=config["minimum_quality"],
            minimum_margin=config["minimum_margin"],
        )
        match = tracker.detect(ceiling_view(image, index), tuple(view["search_xywh"]))
        if match is None:
            raise ValueError(
                "Sprout body/feet lost or changed; body envelope unavailable"
            )
        x, y, w, h = view["patch_xywh"]
        shift = np.asarray(match.center_uv) - [x + w / 2, y + h / 2]
        x, y, w, h = view["envelope_xywh"]
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) + shift
        height, width = image.shape[0] // 2, image.shape[1]
        if np.any(corners < 0) or np.any(corners > [width, height]):
            raise ValueError(
                "Body envelope clipped by camera; capture/reselect reference"
            )
        for z in (0.0, maximum_height):
            points.extend(project_pixels(corners, view["projection"], z))
        matches.append(
            {
                "camera_index": index,
                "corners_uv": corners.tolist(),
                "quality": match.quality,
            }
        )
    xyz = np.asarray(points)
    low, high = xyz.min(0), xyz.max(0)
    low[:2] -= margin
    high[:2] += margin
    if np.any(high[:2] - low[:2] < 0.25) or np.any(high[:2] - low[:2] > 3):
        raise ValueError("Implausible projected body envelope")
    return {
        "name": "sprout_whole_body",
        "center": ((low + high) / 2).tolist(),
        "half_size": ((high - low) / 2).tolist(),
        "margin_m": margin,
        "views": matches,
        "scope": "base_only",
        "method": "reference-pose two-view envelope; union of floor/top ray projections",
    }


def check_contained(current: dict, reviewed: dict, tolerance: float = 0.01) -> None:
    """Reject body motion outside the reviewed obstacle envelope."""
    c, h = np.asarray(current["center"]), np.asarray(current["half_size"])
    rc, rh = np.asarray(reviewed["center"]), np.asarray(reviewed["half_size"])
    if np.any(np.abs(c[:2] - rc[:2]) + h[:2] > rh[:2] + tolerance):
        raise RuntimeError("Sprout body left reviewed keep-out envelope; preview again")


def draw_body(image: np.ndarray, body: dict) -> np.ndarray:
    """Overlay the tracked body envelope on the stacked ceiling images."""
    result = image.copy()
    for view in body["views"]:
        corners = np.array(view["corners_uv"], np.int32)
        corners[:, 1] += view["camera_index"] * (image.shape[0] // 2)
        cv2.polylines(result, [corners], True, (255, 150, 30), 3)
    cv2.putText(
        result,
        "SPROUT BODY KEEP-OUT (two-camera envelope)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 150, 30),
        2,
    )
    return result
