"""Build or verify an alphatamp Restock3D `scene.yaml` from marker detections.

This is the real-to-sim leg of the Restock3D replay pipeline (see
`docs/restock3d_real_robot_plan.md`). Each object to restock stands on a
floor ArUco target marker; the ceiling marker detector reports the marker
centres in the map frame, and this module transforms them into alphatamp's
home frame and writes the `scene.yaml` that `deploy.py` plans from. Sizes
are not perceived: they come from a hand-measured table in the lab config,
the analogue of the cylinder-shelf env's `cylinders` list.

The home frame is derived from the staged shelf, not from the robot:
Restock3D fixes the shelf at home-frame `(0.4, 1.4)` with its open front
facing home −y, so given the shelf's measured map pose the home origin and
yaw follow. `yaw_map` in the lab config is the rotation of the home frame's
axes in the map frame (with `yaw_map=0`, home +y is map +y and the shelf
front faces map −y).

Lab config yaml (see `conf/restock3d/objects.yaml`):

    shelf:
      map_xy: [1.5, 1.5]   # measured shelf centre, map frame
      yaw_map: 0.0         # home-frame yaw in the map frame (see above)
    objects:
      - marker_id: 35      # floor target marker under the object
        name: obj_goal1    # optional; defaults to obj_goal{i} in order
        width: 0.070       # full x-extent (m), the graspable dimension
        height: 0.125      # full z-extent (m), decides the shelf section
        depth: 0.070       # full y-extent (m); optional

Besides writing the scene, `verify_scene` compares a previously written
scene against fresh detections — the lab-day staging check: after taping
markers at the staging sheet's coordinates and setting the objects down,
the per-object residuals say whether the floor matches the planned scene.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf

from prpl_tidybot.marker_detector.client import MarkerDetectorClient

# Restock3D pins the shelf here in the home frame (kinematic_env.py's
# Restock3DEnvConfig.shelf_pose); the home frame is defined so the measured
# shelf lands on it.
SHELF_HOME_XY = (0.4, 1.4)

# Soft bounds from the deploy kit's validator (deploy_scene.validate_scene);
# violations are warnings there too, so they warn here, where fixing the
# staging is still cheap.
FLOOR_X_RANGE = (-0.80, -0.20)
FLOOR_Y_RANGE = (0.60, 1.20)
MIN_OBJECT_SPACING = 0.12
WIDTH_RANGE = (0.02, 0.08)
MAX_HEIGHT = 0.17


@dataclass(frozen=True)
class HomeFrame:
    """The alphatamp home frame's SE2 pose in the map frame."""

    origin_x: float
    origin_y: float
    yaw: float

    def map_to_home(self, x: float, y: float) -> tuple[float, float]:
        """Transform a map-frame point into the home frame."""
        dx, dy = x - self.origin_x, y - self.origin_y
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)

    def home_to_map(self, x: float, y: float) -> tuple[float, float]:
        """Transform a home-frame point into the map frame."""
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return (
            self.origin_x + cos_yaw * x - sin_yaw * y,
            self.origin_y + sin_yaw * x + cos_yaw * y,
        )


def home_frame_from_shelf(shelf_map_xy: Sequence[float], yaw_map: float) -> HomeFrame:
    """Place the home frame so the measured shelf lands at `SHELF_HOME_XY`."""
    cos_yaw, sin_yaw = math.cos(yaw_map), math.sin(yaw_map)
    sx, sy = SHELF_HOME_XY
    return HomeFrame(
        origin_x=float(shelf_map_xy[0]) - (cos_yaw * sx - sin_yaw * sy),
        origin_y=float(shelf_map_xy[1]) - (sin_yaw * sx + cos_yaw * sy),
        yaw=yaw_map,
    )


def load_lab_config(path: Path) -> dict[str, Any]:
    """Parse the lab config; raise ValueError on a malformed file."""
    raw = cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )
    if not isinstance(raw, dict) or "objects" not in raw or "shelf" not in raw:
        raise ValueError(f"{path}: expected a mapping with 'shelf' and 'objects'")
    for i, entry in enumerate(raw["objects"], start=1):
        for key in ("marker_id", "width", "height"):
            if key not in entry:
                raise ValueError(f"{path}: object #{i} is missing '{key}'")
    return raw


def read_payload_from_json(path: Path) -> dict[str, Any]:
    """A recorded marker-detector payload (`--from-json`); JSON stringifies the
    integer marker ids, so normalize them back."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["targets"] = {int(k): v for k, v in payload.get("targets", {}).items()}
    payload["poses"] = {int(k): v for k, v in payload.get("poses", {}).items()}
    return payload


def build_scene(
    payload: Mapping[str, Any],
    lab_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Detections + measured sizes -> (alphatamp scene dict, warnings)."""
    frame = home_frame_from_shelf(
        lab_config["shelf"]["map_xy"], float(lab_config["shelf"].get("yaw_map", 0.0))
    )
    targets = payload.get("targets", {})
    objects: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, entry in enumerate(lab_config["objects"], start=1):
        marker_id = int(entry["marker_id"])
        name = str(entry.get("name") or f"obj_goal{i}")
        if marker_id not in targets:
            raise ValueError(
                f"marker {marker_id} ({name}) was not detected; every object's "
                "marker must be visible when the scene is captured"
            )
        x, y = frame.map_to_home(*targets[marker_id])
        obj: dict[str, Any] = {
            "name": name,
            "width": float(entry["width"]),
            "height": float(entry["height"]),
            "floor": [round(x, 4), round(y, 4)],
        }
        if "depth" in entry:
            obj["depth"] = float(entry["depth"])
        objects.append(obj)
        if not (
            FLOOR_X_RANGE[0] <= x <= FLOOR_X_RANGE[1]
            and FLOOR_Y_RANGE[0] <= y <= FLOOR_Y_RANGE[1]
        ):
            warnings.append(
                f"{name}: floor ({x:.2f}, {y:.2f}) is outside the staging band "
                f"x {FLOOR_X_RANGE} y {FLOOR_Y_RANGE}"
            )
        if not WIDTH_RANGE[0] <= obj["width"] <= WIDTH_RANGE[1]:
            warnings.append(
                f"{name}: width {obj['width']:.3f} outside {WIDTH_RANGE} "
                "(gripper aperture)"
            )
        if obj["height"] > MAX_HEIGHT:
            warnings.append(
                f"{name}: height {obj['height']:.3f} > {MAX_HEIGHT} fits no section"
            )
    for a_idx, a in enumerate(objects):
        for b in objects[a_idx + 1 :]:
            dist = math.hypot(
                a["floor"][0] - b["floor"][0], a["floor"][1] - b["floor"][1]
            )
            if dist < MIN_OBJECT_SPACING:
                warnings.append(
                    f"{a['name']} and {b['name']} are {dist:.2f} m apart "
                    f"(< {MIN_OBJECT_SPACING})"
                )
    return {"objects": objects}, warnings


def verify_scene(
    payload: Mapping[str, Any],
    lab_config: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> dict[str, float]:
    """Per-object distance (m) between the scene's planned floor position and
    the freshly detected one — the lab-day staging check."""
    fresh, _ = build_scene(payload, lab_config)
    planned = {o["name"]: o["floor"] for o in scene["objects"]}
    residuals: dict[str, float] = {}
    for obj in fresh["objects"]:
        name = obj["name"]
        if name not in planned:
            raise ValueError(f"scene has no object named {name}")
        residuals[name] = math.hypot(
            obj["floor"][0] - planned[name][0], obj["floor"][1] - planned[name][1]
        )
    return residuals


def robot_home_poses(
    payload: Mapping[str, Any],
    lab_config: Mapping[str, Any],
) -> dict[int, tuple[float, float, float]]:
    """Each detected robot's SE2 pose in the home frame (staging check: the
    robot should be near the origin with heading near zero)."""
    frame = home_frame_from_shelf(
        lab_config["shelf"]["map_xy"], float(lab_config["shelf"].get("yaw_map", 0.0))
    )
    poses: dict[int, tuple[float, float, float]] = {}
    for robot_idx, (x, y, theta) in payload.get("poses", {}).items():
        hx, hy = frame.map_to_home(x, y)
        heading = math.atan2(math.sin(theta - frame.yaw), math.cos(theta - frame.yaw))
        poses[int(robot_idx)] = (hx, hy, heading)
    return poses


def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; see `scripts/export_restock_scene.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/restock3d/objects.yaml"),
        help="lab config: shelf staging + per-marker object sizes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="scene.yaml to write (or to verify against with --verify)",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="use a recorded marker-detector payload instead of a live query",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="marker-detector host (default: the client's built-in hostname)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare fresh detections against the existing --out scene "
        "instead of writing it",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max per-object staging residual (m) for --verify to pass",
    )
    args = parser.parse_args(argv)

    lab_config = load_lab_config(args.config)
    if args.from_json is not None:
        payload = read_payload_from_json(args.from_json)
    else:
        client = (
            MarkerDetectorClient(host=args.host)
            if args.host is not None
            else MarkerDetectorClient()
        )
        try:
            payload = client.get_latest()
        finally:
            client.close()

    for robot_idx, (x, y, heading) in robot_home_poses(payload, lab_config).items():
        print(
            f"robot {robot_idx}: home-frame pose ({x:.3f}, {y:.3f}, {heading:.3f}) "
            "(staging target: 0, 0, 0)"
        )

    if args.verify:
        scene = cast(
            dict[str, Any],
            OmegaConf.to_container(OmegaConf.load(args.out), resolve=True),
        )
        residuals = verify_scene(payload, lab_config, scene)
        worst = max(residuals.values(), default=0.0)
        for name, residual in sorted(residuals.items()):
            marker = "OK " if residual <= args.tolerance else "BAD"
            print(f"{marker} {name}: {residual * 100:.1f} cm from the planned spot")
        return 0 if worst <= args.tolerance else 1

    scene, warnings = build_scene(payload, lab_config)
    for warning in warnings:
        print(f"WARNING: {warning}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(scene), f=args.out)
    print(f"Wrote {len(scene['objects'])} objects to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
