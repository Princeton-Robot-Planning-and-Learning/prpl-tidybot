"""Tests for `prpl_tidybot.restock_scene` (marker detections -> scene.yaml)."""

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from prpl_tidybot.restock_scene import (
    SHELF_HOME_XY,
    build_scene,
    cli,
    home_frame_from_shelf,
    robot_home_poses,
    staging_sheet,
    verify_scene,
)

# Shelf staged as in the plan doc: map (1.5, 1.5), home frame axis-aligned
# with the map, so the home origin is map (1.1, 0.1).
_LAB_CONFIG = {
    "shelf": {"map_xy": [1.5, 1.5], "yaw_map": 0.0},
    "objects": [
        {"marker_id": 35, "width": 0.07, "height": 0.125, "depth": 0.07},
        {"marker_id": 36, "name": "obj_goal2", "width": 0.05, "height": 0.10},
    ],
}

# Home-frame floor spots (-0.5, 0.8) and (-0.3, 1.0) expressed in the map
# frame (home origin (1.1, 0.1)).
_PAYLOAD = {
    "poses": {0: (1.1, 0.1, 0.02)},
    "targets": {35: (0.6, 0.9), 36: (0.8, 1.1)},
}


def test_home_frame_round_trip() -> None:
    """The home frame puts the measured shelf at Restock3D's shelf position, at any
    staging yaw."""
    frame = home_frame_from_shelf([1.5, 1.5], yaw_map=math.pi / 3)
    assert frame.map_to_home(1.5, 1.5) == pytest.approx(SHELF_HOME_XY)
    x, y = frame.home_to_map(-0.5, 0.8)
    assert frame.map_to_home(x, y) == pytest.approx((-0.5, 0.8))


def test_build_scene_transforms_targets() -> None:
    """Marker positions land at the expected home-frame floor spots; names default in
    order; depth is passed through only when given."""
    scene, warnings = build_scene(_PAYLOAD, _LAB_CONFIG)
    assert not warnings
    first, second = scene["objects"][0], scene["objects"][1]
    assert first["name"] == "obj_goal1"
    assert first["floor"] == pytest.approx([-0.5, 0.8])
    assert first["depth"] == pytest.approx(0.07)
    assert second["name"] == "obj_goal2"
    assert second["floor"] == pytest.approx([-0.3, 1.0])
    assert "depth" not in second


def test_build_scene_warns_and_rejects() -> None:
    """A floor spot outside the staging band or too close to a neighbour warns; an
    undetected marker is an error."""
    payload = {"targets": {35: (0.6, 0.9), 36: (0.61, 0.9)}}
    _, warnings = build_scene(payload, _LAB_CONFIG)
    assert any("apart" in w for w in warnings)
    payload = {"targets": {35: (1.4, 0.9), 36: (0.8, 1.1)}}
    _, warnings = build_scene(payload, _LAB_CONFIG)
    assert any("staging band" in w for w in warnings)
    with pytest.raises(ValueError, match="marker 36"):
        build_scene({"targets": {35: (0.6, 0.9)}}, _LAB_CONFIG)


def test_verify_scene_residuals() -> None:
    """Residuals measure how far each staged object is from the planned spot."""
    scene, _ = build_scene(_PAYLOAD, _LAB_CONFIG)
    moved = {
        "poses": {},
        "targets": {35: (0.6, 0.93), 36: (0.8, 1.1)},
    }
    residuals = verify_scene(moved, _LAB_CONFIG, scene)
    assert residuals["obj_goal1"] == pytest.approx(0.03)
    assert residuals["obj_goal2"] == pytest.approx(0.0)


def test_robot_home_poses() -> None:
    """The robot's map pose transforms to a near-origin home pose when staged
    correctly."""
    poses = robot_home_poses(_PAYLOAD, _LAB_CONFIG)
    x, y, heading = poses[0]
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert heading == pytest.approx(0.02)


def test_cli_write_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """End to end through the CLI: write a scene from a recorded payload, then verify
    against it (passing), then fail verification past the tolerance."""
    config_path = tmp_path / "objects.yaml"
    OmegaConf.save(config=OmegaConf.create(_LAB_CONFIG), f=config_path)
    payload_path = tmp_path / "payload.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(_PAYLOAD, f)
    out_path = tmp_path / "scene.yaml"

    common = ["--config", str(config_path), "--out", str(out_path)]
    assert cli(common + ["--from-json", str(payload_path)]) == 0
    scene = OmegaConf.to_container(OmegaConf.load(out_path), resolve=True)
    assert isinstance(scene, dict) and len(scene["objects"]) == 2

    assert cli(common + ["--from-json", str(payload_path), "--verify"]) == 0

    moved = {"poses": {}, "targets": {35: (0.6, 1.0), 36: (0.8, 1.1)}}
    moved_path = tmp_path / "moved.json"
    with open(moved_path, "w", encoding="utf-8") as f:
        json.dump(moved, f)
    assert cli(common + ["--from-json", str(moved_path), "--verify"]) == 1
    assert "BAD" in capsys.readouterr().out


def test_staging_sheet_round_trips_floor_positions() -> None:
    """The sheet's map-frame taping coordinates transform back to the scene's planned
    home-frame floor spots."""
    scene, _ = build_scene(_PAYLOAD, _LAB_CONFIG)
    lines = staging_sheet(_LAB_CONFIG, scene)
    assert any("robot start: map (1.100, 0.100)" in line for line in lines)
    assert any(
        "obj_goal1 (marker 35): tape at map (0.600, 0.900)" in line for line in lines
    )
    assert any(
        "obj_goal2 (marker 36): tape at map (0.800, 1.100)" in line for line in lines
    )


def test_cli_staging_sheet(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--staging-sheet prints taping coordinates from an existing scene without
    touching the marker detector."""
    config_path = tmp_path / "objects.yaml"
    OmegaConf.save(config=OmegaConf.create(_LAB_CONFIG), f=config_path)
    payload_path = tmp_path / "payload.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(_PAYLOAD, f)
    out_path = tmp_path / "scene.yaml"
    common = ["--config", str(config_path), "--out", str(out_path)]
    assert cli(common + ["--from-json", str(payload_path)]) == 0
    capsys.readouterr()
    assert cli(common + ["--staging-sheet"]) == 0
    out = capsys.readouterr().out
    assert "tape at map (0.600, 0.900)" in out
