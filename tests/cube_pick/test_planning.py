"""The arm model the pick plans with must have the same wrist the robot executes with."""

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("pybullet_helpers")

from prpl_tidybot.cube_pick.planning import planning_arm_urdf  # noqa: E402


def _collision_meshes(link_name: str) -> list[str]:
    root = ET.parse(planning_arm_urdf()).getroot()
    (link,) = [link for link in root.iter("link") if link.get("name") == link_name]
    return [m.get("filename", "") for c in link.iter("collision") for m in c.iter("mesh")]


def test_the_bracelet_collides_with_its_vision_module() -> None:
    # KINDER's MuJoCo model and the real TidyBot both carry the wrist camera inside the
    # bracelet; planning with the bare bracelet lets the camera hit container walls.
    meshes = _collision_meshes("bracelet_link")
    assert meshes and all(m.endswith("bracelet_with_vision_link.STL") for m in meshes)


def test_the_realsense_stand_in_box_is_gone() -> None:
    root = ET.parse(planning_arm_urdf()).getroot()
    boxes = [b for b in root.iter("box") if b.get("size") == "0.025 0.09 0.035"]
    assert not boxes
