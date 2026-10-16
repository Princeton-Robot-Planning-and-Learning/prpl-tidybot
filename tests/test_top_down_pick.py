"""Tests for the top-down cube pick ported from Jimmy Wu's TidyBot ``pick_object``."""

import math
from importlib import resources
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest
from omegaconf import OmegaConf

from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    arm_frame_target,
    forward_kinematics,
    geometry_from_config,
    grasp_succeeded,
    standoff_pose,
    top_down_joint_targets,
)

FLOOR_BOUNDS = (-1.5, 1.7, -1.5, 1.5)
GEOM = TopDownPickGeometry()
_CONF = Path(__file__).resolve().parent.parent / "conf" / "toss_cycles.yaml"


def _pad_bottom_z(joints):
    """Lowest point of the open Robotiq 2F-85's finger pads, in the arm frame, from
    KINDER's Kinova Gen3 + 2F-85 model."""
    xml = resources.files("kinder") / "envs/dynamic3d/models/kinova_gen3/gen3_2f85.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))  # pylint: disable=no-member
    data = mujoco.MjData(model)  # pylint: disable=no-member
    data.qpos[:7] = joints
    mujoco.mj_forward(model, data)  # pylint: disable=no-member
    lowest = math.inf
    for g in range(model.ngeom):
        if not model.body(model.geom_bodyid[g]).name.endswith("_pad"):
            continue
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        corners = (
            data.geom_xpos[g]
            + (signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T
        )
        lowest = min(lowest, corners[:, 2].min())
    return lowest


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize(
    "robot,cube",
    [
        ((1.32, -0.18, math.pi), (0.65, 0.65)),
        ((0.49, -0.49, -3.13), (-0.8, -0.5)),
        ((0.0, 0.0, 0.0), (1.0, 0.2)),
    ],
)
def test_standoff_faces_the_cube_at_the_pick_range(robot, cube):
    """The base parks `standoff` from the cube, on the robot's side, facing it."""
    x, y, theta = standoff_pose(robot, cube, GEOM, FLOOR_BOUNDS)
    assert math.hypot(cube[0] - x, cube[1] - y) == pytest.approx(
        GEOM.standoff, abs=1e-6
    )
    assert _wrap(math.atan2(cube[1] - y, cube[0] - x) - theta) == pytest.approx(
        0.0, abs=1e-6
    )
    assert (
        FLOOR_BOUNDS[0] <= x <= FLOOR_BOUNDS[1]
        and FLOOR_BOUNDS[2] <= y <= FLOOR_BOUNDS[3]
    )


def test_standoff_swings_round_when_the_direct_approach_leaves_the_floor():
    """A cube near the edge is approached from a side whose standoff stays drivable."""
    x, y, _ = standoff_pose((1.6, 0.0, math.pi), (1.3, 1.3), GEOM, FLOOR_BOUNDS)
    assert FLOOR_BOUNDS[2] <= y <= FLOOR_BOUNDS[3]
    assert math.hypot(1.3 - x, 1.3 - y) == pytest.approx(GEOM.standoff, abs=1e-6)


def test_arm_frame_target_of_a_cube_straight_ahead():
    """Facing the cube from the standoff puts it straight ahead of the arm mount."""
    pose = (0.0, 0.0, 0.3)
    cube = (GEOM.standoff * math.cos(0.3), GEOM.standoff * math.sin(0.3))
    ax, ay = arm_frame_target(pose, cube, GEOM)
    assert ax == pytest.approx(GEOM.standoff - GEOM.mounting_offset, abs=1e-9)
    assert ay == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("cube_yaw_rel", [0.0, 0.3, -0.6, 0.78])
def test_joint_targets_put_the_gripper_above_then_at_the_cube_pointing_down(
    cube_yaw_rel,
):
    """IK targets reach the hover and grasp points, the tool pointing straight down and
    the fingers turned with the cube (modulo the cube's 90-degree symmetry)."""
    ax, ay = GEOM.standoff - GEOM.mounting_offset, 0.02
    hover, grasp = top_down_joint_targets(ax, ay, cube_yaw_rel, GEOM)
    for q, z in (
        (hover, GEOM.floor_z + GEOM.grasp_height + GEOM.hover_height),
        (grasp, GEOM.floor_z + GEOM.grasp_height),
    ):
        pos, approach, closing = forward_kinematics(q, GEOM)
        assert np.allclose(pos, [ax, ay, z], atol=0.005)
        assert approach[2] < -0.995  # pointing straight down
        finger_yaw = math.atan2(closing[1], closing[0])
        diff = (finger_yaw - cube_yaw_rel - GEOM.wrist_yaw_offset + math.pi / 4) % (
            math.pi / 2
        ) - math.pi / 4
        assert abs(diff) < math.radians(2)


def test_grasp_check_matches_the_measured_gripper_readings():
    """On the 5 cm cube the gripper read 0.49; closing on nothing reads ~1.0."""
    assert grasp_succeeded(0.49, GEOM)
    assert not grasp_succeeded(0.996, GEOM)
    assert not grasp_succeeded(0.05, GEOM)  # did not close at all


def test_the_configured_grasp_stops_the_finger_pads_just_above_the_floor():
    """The pads reach 3.5 cm below the IK's pinch point; aiming the pinch point at the
    cube's middle drove them 1 cm into the floor on the robot (2026-10-14). The
    configured grasp leaves them a centimetre or so clear, low enough to grip the cube.
    """
    geom = geometry_from_config(OmegaConf.load(_CONF).pick.top_down)
    _, grasp = top_down_joint_targets(
        geom.standoff - geom.mounting_offset, 0.0, 0.3, geom
    )
    clearance = _pad_bottom_z(grasp) - geom.floor_z
    assert 0.005 < clearance < 0.02
