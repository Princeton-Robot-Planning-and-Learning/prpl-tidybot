"""The pick out of the bin's grasps (conf/same_side_practice.yaml's
``pick.top_down.in_bin``), the near-wall controller by default:

the gripper's yaw up to 25 degrees off the cube's faces, tilts about the fingers' travel
axis (kinder-baselines #178), fully open and then at the cube's width plus 2 cm, checked
against the real 2F-85's links, every link ``wall_margin`` off the walls. The usual
face-aligned, untilted, fully open grasps come first. ``finger_allowance`` (light wall
contact for the finger linkage only) stays off.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.gripper_shape import (
    FINGER_LINKS,
    gripper_link_points,
    opening_command,
)
from prpl_tidybot.robot_skills import (
    InBinPick,
    _bin_grasp_geometry,
    bin_grasp_candidates,
    part_wall_problem,
)
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    arm_frame_target,
    forward_kinematics,
    geometry_from_config,
    top_down_joint_targets,
)

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
BIN = BinFit(-0.29, -0.77, math.pi / 2, 0.39, 0.29, 0.21, 0.005)
BASE = (BIN.x + 0.67, BIN.y, math.pi)


def _compose(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=["mode=fake", *overrides])


YAWS = [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0]
TILTS = [0.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0]
# The controller the real robot ran through run 10: the usual two grasps, the box.
LEGACY = (
    "pick.top_down.in_bin.yaw_offsets_deg=[0]",
    "pick.top_down.in_bin.tilts_deg=[0]",
    "pick.top_down.in_bin.openings_mm=[85]",
    "pick.top_down.in_bin.gripper_model=box",
)


def test_the_defaults_are_the_near_wall_controller():
    """Yaw within +-25 deg of the faces, tilts 0/+-10/+-15/+-20, 85 then 70 mm, the real
    gripper's links, the 1 cm margin for every link (no finger allowance)."""
    in_bin = _compose().pick.top_down.in_bin
    assert [float(v) for v in in_bin.yaw_offsets_deg] == YAWS
    assert [float(v) for v in in_bin.tilts_deg] == TILTS
    assert [float(v) for v in in_bin.openings_mm] == [85.0, 70.0]
    assert in_bin.gripper_model == "mesh"
    assert in_bin.finger_allowance == 0.0
    assert in_bin.wall_margin == 0.01
    default = InBinPick()
    assert [float(v) for v in default.yaw_offsets_deg] == YAWS
    assert [float(v) for v in default.tilts_deg] == TILTS
    assert default.openings_mm == (85.0, 70.0)
    assert (default.gripper_model, default.finger_allowance) == ("mesh", 0.0)
    assert default.wall_margin == 0.01
    geom = TopDownPickGeometry()
    assert (geom.tilt, geom.open_position) == (0.0, 0.0)


def _key(g, geom):
    return (
        round(math.degrees(g.wrist_yaw_offset - geom.wrist_yaw_offset), 6),
        round(math.degrees(g.tilt), 6),
        round(math.degrees(g.closing_turn), 6),
        round(g.open_position, 6),
    )


def test_the_usual_two_grasps_come_first_so_every_pick_that_works_keeps_its_grasp():
    """Face-aligned, untilted, fully open on the first pair of faces and then the
    second, exactly the old controller's two grasps, before any new one; no grasp is
    tried twice."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    cands = _candidates(cfg)
    keys = [_key(g, geom) for g in cands]
    assert keys[:2] == [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 90.0, 0.0)]
    legacy = _candidates(_compose(*LEGACY))
    assert [_key(g, geom) for g in legacy] == keys[:2]
    assert len(set(keys)) == len(keys)
    # 11 yaws x 7 tilts x 2 pairs of faces x 2 openings, less 70 mm at +-25 deg.
    assert len(keys) == 11 * 7 * 2 * 2 - 2 * 7 * 2


def test_70_mm_is_skipped_where_the_cube_spans_more_than_it_can_take():
    """At 25 deg off the faces the cube spans 6.6 cm across the closing axis, too wide
    for the 70 mm opening with 5 mm to spare: tried fully open only; at 20 deg (6.4
    cm) both openings."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    by_yaw: dict[float, set[float]] = {}
    for g in _candidates(cfg):
        by_yaw.setdefault(_key(g, geom)[0], set()).add(round(g.open_position, 6))
    partial = round(opening_command(0.070), 6)
    assert by_yaw[25.0] == by_yaw[-25.0] == {0.0}
    assert by_yaw[20.0] == by_yaw[-20.0] == by_yaw[0.0] == {0.0, partial}


def test_an_untilted_grasp_is_the_current_one_bit_for_bit():
    """Tilt 0 takes the existing code path: the same joints as before."""
    geom = geometry_from_config(_compose().pick.top_down)
    ax, ay = arm_frame_target(BASE, (BIN.x, BIN.y), geom)
    before = top_down_joint_targets(ax, ay, 0.3, geom)
    after = top_down_joint_targets(ax, ay, 0.3, dataclasses.replace(geom, tilt=0.0))
    assert before == after


@pytest.mark.parametrize("tilt_deg", [10.0, -15.0, 20.0])
def test_a_tilt_turns_the_tool_about_the_fingers_travel(tilt_deg):
    """#178's semantics: the grasp frame turns about the axis the fingers travel along,
    so the pads stay square to the faces they close on.

    The pinch point stays at the grasp, and the hover is ``hover_height`` higher, back
    along the tilted tool axis.
    """
    geom = geometry_from_config(_compose().pick.top_down)
    ax, ay = arm_frame_target(BASE, (BIN.x, BIN.y), geom)
    flat_hover, flat = top_down_joint_targets(ax, ay, 0.3, geom)
    tilted = dataclasses.replace(geom, tilt=math.radians(tilt_deg))
    hover, grasp = top_down_joint_targets(ax, ay, 0.3, tilted, seed=flat_hover)
    p0, a0, c0 = forward_kinematics(flat, geom)
    p1, a1, c1 = forward_kinematics(grasp, tilted)
    travel0, travel1 = np.cross(c0, a0), np.cross(c1, a1)
    assert np.linalg.norm(p1 - p0) < 0.005
    assert abs(float(travel0 @ travel1)) > math.cos(math.radians(1.0))
    assert math.degrees(math.acos(-a1[2])) == pytest.approx(abs(tilt_deg), abs=1.0)
    ph, ah, _ = forward_kinematics(hover, tilted)
    assert ph[2] - p1[2] == pytest.approx(geom.hover_height, abs=0.005)
    back = (ph - p1) / np.linalg.norm(ph - p1)
    assert float(back @ -a1) > math.cos(math.radians(2.5))
    assert float(ah @ a1) > math.cos(math.radians(2.5))


def test_the_mesh_gripper_fully_open_spans_the_measured_box():
    """KINDER's gen3_2f85, in the server's tool frame (x the fingers' travel, y across,
    z the approach): fully open it fills the 14.8 x 7.6 cm box, 12.4 cm up and 2.9 cm
    down from the pinch point."""
    points = np.vstack(list(gripper_link_points(0.0).values()))
    assert np.abs(points[:, 0]).max() == pytest.approx(0.0741, abs=0.003)
    assert np.abs(points[:, 1]).max() == pytest.approx(0.0375, abs=0.003)
    assert points[:, 2].min() == pytest.approx(-0.124, abs=0.004)
    assert points[:, 2].max() == pytest.approx(0.029, abs=0.004)
    assert FINGER_LINKS <= set(gripper_link_points(0.0))


def test_the_partial_opening_is_178s_70_mm_command():
    """#178 opens to the cube's width plus 2 cm, commanded through its measured
    MuJoCo gap table: 70 mm is a command of about 0.2."""
    assert opening_command(0.085) == 0.0
    assert opening_command(0.070) == pytest.approx(0.198, abs=0.005)
    open_span = np.abs(np.vstack(list(gripper_link_points(0.0).values()))[:, 0]).max()
    partial = np.vstack(list(gripper_link_points(opening_command(0.070)).values()))
    assert np.abs(partial[:, 0]).max() < open_span - 0.003


def _part(u_past_inner):
    """A small block of points below the rim, inside the bin and reaching
    ``u_past_inner`` past the inner face of the +u wall (bin frame)."""
    inner = 0.5 * BIN.length - BIN.wall
    u = np.linspace(inner - 0.03, inner + u_past_inner, 8)
    return np.c_[u, np.zeros_like(u)]


@pytest.mark.parametrize(
    "name, past, allowance, touches",
    [
        ("left_follower", 0.004, 0.0, True),
        ("left_follower", 0.004, 0.005, False),
        ("left_follower", 0.008, 0.005, True),
        ("right_pad", -0.002, 0.005, False),
        ("bracelet_link", 0.004, 0.02, True),
        ("base", 0.004, 0.02, True),
    ],
)
def test_finger_allowance_lets_only_the_linkage_touch(name, past, allowance, touches):
    """B: the finger links may reach up to ``finger_allowance`` past the inner face; the
    gripper housing and the arm links keep the hard ``wall_margin``."""
    in_bin = dataclasses.replace(InBinPick(), finger_allowance=allowance)
    problem = part_wall_problem(name, _part(past), BIN, in_bin)
    assert (problem is not None) == touches


def test_a_refusal_names_both_pairs_of_faces():
    """A cube against a wall is refused with the reason for each pair of faces, so a
    bracelet collision on the other pair is visible."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    in_bin = cfg.pick.top_down.in_bin
    cube = {"x": BIN.x + 0.15, "y": BIN.y + 0.11, "yaw_deg": 45.0}  # a corner
    chosen, problem = _bin_grasp_geometry(
        geom, BASE, cube, BIN, in_bin, RETRACT_ARM_CONF
    )
    assert chosen is None
    assert "faces 0 deg" in problem and "faces 90 deg" in problem


def test_the_tilted_candidates_pick_a_cube_the_current_grasp_refuses():
    """6.0 cm from the far wall (the one away from the robot), at 30 degrees: the old
    controller refuses; a tilted grasp of the default clears it."""
    cube = {"x": BIN.x - 0.08, "y": BIN.y, "yaw_deg": 30.0}
    current = _compose(*LEGACY)
    geom = geometry_from_config(current.pick.top_down)
    chosen, _ = _bin_grasp_geometry(
        geom, BASE, cube, BIN, current.pick.top_down.in_bin, RETRACT_ARM_CONF
    )
    assert chosen is None
    cfg = _compose()
    chosen, problem = _bin_grasp_geometry(
        geometry_from_config(cfg.pick.top_down),
        BASE,
        cube,
        BIN,
        cfg.pick.top_down.in_bin,
        RETRACT_ARM_CONF,
    )
    assert problem is None, problem
    assert chosen is not None
    assert chosen.tilt != 0.0


def _candidates(cfg):
    return bin_grasp_candidates(
        geometry_from_config(cfg.pick.top_down), cfg.pick.top_down.in_bin
    )


def test_a_yaw_relaxed_grasp_turns_the_closing_axis_off_the_faces():
    """C1: with an offset the closing axis sits that far off the faces' normal."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    ax, ay = arm_frame_target(BASE, (BIN.x, BIN.y), geom)
    _, flat = top_down_joint_targets(ax, ay, 0.0, geom)
    relaxed = dataclasses.replace(geom, wrist_yaw_offset=math.radians(15.0))
    _, turned = top_down_joint_targets(ax, ay, 0.0, relaxed)
    c0 = forward_kinematics(flat, geom)[2]
    c1 = forward_kinematics(turned, relaxed)[2]
    assert math.degrees(math.acos(abs(float(c0 @ c1)))) == pytest.approx(15.0, abs=1.0)
