"""Candidate controllers for the pick out of the bin, each a config option that is off
by default (conf/same_side_practice.yaml's ``pick.top_down.in_bin``):

A. the tilted, partly opened grasp of kinder-baselines #178: tilts about the fingers'
travel axis, each tried fully open and then at the cube's width plus 2 cm, checked
against the real 2F-85's links (``gripper_model: mesh``) instead of one box; B. light
wall contact for the finger linkage only (``finger_allowance``, m past the    inner wall
face), the housing, wrist and arm keeping ``wall_margin``.
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
TILTS_A = "[0,10,-10,15,-15,20,-20]"


def _compose(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=["mode=fake", *overrides])


def _variant_a(*extra):
    return _compose(
        f"pick.top_down.in_bin.tilts_deg={TILTS_A}",
        "pick.top_down.in_bin.openings_mm=[85,70]",
        "pick.top_down.in_bin.gripper_model=mesh",
        *extra,
    )


def test_the_defaults_are_the_current_controller():
    """Nothing changes unless asked: one untilted, fully open grasp per pair of faces,
    checked against the box, no wall contact allowed."""
    in_bin = _compose().pick.top_down.in_bin
    assert list(in_bin.tilts_deg) == [0.0]
    assert list(in_bin.openings_mm) == [85.0]
    assert in_bin.gripper_model == "box"
    assert in_bin.finger_allowance == 0.0
    default = InBinPick()
    assert (default.tilts_deg, default.openings_mm) == ((0.0,), (85.0,))
    assert (default.gripper_model, default.finger_allowance) == ("box", 0.0)
    geom = TopDownPickGeometry()
    assert (geom.tilt, geom.open_position) == (0.0, 0.0)


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
    """With the candidates on, a cube against a wall is refused with the reason for each
    pair of faces, so a bracelet collision on the other pair is visible."""
    cfg = _variant_a()
    geom = geometry_from_config(cfg.pick.top_down)
    in_bin = cfg.pick.top_down.in_bin
    cube = {"x": BIN.x + 0.15, "y": BIN.y + 0.11, "yaw_deg": 45.0}  # a corner
    chosen, problem = _bin_grasp_geometry(
        geom, BASE, cube, BIN, in_bin, RETRACT_ARM_CONF
    )
    assert chosen is None
    assert "faces 0 deg" in problem and "faces 90 deg" in problem


def test_the_tilted_candidates_pick_a_cube_the_current_grasp_refuses():
    """6.0 cm from the far wall (the one away from the robot), at 30 degrees: the
    current grasp refuses; a tilted candidate of A clears it."""
    cube = {"x": BIN.x - 0.08, "y": BIN.y, "yaw_deg": 30.0}
    current = _compose()
    geom = geometry_from_config(current.pick.top_down)
    chosen, _ = _bin_grasp_geometry(
        geom, BASE, cube, BIN, current.pick.top_down.in_bin, RETRACT_ARM_CONF
    )
    assert chosen is None
    cfg = _variant_a()
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


def test_the_gripper_yaw_is_tied_to_the_faces_by_default():
    """C is off by default: one gripper yaw per pair of faces."""
    assert list(_compose().pick.top_down.in_bin.yaw_offsets_deg) == [0.0]
    assert InBinPick().yaw_offsets_deg == (0.0,)


def _candidates(cfg):
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        bin_grasp_candidates,
    )

    return bin_grasp_candidates(
        geometry_from_config(cfg.pick.top_down), cfg.pick.top_down.in_bin
    )


def test_an_edge_grasp_closes_across_the_diagonal_fully_open_only():
    """C2: the fingers close on two opposite vertical edges, the gripper turned 45
    degrees off the faces.

    The diagonal (7.07 cm) does not fit the 70 mm opening, so an edge grasp is tried
    fully open only; face-aligned grasps come first.
    """
    cfg = _variant_a("pick.top_down.in_bin.yaw_offsets_deg=[0,45]")
    cands = _candidates(cfg)
    offsets = [round(math.degrees(g.wrist_yaw_offset)) for g in cands]
    assert offsets[0] == 0 and offsets[-1] == 45
    assert offsets.index(45) > max(i for i, o in enumerate(offsets) if o == 0)
    edge = [g for g in cands if round(math.degrees(g.wrist_yaw_offset)) == 45]
    assert edge and all(g.open_position == 0.0 for g in edge)
    face = [g for g in cands if g.wrist_yaw_offset == 0.0]
    assert {g.open_position for g in face} == {0.0, opening_command(0.070)}


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
