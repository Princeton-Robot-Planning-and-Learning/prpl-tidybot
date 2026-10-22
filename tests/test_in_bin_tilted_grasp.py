"""The pick out of the bin's grasps (conf/same_side_practice.yaml's
``pick.top_down.in_bin``), the near-wall controller and then the oblique extension by
default:

the gripper's yaw up to 25 degrees off the cube's faces, tilts about the fingers' travel
axis (kinder-baselines #178), fully open and then at the cube's width plus 2 cm, checked
against the real 2F-85's links, every link ``wall_margin`` off the walls; then yaw up to
45 degrees, tilts up to 30 and 85/80/75/70 mm. The usual face-aligned, untilted, fully
open grasps come first, then the near-wall set in its own order. ``finger_allowance``
(light wall contact for the finger linkage only) stays off.
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
    _bin_pick_targets,
    _BinGraspSearch,
    bin_grasp_candidates,
    part_wall_problem,
)
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    arm_frame_target,
    arm_link_points,
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
EXT_YAWS = [0.0] + [s * k for k in range(5, 50, 5) for s in (1.0, -1.0)]
EXT_TILTS = [0.0] + [s * k for k in range(5, 35, 5) for s in (1.0, -1.0)]
EXT_OPENINGS = [85.0, 80.0, 75.0, 70.0]
# kinder-baselines #178's MuJoCo gap table, interpolated: 85 fully open, 70 measured
# between 0.1 and 0.2, 80 and 75 interpolated.
COMMANDS = {85.0: 0.0, 80.0: 0.06892, 75.0: 0.13418, 70.0: 0.19747}
# The controller the real robot ran through run 10: the usual two grasps, the box.
LEGACY = (
    "pick.top_down.in_bin.yaw_offsets_deg=[0]",
    "pick.top_down.in_bin.tilts_deg=[0]",
    "pick.top_down.in_bin.openings_mm=[85]",
    "pick.top_down.in_bin.more_grasps=[]",
    "pick.top_down.in_bin.gripper_model=box",
)
# The default before the oblique extension (bd85a79): the near-wall set only.
NEAR_WALL = ("pick.top_down.in_bin.more_grasps=[]",)


def test_the_defaults_are_the_near_wall_controller_then_the_oblique_extension():
    """Yaw within +-25 deg of the faces, tilts 0/+-10/+-15/+-20, 85 then 70 mm; then yaw
    within +-45 deg, tilts every 5 deg to +-30, 85/80/75/70 mm; the real gripper's
    links, the 1 cm margin for every link (no finger allowance)."""
    in_bin = _compose().pick.top_down.in_bin
    assert [float(v) for v in in_bin.yaw_offsets_deg] == YAWS
    assert [float(v) for v in in_bin.tilts_deg] == TILTS
    assert [float(v) for v in in_bin.openings_mm] == [85.0, 70.0]
    (more,) = in_bin.more_grasps
    assert [float(v) for v in more.yaw_offsets_deg] == EXT_YAWS
    assert [float(v) for v in more.tilts_deg] == EXT_TILTS
    assert [float(v) for v in more.openings_mm] == EXT_OPENINGS
    assert in_bin.gripper_model == "mesh"
    assert in_bin.finger_allowance == 0.0
    assert in_bin.wall_margin == 0.01
    default = InBinPick()
    assert [float(v) for v in default.yaw_offsets_deg] == YAWS
    assert [float(v) for v in default.tilts_deg] == TILTS
    assert default.openings_mm == (85.0, 70.0)
    (ext,) = default.more_grasps
    assert list(ext.yaw_offsets_deg) == EXT_YAWS
    assert list(ext.tilts_deg) == EXT_TILTS
    assert list(ext.openings_mm) == EXT_OPENINGS
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
    # The near-wall set: 11 yaws x 7 tilts x 2 pairs of faces x 2 openings, less 70 mm
    # at +-25 deg (280); the oblique extension: 19 yaws x 13 tilts x 2 pairs of faces
    # x 4 openings, less 70 mm at +-25..+-45 deg and 75 mm at +-40..+-45 deg (1612),
    # the 280 among them.
    near_wall = _candidates(_compose(*NEAR_WALL))
    assert len(near_wall) == 11 * 7 * 2 * 2 - 2 * 7 * 2 == 280
    assert len(keys) == 19 * 13 * 2 * 4 - 10 * 13 * 2 - 4 * 13 * 2 == 1612


def _spec():
    """The oblique extension's candidate list, written out from the spec (hitl-pmp-
    results/kb-restack-notes-workstream-c-geometry.md, "Oblique extension"):

    (yaw off the faces, tilt, faces, gripper command), in try order.
    """

    def grasps(yaws, tilts, openings):
        for y in yaws:
            span = 0.05 * (
                abs(math.cos(math.radians(y))) + abs(math.sin(math.radians(y)))
            )
            for t in tilts:
                for f in (0.0, 90.0):
                    for o in openings:
                        if not (y and o / 1000 < span + 0.005):
                            yield (y, t, f, o)

    near_wall = list(grasps(YAWS, TILTS, [85.0, 70.0]))
    usual = [g for g in near_wall if g[0] == 0 and g[1] == 0 and g[3] == 85.0]
    order = usual + [g for g in near_wall if g not in usual]
    order += [g for g in grasps(EXT_YAWS, EXT_TILTS, EXT_OPENINGS) if g not in order]
    return [(y, t, f, COMMANDS[o]) for y, t, f, o in order]


def test_the_candidate_list_is_the_spec_in_order():
    """Exactly the spec's 1612 grasps, in its order: the usual two, the rest of the
    near-wall set as bd85a79 tried it, then the oblique extension's new grasps."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    keys = [_key(g, geom) for g in _candidates(cfg)]
    spec = _spec()
    assert len(spec) == 1612
    assert [k[:3] for k in keys] == [s[:3] for s in spec]
    assert [k[3] for k in keys] == pytest.approx([s[3] for s in spec], abs=1e-5)


def test_bd85a79s_280_grasps_come_first_in_their_own_order():
    """So every pick the near-wall set clears keeps its exact grasp."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    keys = [_key(g, geom) for g in _candidates(cfg)]
    near_wall = [_key(g, geom) for g in _candidates(_compose(*NEAR_WALL))]
    assert keys[:280] == near_wall


def test_75_mm_is_skipped_too_where_the_cube_spans_7_cm():
    """At 40 and 45 deg off the faces the cube spans 7.0 and 7.1 cm: 85 and 80 mm only;
    at 25-35 deg (6.6-7.0 cm) every opening but 70 mm; to 20 deg all four."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    by_yaw: dict[float, set[float]] = {}
    for g in _candidates(cfg):
        by_yaw.setdefault(_key(g, geom)[0], set()).add(round(g.open_position, 5))
    c = {mm: round(opening_command(mm / 1000), 5) for mm in EXT_OPENINGS}
    for yaw in (40.0, -40.0, 45.0, -45.0):
        assert by_yaw[yaw] == {c[85.0], c[80.0]}
    for yaw in (25.0, -25.0, 30.0, -30.0, 35.0, -35.0):
        assert by_yaw[yaw] == {c[85.0], c[80.0], c[75.0]}
    for yaw in (0.0, 5.0, -10.0, 20.0, -20.0):
        assert by_yaw[yaw] == set(c.values())


def test_80_and_75_mm_are_interpolated_in_178s_gap_table():
    """Between the measured 0.0851 (0.0), 0.0777 (0.1) and 0.0698 (0.2) m gaps."""
    for mm, command in COMMANDS.items():
        assert opening_command(mm / 1000) == pytest.approx(command, abs=1e-5)


def test_70_mm_is_skipped_where_the_cube_spans_more_than_it_can_take():
    """At 25 deg off the faces the cube spans 6.6 cm across the closing axis, too wide
    for the 70 mm opening with 5 mm to spare: tried fully open only; at 20 deg (6.4
    cm) both openings (the near-wall set)."""
    cfg = _compose(*NEAR_WALL)
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


# The test bin's walls as the robot at BASE (facing -x) sees them.
_HL, _HW = 0.5 * BIN.length - BIN.wall, 0.5 * BIN.width - BIN.wall
SIDE_WALL_4CM = {"x": BIN.x, "y": BIN.y + _HL - 0.04, "yaw_deg": 35.0}
FAR_WALL_4CM = {"x": BIN.x - _HW + 0.04, "y": BIN.y, "yaw_deg": 35.0}
TIGHT_CORNER = {"x": BIN.x - _HW + 0.03, "y": BIN.y + _HL - 0.03, "yaw_deg": 35.0}


def _search(cfg, cube):
    return _bin_grasp_geometry(
        geometry_from_config(cfg.pick.top_down),
        BASE,
        cube,
        BIN,
        cfg.pick.top_down.in_bin,
        RETRACT_ARM_CONF,
    )


@pytest.mark.parametrize("cube", [SIDE_WALL_4CM, FAR_WALL_4CM])
def test_the_oblique_extension_picks_an_oblique_cube_4_cm_from_one_wall(cube):
    """4 cm from one wall at 35 deg to it: the near-wall set refuses it; the default
    clears it with one of the extension's new grasps, every link 1 cm off the walls."""
    chosen, _ = _search(_compose(*NEAR_WALL), cube)
    assert chosen is None
    cfg = _compose()
    chosen, problem = _search(cfg, cube)
    assert problem is None and chosen is not None
    geom = geometry_from_config(cfg.pick.top_down)
    near_wall = {_key(g, geom) for g in _candidates(_compose(*NEAR_WALL))}
    assert _key(chosen, geom) not in near_wall
    assert _reference_problem(chosen, cube, cfg.pick.top_down.in_bin) is None


def test_a_tight_corner_is_refused_naming_the_corner_and_what_binds():
    """3 cm from the far wall and 3 cm from the right wall no grasp of the 1612 keeps
    the margin: the refusal says it is a corner, which walls, and for each pair of faces
    the part that binds most often."""
    chosen, problem = _search(_compose(), TIGHT_CORNER)
    assert chosen is None
    assert problem.startswith("cube in a corner: 3.0 cm from the "), problem
    assert "far wall" in problem.split(";")[0] and "right wall" in problem.split(";")[0]
    assert "within the open fingers' reach (8.4 cm)" in problem
    for faces in ("faces 0 deg: ", "faces 90 deg: "):
        clause = problem.split(faces)[1].split(" grasps (first: ")[0]
        assert " binds in " in clause and clause.endswith("/806"), clause
    assert problem.endswith("; 1612 candidates")


def _reference_problem(geom, cube, in_bin):
    """Bd85a79's check of one grasp, every gripper point transformed (no caching, no
    bounding spheres): the search must give the same answer, the same reason."""
    ax, ay = arm_frame_target(BASE, (cube["x"], cube["y"]), geom)
    yaw = math.radians(cube["yaw_deg"]) - BASE[2]
    try:
        above = _bin_pick_targets(geom, (ax, ay, yaw), BIN, in_bin, RETRACT_ARM_CONF)
        hover, grasp = top_down_joint_targets(ax, ay, yaw, geom, seed=above)
    except ValueError as e:
        return f"no top-down pick into the bin: {e}"
    margin = float(in_bin.wall_margin)
    mx = BASE[0] + geom.mounting_offset * math.cos(BASE[2])
    my = BASE[1] + geom.mounting_offset * math.sin(BASE[2])
    rim = geom.floor_z + BIN.height + margin
    ct, st = math.cos(BASE[2]), math.sin(BASE[2])
    cb, sb = math.cos(BIN.yaw), math.sin(BIN.yaw)
    for phase, joints in (("above", above), ("hover", hover), ("grasp", grasp)):
        pos, approach, closing = forward_kinematics(joints, geom)
        tool = np.column_stack([np.cross(closing, approach), closing, approach])
        parts = dict(arm_link_points(joints, geom))
        for name, local in gripper_link_points(geom.open_position).items():
            parts[f"gripper {name}"] = pos + np.einsum("ij,kj->ik", local, tool)
        for name, points in parts.items():
            low = points[points[:, 2] < rim]
            if len(low) == 0:
                continue
            wx = mx + ct * low[:, 0] - st * low[:, 1] - BIN.x
            wy = my + st * low[:, 0] + ct * low[:, 1] - BIN.y
            local = np.c_[cb * wx + sb * wy, -sb * wx + cb * wy]
            problem = part_wall_problem(name, local, BIN, in_bin)
            if problem is not None:
                depth = (BIN.height + geom.floor_z) - float(low[:, 2].min())
                return f"{phase}: {problem} ({depth:.3f} m below the rim)"
    return None


@pytest.mark.parametrize("cube", [TIGHT_CORNER, SIDE_WALL_4CM])
def test_the_cheaper_search_answers_as_the_full_check_does(cube):
    """Every 23rd candidate (each opening, yaw and pair of faces among them): the
    search's verdict and reason equal the full point-by-point check's."""
    cfg = _compose()
    in_bin = cfg.pick.top_down.in_bin
    search = _BinGraspSearch(BASE, cube, BIN, in_bin, RETRACT_ARM_CONF)
    for g in _candidates(cfg)[::23]:
        assert search.problem(g) == _reference_problem(g, cube, in_bin)
