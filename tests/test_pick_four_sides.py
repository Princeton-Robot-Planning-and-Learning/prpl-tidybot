"""The pick may stand on any of the cube's four sides (kinder-baselines #189's real
counterpart).

Today's stand (the straight line from the robot, :func:`standoff_pose`) is tried first,
unchanged, so every pick that planned before plans the same stand and grasp. Only if no
stand exists there, the drive there is refused, or the grasp search refuses there, the
stands on the cube's four face normals are tried, nearest the robot's base yaw first,
each checked for the drivable box, the barrier's and the bin's clearance, the straight
drive and the full grasp search. The cube past or on the barrier stays final.
"""

import math
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.aimed_toss import Workspace
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.in_bin_pick_benchmark import load_cases
from prpl_tidybot.robot_skills import (
    _bin_grasp_geometry,
    _floor_grasp_geometry,
    bin_margins,
    plan_pick,
)
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    face_standoffs,
    geometry_from_config,
    standoff_pose,
)

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
SEED = list(RETRACT_ARM_CONF)


def _cfg() -> Any:
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=["mode=sim"])


CFG = _cfg()
FLOOR = tuple(float(v) for v in CFG.preview.floor_bounds)
GEOM = geometry_from_config(CFG.pick.top_down)
WS = Workspace.from_config(CFG.barrier, FLOOR)
IN_BIN = CFG.pick.top_down.in_bin

# The 2026-10-21 10:39 scene's bin (test_same_side_practice.WALL_BIN).
WALL_BIN = BinFit(-0.305, -0.395, 1.55, 0.39, 0.29, 0.21, 0.005)
TOSS_POSE = (1.10, -0.78, math.pi)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _plan(robot, cube, fit, seed=None, workspace=WS, floor=FLOOR):
    return plan_pick(
        robot, cube, GEOM, floor, workspace, fit, IN_BIN, SEED if seed is None else seed
    )


def _assert_stand_checks(robot, stand, cube, fit):
    """The stand is on a face normal, standoff from the cube, facing it, in the drivable
    box, clear of the barrier and the bin, with a straight drive there."""
    assert math.hypot(cube["x"] - stand[0], cube["y"] - stand[1]) == pytest.approx(
        GEOM.standoff, abs=1e-6
    )
    facing = math.atan2(cube["y"] - stand[1], cube["x"] - stand[0])
    assert _wrap(facing - stand[2]) == pytest.approx(0.0, abs=1e-6)
    off_face = _wrap(stand[2] - math.radians(cube["yaw_deg"]))
    assert min(abs(_wrap(off_face - k * math.pi / 2)) for k in range(4)) < 1e-6
    assert FLOOR[0] <= stand[0] <= FLOOR[1] and FLOOR[2] <= stand[1] <= FLOOR[3]
    assert WS.pose_problem(stand, fit) is None
    assert WS.path_problem(robot, stand, fit) is None


# --------------------------------------------------------------------- ordering


def test_face_standoffs_are_the_four_face_normals_nearest_the_base_yaw_first():
    """Heading cube yaw + k 90 deg, sorted by the turn from the robot's base yaw."""
    robot = (0.0, 0.0, math.radians(100.0))
    sides = face_standoffs(robot, (1.0, 0.5), math.radians(10.0), GEOM)
    headings = [round(math.degrees(p[2])) for _, p in sides]
    assert [k for k, _ in sides] == [1, 0, 2, 3]
    assert headings == [100, 10, -170, -80]
    for _, (x, y, h) in sides:
        assert x == pytest.approx(1.0 - GEOM.standoff * math.cos(h))
        assert y == pytest.approx(0.5 - GEOM.standoff * math.sin(h))


def test_face_standoffs_break_ties_to_the_lower_k():
    """A base yaw halfway between two face normals tries the lower k first."""
    robot = (0.0, 0.0, math.radians(45.0))
    sides = face_standoffs(robot, (1.0, 0.0), 0.0, GEOM)
    assert [k for k, _ in sides] == [0, 1, 2, 3]


def test_face_standoffs_skip_the_one_within_10_deg_of_todays_heading():
    """Today's stand was just tried: a face normal within 10 deg of it is not tried
    again; one 11 deg off is."""
    robot = (0.0, 0.0, 0.0)
    near = face_standoffs(robot, (1.0, 0.0), 0.0, GEOM, tried=math.radians(9.0))
    assert [k for k, _ in near] == [1, 3, 2]
    far = face_standoffs(robot, (1.0, 0.0), 0.0, GEOM, tried=math.radians(11.0))
    assert [k for k, _ in far] == [0, 1, 3, 2]


# --------------------------------------------------------------- today first


def test_a_healthy_floor_pick_keeps_todays_stand():
    """A floor cube well clear of the bin: the stand is today's, exactly."""
    cube = {"x": 0.40, "y": -0.60, "yaw_deg": 20.0, "in_bin": False}
    stand, why = _plan(TOSS_POSE, cube, WALL_BIN)
    assert why is None
    today = standoff_pose(
        TOSS_POSE, (cube["x"], cube["y"]), GEOM, FLOOR, WS.valid(WALL_BIN)
    )
    assert stand == today


@pytest.mark.parametrize(
    "case_id", ["2026-10-20/18-10-36#34", "2026-10-21/10-29-48#33"]
)
def test_a_healthy_in_bin_pick_keeps_todays_stand_and_grasp(case_id):
    """Real in-bin picks: today's stand, and so the same grasp there."""
    case = next(c for c in load_cases() if c["id"] == case_id)
    b = case["bin"]
    fit = BinFit(
        b["x"], b["y"], b["yaw"], b["length"], b["width"], b["height"], b["wall"]
    )
    cube = dict(case["cube"], in_bin=True)
    stand, why = _plan(case["robot"], cube, fit, seed=case["arm_seed"])
    assert why is None
    today = standoff_pose(
        case["robot"], (cube["x"], cube["y"]), GEOM, FLOOR, WS.valid(fit)
    )
    assert stand == today
    chosen, _ = _bin_grasp_geometry(GEOM, stand, cube, fit, IN_BIN, case["arm_seed"])
    assert chosen is not None


# ------------------------------------------------------------- other sides


# A floor cube 5 cm outside the bin's -v wall: from the toss pose today's stand looks
# across the wall and its grasp touches it; the stand on the cube's +90 face normal
# clears.
FLOOR_WALL_CUBE = {"x": -0.1100, "y": -0.3991, "yaw_deg": 0.0, "in_bin": False}


def test_a_floor_cube_whose_direct_grasp_touches_the_bin_is_picked_from_another_side():
    """Today's grasp touches the bin; another face normal's stand clears it."""
    today = standoff_pose(
        TOSS_POSE,
        (FLOOR_WALL_CUBE["x"], FLOOR_WALL_CUBE["y"]),
        GEOM,
        FLOOR,
        WS.valid(WALL_BIN),
    )
    assert (
        _floor_grasp_geometry(
            GEOM, today, FLOOR_WALL_CUBE, WALL_BIN, bin_margins(IN_BIN), SEED
        )[0]
        is None
    )
    stand, why = _plan(TOSS_POSE, FLOOR_WALL_CUBE, WALL_BIN)
    assert why is None and stand is not None
    assert stand != today
    _assert_stand_checks(TOSS_POSE, stand, FLOOR_WALL_CUBE, WALL_BIN)
    chosen, _ = _floor_grasp_geometry(
        GEOM, stand, FLOOR_WALL_CUBE, WALL_BIN, bin_margins(IN_BIN), SEED
    )
    assert chosen is not None


# The real refusal 2026-10-20/17-27-17 observe 19: today's drive brushes the barrier;
# the cube's other face normals drive straight.
DRIVE_ROBOT = (0.2191000279161867, -0.014893669300127477, 1.5691008885675153)
DRIVE_CUBE = {
    "x": 0.15715403036650655,
    "y": -0.7999990048297656,
    "yaw_deg": math.degrees(0.23561944901923448),
    "in_bin": False,
}
DRIVE_BIN = BinFit(-0.301875, -0.45, 1.531526418625024, 0.39, 0.29, 0.21, 0.005)


def test_a_floor_cube_whose_direct_drive_is_refused_is_picked_from_another_side():
    """Today's drive is refused; another face normal's stand drives straight."""
    today = standoff_pose(
        DRIVE_ROBOT,
        (DRIVE_CUBE["x"], DRIVE_CUBE["y"]),
        GEOM,
        FLOOR,
        WS.valid(DRIVE_BIN),
    )
    assert WS.path_problem(DRIVE_ROBOT, today, DRIVE_BIN) is not None
    stand, why = _plan(DRIVE_ROBOT, DRIVE_CUBE, DRIVE_BIN)
    assert why is None and stand is not None
    _assert_stand_checks(DRIVE_ROBOT, stand, DRIVE_CUBE, DRIVE_BIN)


def test_a_cube_with_no_stand_today_is_picked_from_a_face_normal():
    """Only the stand on the 7 deg face normal fits the drivable box: no 15-degree swing
    of today's approach does, so standoff_pose finds none."""
    cube = {"x": 1.0, "y": 0.0, "yaw_deg": 7.0, "in_bin": False}
    h = math.radians(7.0)
    x, y = 1.0 - GEOM.standoff * math.cos(h), -GEOM.standoff * math.sin(h)
    floor = (x - 0.02, x + 0.02, y - 0.02, y + 0.02)
    with pytest.raises(ValueError):
        standoff_pose((0.0, 0.0, 0.0), (1.0, 0.0), GEOM, floor)
    stand, why = _plan((0.0, 0.0, 0.0), cube, None, workspace=None, floor=floor)
    assert why is None
    assert stand == pytest.approx((x, y, h))


# The in-bin coverage grid's cell u -0.04, v +0.11, yaw 0 (run 10's look-96 scene): 3 cm
# from the bin's far wall as the robot at the toss pose sees it. From today's stand
# every one of the 1612 grasps binds; from the face normal at +89 deg (the wall then
# beside the arm, not across it) a grasp clears.
SCENE_ROBOT = (1.1046652117768665, -0.7918998775923151, -3.088889183598109)
SCENE_SEED = [-0.0, -0.3358, -3.1416, -2.5552, -0.0, -0.83, 1.5713]
SCENE_BIN = BinFit(
    -0.30697234717704597,
    -0.7897929905856097,
    1.5620696805349246,
    0.39,
    0.29,
    0.21,
    0.005,
)
FAR_WALL_CUBE = {
    "x": -0.41731722013403977,
    "y": -0.8288315486033554,
    "yaw_deg": 89.5,
    "in_bin": True,
}


def test_an_in_bin_cube_refused_at_todays_stand_is_picked_from_another_side():
    """Today's stand refuses every grasp; another face normal's stand clears one."""
    xy = (FAR_WALL_CUBE["x"], FAR_WALL_CUBE["y"])
    today = standoff_pose(SCENE_ROBOT, xy, GEOM, FLOOR, WS.valid(SCENE_BIN))
    assert (
        _bin_grasp_geometry(GEOM, today, FAR_WALL_CUBE, SCENE_BIN, IN_BIN, SCENE_SEED)[
            0
        ]
        is None
    )
    stand, why = _plan(SCENE_ROBOT, FAR_WALL_CUBE, SCENE_BIN, seed=SCENE_SEED)
    assert why is None and stand is not None
    _assert_stand_checks(SCENE_ROBOT, stand, FAR_WALL_CUBE, SCENE_BIN)
    assert (
        _bin_grasp_geometry(GEOM, stand, FAR_WALL_CUBE, SCENE_BIN, IN_BIN, SCENE_SEED)[
            0
        ]
        is not None
    )


def test_a_refusal_lists_the_sides_tried():
    """Real run 10 refusal #87 (a cube in a corner): no side clears; the reason keeps
    today's and then names each side tried and why."""
    case = next(c for c in load_cases() if c["id"] == "2026-10-21/10-29-48#87")
    b = case["bin"]
    fit = BinFit(
        b["x"], b["y"], b["yaw"], b["length"], b["width"], b["height"], b["wall"]
    )
    cube = dict(case["cube"], in_bin=True)
    stand, why = _plan(case["robot"], cube, fit, seed=case["arm_seed"])
    assert stand is None
    assert why.startswith("cube too close to the bin wall to grasp (")
    assert "; other sides tried: " in why
    tried = why.split("; other sides tried: ", 1)[1]
    sides = face_standoffs(
        case["robot"],
        (cube["x"], cube["y"]),
        math.radians(cube["yaw_deg"]),
        GEOM,
        tried=standoff_pose(
            case["robot"], (cube["x"], cube["y"]), GEOM, FLOOR, WS.valid(fit)
        )[2],
    )
    assert len(sides) >= 3
    for _, pose in sides:
        assert f"{math.degrees(pose[2]):+.0f} deg: " in tried


def test_a_cube_past_the_barrier_is_still_refused_outright():
    """The barrier is final: no other side is tried."""
    cube = {"x": 0.0, "y": 0.6, "yaw_deg": 0.0, "in_bin": False}
    stand, why = _plan(TOSS_POSE, cube, WALL_BIN)
    assert stand is None and why == "cube past the barrier"
