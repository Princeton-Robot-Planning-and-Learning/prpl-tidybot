"""Tests for the real toss bounds, the barrier and the aimed toss pose."""

import math

import numpy as np
import pytest

from prpl_tidybot.aimed_toss import (
    BASE_LENGTH,
    Barrier,
    RealTossBounds,
    Workspace,
    aimed_toss_pose,
    choose_toss_direction,
    footprint,
    plan_aimed_toss,
)
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    forward_kinematics,
    standoff_pose,
)
from prpl_tidybot.toss import plan_toss_swing, toss_windup_pose

FLOOR = (-1.5, 1.7, -1.5, 1.5)
BARRIER = Barrier()  # x = 0, along y, 1.8 m long, 5 cm thick
WS = Workspace(floor_bounds=FLOOR, barrier=BARRIER, clearance=0.05)


def _bin(x=-0.6, y=0.0, yaw=0.0):
    return BinFit(x=x, y=y, yaw=yaw, length=0.40, width=0.32, height=0.19)


@pytest.mark.parametrize(
    "params",
    [(0.9, 100.0, 500.0), (1.45, 140.0, 850.0), (1.2, 120.0, 600.0)],
)
def test_parameters_inside_the_real_box_are_accepted(params):
    """The corners and the middle of the box pass."""
    RealTossBounds().validate(*params)


@pytest.mark.parametrize(
    "params, word",
    [
        ((0.89, 120.0, 600.0), "standoff"),
        ((1.46, 120.0, 600.0), "standoff"),
        ((1.2, 99.0, 600.0), "speed"),
        ((1.2, 141.0, 600.0), "speed"),
        ((1.2, 120.0, 499.0), "release_ms"),
        ((1.2, 120.0, 851.0), "release_ms"),
        ((1.2, float("nan"), 600.0), "speed"),
    ],
)
def test_parameters_outside_the_real_box_are_rejected(params, word):
    """Anything outside the box raises, naming the parameter."""
    with pytest.raises(ValueError, match=word):
        RealTossBounds().validate(*params)


def test_every_toss_in_the_box_is_thrown_forward_and_up():
    """The derivation the bounds rest on: over the whole (speed, release) box the pinch
    point moves forward and upward at the release cycle (arm frame +x, +z)."""
    geom = TopDownPickGeometry()
    bounds = RealTossBounds()
    for speed in np.linspace(*bounds.speed, 5):
        for release in np.linspace(*bounds.release_ms, 8).astype(int):
            traj = plan_toss_swing(
                toss_windup_pose(0.0), 0.0, max_vel=speed, gripper_release_ms=release
            )
            here = forward_kinematics(np.deg2rad(traj[release]), geom)[0]
            after = forward_kinematics(np.deg2rad(traj[release + 1]), geom)[0]
            velocity = (after - here) / 0.001
            assert velocity[0] > 0.0, (speed, release)
            assert velocity[2] > 0.1, (speed, release)
            assert abs(velocity[1]) < 0.05, (speed, release)


def test_the_barriers_robot_side_is_its_plus_x_face_by_default():
    """Along y and 5 cm thick, the robot's side starts 2.5 cm past its centre."""
    assert BARRIER.robot_side == (1.0, 0.0)
    assert BARRIER.depth((0.025, 0.7)) == pytest.approx(0.0)
    assert BARRIER.depth((0.5, -3.0)) == pytest.approx(0.475)
    assert Barrier(x=0.3, yaw=0.0, length=0.2, thickness=0.05).depth(
        (0.4, 0.0)
    ) == pytest.approx(0.0)
    assert set(BARRIER.as_dict()) == {
        "x",
        "y",
        "length",
        "thickness",
        "yaw",
        "robot_side",
    }
    assert BARRIER.as_dict()["robot_side"] == [1.0, 0.0]


# The real room (2026-10-20): black tape across the whole floor, its robot-side edge
# at y = +0.94 and the robot on the -y side; the bin fixed against the back wall
# (measured x +0.02..+0.44, y +1.85..~2.05); throws from (0.23, 0.61) facing +y.
TAPE = Barrier(
    x=0.1, y=0.965, length=3.2, thickness=0.05, yaw=0.0, robot_side=(0.0, -1.0)
)
REAL_WS = Workspace(floor_bounds=FLOOR, barrier=TAPE, clearance=0.05)
REAL_BIN = BinFit(x=0.23, y=1.95, yaw=0.0, length=0.42, width=0.20, height=0.19)
REAL_TOSS_POSE = (0.23, 0.61, math.pi / 2)


def test_a_barrier_with_its_robot_side_towards_minus_y():
    """The tape's robot-side edge is its -y face; depth is how far a point is past
    it into the robot's side."""
    assert TAPE.depth((0.0, 0.94)) == pytest.approx(0.0)
    assert TAPE.depth((-1.2, 0.61)) == pytest.approx(0.33)
    assert TAPE.depth((0.3, 1.2)) < 0.0
    assert Barrier(robot_side=(0.0, -2.0)).robot_side == (0.0, -1.0)
    with pytest.raises(ValueError, match="robot_side"):
        Barrier(robot_side=(0.0, 0.0))


def test_the_real_toss_pose_keeps_the_base_clear_of_the_tape():
    """Facing +y the base's front edge reaches y = 0.884, inside the 5 cm clearance
    (0.89); 9 cm closer it would not be."""
    assert REAL_WS.pose_problem(REAL_TOSS_POSE, REAL_BIN) is None
    problem = REAL_WS.pose_problem((0.23, 0.70, math.pi / 2), REAL_BIN)
    assert problem is not None and "barrier" in problem


def test_the_real_aimed_toss_stands_at_the_toss_pose():
    """1.34 m from the bin centre, from straight -y (bin -> robot azimuth -90 deg)."""
    pose = aimed_toss_pose(REAL_BIN, 1.34, -math.pi / 2, REAL_WS)
    assert pose == pytest.approx(REAL_TOSS_POSE)
    with pytest.raises(ValueError, match="barrier"):
        aimed_toss_pose(REAL_BIN, 1.0, -math.pi / 2, REAL_WS)


def test_drives_must_stay_on_the_tapes_robot_side():
    """A drive along the tape is fine; one towards the bin across it is refused."""
    assert REAL_WS.path_problem((-1.0, 0.3, 0.0), (1.0, 0.3, math.pi / 2)) is None
    problem = REAL_WS.path_problem(REAL_TOSS_POSE, (0.23, 1.2, math.pi / 2))
    assert problem is not None and "barrier" in problem


@pytest.mark.parametrize(
    "xy, yaw, problem",
    [
        ((0.3, 0.5), 0.0, None),
        ((0.3, 0.912), 0.0, None),  # its +y face 3 mm short of the edge
        ((0.3, 0.92), 0.0, "overlapping"),  # its +y face 5 mm onto the tape
        ((0.3, 0.912), math.radians(45), "overlapping"),  # a corner onto the tape
        ((0.3, 0.95), 0.0, "past"),
        ((0.3, 1.95), 0.0, "past"),  # in the bin
    ],
)
def test_a_cube_on_or_past_the_tape_is_not_pickable(xy, yaw, problem):
    """With the cube's half-size, a cube whose footprint reaches the tape is refused
    as overlapping it; one whose centre is past the edge, as past the barrier."""
    ws = Workspace(floor_bounds=FLOOR, barrier=TAPE, clearance=0.05, cube_half=0.025)
    found = ws.cube_problem(xy, yaw)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found and "barrier" in found


def test_without_a_cube_size_only_the_cubes_centre_counts():
    """The default (cube_half 0) is the old rule: the centre past the robot-side
    face."""
    assert WS.cube_problem((0.03, 0.0), 0.0) is None
    assert "past" in (WS.cube_problem((0.02, 0.0), 0.0) or "")
    assert Workspace(FLOOR).cube_problem((-5.0, 0.0), 0.0) is None


def test_the_toss_pose_faces_the_bin_from_the_given_direction():
    """Standoff from the bin centre along the direction, heading at the bin."""
    pose = aimed_toss_pose(_bin(), 1.2, 0.0, WS)
    assert pose == pytest.approx((0.6, 0.0, math.pi)) or pose == pytest.approx(
        (0.6, 0.0, -math.pi)
    )
    pose = aimed_toss_pose(_bin(-0.8, 0.5), 1.3, math.radians(20), WS)
    assert math.hypot(pose[0] + 0.8, pose[1] - 0.5) == pytest.approx(1.3)
    bearing = math.atan2(0.5 - pose[1], -0.8 - pose[0])
    assert math.cos(bearing - pose[2]) == pytest.approx(1.0)


def test_a_toss_pose_whose_footprint_crosses_the_barrier_is_rejected():
    """At 0.9 m from a bin 0.6 m behind the barrier the base's front edge would
    reach x = 0.026 < 0.075: rejected before anything moves."""
    with pytest.raises(ValueError, match="barrier"):
        aimed_toss_pose(_bin(), 0.9, 0.0, WS)
    # Rotating the base around the bin keeps it behind the line at all.
    with pytest.raises(ValueError, match="barrier"):
        aimed_toss_pose(_bin(), 1.2, math.radians(120), WS)


def test_a_toss_pose_off_the_floor_or_on_the_bin_is_rejected():
    """The drivable box bounds the base centre; the footprint stays off the bin."""
    with pytest.raises(ValueError, match="drivable box"):
        aimed_toss_pose(_bin(1.0, 1.0), 1.0, math.radians(45), Workspace(FLOOR))
    with pytest.raises(ValueError, match="overlaps the bin"):
        # No barrier, bin on the robot's side, a standoff shorter than the bin
        # half-length plus the base: the footprint lands on it.
        aimed_toss_pose(
            BinFit(0.8, 0.0, 0.0, 0.40, 0.32, 0.19),
            0.3,
            0.0,
            Workspace(FLOOR),
        )


def test_choose_direction_prefers_the_robots_bearing_and_swings_round():
    """From the robot's bearing if valid; otherwise the nearest valid one."""
    robot = (1.0, 0.0, 0.0)
    assert choose_toss_direction(_bin(), 1.2, robot, WS) == pytest.approx(0.0)
    # Robot at the top of the room: straight along its bearing the base would cross
    # the barrier, so the direction swings towards +x.
    d = choose_toss_direction(_bin(-0.2, 0.0), 1.2, (-0.2, 1.4, 0.0), WS)
    pose = aimed_toss_pose(_bin(-0.2, 0.0), 1.2, d, WS)
    assert min(BARRIER.depth(c) for c in footprint(pose)) >= 0.05 - 1e-9


def test_choose_direction_raises_when_no_pose_is_valid():
    """A bin too deep behind the barrier for the longest standoff has no toss."""
    with pytest.raises(ValueError, match="No valid toss pose"):
        choose_toss_direction(_bin(-1.3, 0.0), 1.2, (1.0, 0.0, 0.0), WS)


def test_paths_must_stay_on_the_robots_side():
    """A straight drive that sweeps the footprint across the barrier is refused, one
    that stays on the +x side is fine."""
    assert WS.path_problem((1.0, -1.0, 0.0), (1.0, 1.0, math.pi)) is None
    problem = WS.path_problem((1.0, 0.0, 0.0), (0.2, 0.0, 0.0))
    assert problem is not None and "barrier" in problem


def test_the_base_footprint_is_centre_referenced():
    """0.548 m along the heading, 0.508 m across."""
    corners = footprint((0.0, 0.0, 0.0))
    assert corners[:, 0].max() == pytest.approx(BASE_LENGTH / 2)
    assert corners[:, 1].max() == pytest.approx(0.254)


def test_workspace_from_config_off_by_default():
    """Without ``enabled`` the barrier is ignored (existing toss_cycles behaviour)."""
    assert Workspace.from_config({"enabled": False}, FLOOR).barrier is None
    ws = Workspace.from_config(
        {
            "enabled": True,
            "center": [0.1, 0.0],
            "length": 1.8,
            "thickness": 0.05,
            "yaw": math.pi / 2,
            "clearance": 0.07,
        },
        FLOOR,
    )
    assert ws.barrier is not None and ws.barrier.x == 0.1 and ws.clearance == 0.07
    assert ws.barrier.robot_side == (1.0, 0.0) and ws.cube_half == 0.0


def test_workspace_from_config_reads_the_robot_side_and_the_cube_size():
    """The real tape's block."""
    ws = Workspace.from_config(
        {
            "enabled": True,
            "center": [0.1, 0.965],
            "length": 3.2,
            "thickness": 0.05,
            "yaw": 0.0,
            "robot_side": [0.0, -1.0],
            "clearance": 0.05,
            "cube_half": 0.025,
        },
        FLOOR,
    )
    assert ws.barrier == TAPE and ws.cube_half == 0.025


def test_pick_standoffs_keep_the_base_on_the_robots_side():
    """A cube just on the robot's side of the barrier, approached from straight
    across in y, is still picked from a standoff whose footprint stays clear of the
    barrier: the approach swings round towards +x."""
    geom = TopDownPickGeometry()
    robot = (0.25, 1.2, 0.0)
    cube = (0.25, 0.0)
    plain = standoff_pose(robot, cube, geom, FLOOR)
    assert WS.pose_problem(plain) is not None  # the direct approach would cross
    pose = standoff_pose(robot, cube, geom, FLOOR, is_valid=WS.valid())
    assert WS.pose_problem(pose) is None
    assert math.hypot(pose[0] - cube[0], pose[1] - cube[1]) == pytest.approx(
        geom.standoff
    )


def test_plan_aimed_toss_checks_bounds_first_then_pose_then_path():
    """Bounds are checked before the pose, the pose before the drive."""
    bounds = RealTossBounds()
    robot = (1.0, 0.0, math.pi)
    pose = plan_aimed_toss(_bin(), 1.2, 120.0, 600.0, 0.0, robot, WS, bounds)
    assert pose[:2] == pytest.approx((0.6, 0.0))
    with pytest.raises(ValueError, match="speed"):
        plan_aimed_toss(_bin(), 0.9, 160.0, 600.0, 0.0, robot, WS, bounds)
    with pytest.raises(ValueError, match="barrier"):
        plan_aimed_toss(_bin(), 0.9, 120.0, 600.0, 0.0, robot, WS, bounds)
    # Chosen direction when none is given.
    pose = plan_aimed_toss(_bin(), 1.2, 120.0, 600.0, None, robot, WS, bounds)
    assert WS.pose_problem(pose) is None
