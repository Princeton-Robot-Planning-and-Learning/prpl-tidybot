"""Tests for toss.py.

The reference numbers are Jimmy Wu's real-robot toss (`KinovaArm.toss` and
`MoveJController.trapezoidal_motion_profile` in tidybot_real's `robot/kinova.py`):
wind-up ``[h, 50, 180, 250, 0, 260, 90]``, release pose ``[h, 20, 180, 325, 0, 25,
90]``, 140 deg/s, 300 / 200 deg/s^2, gripper opened at cycle 600 of a 1 kHz loop.
The expected lengths and positions below were produced by running his function
verbatim.
"""

import numpy as np
import pytest

from prpl_tidybot.toss import (
    CYCLIC_STEP_SIZE,
    TOSS_GRIPPER_RELEASE_MS,
    TOSS_MAX_ACCEL,
    TOSS_MAX_DECEL,
    TOSS_MAX_VEL,
    describe_toss,
    joint_trajectory,
    max_joint_error_deg,
    plan_toss_swing,
    toss_release_pose,
    toss_windup_pose,
    trapezoidal_motion_profile,
    wrapped_joint_differences,
)


def test_toss_defaults_match_jimmys_toss():
    """The default speed, accelerations and release cycle are Jimmy's."""
    assert TOSS_MAX_VEL == 140.0
    assert TOSS_MAX_ACCEL == 300.0
    assert TOSS_MAX_DECEL == 200.0
    assert TOSS_GRIPPER_RELEASE_MS == 600
    assert CYCLIC_STEP_SIZE == 0.001


def test_toss_poses_match_jimmys_toss():
    """The heading goes into joint 1; the other six joints are Jimmy's constants."""
    assert toss_windup_pose(0.0) == [0.0, 50.0, 180.0, 250.0, 0.0, 260.0, 90.0]
    assert toss_release_pose(0.0) == [0.0, 20.0, 180.0, 325.0, 0.0, 25.0, 90.0]
    assert toss_windup_pose(30.0)[0] == 30.0
    assert toss_release_pose(-15.0)[0] == -15.0


def test_wrapped_joint_differences_take_the_short_way():
    """Joint 6 goes 260 -> 25 the short way (+125 through 360), not -235."""
    diffs = wrapped_joint_differences(toss_windup_pose(0.0), toss_release_pose(0.0))
    assert np.allclose(diffs, [0.0, -30.0, 0.0, 75.0, 0.0, 125.0, 0.0])


def test_profile_at_jimmys_defaults():
    """125 deg at 140 deg/s: accelerate, cruise at 140, decelerate; 1478 cycles."""
    t, pos = trapezoidal_motion_profile(
        125.0, max_vel=140.0, max_accel=300.0, max_decel=200.0
    )
    assert len(t) == len(pos) == 1478
    assert t[-1] == pytest.approx(1.477, abs=1e-6)
    assert pos[0] == 0.0
    assert pos[-1] == pytest.approx(125.0, abs=1e-3)
    vel = np.diff(pos) / CYCLIC_STEP_SIZE
    # The final sample lands a hair past the apex of the deceleration parabola
    # (float32), so allow a sub-millidegree step back there.
    assert np.all(vel >= -0.1), "the profile never moves backwards"
    assert vel.max() == pytest.approx(140.0, abs=0.1)
    # Accelerating for 140/300 s, then cruising, then decelerating for 140/200 s.
    assert vel[100] == pytest.approx(300.0 * 0.1005, abs=0.5)
    assert vel[600] == pytest.approx(140.0, abs=0.1)
    assert vel[-50] < 20.0
    # The gripper opens at cycle 600, 51.3 deg into the 125 deg swing.
    assert pos[600] == pytest.approx(51.3333, abs=1e-3)


def test_profile_without_a_cruise_phase():
    """A short move never reaches max_vel: accelerate straight into deceleration.

    Jimmy's original solves this branch for half the distance (its accel duration is
    ``sqrt(d / (a + a^2 / d'))`` where covering ``d`` needs ``sqrt(2 d / ...)``), so it
    stops at 10 of 20 deg. The toss never takes this branch (its 125 deg swing is
    longer than the 81.7 deg needed to reach even 140 deg/s), so the port fixes it.
    """
    _, pos = trapezoidal_motion_profile(
        20.0, max_vel=140.0, max_accel=300.0, max_decel=200.0
    )
    vel = np.diff(pos) / CYCLIC_STEP_SIZE
    assert pos[-1] == pytest.approx(20.0, abs=1e-3)
    assert vel.max() < 140.0


def test_profile_rejects_negative_distance():
    """Distances are magnitudes; a negative one is a caller bug."""
    with pytest.raises(ValueError):
        trapezoidal_motion_profile(-1.0, max_vel=60.0, max_accel=80.0, max_decel=80.0)


def test_joint_trajectory_scales_each_joint_by_its_share_of_the_largest_move():
    """All joints start and stop together; joint 6 moves furthest and sets the pace."""
    start = toss_windup_pose(0.0)
    traj = joint_trajectory(
        start, toss_release_pose(0.0), max_vel=140.0, max_accel=300.0, max_decel=200.0
    )
    assert traj.shape == (1478, 7)
    assert np.allclose(traj[0], start)
    # Not re-wrapped: joint 6 ends at 385 = 25 + 360, exactly as Jimmy commands it.
    assert np.allclose(traj[-1], [0.0, 20.0, 180.0, 325.0, 0.0, 385.0, 90.0], atol=1e-3)
    joint_speeds = np.abs(np.diff(traj, axis=0)).max(axis=0) / CYCLIC_STEP_SIZE
    assert joint_speeds[5] == pytest.approx(140.0, abs=0.1)
    assert joint_speeds[3] == pytest.approx(140.0 * 75.0 / 125.0, abs=0.1)
    assert joint_speeds[1] == pytest.approx(140.0 * 30.0 / 125.0, abs=0.1)
    assert joint_speeds[0] == joint_speeds[2] == joint_speeds[4] == 0.0


def test_plan_toss_swing_defaults_are_jimmys_swing():
    """From the wind-up pose, the default swing is Jimmy's 1478-cycle trajectory."""
    traj = plan_toss_swing(toss_windup_pose(0.0), heading=0.0)
    assert traj.shape == (1478, 7)


def test_plan_toss_swing_at_first_run_speed():
    """At 60 deg/s the same swing takes 2335 cycles."""
    traj = plan_toss_swing(toss_windup_pose(0.0), heading=0.0, max_vel=60.0)
    assert traj.shape == (2335, 7)


@pytest.mark.parametrize("max_vel", [0.0, -10.0, 140.5])
def test_plan_toss_swing_rejects_speeds_outside_jimmys_range(max_vel):
    """Speeds must be positive and no faster than Jimmy's 140 deg/s."""
    with pytest.raises(ValueError):
        plan_toss_swing(toss_windup_pose(0.0), heading=0.0, max_vel=max_vel)


def test_plan_toss_swing_rejects_a_release_after_the_swing_ends():
    """A release cycle past the end of the swing would never open the gripper."""
    with pytest.raises(ValueError):
        plan_toss_swing(toss_windup_pose(0.0), heading=0.0, gripper_release_ms=1478)


def test_max_joint_error_wraps_around_360():
    """Jimmy's 1 deg arrival check treats 359.5 and 0.2 as 0.7 deg apart."""
    assert max_joint_error_deg([359.5, 10.0], [0.2, 10.0]) == pytest.approx(0.7)


def test_describe_toss_reports_where_the_gripper_opens():
    """At Jimmy's settings the gripper opens 0.6 s in, 41% of the way through the
    swing; at 60 deg/s the same 600 ms is only 24% of the way through."""
    jimmy = describe_toss(heading=0.0)
    assert jimmy["duration_s"] == pytest.approx(1.477, abs=1e-3)
    assert jimmy["release_time_s"] == pytest.approx(0.6)
    assert jimmy["release_fraction"] == pytest.approx(51.3333 / 125.0, abs=1e-4)
    slow = describe_toss(heading=0.0, max_vel=60.0)
    assert slow["duration_s"] == pytest.approx(2.334, abs=1e-3)
    assert slow["release_fraction"] == pytest.approx(30.0 / 125.0, abs=1e-4)
    # The joint configuration at release, for the operator log.
    assert len(slow["release_joints_deg"]) == 7
