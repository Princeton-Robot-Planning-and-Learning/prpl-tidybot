"""Tests for real_sim/plan_executors/kinematic3d.py.

Two executors live in the same module: PurePursuitKinematic3DPlanExecutor (the default;
smooths motion across consecutive waypoints) and SettleKinematic3DPlanExecutor (the per-
waypoint-convergence fallback). The grounding helpers (delta -> absolute target, gripper
command mapping) are shared; the test file is organised top-down: shared fixtures, then
one section per executor.
"""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.kinematic3d import (
    PurePursuitKinematic3DPlanExecutor,
    SettleKinematic3DPlanExecutor,
)
from prpl_tidybot.structs import TidyBotObservation


def _make_state(
    *,
    base_xytheta: tuple[float, float, float] = (1.0, 2.0, 0.5),
    arm_conf: list[float] | None = None,
    gripper: float = 0.4,
):
    """Build a sim state by running the perceiver on a hand-built obs."""
    obs = TidyBotObservation(
        arm_conf=arm_conf or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=base_xytheta[0], y=base_xytheta[1], theta=base_xytheta[2]),
        gripper=gripper,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


# ---------------------------------------------------------------------------
# PurePursuitKinematic3DPlanExecutor
# ---------------------------------------------------------------------------


def test_pure_pursuit_single_pair_drives_to_implied_final_waypoint():
    """A single (state, delta) pair builds a 2-waypoint path; the lookahead clamps to
    the final waypoint when the path is shorter than `lookahead_distance`."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.05  # 5 cm, well below default lookahead_distance=0.2
    action[1] = -0.05
    executor = PurePursuitKinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.base_pose_target_map.x == pytest.approx(1.05)
    assert real_action.base_pose_target_map.y == pytest.approx(1.95)


def test_pure_pursuit_commands_lookahead_along_long_path():
    """On a path longer than `lookahead_distance`, the commanded target sits exactly
    `lookahead_distance` ahead of the robot's projection (cursor)."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    # Three deltas of +0.5 m in x — total path length 1.5 m.
    delta = np.zeros(11)
    delta[0] = 0.5
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    s2 = _make_state(base_xytheta=(1.0, 0.0, 0.0))
    executor = PurePursuitKinematic3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, delta), (s1, delta), (s2, delta)])

    # Robot at start: lookahead point is 0.3 m along the path.
    real_action, _ = executor.step(s0)
    assert real_action.base_pose_target_map.x == pytest.approx(0.3)

    # Robot at (0.4, 0): cursor advances to 0.4, lookahead at 0.7.
    real_action, _ = executor.step(_make_state(base_xytheta=(0.4, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.7)


def test_pure_pursuit_clamps_lookahead_to_final_waypoint():
    """Near the end of the path, the lookahead saturates at the final waypoint."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    delta = np.zeros(11)
    delta[0] = 1.0
    executor = PurePursuitKinematic3DPlanExecutor(lookahead_distance=0.5)
    executor.set_trajectory([(s0, delta)])

    # Robot at (0.8, 0): cursor=0.8, lookahead would be 1.3 but clamps to 1.0.
    real_action, _ = executor.step(_make_state(base_xytheta=(0.8, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(1.0)


def test_pure_pursuit_cursor_is_monotonic_under_perception_jitter():
    """Brief perception jitter back along the path doesn't make the lookahead
    target regress: the cursor only ever advances."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    delta = np.zeros(11)
    delta[0] = 1.0
    executor = PurePursuitKinematic3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, delta)])

    executor.step(_make_state(base_xytheta=(0.5, 0.0, 0.0)))  # cursor -> 0.5
    # Now the perceived state jitters back to 0.45 — cursor must stay at 0.5,
    # not retreat to 0.45.
    real_action, _ = executor.step(_make_state(base_xytheta=(0.45, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.8)  # 0.5 + 0.3


def test_pure_pursuit_done_only_at_final_waypoint():
    """Done() returns False until the perceived state matches the final waypoint within
    all tolerances (position, angle, joint, gripper)."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    delta = np.zeros(11)
    delta[0] = 1.0
    executor = PurePursuitKinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, delta)])

    assert executor.done(s0) is False
    assert executor.done(_make_state(base_xytheta=(0.5, 0.0, 0.0))) is False
    # Within position tolerance of (1.0, 0, 0); arm + gripper unchanged so
    # those tolerances pass trivially.
    final = _make_state(base_xytheta=(1.0, 0.0, 0.0))
    assert executor.done(final) is True


def test_pure_pursuit_done_at_max_iter_safety_cap():
    """If `max_iter` ticks elapse without convergence, done() returns True anyway — the
    rollout doesn't get stuck on an unreachable goal."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    delta = np.zeros(11)
    delta[0] = 1.0
    executor = PurePursuitKinematic3DPlanExecutor(position_tolerance=1e-9, max_iter=3)
    executor.set_trajectory([(s0, delta)])
    for _ in range(3):
        executor.step(s0)
    assert executor.done(s0) is True


def test_pure_pursuit_empty_trajectory_is_immediately_done():
    """An empty trajectory yields done=True without any step() call."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = PurePursuitKinematic3DPlanExecutor()
    executor.set_trajectory([])
    assert executor.done(s0) is True


def test_pure_pursuit_gripper_close_command():
    """Gripper command <-0.5 in the trajectory's closest waypoint becomes
    TidyBotAction.gripper_goal=1.0."""
    s0 = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = -1.0
    executor = PurePursuitKinematic3DPlanExecutor()
    executor.set_trajectory([(s0, action)])
    real_action, _ = executor.step(s0)
    assert real_action.gripper_goal == 1.0


def test_pure_pursuit_gripper_no_change_passes_through_current_finger():
    """Gripper command in [-0.5, 0.5] passes through the current finger_state."""
    s0 = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 0.0
    executor = PurePursuitKinematic3DPlanExecutor()
    executor.set_trajectory([(s0, action)])
    real_action, _ = executor.step(s0)
    assert real_action.gripper_goal == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# SettleKinematic3DPlanExecutor
# ---------------------------------------------------------------------------


def _settle_single_action_target(action: np.ndarray, state) -> "tuple":
    """Drive a one-pair trajectory through the settle executor and return its first
    command."""
    executor = SettleKinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])
    return executor.step(state)


def test_settle_base_delta_becomes_absolute_target():
    """Base components of the sim action add componentwise to the current world pose."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.1
    action[1] = -0.2
    action[2] = 0.05

    real_action, _ = _settle_single_action_target(action, state)

    assert real_action.base_pose_target_map.x == pytest.approx(1.1)
    assert real_action.base_pose_target_map.y == pytest.approx(1.8)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.55)


def test_settle_arm_delta_summed_with_current_joints():
    """Arm deltas at action[3..10] are added per-joint to the current arm conf."""
    state = _make_state(arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.zeros(11)
    action[3:10] = [0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.04]

    real_action, _ = _settle_single_action_target(action, state)

    expected = [0.11, 0.22, 0.33, 0.39, 0.48, 0.57, 0.74]
    assert real_action.arm_goal == pytest.approx(expected)


def test_settle_gripper_close_command():
    """Gripper command <-0.5 becomes TidyBotAction.gripper_goal=1.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = -1.0
    real_action, _ = _settle_single_action_target(action, state)
    assert real_action.gripper_goal == 1.0


def test_settle_gripper_open_command():
    """Gripper command >0.5 becomes TidyBotAction.gripper_goal=0.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 1.0
    real_action, _ = _settle_single_action_target(action, state)
    assert real_action.gripper_goal == 0.0


def test_settle_gripper_no_change_passes_through_current():
    """Gripper command in [-0.5, 0.5] passes through the current finger_state."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 0.0
    real_action, _ = _settle_single_action_target(action, state)
    assert real_action.gripper_goal == pytest.approx(0.4)


def test_settle_executor_reissues_cached_target_until_converged():
    """Across multiple ticks on the same waypoint the settle executor reissues the same
    absolute target it computed on the first tick — even if the perceived state drifts
    in between.

    This is the per-action settle behaviour the old `RealTidyBotEnv.step` inner loop
    produced inline.
    """
    initial_state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    drifted_state = _make_state(base_xytheta=(1.05, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.1
    executor = SettleKinematic3DPlanExecutor(position_tolerance=1e-3, max_iter=100)
    executor.set_trajectory([(initial_state, action)])

    real_action_1, _ = executor.step(initial_state)
    real_action_2, _ = executor.step(drifted_state)

    assert real_action_1 is real_action_2
    assert real_action_1.base_pose_target_map.x == pytest.approx(1.1)


def test_settle_executor_advances_when_within_tolerance():
    """When the perceived state matches the cached absolute target within tolerance, the
    settle executor advances to the next waypoint on the next done() check."""
    initial_state = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action0 = np.zeros(11)
    action0[0] = 0.5
    action1 = np.zeros(11)
    action1[1] = 0.5
    executor = SettleKinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory(
        [(initial_state, action0), (_make_state(base_xytheta=(0.5, 0.0, 0.0)), action1)]
    )

    assert executor.done(initial_state) is False
    executor.step(initial_state)
    converged_state = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    assert executor.done(converged_state) is False  # advances to waypoint 1
    real_action, _ = executor.step(converged_state)
    # Second waypoint: snapshot was converged_state (0.5, 0, 0); delta +0.5 in y.
    assert real_action.base_pose_target_map.x == pytest.approx(0.5)
    assert real_action.base_pose_target_map.y == pytest.approx(0.5)


def test_settle_executor_advances_at_max_iter_even_without_convergence():
    """If max_iter ticks elapse without convergence the settle executor still advances;
    this is the safety cap that prevents the rollout from getting stuck on an
    unreachable target."""
    initial_state = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action = np.zeros(11)
    action[0] = 1.0
    executor = SettleKinematic3DPlanExecutor(position_tolerance=1e-9, max_iter=3)
    executor.set_trajectory([(initial_state, action)])

    for _ in range(3):
        executor.step(initial_state)
    assert executor.done(initial_state) is True
