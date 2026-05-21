"""Tests for real_sim/plan_executors/kinematic3d.py.

The unified :class:`Kinematic3DPlanExecutor` enforces a strict invariant: each `(state,
action)` pair moves ONLY the base OR the arm, never both. It then segments the
trajectory and dispatches per-segment between pure-pursuit on the base (configurable)
and settle-then-advance on the arm. These tests cover that segmentation and the two
strategy paths.
"""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.structs import TidyBotObservation


def _make_state(
    *,
    base_xytheta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    arm_conf: list[float] | None = None,
    gripper: float = 0.4,
):
    """Build a sim state by running the perceiver on a hand-built obs."""
    obs = TidyBotObservation(
        arm_conf=arm_conf or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=base_xytheta[0], y=base_xytheta[1], theta=base_xytheta[2]),
        gripper=gripper,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


def _base_action(dx: float = 0.0, dy: float = 0.0, drot: float = 0.0) -> np.ndarray:
    action = np.zeros(11)
    action[0] = dx
    action[1] = dy
    action[2] = drot
    return action


def _arm_action(
    arm_deltas: list[float] | None = None, gripper_cmd: float = 0.0
) -> np.ndarray:
    action = np.zeros(11)
    if arm_deltas is not None:
        action[3:10] = arm_deltas
    action[10] = gripper_cmd
    return action


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_mixed_base_and_arm_motion_in_same_pair():
    """A pair with both base and arm deltas violates the executor's invariant."""
    mixed = np.zeros(11)
    mixed[0] = 0.1  # base
    mixed[3] = 0.05  # arm joint 1
    executor = Kinematic3DPlanExecutor()
    with pytest.raises(ValueError, match="ONLY the base OR the arm"):
        executor.set_trajectory([(_make_state(), mixed)])


def test_rejects_invalid_base_strategy():
    """Constructor rejects unknown base_strategy values."""
    with pytest.raises(ValueError, match="base_strategy"):
        Kinematic3DPlanExecutor(base_strategy="banana")


# ---------------------------------------------------------------------------
# Arm segments (always settle)
# ---------------------------------------------------------------------------


def test_arm_segment_grounds_target_from_snapshot_and_holds_base():
    """For an arm-motion pair the commanded TidyBotAction has arm = snapshot + delta
    and base = snapshot.base (the segment must not move the base)."""
    state = _make_state(
        base_xytheta=(1.0, 2.0, 0.5),
        arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    action = _arm_action(arm_deltas=[0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.04])
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.arm_goal == pytest.approx(
        [0.11, 0.22, 0.33, 0.39, 0.48, 0.57, 0.74]
    )
    # Base unchanged.
    assert real_action.base_pose_target_map.x == pytest.approx(1.0)
    assert real_action.base_pose_target_map.y == pytest.approx(2.0)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.5)


def test_arm_segment_reissues_cached_target_until_converged():
    """Across multiple ticks on the same arm pair the executor reissues the same
    grounded target, even if perception drifts in between."""
    initial = _make_state(arm_conf=[0.0] * 7)
    action = _arm_action(arm_deltas=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    drifted = _make_state(arm_conf=[0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    executor = Kinematic3DPlanExecutor(joint_tolerance=1e-3, max_iter_per_pair=100)
    executor.set_trajectory([(initial, action)])

    target_1, _ = executor.step(initial)
    target_2, _ = executor.step(drifted)
    assert target_1 is target_2
    assert target_1.arm_goal[0] == pytest.approx(0.1)


def test_arm_segment_advances_when_within_tolerance():
    """Once the perceived state matches the cached target within tolerance, the executor
    advances to the next arm pair on the next done() check."""
    s0 = _make_state(arm_conf=[0.0] * 7)
    a0 = _arm_action(arm_deltas=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s1 = _make_state(arm_conf=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a1 = _arm_action(arm_deltas=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    executor = Kinematic3DPlanExecutor(joint_tolerance=1e-3)
    executor.set_trajectory([(s0, a0), (s1, a1)])

    executor.step(s0)
    converged = _make_state(arm_conf=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # done() advances past pair 0; not yet done overall.
    assert executor.done(converged) is False
    real_action, _ = executor.step(converged)
    # New snapshot is `converged`; pair 1 adds 0.5 to joint_2.
    assert real_action.arm_goal[1] == pytest.approx(0.5)


def test_arm_segment_advances_at_max_iter_even_without_convergence():
    """If max_iter_per_pair ticks elapse without convergence, the executor still
    advances; this caps the rollout if a target is unreachable."""
    s0 = _make_state(arm_conf=[0.0] * 7)
    action = _arm_action(arm_deltas=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    executor = Kinematic3DPlanExecutor(joint_tolerance=1e-9, max_iter_per_pair=3)
    executor.set_trajectory([(s0, action)])

    for _ in range(3):
        executor.step(s0)
    assert executor.done(s0) is True


def test_arm_segment_keeps_base_at_snapshot_pose_even_under_drift():
    """The settle path uses the snapshot at the start of the pair, so even if the base
    drifts during the segment the commanded base is the snapshot pose (not the current
    pose).

    Verifies the base goal is locked once per pair.
    """
    snapshot = _make_state(base_xytheta=(0.0, 0.0, 0.0), arm_conf=[0.0] * 7)
    action = _arm_action(arm_deltas=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(snapshot, action)])

    cmd_initial, _ = executor.step(snapshot)
    drifted_base = _make_state(base_xytheta=(0.5, 0.0, 0.0), arm_conf=[0.0] * 7)
    cmd_drifted, _ = executor.step(drifted_base)
    assert cmd_drifted.base_pose_target_map.x == pytest.approx(
        cmd_initial.base_pose_target_map.x
    )


# ---------------------------------------------------------------------------
# Base segments — pure_pursuit (default)
# ---------------------------------------------------------------------------


def test_base_segment_pure_pursuit_clamps_to_final_when_path_is_short():
    """Single base pair (path < lookahead): the commanded target is the final
    waypoint."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.0))
    action = _base_action(dx=0.05, dy=-0.05)
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.base_pose_target_map.x == pytest.approx(1.05)
    assert real_action.base_pose_target_map.y == pytest.approx(1.95)


def test_base_segment_pure_pursuit_commands_lookahead_along_long_path():
    """On a path longer than lookahead_distance, target sits at cursor + lookahead."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    s2 = _make_state(base_xytheta=(1.0, 0.0, 0.0))
    delta = _base_action(dx=0.5)
    executor = Kinematic3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, delta), (s1, delta), (s2, delta)])

    real_action, _ = executor.step(s0)
    assert real_action.base_pose_target_map.x == pytest.approx(0.3)

    real_action, _ = executor.step(_make_state(base_xytheta=(0.4, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.7)


def test_base_segment_pure_pursuit_cursor_is_monotonic_under_perception_jitter():
    """Brief perception jitter back along the path doesn't make the cursor regress."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = Kinematic3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    executor.step(_make_state(base_xytheta=(0.5, 0.0, 0.0)))  # cursor -> 0.5
    real_action, _ = executor.step(_make_state(base_xytheta=(0.45, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.8)


def test_base_segment_pure_pursuit_done_only_at_final_waypoint():
    """Done() returns False until the perceived base is within position + angle
    tolerance of the final waypoint."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    assert executor.done(s0) is False
    assert executor.done(_make_state(base_xytheta=(0.5, 0.0, 0.0))) is False
    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True


# ---------------------------------------------------------------------------
# Base segments — settle
# ---------------------------------------------------------------------------


def test_base_segment_settle_reissues_per_pair_target():
    """With base_strategy='settle', a base pair settles on its absolute target (not a
    lookahead).

    Across multiple ticks the same target is reissued.
    """
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action = _base_action(dx=0.5)
    drifted = _make_state(base_xytheta=(0.2, 0.0, 0.0))
    executor = Kinematic3DPlanExecutor(
        base_strategy="settle", position_tolerance=1e-3, max_iter_per_pair=100
    )
    executor.set_trajectory([(s0, action)])

    cmd_1, _ = executor.step(s0)
    cmd_2, _ = executor.step(drifted)
    assert cmd_1 is cmd_2
    assert cmd_1.base_pose_target_map.x == pytest.approx(0.5)


def test_base_segment_settle_advances_per_pair():
    """Multi-pair base segment with settle: each pair converges before the next."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    executor = Kinematic3DPlanExecutor(base_strategy="settle", position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=0.5)), (s1, _base_action(dx=0.5))])

    executor.step(s0)
    converged_to_s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    assert executor.done(converged_to_s1) is False  # advances pair, not whole segment
    cmd, _ = executor.step(converged_to_s1)
    # New snapshot at (0.5, 0, 0); next target is (1.0, 0, 0).
    assert cmd.base_pose_target_map.x == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mixed (interleaved) trajectories
# ---------------------------------------------------------------------------


def test_mixed_trajectory_segments_base_then_arm():
    """A trajectory of [base, arm] becomes two segments; each is driven by its own
    strategy.

    After the base segment finishes, the arm segment starts and commands the arm target
    while holding the base.
    """
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0), arm_conf=[0.0] * 7)
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0), arm_conf=[0.0] * 7)
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory(
        [
            (s0, _base_action(dx=0.5)),  # base segment
            (s1, _arm_action(arm_deltas=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # arm
        ]
    )

    # Base segment runs.
    executor.step(s0)
    # Robot reaches the base final waypoint; done() advances past the base segment.
    s_after_base = _make_state(base_xytheta=(0.5, 0.0, 0.0), arm_conf=[0.0] * 7)
    assert executor.done(s_after_base) is False  # arm segment still pending
    # Arm segment now commands arm target relative to its snapshot.
    cmd, _ = executor.step(s_after_base)
    assert cmd.arm_goal[0] == pytest.approx(0.2)
    # And the base stays where the snapshot says.
    assert cmd.base_pose_target_map.x == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Done semantics
# ---------------------------------------------------------------------------


def test_empty_trajectory_is_immediately_done():
    """Empty trajectory yields done=True without any step() call."""
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([])
    assert executor.done(_make_state()) is True


def test_done_is_sticky():
    """Once done() reports True, it stays True even if perception subsequently drifts
    outside tolerance — same end-of-trajectory oscillation fix as before."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True
    # Drifted reading would have undone done() pre-latch.
    assert executor.done(_make_state(base_xytheta=(0.9, 0.0, 0.0))) is True


# ---------------------------------------------------------------------------
# Gripper (rides on arm segments)
# ---------------------------------------------------------------------------


def test_gripper_close_command_emitted_in_arm_segment():
    """A gripper-close (<-0.5) in an arm pair becomes TidyBotAction.gripper_goal=1."""
    state = _make_state(gripper=0.4, arm_conf=[0.0] * 7)
    action = _arm_action(gripper_cmd=-1.0)
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == 1.0


def test_gripper_no_change_passes_through_current_finger():
    """A gripper command in [-0.5, 0.5] passes through the perceived finger_state."""
    state = _make_state(gripper=0.4, arm_conf=[0.0] * 7)
    action = _arm_action(gripper_cmd=0.0)
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == pytest.approx(0.4)
