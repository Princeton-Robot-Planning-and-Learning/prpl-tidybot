"""Tests for the kinematic3d arm-motion plan executors.

`ArmMotion3DPlanExecutor` is abstract; `StreamingArmMotion3DPlanExecutor` is the
concrete crossover-advance subclass. Shared concerns (arm-only validation) are tested
against the concrete subclass since the abstract base can't be instantiated.

Distance function for tests is a plain L1 (no wrap), built to be easy to reason about.
Production wires in pybullet-helpers' weighted joint distance.
"""

from typing import Sequence

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    StreamingArmMotion3DPlanExecutor,
)
from prpl_tidybot.structs import TidyBotObservation


def _l1_distance(q1: Sequence[float], q2: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.array(q1) - np.array(q2))))


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


def _arm_action(
    arm_deltas: list[float] | None = None, gripper_cmd: float = 0.0
) -> np.ndarray:
    action = np.zeros(11)
    if arm_deltas is not None:
        action[3:10] = arm_deltas
    action[10] = gripper_cmd
    return action


def _base_action(dx: float) -> np.ndarray:
    action = np.zeros(11)
    action[0] = dx
    return action


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_nonpositive_advance_radius():
    """advance_radius must be > 0."""
    with pytest.raises(ValueError, match="advance_radius"):
        StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance, advance_radius=0.0)


def test_constructor_rejects_nonpositive_arrival_tolerance():
    """arrival_tolerance must be > 0."""
    with pytest.raises(ValueError, match="arrival_tolerance"):
        StreamingArmMotion3DPlanExecutor(
            distance_fn=_l1_distance, arrival_tolerance=0.0
        )


# ---------------------------------------------------------------------------
# Set-trajectory validation (arm-only pairs)
# ---------------------------------------------------------------------------


def test_set_trajectory_rejects_pair_with_base_motion():
    """A pair with any base delta raises ValueError at set_trajectory time."""
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    with pytest.raises(ValueError, match="arm-only pairs"):
        executor.set_trajectory([(_make_state(), _base_action(dx=0.1))])


def test_empty_trajectory_is_immediately_done():
    """An empty trajectory yields done=True without any step() call."""
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory([])
    assert executor.done(_make_state()) is True


# ---------------------------------------------------------------------------
# Cursor advance
# ---------------------------------------------------------------------------


def test_cursor_does_not_advance_when_far_from_current_target():
    """Perceived joints far from waypoints[0] → cursor stays at 0, command
    waypoint[0]."""
    s0 = _make_state(arm_conf=[0.0] * 7)
    # Three sequential pairs each adding 0.1 to joint 1.
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.1, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.2, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(pairs)

    real_action, _ = executor.step(s0)
    # waypoints are [0.1, 0.2, 0.3]; cursor=0 → target [0.1, 0, 0, ...].
    assert real_action.arm_goal[0] == pytest.approx(0.1)


def test_cursor_advances_when_within_radius():
    """Once perceived is within advance_radius of waypoints[cursor], advance to next."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.1, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(pairs)
    # Perceived equals waypoints[0] → cursor advances to 1, commanded target is
    # waypoints[1].
    real_action, _ = executor.step(_make_state(arm_conf=[0.1, 0, 0, 0, 0, 0, 0]))
    assert real_action.arm_goal[0] == pytest.approx(0.2)


def test_cursor_can_advance_multiple_waypoints_in_one_tick():
    """If perceived is within advance_radius of several consecutive waypoints, cursor
    jumps straight to the furthest one within radius."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.01, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.01, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.01, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.02, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.01, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(pairs)
    # waypoints = [0.01, 0.02, 0.03] for joint 1. Perceived 0.025 is within 0.05 of
    # waypoints[0] and waypoints[1] → cursor jumps to 2, commands waypoints[2] = 0.03.
    real_action, _ = executor.step(_make_state(arm_conf=[0.025, 0, 0, 0, 0, 0, 0]))
    assert real_action.arm_goal[0] == pytest.approx(0.03)


def test_cursor_caps_at_final_waypoint():
    """Cursor never advances past the last pair; final target is reissued each tick."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(pairs)
    # Perceived at the final waypoint — cursor can't advance past 0; command stays.
    cmd1, _ = executor.step(_make_state(arm_conf=[0.1, 0, 0, 0, 0, 0, 0]))
    cmd2, _ = executor.step(_make_state(arm_conf=[0.1, 0, 0, 0, 0, 0, 0]))
    assert cmd1.arm_goal[0] == pytest.approx(0.1)
    assert cmd2.arm_goal[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Done semantics
# ---------------------------------------------------------------------------


def test_done_not_immediate_when_final_target_equals_initial_perceived():
    """Multi-waypoint arm segment is not immediately done even when the final target
    equals the initial perceived position.

    Regression test for the merged approach+retract arm segment bug: the Pick
    skill produces one "arm" segment whose final waypoint is HOME (retract), the
    same position the robot starts at. Without the cursor-guard in done(), the
    distance check fires before a single step() call and the arm never moves.
    """
    # Simulate: approach from [0.0]*7 out to [1.0, 0, ...] and back to [0.0]*7.
    home = [0.0] * 7
    pairs = [
        # approach leg
        (_make_state(arm_conf=home), _arm_action(arm_deltas=[0.5, 0, 0, 0, 0, 0, 0])),
        (
            _make_state(arm_conf=[0.5, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.5, 0, 0, 0, 0, 0, 0]),
        ),
        # retract leg — final target is home ([1.0 - 0.5 - 0.5, ...] = [0.0, ...])
        (
            _make_state(arm_conf=[1.0, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[-0.5, 0, 0, 0, 0, 0, 0]),
        ),
        (
            _make_state(arm_conf=[0.5, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[-0.5, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.1, arrival_tolerance=0.05
    )
    executor.set_trajectory(pairs)

    # Before any step(): perceived = home = final_target — must NOT be done.
    assert executor.done(_make_state(arm_conf=home)) is False


def test_done_true_when_within_arrival_tolerance_of_final_waypoint():
    """Done flips True once perceived joints are within arrival_tolerance of the final
    waypoint."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.5, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, arrival_tolerance=0.01
    )
    executor.set_trajectory(pairs)

    assert executor.done(_make_state(arm_conf=[0.0] * 7)) is False
    assert executor.done(_make_state(arm_conf=[0.5, 0, 0, 0, 0, 0, 0])) is True


def test_done_is_sticky():
    """Once done is reported, drift back outside tolerance does not undo it."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.5, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, arrival_tolerance=0.01
    )
    executor.set_trajectory(pairs)

    assert executor.done(_make_state(arm_conf=[0.5, 0, 0, 0, 0, 0, 0])) is True
    # Drifted away — would have undone done() pre-latch.
    assert executor.done(_make_state(arm_conf=[0.0] * 7)) is True


def test_done_true_at_max_iter_total_even_without_convergence():
    """Done flips True once max_iter_total ticks elapse, even if perceived joints never
    reach the final waypoint."""
    pairs = [
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[0.5, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, arrival_tolerance=1e-9, max_iter_total=3
    )
    executor.set_trajectory(pairs)

    for _ in range(3):
        executor.step(_make_state(arm_conf=[0.0] * 7))
    assert executor.done(_make_state(arm_conf=[0.0] * 7)) is True


# ---------------------------------------------------------------------------
# Commanded action shape
# ---------------------------------------------------------------------------


def test_commanded_action_holds_base_at_perceived_pose():
    """An arm pair's commanded TidyBotAction has base = perceived base pose."""
    state = _make_state(
        base_xytheta=(1.0, 2.0, 0.5),
        arm_conf=[0.0] * 7,
    )
    pairs = [(state, _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]))]
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory(pairs)

    real_action, _ = executor.step(state)
    assert real_action.base_pose_target_map.x == pytest.approx(1.0)
    assert real_action.base_pose_target_map.y == pytest.approx(2.0)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.5)


def test_gripper_close_not_skipped_by_advance_cursor():
    """Gripper-close pairs (arm_delta=0) are not skipped when the arm is already
    at the target joint position.

    Regression test: the cursor crossover advance skips any pair whose target
    equals the perceived joints. Gripper-close pairs have arm_delta=0, so their
    target is the current grasp position — the cursor was jumping past them and
    the gripper command was never issued.
    """
    grasp_joints = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pairs = [
        # approach: move from home to grasp
        (
            _make_state(arm_conf=[0.0] * 7),
            _arm_action(arm_deltas=[1.0, 0, 0, 0, 0, 0, 0]),
        ),
        # gripper close: arm holds, gripper closes
        (
            _make_state(arm_conf=grasp_joints),
            _arm_action(arm_deltas=[0.0] * 7, gripper_cmd=-1.0),
        ),
        # retract: move back toward home
        (
            _make_state(arm_conf=grasp_joints),
            _arm_action(arm_deltas=[-1.0, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.5
    )
    executor.set_trajectory(pairs)

    # Perceive the arm at the grasp position (approach just completed).
    perceived_at_grasp = _make_state(arm_conf=grasp_joints)

    # The cursor should stop at the gripper pair, not jump straight to retract.
    real_action, sim_action = executor.step(perceived_at_grasp)
    assert real_action.gripper_goal == pytest.approx(1.0), (
        "gripper-close command must be emitted on the tick the arm arrives at grasp"
    )


def test_gripper_close_command_emitted():
    """A gripper-close (<-0.5) becomes TidyBotAction.gripper_goal=1.0."""
    state = _make_state(gripper=0.4, arm_conf=[0.0] * 7)
    pairs = [(state, _arm_action(gripper_cmd=-1.0))]
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory(pairs)

    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == 1.0


def test_gripper_no_change_uses_planned_finger():
    """A gripper command in [-0.5, 0.5] uses the planned state's finger_state as the
    hold target (not the perceived finger). This keeps the gripper closed throughout
    retract when the planned state already has finger_state=1.0."""
    planned_state = _make_state(gripper=0.4, arm_conf=[0.0] * 7)
    pairs = [(planned_state, _arm_action(gripper_cmd=0.0))]
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory(pairs)

    # Perceived finger differs from planned finger to make the distinction explicit.
    perceived_state = _make_state(gripper=0.7, arm_conf=[0.0] * 7)
    real_action, _ = executor.step(perceived_state)
    assert real_action.gripper_goal == pytest.approx(0.4)


def test_gripper_stays_closed_during_retract_after_grasp():
    """After a gripper-close, retract pairs maintain gripper_goal=1.0 even when
    the perceived finger has not yet reached 1.0.

    The planned state for retract pairs has finger_state=1.0 (post-grasp). Using
    the planned finger prevents the arm executor from re-issuing the partially-closed
    perceived position as the gripper hold target on every retract tick.
    """
    grasp_joints = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pairs = [
        # gripper-close pair: arm holds, gripper closes
        (
            _make_state(arm_conf=grasp_joints, gripper=0.4),
            _arm_action(arm_deltas=[0.0] * 7, gripper_cmd=-1.0),
        ),
        # retract pair: planned state has finger=1.0 (post-grasp)
        (
            _make_state(arm_conf=grasp_joints, gripper=1.0),
            _arm_action(arm_deltas=[-1.0, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    # gripper_dwell_ticks=0: cursor advances immediately after one gripper tick
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance,
        advance_radius=0.5,
        arrival_tolerance=0.05,
        gripper_dwell_ticks=0,
    )
    executor.set_trajectory(pairs)

    # Tick 1: arm at grasp — gripper-close issued, cursor advances to retract
    executor.step(_make_state(arm_conf=grasp_joints, gripper=0.4))

    # Tick 2: retract phase; perceived finger still partially closed (0.4)
    real_action, _ = executor.step(_make_state(arm_conf=grasp_joints, gripper=0.4))
    assert real_action.gripper_goal == pytest.approx(1.0), (
        "retract phase must hold gripper_goal=1.0 using planned finger, not perceived"
    )


def test_gripper_dwell_holds_arm_at_grasp():
    """gripper_dwell_ticks > 0 keeps the arm at the grasp position for that many
    extra ticks after issuing the close command, before advancing to retract.

    This lets the Kinova gripper physically close around the object before the
    arm starts retracting.
    """
    grasp_joints = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pairs = [
        (
            _make_state(arm_conf=grasp_joints, gripper=0.4),
            _arm_action(arm_deltas=[0.0] * 7, gripper_cmd=-1.0),
        ),
        (
            _make_state(arm_conf=grasp_joints, gripper=1.0),
            _arm_action(arm_deltas=[-1.0, 0, 0, 0, 0, 0, 0]),
        ),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance,
        advance_radius=0.5,
        arrival_tolerance=0.05,
        gripper_dwell_ticks=2,
    )
    executor.set_trajectory(pairs)

    perceived = _make_state(arm_conf=grasp_joints, gripper=0.4)

    # Ticks 1–3: dwell counts down (2→1→0), cursor does not advance yet; arm at grasp.
    # The cursor advances at the END of the tick when dwell_remaining hits 0, so the
    # retract target first appears on tick 4.
    for tick in range(1, 4):
        a, _ = executor.step(perceived)
        assert a.gripper_goal == pytest.approx(1.0), f"tick {tick}: gripper_goal"
        assert a.arm_goal[0] == pytest.approx(1.0), f"tick {tick}: arm must hold at grasp"

    # Tick 4: cursor is now at the retract pair → arm moves to home (0.0)
    action4, _ = executor.step(perceived)
    assert action4.gripper_goal == pytest.approx(1.0)
    assert action4.arm_goal[0] == pytest.approx(0.0), "arm must retract after dwell ends"
