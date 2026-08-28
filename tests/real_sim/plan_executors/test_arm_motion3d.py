"""Tests for the kinematic3d arm-motion plan executors.

`ArmMotion3DPlanExecutor` is abstract; `StreamingArmMotion3DPlanExecutor` is the
concrete crossover-advance subclass. Shared concerns (arm-only validation) are tested
against the concrete subclass since the abstract base can't be instantiated.

Distance function for tests is a plain L1 (no wrap), built to be easy to reason about.
Production wires in pybullet-helpers' weighted joint distance.
"""

import logging
from typing import Sequence

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    CarrotArmMotion3DPlanExecutor,
    StreamingArmMotion3DPlanExecutor,
    _path_progress,
)
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
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


def test_done_raises_execution_failure_at_max_iter_total(caplog):
    """Once max_iter_total ticks elapse without reaching the final waypoint, done()
    logs a warning with the remaining distances and raises ExecutionFailure instead of
    reporting the segment complete."""
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
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ExecutionFailure, match="gave up after 3 ticks"):
            executor.done(_make_state(arm_conf=[0.0] * 7))
    assert "distance 0.500 to the cursor target" in caplog.text


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
    real_action, _ = executor.step(perceived_at_grasp)
    assert real_action.gripper_goal == pytest.approx(
        1.0
    ), "gripper-close command must be emitted on the tick the arm arrives at grasp"


def test_gripper_close_command_emitted():
    """A gripper-close (<-0.5) becomes TidyBotAction.gripper_goal=1.0."""
    state = _make_state(gripper=0.4, arm_conf=[0.0] * 7)
    pairs = [(state, _arm_action(gripper_cmd=-1.0))]
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory(pairs)

    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == 1.0


def test_gripper_hold_before_any_command_uses_perceived():
    """Before any explicit open/close command, a hold action (|cmd| <= 0.5) uses
    the perceived finger_state as the goal — there is no prior command to remember."""
    pairs = [(_make_state(arm_conf=[0.0] * 7), _arm_action(gripper_cmd=0.0))]
    executor = StreamingArmMotion3DPlanExecutor(distance_fn=_l1_distance)
    executor.set_trajectory(pairs)

    perceived_state = _make_state(gripper=0.7, arm_conf=[0.0] * 7)
    real_action, _ = executor.step(perceived_state)
    assert real_action.gripper_goal == pytest.approx(0.7)


def test_gripper_hold_after_close_uses_last_goal():
    """After an explicit close command, hold ticks maintain gripper_goal=1.0 regardless
    of the perceived finger_state. The kinder planning sim does not update finger_state
    after close actions, so we cannot read it from the planned state."""
    grasp_joints = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pairs = [
        (_make_state(arm_conf=grasp_joints), _arm_action(gripper_cmd=-1.0)),  # close
        (_make_state(arm_conf=grasp_joints), _arm_action(gripper_cmd=0.0)),  # hold
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.5
    )
    executor.set_trajectory(pairs)

    # Tick 1: close command — advances cursor (dwell=0)
    executor.step(_make_state(arm_conf=grasp_joints, gripper=0.0))
    # Tick 2: hold — perceived finger still 0.0, but last_gripper_goal=1.0
    action, _ = executor.step(_make_state(arm_conf=grasp_joints, gripper=0.0))
    assert action.gripper_goal == pytest.approx(1.0)


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
    assert real_action.gripper_goal == pytest.approx(
        1.0
    ), "retract phase must hold gripper_goal=1.0 using planned finger, not perceived"


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
        assert a.arm_goal[0] == pytest.approx(1.0), f"tick {tick}: arm hold at grasp"

    # Tick 4: cursor is now at the retract pair → arm moves to home (0.0)
    action4, _ = executor.step(perceived)
    assert action4.gripper_goal == pytest.approx(1.0)
    assert action4.arm_goal[0] == pytest.approx(
        0.0
    ), "arm must retract after dwell ends"


# ---------------------------------------------------------------------------
# CarrotArmMotion3DPlanExecutor
# ---------------------------------------------------------------------------


def _joint1_chain(num_waypoints: int, step: float = 0.1):
    """Sequential pairs each adding `step` to joint 1."""
    pairs = []
    for i in range(num_waypoints):
        pairs.append(
            (
                _make_state(arm_conf=[i * step, 0, 0, 0, 0, 0, 0]),
                _arm_action(arm_deltas=[step, 0, 0, 0, 0, 0, 0]),
            )
        )
    return pairs


def test_carrot_commands_constant_lookahead_point():
    """The commanded target sits exactly `lookahead` ahead of the perceived
    joints along the plan polyline, interpolated between waypoints."""
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, lookahead=0.25
    )
    executor.set_trajectory(_joint1_chain(8))
    # Waypoints at 0.1, 0.2, ..., 0.8 on joint 1.
    s = _make_state(arm_conf=[0.12, 0, 0, 0, 0, 0, 0])
    real_action, _ = executor.step(s)
    # Cursor advances to the 0.2 waypoint (0.1 is within radius... it is not:
    # |0.12-0.1|=0.02 <= 0.05, so cursor -> 1 with target 0.2). The carrot sits
    # 0.25 ahead of 0.12 along the polyline: 0.37, between waypoints 0.3, 0.4.
    assert real_action.arm_goal[0] == pytest.approx(0.37)


def test_carrot_far_from_plan_commands_cursor_waypoint():
    """When the cursor waypoint is already >= lookahead away, command it
    unchanged (no interpolation toward the arm)."""
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, lookahead=0.25
    )
    executor.set_trajectory(_joint1_chain(8))
    s = _make_state(arm_conf=[-0.5, 0, 0, 0, 0, 0, 0])
    real_action, _ = executor.step(s)
    assert real_action.arm_goal[0] == pytest.approx(0.1)


def test_carrot_never_commands_past_gripper_waypoint():
    """The lookahead walk stops at a gripper command; the arm holds there
    while the dwell runs."""
    pairs = _joint1_chain(3)  # waypoints 0.1, 0.2, 0.3
    # A close-gripper pair at the 0.3 posture, then further arm motion.
    pairs.append(
        (
            _make_state(arm_conf=[0.3, 0, 0, 0, 0, 0, 0]),
            _arm_action(gripper_cmd=-1.0),
        )
    )
    pairs.extend(
        (
            _make_state(arm_conf=[(3 + i) * 0.1, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[0.1, 0, 0, 0, 0, 0, 0]),
        )
        for i in range(1, 3)
    )
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, lookahead=0.4
    )
    executor.set_trajectory(pairs)
    s = _make_state(arm_conf=[0.19, 0, 0, 0, 0, 0, 0])
    real_action, _ = executor.step(s)
    # A 0.4 lookahead from 0.19 would reach 0.59, but the gripper waypoint at
    # 0.3 caps the command.
    assert real_action.arm_goal[0] == pytest.approx(0.3)


def test_carrot_caps_at_final_waypoint():
    """Near the end of the plan the carrot clamps to the final waypoint, so
    the arm decelerates naturally into it."""
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, lookahead=0.5
    )
    executor.set_trajectory(_joint1_chain(3))  # waypoints 0.1, 0.2, 0.3
    s = _make_state(arm_conf=[0.28, 0, 0, 0, 0, 0, 0])
    real_action, _ = executor.step(s)
    assert real_action.arm_goal[0] == pytest.approx(0.3)


def test_carrot_rejects_nonpositive_lookahead():
    """lookahead must be positive."""
    with pytest.raises(ValueError, match="lookahead"):
        CarrotArmMotion3DPlanExecutor(distance_fn=_l1_distance, lookahead=0.0)


def test_gripper_close_position_caps_the_close_and_the_hold():
    """A close command targets `gripper_close_position` instead of 1.0, and later hold
    ticks keep re-issuing that same partial close."""
    state = _make_state(gripper=0.0, arm_conf=[0.0] * 7)
    pairs = [
        (state, _arm_action(gripper_cmd=-1.0)),
        (state, _arm_action(arm_deltas=[0.5] + [0.0] * 6)),
    ]
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, gripper_close_position=0.7
    )
    executor.set_trajectory(pairs)
    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == pytest.approx(0.7)
    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == pytest.approx(0.7)
    # Open is unaffected.
    executor.set_trajectory([(state, _arm_action(gripper_cmd=1.0))])
    real_action, _ = executor.step(state)
    assert real_action.gripper_goal == 0.0


def test_constructor_rejects_bad_gripper_close_position():
    """gripper_close_position must be in (0, 1]."""
    for value in (0.0, 1.5):
        with pytest.raises(ValueError, match="gripper_close_position"):
            StreamingArmMotion3DPlanExecutor(
                distance_fn=_l1_distance, gripper_close_position=value
            )


# ---------------------------------------------------------------------------
# Path-progress advance (robust to offsets on untracked joints)
# ---------------------------------------------------------------------------


def test_cursor_advances_by_path_progress_despite_untracked_joint_offset():
    """An offset on a joint the plan never moves exceeds advance_radius on its own,
    but the cursor still advances once the tracked joint passes each waypoint."""
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(_joint1_chain(5))  # joint 1 waypoints 0.1 ... 0.5
    # Joint 5 sits 0.2 off the plan for the whole trajectory.
    offset = [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]

    real_action, _ = executor.step(_make_state(arm_conf=offset))
    assert real_action.arm_goal[0] == pytest.approx(0.1)  # cursor 0
    # Joint 1 has passed the 0.1 and 0.2 waypoints (distance to each is
    # 0.2 + |dj1| > radius, but the projection is past them).
    perceived = list(offset)
    perceived[0] = 0.23
    real_action, _ = executor.step(_make_state(arm_conf=perceived))
    assert real_action.arm_goal[0] == pytest.approx(0.3)
    # Done still needs the final target within arrival_tolerance.
    perceived[0] = 0.5
    for _ in range(3):
        executor.step(_make_state(arm_conf=perceived))
    assert not executor.done(_make_state(arm_conf=perceived))
    assert executor.done(_make_state(arm_conf=[0.5, 0, 0, 0, 0.05, 0, 0]))


def test_cursor_does_not_advance_on_progress_before_the_target():
    """A point beside the segment but short of the target does not advance."""
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05
    )
    executor.set_trajectory(_joint1_chain(3))
    perceived = [0.04, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    real_action, _ = executor.step(_make_state(arm_conf=perceived))
    assert real_action.arm_goal[0] == pytest.approx(0.1)


def test_path_progress_wraps_continuous_joints():
    """Projection uses wrapped joint differences, so a target on the far side of the
    +/-pi seam still reads as passed."""
    start = [3.0, 0, 0, 0, 0, 0, 0]
    end = [3.3, 0, 0, 0, 0, 0, 0]  # == -2.98 after wrapping
    assert _path_progress(start, end, [-2.9, 0, 0, 0, 0, 0, 0]) > 1.0
    assert _path_progress(start, end, [3.15, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.5)


def test_carrot_lookahead_measured_along_path_not_raw_distance():
    """With an orthogonal offset, the carrot still commands a full lookahead ahead
    along the path (the raw distance to the cursor target would eat into it)."""
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, lookahead=0.25
    )
    executor.set_trajectory(_joint1_chain(8))
    s = _make_state(arm_conf=[0.12, 0, 0, 0, 0.2, 0, 0])
    real_action, _ = executor.step(s)
    # Same as the no-offset case: 0.25 ahead of 0.12 along the polyline.
    assert real_action.arm_goal[0] == pytest.approx(0.37)


def test_stall_warning_logged_once_after_no_advance(caplog):
    """If the cursor does not move for stall_warning_ticks ticks a single warning with
    the distance and progress is logged."""
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, stall_warning_ticks=3
    )
    executor.set_trajectory(_joint1_chain(3))
    stuck = _make_state(arm_conf=[-0.5, 0, 0, 0, 0, 0, 0])
    with caplog.at_level(logging.WARNING):
        for _ in range(6):
            executor.step(stuck)
    warnings = [r for r in caplog.records if "has not advanced" in r.getMessage()]
    assert len(warnings) == 1
    assert "waypoint 1/3" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Stalled arrival (steady-state controller error)
# ---------------------------------------------------------------------------


def _approach_then_open(num_waypoints: int = 4):
    """Joint-1 approach waypoints followed by a gripper open and one retract pair."""
    pairs = _joint1_chain(num_waypoints)
    last = num_waypoints * 0.1
    pairs.append(
        (_make_state(arm_conf=[last, 0, 0, 0, 0, 0, 0]), _arm_action(gripper_cmd=1.0))
    )
    pairs.append(
        (
            _make_state(arm_conf=[last, 0, 0, 0, 0, 0, 0]),
            _arm_action(arm_deltas=[-0.1, 0, 0, 0, 0, 0, 0]),
        )
    )
    return pairs


def test_stationary_stall_before_gripper_command_counts_as_arrival(caplog):
    """Stopped short of the last approach waypoint (progress >= 0.5) and not moving
    for stall_advance_ticks ticks, the executor advances to the gripper pair and
    issues the gripper command."""
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance,
        advance_radius=0.05,
        stall_advance_ticks=5,
        stall_advance_min_progress=0.5,
    )
    executor.set_trajectory(_approach_then_open(4))  # waypoints 0.1..0.4, open, retract
    # Joint 1 at 0.36 (progress 0.6 on 0.3 -> 0.4) with an offset on joint 5 that
    # keeps the raw distance above advance_radius.
    stuck = _make_state(arm_conf=[0.36, 0, 0, 0, 0.2, 0, 0])
    with caplog.at_level(logging.WARNING):
        # Stillness is counted from the second tick on: five ticks of holding.
        for _ in range(5):
            real_action, _ = executor.step(stuck)
            assert real_action.gripper_goal == pytest.approx(0.4)  # hold perceived
        real_action, _ = executor.step(stuck)
    assert real_action.gripper_goal == 0.0  # the open command
    assert any("treating it as reached" in r.getMessage() for r in caplog.records)


def test_stationary_stall_mid_path_advances_one_waypoint():
    """A stationary stall past half a mid-path segment advances by one waypoint so the
    carrot leads further; it does not run ahead through the plan."""
    executor = CarrotArmMotion3DPlanExecutor(
        distance_fn=_l1_distance,
        advance_radius=0.05,
        lookahead=0.15,
        stall_advance_ticks=3,
    )
    executor.set_trajectory(_joint1_chain(6))  # 0.1 .. 0.6
    stuck = _make_state(arm_conf=[0.16, 0, 0, 0, 0.2, 0, 0])  # progress 0.6 on 0.1->0.2
    goals = []
    for _ in range(4):
        real_action, _ = executor.step(stuck)
        goals.append(real_action.arm_goal[0])
    # Before the stall rule fires the carrot leads 0.15 past the projected 0.16:
    # 0.31. After one stall-advance the cursor is at 0.3 and the lead is measured
    # from that segment's start (0.2), so the carrot moves out to 0.35.
    assert goals[0] == pytest.approx(0.31)
    assert goals[-1] == pytest.approx(0.35)


def test_stall_needs_progress_and_stillness():
    """Short of half the segment, or still moving, a stall does not advance."""
    executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=_l1_distance, advance_radius=0.05, stall_advance_ticks=3
    )
    executor.set_trajectory(_approach_then_open(4))
    early = _make_state(arm_conf=[0.32, 0, 0, 0, 0.2, 0, 0])  # progress 0.2
    for _ in range(6):
        real_action, _ = executor.step(early)
    assert real_action.gripper_goal == pytest.approx(0.4)

    executor.set_trajectory(_approach_then_open(4))
    for i in range(6):  # creeping within the segment: never still
        moving = _make_state(arm_conf=[0.36 + 0.006 * i, 0, 0, 0, 0.2, 0, 0])
        real_action, _ = executor.step(moving)
    assert real_action.gripper_goal == pytest.approx(0.4)
