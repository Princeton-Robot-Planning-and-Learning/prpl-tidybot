"""Tests for real_sim/plan_executors/kinematic3d.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.kinematic3d import (
    Kinematic3DPlanExecutor,
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


def _single_action_target(action: np.ndarray, state) -> "tuple":
    """Drive a one-pair trajectory through the executor and return its first command."""
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, action)])
    real_action, returned_sim_action = executor.step(state)
    return real_action, returned_sim_action


def test_base_delta_becomes_absolute_target():
    """Base components of the sim action add componentwise to the current world pose."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.1
    action[1] = -0.2
    action[2] = 0.05

    real_action, _ = _single_action_target(action, state)

    assert real_action.base_pose_target_map.x == pytest.approx(1.1)
    assert real_action.base_pose_target_map.y == pytest.approx(1.8)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.55)


def test_arm_delta_summed_with_current_joints():
    """Arm deltas at action[3..10] are added per-joint to the current arm conf."""
    state = _make_state(arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.zeros(11)
    action[3:10] = [0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.04]

    real_action, _ = _single_action_target(action, state)

    expected = [0.11, 0.22, 0.33, 0.39, 0.48, 0.57, 0.74]
    assert real_action.arm_goal == pytest.approx(expected)


def test_gripper_close_command():
    """Gripper command <-0.5 becomes TidyBotAction.gripper_goal=1.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = -1.0
    real_action, _ = _single_action_target(action, state)
    assert real_action.gripper_goal == 1.0


def test_gripper_open_command():
    """Gripper command >0.5 becomes TidyBotAction.gripper_goal=0.0."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 1.0
    real_action, _ = _single_action_target(action, state)
    assert real_action.gripper_goal == 0.0


def test_gripper_no_change_passes_through_current():
    """Gripper command in [-0.5, 0.5] passes through the current finger_state."""
    state = _make_state(gripper=0.4)
    action = np.zeros(11)
    action[10] = 0.0
    real_action, _ = _single_action_target(action, state)
    assert real_action.gripper_goal == pytest.approx(0.4)


def test_executor_reissues_cached_target_until_converged():
    """Across multiple ticks on the same waypoint the executor reissues the same
    absolute target it computed on the first tick — even if the perceived state drifts
    in between.

    Settle-then-advance behaviour matches the old per-action inner loop in
    RealTidyBotEnv.step.
    """
    initial_state = _make_state(base_xytheta=(1.0, 2.0, 0.5))
    drifted_state = _make_state(base_xytheta=(1.05, 2.0, 0.5))
    action = np.zeros(11)
    action[0] = 0.1  # base dx
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-3, max_iter=100)
    executor.set_trajectory([(initial_state, action)])

    real_action_1, _ = executor.step(initial_state)
    real_action_2, _ = executor.step(drifted_state)

    assert real_action_1 is real_action_2
    assert real_action_1.base_pose_target_map.x == pytest.approx(1.1)


def test_executor_advances_when_within_tolerance():
    """When the perceived state matches the cached absolute target within tolerance, the
    executor advances to the next waypoint on the next done() check."""
    initial_state = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action0 = np.zeros(11)
    action0[0] = 0.5
    action1 = np.zeros(11)
    action1[1] = 0.5
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory(
        [(initial_state, action0), (_make_state(base_xytheta=(0.5, 0.0, 0.0)), action1)]
    )

    assert executor.done(initial_state) is False
    executor.step(initial_state)
    # Perceived state now matches the first waypoint's absolute target.
    converged_state = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    assert executor.done(converged_state) is False  # advances to waypoint 1
    real_action, _ = executor.step(converged_state)
    # Second waypoint: snapshot was converged_state (0.5, 0, 0); delta +0.5 in y.
    assert real_action.base_pose_target_map.x == pytest.approx(0.5)
    assert real_action.base_pose_target_map.y == pytest.approx(0.5)


def test_executor_advances_at_max_iter_even_without_convergence():
    """If max_iter ticks elapse without convergence the executor still advances; this is
    the safety cap that prevents the rollout from getting stuck on an unreachable
    target."""
    initial_state = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action = np.zeros(11)
    action[0] = 1.0  # absolute target 1.0 in x; we'll keep observing 0.0
    executor = Kinematic3DPlanExecutor(position_tolerance=1e-9, max_iter=3)
    executor.set_trajectory([(initial_state, action)])

    for _ in range(3):
        executor.step(initial_state)
    assert executor.done(initial_state) is True
