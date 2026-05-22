"""Tests for the kinematic3d base-motion plan executors.

`BaseMotion3DPlanExecutor` is abstract; the two concrete subclasses implement different
tracking strategies and have separate test sections below. Shared concerns (base-only
validation, empty trajectory, etc.) are also tested against both subclasses.
"""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
    SettleBaseMotion3DPlanExecutor,
)
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


# ---------------------------------------------------------------------------
# Shared validation (any base-motion subclass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "executor_cls",
    [PurePursuitBaseMotion3DPlanExecutor, SettleBaseMotion3DPlanExecutor],
)
def test_rejects_pair_with_arm_motion(executor_cls):
    """A pair with any arm joint delta raises ValueError at set_trajectory time."""
    action = np.zeros(11)
    action[3] = 0.05  # arm joint 1
    executor = executor_cls()
    with pytest.raises(ValueError, match="base-only pairs"):
        executor.set_trajectory([(_make_state(), action)])


@pytest.mark.parametrize(
    "executor_cls",
    [PurePursuitBaseMotion3DPlanExecutor, SettleBaseMotion3DPlanExecutor],
)
def test_rejects_pair_with_gripper_command(executor_cls):
    """A pair with any gripper command raises ValueError at set_trajectory time."""
    action = np.zeros(11)
    action[10] = -1.0
    executor = executor_cls()
    with pytest.raises(ValueError, match="base-only pairs"):
        executor.set_trajectory([(_make_state(), action)])


@pytest.mark.parametrize(
    "executor_cls",
    [PurePursuitBaseMotion3DPlanExecutor, SettleBaseMotion3DPlanExecutor],
)
def test_empty_trajectory_is_immediately_done(executor_cls):
    """An empty trajectory yields done=True without any step() call."""
    executor = executor_cls()
    executor.set_trajectory([])
    assert executor.done(_make_state()) is True


# ---------------------------------------------------------------------------
# PurePursuitBaseMotion3DPlanExecutor
# ---------------------------------------------------------------------------


def test_pure_pursuit_rejects_nonpositive_lookahead():
    """Constructor rejects a non-positive lookahead distance."""
    with pytest.raises(ValueError, match="lookahead_distance"):
        PurePursuitBaseMotion3DPlanExecutor(lookahead_distance=0.0)


def test_pure_pursuit_clamps_to_final_when_path_is_short():
    """Single base pair (path < lookahead): the commanded target is the final
    waypoint."""
    state = _make_state(base_xytheta=(1.0, 2.0, 0.0))
    action = _base_action(dx=0.05, dy=-0.05)
    executor = PurePursuitBaseMotion3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.base_pose_target_map.x == pytest.approx(1.05)
    assert real_action.base_pose_target_map.y == pytest.approx(1.95)


def test_pure_pursuit_commands_lookahead_along_long_path():
    """On a path longer than lookahead_distance, target sits at cursor + lookahead."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    s2 = _make_state(base_xytheta=(1.0, 0.0, 0.0))
    delta = _base_action(dx=0.5)
    executor = PurePursuitBaseMotion3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, delta), (s1, delta), (s2, delta)])

    real_action, _ = executor.step(s0)
    assert real_action.base_pose_target_map.x == pytest.approx(0.3)

    real_action, _ = executor.step(_make_state(base_xytheta=(0.4, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.7)


def test_pure_pursuit_cursor_is_monotonic_under_perception_jitter():
    """Brief perception jitter back along the path doesn't make the cursor regress."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = PurePursuitBaseMotion3DPlanExecutor(lookahead_distance=0.3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    executor.step(_make_state(base_xytheta=(0.5, 0.0, 0.0)))  # cursor -> 0.5
    real_action, _ = executor.step(_make_state(base_xytheta=(0.45, 0.0, 0.0)))
    assert real_action.base_pose_target_map.x == pytest.approx(0.8)


def test_pure_pursuit_done_only_at_final_waypoint():
    """Done() returns False until the perceived base is within position + angle
    tolerance of the final waypoint."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = PurePursuitBaseMotion3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    assert executor.done(s0) is False
    assert executor.done(_make_state(base_xytheta=(0.5, 0.0, 0.0))) is False
    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True


def test_pure_pursuit_done_is_sticky():
    """Once done() reports True, drift back outside tolerance does not undo it."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = PurePursuitBaseMotion3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True
    assert executor.done(_make_state(base_xytheta=(0.9, 0.0, 0.0))) is True


# ---------------------------------------------------------------------------
# SettleBaseMotion3DPlanExecutor
# ---------------------------------------------------------------------------


def test_settle_grounds_target_from_snapshot_and_holds_arm():
    """For a base-motion pair the commanded TidyBotAction has base = snapshot +
    delta and the arm + gripper are held at the perceived state."""
    state = _make_state(
        base_xytheta=(1.0, 2.0, 0.5),
        arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        gripper=0.7,
    )
    action = _base_action(dx=0.1, dy=-0.2, drot=0.3)
    executor = SettleBaseMotion3DPlanExecutor()
    executor.set_trajectory([(state, action)])

    real_action, _ = executor.step(state)
    assert real_action.base_pose_target_map.x == pytest.approx(1.1)
    assert real_action.base_pose_target_map.y == pytest.approx(1.8)
    assert real_action.base_pose_target_map.theta() == pytest.approx(0.8)
    assert real_action.arm_goal == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert real_action.gripper_goal == pytest.approx(0.7)


def test_settle_reissues_cached_target_until_converged():
    """Across multiple ticks on the same pair the executor reissues the same grounded
    target, even if perception drifts in between."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action = _base_action(dx=0.5)
    drifted = _make_state(base_xytheta=(0.2, 0.0, 0.0))
    executor = SettleBaseMotion3DPlanExecutor(
        position_tolerance=1e-3, max_iter_per_pair=100
    )
    executor.set_trajectory([(s0, action)])

    cmd_1, _ = executor.step(s0)
    cmd_2, _ = executor.step(drifted)
    assert cmd_1 is cmd_2
    assert cmd_1.base_pose_target_map.x == pytest.approx(0.5)


def test_settle_advances_when_within_tolerance():
    """Once the perceived state matches the cached target within tolerance, the executor
    advances to the next pair on the next done() check."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    executor = SettleBaseMotion3DPlanExecutor(position_tolerance=1e-3)
    executor.set_trajectory([(s0, _base_action(dx=0.5)), (s1, _base_action(dx=0.5))])

    executor.step(s0)
    converged = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    assert executor.done(converged) is False  # advances pair, not whole trajectory
    cmd, _ = executor.step(converged)
    # New snapshot at (0.5, 0, 0); next target is (1.0, 0, 0).
    assert cmd.base_pose_target_map.x == pytest.approx(1.0)


def test_settle_advances_at_max_iter_even_without_convergence():
    """If max_iter_per_pair ticks elapse without convergence, the executor still
    advances; this caps the rollout if a target is unreachable."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    action = _base_action(dx=0.5)
    executor = SettleBaseMotion3DPlanExecutor(
        position_tolerance=1e-9, max_iter_per_pair=3
    )
    executor.set_trajectory([(s0, action)])

    for _ in range(3):
        executor.step(s0)
    assert executor.done(s0) is True
