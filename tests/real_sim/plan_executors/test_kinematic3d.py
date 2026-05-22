"""Tests for the Kinematic3DPlanExecutor dispatcher.

The dispatcher's job is narrow: validate that each (state, action) pair moves either the
base XOR the arm/gripper, split the trajectory into maximal same-kind segments, and
delegate each segment to the appropriate sub-executor. Strategy-specific tracking
behaviour is covered in test_base_motion3d.py; the arm sub-executor's
NotImplementedError surface is covered in test_arm_motion3d.py.
"""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.structs import TidyBotObservation


def _tight_executor() -> Kinematic3DPlanExecutor:
    """Dispatcher with a tight-tolerance pure-pursuit base sub-executor."""
    return Kinematic3DPlanExecutor(
        base_executor=PurePursuitBaseMotion3DPlanExecutor(position_tolerance=1e-3),
    )


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


def test_rejects_pair_that_mixes_base_and_arm_motion():
    """A pair with both base and arm deltas violates the dispatcher's invariant."""
    mixed = np.zeros(11)
    mixed[0] = 0.1  # base
    mixed[3] = 0.05  # arm joint 1
    executor = Kinematic3DPlanExecutor()
    with pytest.raises(ValueError, match="ONLY the base OR the arm"):
        executor.set_trajectory([(_make_state(), mixed)])


def test_rejects_pair_that_mixes_base_and_gripper_command():
    """Gripper rides on the arm side, so base + gripper is also a mixed pair."""
    mixed = np.zeros(11)
    mixed[0] = 0.1
    mixed[10] = -1.0
    executor = Kinematic3DPlanExecutor()
    with pytest.raises(ValueError, match="ONLY the base OR the arm"):
        executor.set_trajectory([(_make_state(), mixed)])


# ---------------------------------------------------------------------------
# Segmentation / delegation
# ---------------------------------------------------------------------------


def test_base_only_trajectory_delegates_to_base_executor():
    """A base-only trajectory drives the base sub-executor and reaches the final
    waypoint."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    executor = _tight_executor()
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    real_action, _ = executor.step(s0)
    # Pure-pursuit commands lookahead along the path.
    assert real_action.base_pose_target_map.x == pytest.approx(0.2)
    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True


def test_trajectory_with_arm_segment_raises_when_no_arm_executor_configured():
    """An arm segment with no arm_executor wired in surfaces a clear NotImplementedError
    from the dispatcher when the segment is loaded."""
    s0 = _make_state()
    arm_pair = _arm_action(arm_deltas=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    executor = Kinematic3DPlanExecutor()  # no arm_executor
    with pytest.raises(NotImplementedError, match="arm_executor was configured"):
        executor.set_trajectory([(s0, arm_pair)])


def test_base_then_arm_trajectory_runs_base_then_raises_at_arm_segment():
    """A trajectory of [base, arm] runs the base segment to completion, then raises
    NotImplementedError when the dispatcher tries to load the arm segment with no
    arm_executor configured."""
    s0 = _make_state(base_xytheta=(0.0, 0.0, 0.0))
    s1 = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    executor = _tight_executor()  # no arm_executor
    executor.set_trajectory(
        [
            (s0, _base_action(dx=0.5)),
            (s1, _arm_action(arm_deltas=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ]
    )

    # Base segment runs.
    executor.step(s0)
    # Robot reaches the base final waypoint; done() then tries to load the arm
    # segment, which raises from the dispatcher.
    s_after_base = _make_state(base_xytheta=(0.5, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match="arm_executor was configured"):
        executor.done(s_after_base)


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
    executor = _tight_executor()
    executor.set_trajectory([(s0, _base_action(dx=1.0))])

    assert executor.done(_make_state(base_xytheta=(1.0, 0.0, 0.0))) is True
    # Drifted reading would have undone done() pre-latch.
    assert executor.done(_make_state(base_xytheta=(0.9, 0.0, 0.0))) is True
