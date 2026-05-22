"""Tests for the kinematic3d arm/gripper plan executor stub.

`ArmMotion3DPlanExecutor` is unimplemented; all `PlanExecutor` methods raise
NotImplementedError. These tests pin that behaviour so a future re-implementation has to
explicitly replace it.
"""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import ArmMotion3DPlanExecutor
from prpl_tidybot.structs import TidyBotObservation


def _make_state():
    obs = TidyBotObservation(
        arm_conf=[0.0] * 7,
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        gripper=0.0,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


def test_set_trajectory_raises():
    """`set_trajectory` raises NotImplementedError — the stub never accepts a plan."""
    executor = ArmMotion3DPlanExecutor()
    arm_pair = np.zeros(11)
    arm_pair[3] = 0.1
    with pytest.raises(NotImplementedError, match="unimplemented"):
        executor.set_trajectory([(_make_state(), arm_pair)])


def test_step_raises():
    """`step` raises NotImplementedError on the stub."""
    executor = ArmMotion3DPlanExecutor()
    with pytest.raises(NotImplementedError, match="unimplemented"):
        executor.step(_make_state())


def test_done_raises():
    """`done` raises NotImplementedError on the stub."""
    executor = ArmMotion3DPlanExecutor()
    with pytest.raises(NotImplementedError, match="unimplemented"):
        executor.done(_make_state())
