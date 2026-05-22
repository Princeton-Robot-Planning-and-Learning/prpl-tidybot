"""Plan executor for kinematic3d arm + gripper trajectories — stub.

Drives a sequence of (state, action) pairs whose kinder 11-d action holds
the base delta at zero and encodes arm joint deltas in ``action[3:10]``
and a gripper command in ``action[10]``.

**This class is unimplemented.** The previous in-place implementation
shipped as part of the unified ``Kinematic3DPlanExecutor`` produced
incorrect behaviour on real hardware (joint-angle wrap-around when
grounding ``target = state + delta``; settle semantics that didn't match
the underlying OTG; per-pair tolerances tuned against sim, not the
real arm) and has been removed. All :class:`PlanExecutor` methods raise
:class:`NotImplementedError` until a reworked implementation lands.

The higher-level :class:`Kinematic3DPlanExecutor` dispatcher still
routes arm/gripper segments here, so any trajectory that contains an
arm or gripper pair will raise as soon as the dispatcher reaches that
segment. This is intentional — it surfaces the gap loudly rather than
silently running broken arm logic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState

from prpl_tidybot.structs import TidyBotAction

_NOT_IMPLEMENTED_MSG = (
    "ArmMotion3DPlanExecutor is a stub; arm/gripper plan execution is "
    "unimplemented (the previous implementation was incorrect and was removed). "
    "Trajectories that move the arm joints or gripper cannot be executed yet."
)


class ArmMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Executor for kinematic3d arm/gripper trajectories. NOT IMPLEMENTED."""

    def __init__(self, robot_name: str = "robot") -> None:
        self._robot_name = robot_name

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        del trajectory
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        del sim_state
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def done(self, sim_state: ObjectCentricState) -> bool:
        del sim_state
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
