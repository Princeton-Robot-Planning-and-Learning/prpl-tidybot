"""Per-segment dispatcher between base and arm/gripper sub-executors.

Bilevel-planning trajectories alternate between base-motion and
arm/gripper-motion pairs but never mix the two within a single pair —
mixing complicates per-segment strategy choice (you can't pure-pursue an
arm waypoint) and wasn't useful on the planner side either. This
dispatcher enforces that XOR at :meth:`set_trajectory` time, splits the
trajectory into maximal runs of same-kind pairs ("segments"), and feeds
each segment to the appropriate sub-executor:

* base segments → a :class:`BaseMotion3DPlanExecutor` subclass
  (:class:`PurePursuitBaseMotion3DPlanExecutor` by default;
  :class:`SettleBaseMotion3DPlanExecutor` is also available)
* arm/gripper segments → :class:`ArmMotion3DPlanExecutor` (currently a
  stub that raises :class:`NotImplementedError`; the prior in-place
  implementation produced wrong behaviour on real hardware)

The dispatcher takes the sub-executors as constructor arguments so
Hydra can instantiate the desired concrete classes directly via
``_target_`` — there's no string-based strategy switch here. A
trajectory that contains any arm or gripper motion raises the first
time the dispatcher reaches the corresponding segment. This is
intentional until the arm executor is rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import ArmMotion3DPlanExecutor
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    BaseMotion3DPlanExecutor,
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.structs import TidyBotAction

_BASE_MOTION_EPS = 1e-4
_ARM_MOTION_EPS = 1e-4


@dataclass
class _Segment:
    """A maximal run of (state, action) pairs that all move the same component.

    ``kind`` is ``"base"`` (each pair has nontrivial base motion only) or
    ``"arm"`` (each pair has no base motion; arm and/or gripper may
    move). Pairs that move neither also land in ``"arm"`` — they would
    fall through to the arm sub-executor's NotImplementedError, but the
    planner does not produce such pairs in practice.
    """

    kind: str
    pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]]


class Kinematic3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Dispatch a kinematic3d trajectory between base and arm sub-executors."""

    def __init__(
        self,
        base_executor: BaseMotion3DPlanExecutor | None = None,
        arm_executor: ArmMotion3DPlanExecutor | None = None,
    ) -> None:
        self._base_executor = base_executor or PurePursuitBaseMotion3DPlanExecutor()
        self._arm_executor = arm_executor or ArmMotion3DPlanExecutor()
        self._segments: list[_Segment] = []
        self._segment_idx: int = 0
        self._active: (
            PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState] | None
        ) = None
        self._done_latched: bool = False

    # ------------------------------------------------------------------ Public

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        for _, action in trajectory:
            _validate_no_mixed_motion(action)
        self._segments = _build_segments(trajectory)
        self._segment_idx = 0
        self._done_latched = False
        self._active = None
        if self._segments:
            self._load_current_segment()

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if self._done_latched or self._segment_idx >= len(self._segments):
            raise RuntimeError(
                "Kinematic3DPlanExecutor.step called after the trajectory finished"
            )
        assert self._active is not None
        return self._active.step(sim_state)

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._done_latched:
            return True
        if not self._segments:
            self._done_latched = True
            return True
        while self._segment_idx < len(self._segments):
            assert self._active is not None
            if not self._active.done(sim_state):
                return False
            self._segment_idx += 1
            if self._segment_idx < len(self._segments):
                self._load_current_segment()
        self._done_latched = True
        return True

    # ---------------------------------------------------------------- Internal

    def _load_current_segment(self) -> None:
        """Hand the current segment's pairs to the appropriate sub-executor."""
        segment = self._segments[self._segment_idx]
        self._active = (
            self._base_executor if segment.kind == "base" else self._arm_executor
        )
        self._active.set_trajectory(segment.pairs)


# ============================================================================
# Module-level helpers
# ============================================================================


def _validate_no_mixed_motion(action: NDArray[np.floating]) -> None:
    """Reject pairs that command motion across more than one component group.

    Components: base (action[0:3]), arm joints (action[3:10]), gripper
    (action[10]). Arm joints and gripper count as a single "arm" group —
    they're handled together by ArmMotion3DPlanExecutor.
    """
    base_moves = bool(np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS))
    arm_or_gripper_moves = bool(np.any(np.abs(action[3:10]) > _ARM_MOTION_EPS)) or (
        abs(float(action[10])) > _ARM_MOTION_EPS
    )
    if base_moves and arm_or_gripper_moves:
        raise ValueError(
            "Kinematic3DPlanExecutor requires each (state, action) pair to "
            "move ONLY the base OR the arm/gripper, not both. Got base_delta="
            f"{action[0:3]}, arm_delta={action[3:10]}, gripper_cmd={action[10]}."
        )


def _build_segments(
    trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
) -> list[_Segment]:
    """Split into maximal runs of same-kind pairs.

    A pair is ``"base"`` if any base-delta component is nontrivial,
    otherwise ``"arm"``. The validator in
    :meth:`Kinematic3DPlanExecutor.set_trajectory` already rejects pairs
    that move both groups, so the classification is unambiguous.
    """
    segments: list[_Segment] = []
    current: _Segment | None = None
    for state, action in trajectory:
        kind = "base" if np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS) else "arm"
        if current is None or current.kind != kind:
            if current is not None:
                segments.append(current)
            current = _Segment(kind=kind, pairs=[])
        current.pairs.append((state, action))
    if current is not None:
        segments.append(current)
    return segments
