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
* arm/gripper segments → an :class:`ArmMotion3DPlanExecutor` subclass
  (e.g. :class:`StreamingArmMotion3DPlanExecutor`). The arm executor
  must be wired in explicitly; the dispatcher's ``arm_executor``
  argument defaults to ``None`` and raises a clear
  :class:`NotImplementedError` if a trajectory containing arm motion
  is set without one configured. This avoids silently importing
  pybullet (needed to construct the streaming arm executor's distance
  function) for base-only callers.
* magic segments — a single ``SkillCall`` pair standing in for a skill
  the planner did not simulate — → a ``gap_executor`` (e.g.
  :class:`SettleGapExecutor`), which decides how the skill is actually
  carried out. Like the arm executor it must be wired in explicitly;
  a trajectory containing a ``SkillCall`` raises
  :class:`NotImplementedError` when no gap executor is configured.

The dispatcher takes the sub-executors as constructor arguments so
Hydra can instantiate the desired concrete classes directly via
``_target_`` — there's no string-based strategy switch here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from kinder_models.structs import SkillCall
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import ArmMotion3DPlanExecutor
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    BaseMotion3DPlanExecutor,
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.structs import RealAction

# A planned action: a kinder 11-d delta, or a SkillCall marking a magic gap.
SimAction = NDArray[np.floating] | SkillCall[ObjectCentricState]

_BASE_MOTION_EPS = 1e-4
_ARM_MOTION_EPS = 1e-4


@dataclass
class _Segment:
    """A maximal run of (state, action) pairs handled by one sub-executor.

    ``kind`` is ``"base"`` (each pair has nontrivial base motion only),
    ``"arm"`` (each pair has no base motion; arm and/or gripper may
    move), or ``"magic"`` (exactly one ``SkillCall`` pair; consecutive
    calls form separate segments). Pairs that move neither base nor arm
    land in ``"arm"`` — they would fall through to the arm sub-executor's
    NotImplementedError, but the planner does not produce such pairs in
    practice.
    """

    kind: str
    pairs: list[tuple[ObjectCentricState, Any]]


class Kinematic3DPlanExecutor(PlanExecutor[SimAction, RealAction, ObjectCentricState]):
    """Dispatch a kinematic3d trajectory between base, arm, and gap sub-executors."""

    def __init__(
        self,
        base_executor: BaseMotion3DPlanExecutor | None = None,
        arm_executor: ArmMotion3DPlanExecutor | None = None,
        gap_executor: (
            PlanExecutor[SimAction, RealAction, ObjectCentricState] | None
        ) = None,
    ) -> None:
        self._base_executor = base_executor or PurePursuitBaseMotion3DPlanExecutor()
        self._arm_executor = arm_executor
        self._gap_executor = gap_executor
        self._segments: list[_Segment] = []
        self._segment_idx: int = 0
        self._active: PlanExecutor[Any, Any, ObjectCentricState] | None = None
        self._done_latched: bool = False

    # ------------------------------------------------------------------ Public

    def set_trajectory(
        self,
        trajectory: Sequence[tuple[ObjectCentricState, SimAction]],
    ) -> None:
        for _, action in trajectory:
            if not isinstance(action, SkillCall):
                _validate_no_mixed_motion(action)
        self._segments = _build_segments(trajectory)
        self._segment_idx = 0
        self._done_latched = False
        self._active = None
        if self._segments:
            self._load_current_segment()

    def step(self, sim_state: ObjectCentricState) -> tuple[RealAction, SimAction]:
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
        if segment.kind == "base":
            self._active = self._base_executor
        elif segment.kind == "magic":
            if self._gap_executor is None:
                raise NotImplementedError(
                    "Kinematic3DPlanExecutor reached a SkillCall (magic skill) but "
                    "no gap_executor was configured. Pass a gap executor (e.g. "
                    "SettleGapExecutor) to the constructor or the Hydra "
                    "plan_executor config."
                )
            self._active = self._gap_executor
        else:
            if self._arm_executor is None:
                raise NotImplementedError(
                    "Kinematic3DPlanExecutor reached an arm/gripper segment but "
                    "no arm_executor was configured. Pass a concrete "
                    "ArmMotion3DPlanExecutor subclass (e.g. "
                    "StreamingArmMotion3DPlanExecutor) to the constructor or "
                    "the Hydra plan_executor config."
                )
            self._active = self._arm_executor
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
    trajectory: Sequence[tuple[ObjectCentricState, SimAction]],
) -> list[_Segment]:
    """Split into maximal runs of same-kind pairs.

    A ``SkillCall`` pair is ``"magic"`` and always forms its own segment.
    Otherwise a pair is ``"base"`` if any base-delta component is
    nontrivial and ``"arm"`` if not. The validator in
    :meth:`Kinematic3DPlanExecutor.set_trajectory` already rejects pairs
    that move both groups, so the classification is unambiguous.
    """
    segments: list[_Segment] = []
    current: _Segment | None = None
    for state, action in trajectory:
        if isinstance(action, SkillCall):
            kind = "magic"
        elif np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS):
            kind = "base"
        else:
            kind = "arm"
        if current is None or current.kind != kind or kind == "magic":
            if current is not None:
                segments.append(current)
            current = _Segment(kind=kind, pairs=[])
        current.pairs.append((state, action))
    if current is not None:
        segments.append(current)
    return segments
