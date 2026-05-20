"""A pass-through plan executor for when sim and real actions coincide.

Used by the sim-mode pipeline (KinderSimEnv), where the agent's sim action
and the env's expected action are the same kinder 11-d vector and no
translation is needed. The executor simply emits each planned action in
turn — one real-env tick per (state, action) pair — and reports done when
the trajectory is exhausted.
"""

from typing import Any, Generic, TypeVar

from prpl_utils.real_sim import PlanExecutor

_T = TypeVar("_T")


class PassThroughPlanExecutor(Generic[_T], PlanExecutor[_T, _T, Any]):
    """Identity plan executor: one real-env tick per trajectory entry."""

    def __init__(self) -> None:
        self._actions: list[_T] = []
        self._index = 0

    def set_trajectory(self, trajectory: list[tuple[Any, _T]]) -> None:
        self._actions = [action for _, action in trajectory]
        self._index = 0

    def step(self, sim_state: Any) -> tuple[_T, _T]:
        del sim_state
        sim_action = self._actions[self._index]
        self._index += 1
        return sim_action, sim_action

    def done(self, sim_state: Any) -> bool:
        del sim_state
        return self._index >= len(self._actions)
