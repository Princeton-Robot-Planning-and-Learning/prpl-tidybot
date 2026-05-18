"""A pass-through action grounder for when sim and real actions coincide.

Used by the sim-mode pipeline (PrplLab3DSimEnv), where the agent's
sim action and the env's expected action are the same kinder 11-d
vector and no translation is needed.
"""

from typing import Any, Generic, TypeVar

from prpl_utils.real_sim import ActionGrounder

_T = TypeVar("_T")


class PassThroughActionGrounder(Generic[_T], ActionGrounder[_T, _T, Any]):
    """Identity action grounder: returns the sim action unchanged."""

    def __call__(self, sim_action: _T, sim_state: Any) -> _T:
        del sim_state
        return sim_action
