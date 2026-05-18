"""A pass-through perceiver for when the env already produces the state.

Used by the sim-mode pipeline (PrplLab3DSimEnv), where `_RealObsType` and
`_StateType` are the same ObjectCentricState and no perception is needed.
"""

from typing import Any, Generic, TypeVar

from prpl_utils.real_sim import Perceiver

_T = TypeVar("_T")


class PassThroughPerceiver(Generic[_T], Perceiver[_T, _T]):
    """Identity perceiver: returns the observation as-is."""

    def reset(self, obs: _T, info: dict[str, Any]) -> _T:
        del info
        return obs

    def step(self, obs: _T, info: dict[str, Any]) -> _T:
        del info
        return obs
