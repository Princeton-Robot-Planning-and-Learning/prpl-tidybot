"""Real-to-sim-to-real adapters for prpl_tidybot.

For fake / real backends, compose a `RealTidyBotEnv`, a kinematic3d
perceiver (`PrplLab3DPerceiver` or `BaseMotion3DPerceiver`),
`Kinematic3DActionGrounder`, and a `prpl_utils.gym_agent.Agent` with
`prpl_utils.real_sim.Runner`. For sim, swap in `KinderSimEnv` plus
`PassThroughPerceiver` / `PassThroughActionGrounder` — the env already
produces an ObjectCentricState, so re-encoding through TidyBotObservation
is unnecessary.

`build_planner_env_models` is a tiny adapter for using
`kinder_bilevel_planning.BilevelPlanningAgent` inside this pipeline; see
its docstring.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import kinder
from bilevel_planning.structs import SesameModels
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from prpl_tidybot.real_sim.action_grounders.kinematic3d import (
    Kinematic3DActionGrounder,
)
from prpl_tidybot.real_sim.action_grounders.passthrough import (
    PassThroughActionGrounder,
)
from prpl_tidybot.real_sim.perceivers.kinematic3d import (
    BaseMotion3DPerceiver,
    KinematicRobotPerceiverBase,
    PrplLab3DPerceiver,
)
from prpl_tidybot.real_sim.perceivers.passthrough import PassThroughPerceiver


def build_planner_env_models(
    env_name: str,
    make_kwargs: Mapping[str, Any],
    env_model_kwargs: Mapping[str, Any] | None = None,
) -> SesameModels:
    """Build kinder-bilevel-planning env models for our perceiver pipeline.

    Spins up a one-shot reference kinder env (`kinder.make(**make_kwargs)`)
    purely to source the `ObjectCentricBoxSpace` + action space the
    bilevel-planning factory expects — the env is closed before this
    function returns.

    `kinder_bilevel_planning.env_models.create_bilevel_planning_models`
    returns a `SesameModels` whose `observation_to_state` callback is
    `observation_space.devectorize`. That assumes the consumer hands the
    agent raw vectorized obs and expects the agent to devectorize itself.
    In this pipeline, the perceiver layer (between the env and the agent)
    has already produced an `ObjectCentricState` by the time the agent
    sees it, so we swap `observation_to_state` for an identity.

    Everything else returned by `create_bilevel_planning_models` (types,
    predicates, skills, transition_fn, state_abstractor, goal_deriver) is
    untouched.
    """
    kinder.register_all_environments()
    ref_env = kinder.make(**make_kwargs)
    try:
        base = create_bilevel_planning_models(
            env_name,
            ref_env.observation_space,
            ref_env.action_space,
            **(env_model_kwargs or {}),
        )
    finally:
        ref_env.close()
    return replace(base, observation_to_state=lambda x: x)


__all__ = [
    "BaseMotion3DPerceiver",
    "Kinematic3DActionGrounder",
    "KinematicRobotPerceiverBase",
    "PassThroughActionGrounder",
    "PassThroughPerceiver",
    "PrplLab3DPerceiver",
    "build_planner_env_models",
]
