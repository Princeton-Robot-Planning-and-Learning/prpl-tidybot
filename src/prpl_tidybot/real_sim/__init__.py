"""Real-to-sim-to-real adapters for prpl_tidybot.

For fake / real backends, compose a `RealTidyBotEnv`, a kinematic3d
perceiver (`PrplLab3DPerceiver` or `BaseMotion3DPerceiver`),
`Kinematic3DPlanExecutor` (which dispatches per-segment to a
`BaseMotion3DPlanExecutor` subclass for base motion and to an
`ArmMotion3DPlanExecutor` subclass — currently
`StreamingArmMotion3DPlanExecutor` — for arm/gripper motion), and a
`prpl_utils.planning_agent.PlanningAgent` with
`prpl_utils.real_sim.Runner`. For sim, swap in `KinderSimEnv` plus
`PassThroughPerceiver` / `PassThroughPlanExecutor` — the env already
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

from prpl_tidybot.real_sim.perceivers.kinematic3d import (
    BaseMotion3DPerceiver,
    KinematicRobotPerceiverBase,
    PrplLab3DPerceiver,
)
from prpl_tidybot.real_sim.perceivers.passthrough import PassThroughPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    ArmMotion3DPlanExecutor,
    StreamingArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    BaseMotion3DPlanExecutor,
    PurePursuitBaseMotion3DPlanExecutor,
    SettleBaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.kinematic3d import (
    Kinematic3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.passthrough import (
    PassThroughPlanExecutor,
)


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
    "ArmMotion3DPlanExecutor",
    "BaseMotion3DPerceiver",
    "BaseMotion3DPlanExecutor",
    "Kinematic3DPlanExecutor",
    "KinematicRobotPerceiverBase",
    "PassThroughPerceiver",
    "PassThroughPlanExecutor",
    "PrplLab3DPerceiver",
    "PurePursuitBaseMotion3DPlanExecutor",
    "SettleBaseMotion3DPlanExecutor",
    "StreamingArmMotion3DPlanExecutor",
    "build_planner_env_models",
]
