"""Real-to-sim-to-real adapters for prpl_tidybot.

For fake / real backends, compose a `RealTidyBotEnv`, a kinematic3d
perceiver (`PrplLab3DPerceiver` or `BaseMotion3DPerceiver`),
`Kinematic3DActionGrounder`, and a `prpl_utils.gym_agent.Agent` with
`prpl_utils.real_sim.Runner`. For sim, swap in `KinderSimEnv` plus
`PassThroughPerceiver` / `PassThroughActionGrounder` — the env already
produces an ObjectCentricState, so re-encoding through TidyBotObservation
is unnecessary.
"""

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

__all__ = [
    "BaseMotion3DPerceiver",
    "Kinematic3DActionGrounder",
    "KinematicRobotPerceiverBase",
    "PassThroughActionGrounder",
    "PassThroughPerceiver",
    "PrplLab3DPerceiver",
]
