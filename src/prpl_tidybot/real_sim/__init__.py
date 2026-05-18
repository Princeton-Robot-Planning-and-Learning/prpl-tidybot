"""Real-to-sim-to-real adapters for prpl_tidybot.

For fake / real backends, compose a `RealTidyBotEnv`, `PrplLab3DPerceiver`,
`PrplLab3DActionGrounder`, and a `prpl_utils.gym_agent.Agent` with
`prpl_utils.real_sim.Runner`. For sim, swap in `PrplLab3DSimEnv` plus
`PassThroughPerceiver` / `PassThroughActionGrounder` — the env already
produces an ObjectCentricState, so re-encoding through TidyBotObservation
is unnecessary.
"""

from prpl_tidybot.real_sim.action_grounders.kinematic3d import (
    PrplLab3DActionGrounder,
)
from prpl_tidybot.real_sim.action_grounders.passthrough import (
    PassThroughActionGrounder,
)
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.perceivers.passthrough import PassThroughPerceiver

__all__ = [
    "PassThroughActionGrounder",
    "PassThroughPerceiver",
    "PrplLab3DActionGrounder",
    "PrplLab3DPerceiver",
]
