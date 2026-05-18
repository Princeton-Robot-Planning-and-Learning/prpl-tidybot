"""Real-to-sim-to-real adapters for prpl_tidybot.

Compose a `RealTidyBotEnv`, `PrplLab3DPerceiver`, `PrplLab3DActionGrounder`,
and a `prpl_utils.gym_agent.Agent` with `prpl_utils.real_sim.Runner` to run
an agent against the real environment via the kinder/PrplLab3D-o{1,2}-v0
sim state space. No tidybot-specific Runner subclass is needed.
"""

from prpl_tidybot.real_sim.action_grounders.kinematic3d import (
    PrplLab3DActionGrounder,
)
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver

__all__ = ["PrplLab3DActionGrounder", "PrplLab3DPerceiver"]
