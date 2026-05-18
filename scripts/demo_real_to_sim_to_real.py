"""Demo the real-to-sim-to-real pipeline against a FakeInterface.

Builds a RealTidyBotEnv backed by FakeInterface, wires it to
PrplLab3DPerceiver and PrplLab3DActionGrounder, drives a small random
agent through prpl_utils.real_sim.Runner, and prints the resulting fake-
interface state. This is the runnable equivalent of
tests/real_sim/test_runner_integration.py.

Usage:
    python scripts/demo_real_to_sim_to_real.py --steps 10 --seed 0
"""

import argparse

import numpy as np
from numpy.typing import NDArray
from prpl_utils.gym_agent import Agent
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim import PrplLab3DActionGrounder, PrplLab3DPerceiver


class _RandomActionAgent(Agent[ObjectCentricState, NDArray[np.floating]]):
    """Sample small random 11-d deltas; never command the gripper."""

    def _get_action(self) -> NDArray[np.floating]:
        action = self._rng.uniform(-0.05, 0.05, size=11)
        action[10] = 0.0
        return action


def main() -> None:
    """Parse CLI args, run the pipeline, and print the final fake state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=10, help="rollout length (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="agent random seed (default: 0)"
    )
    args = parser.parse_args()

    interface = FakeInterface()
    env = RealTidyBotEnv(interface)
    runner: Runner = Runner(
        real_env=env,
        perceiver=PrplLab3DPerceiver(),
        agent=_RandomActionAgent(seed=args.seed),
        action_grounder=PrplLab3DActionGrounder(),
    )

    runner.reset()
    runner.run(max_steps=args.steps)

    obs = interface.get_observation()
    print(f"Ran {args.steps} steps with seed {args.seed}.")
    print(
        f"Final base pose: x={obs.base_pose.x:.4f}, "
        f"y={obs.base_pose.y:.4f}, theta={obs.base_pose.theta():.4f}"
    )
    print("Final arm conf:  " + ", ".join(f"{v:.4f}" for v in obs.arm_conf))
    print(f"Final gripper:   {obs.gripper:.4f}")


if __name__ == "__main__":
    main()
