"""Demo the prpl_tidybot pipeline in fake, sim, or real mode.

In `fake` mode the pipeline is the production one: a `FakeInterface`
(stand-in for real hardware) backs a `RealTidyBotEnv`, observations are
`TidyBotObservation`, and the `PrplLab3DPerceiver` / `PrplLab3DActionGrounder`
do the conversions to and from the kinder state/action space.

In `sim` mode the kinder env is driven directly via `PrplLab3DSimEnv`;
observations are already `ObjectCentricState`, so the perceiver and
action grounder are pass-throughs.

In `real` mode a `RealInterface` backs the `RealTidyBotEnv`. The real
hardware drivers are currently skeleton stubs that raise
`NotImplementedError`; running `--mode real` surfaces the next
unimplemented method so you can fill them in piece-by-piece.

Either way the agent loop is identical (kinder ObjectCentricState in,
11-d kinder action out) and the Runner emits a kinder ObjectCentricState
each step, so we can read the same robot features regardless of mode.

Usage:
    python scripts/demo_real_to_sim_to_real.py --mode fake --steps 10
    python scripts/demo_real_to_sim_to_real.py --mode sim --steps 10
    python scripts/demo_real_to_sim_to_real.py --mode real --steps 10
"""

import argparse

import gymnasium
import numpy as np
from numpy.typing import NDArray
from prpl_utils.gym_agent import Agent
from prpl_utils.real_sim import ActionGrounder, Perceiver, Runner
from relational_structs import ObjectCentricState

from prpl_tidybot.interfaces.interface import FakeInterface, RealInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim import (
    PassThroughActionGrounder,
    PassThroughPerceiver,
    PrplLab3DActionGrounder,
    PrplLab3DPerceiver,
)
from prpl_tidybot.sim_env import PrplLab3DSimEnv


class _RandomActionAgent(Agent[ObjectCentricState, NDArray[np.floating]]):
    """Sample small random 11-d deltas; never command the gripper."""

    def _get_action(self) -> NDArray[np.floating]:
        action = self._rng.uniform(-0.05, 0.05, size=11)
        action[10] = 0.0
        return action


def _build_pipeline(
    mode: str,
) -> tuple[gymnasium.Env, Perceiver, ActionGrounder]:
    if mode == "fake":
        env: gymnasium.Env = RealTidyBotEnv(FakeInterface())
        perceiver: Perceiver = PrplLab3DPerceiver()
        grounder: ActionGrounder = PrplLab3DActionGrounder()
        return env, perceiver, grounder
    if mode == "sim":
        return (
            PrplLab3DSimEnv(),
            PassThroughPerceiver[ObjectCentricState](),
            PassThroughActionGrounder[NDArray[np.floating]](),
        )
    if mode == "real":
        env = RealTidyBotEnv(RealInterface())
        perceiver = PrplLab3DPerceiver()
        grounder = PrplLab3DActionGrounder()
        return env, perceiver, grounder
    raise ValueError(f"unknown mode: {mode!r}")


def main() -> None:
    """Parse CLI args, run the pipeline, and print the final state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fake", "sim", "real"),
        default="fake",
        help="pipeline backend (default: fake)",
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="rollout length (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="agent + env random seed (default: 0)",
    )
    args = parser.parse_args()

    env, perceiver, grounder = _build_pipeline(args.mode)
    runner: Runner = Runner(
        real_env=env,
        perceiver=perceiver,
        agent=_RandomActionAgent(seed=args.seed),
        action_grounder=grounder,
    )

    state = runner.reset(seed=args.seed)
    for _ in range(args.steps):
        state, _, _, _, _ = runner.step()

    robot = state.get_object_from_name("robot")
    bx = state.get(robot, "pos_base_x")
    by = state.get(robot, "pos_base_y")
    bt = state.get(robot, "pos_base_rot")
    arm = [state.get(robot, f"joint_{i + 1}") for i in range(7)]
    gripper = state.get(robot, "finger_state")

    print(f"Mode: {args.mode}, steps: {args.steps}, seed: {args.seed}.")
    print(f"Final base pose: x={bx:.4f}, y={by:.4f}, theta={bt:.4f}")
    print("Final arm conf:  " + ", ".join(f"{v:.4f}" for v in arm))
    print(f"Final gripper:   {gripper:.4f}")


if __name__ == "__main__":
    main()
