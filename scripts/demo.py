"""Demo the prpl_tidybot pipeline against an arbitrary kinder env.

Wires `prpl_utils.real_sim.Runner` to:
  - the env yaml's pipeline (env wrapper + perceiver + action grounder),
  - a `BilevelPlanningAgent` constructed via
    `prpl_tidybot.real_sim.build_planner_env_models`.

Each env yaml under `conf/env/` declares all three pipelines
(`fake` / `sim` / `real`); pick one with `mode=...`. There's no env
switch in the script — adding a new env means dropping a new yaml file.

Examples:
    python scripts/demo.py env=base_motion3d mode=sim
    python scripts/demo.py env=base_motion3d mode=fake max_eval_steps=20
    python scripts/demo.py env=prpl3d-o1 mode=sim seed=42
    python scripts/demo.py env=base_motion3d mode=real    # raises
                                                          # NotImplementedError
                                                          # from the first
                                                          # RealInterface
                                                          # hardware read
"""

from pathlib import Path

import hydra
import kinder
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner

from prpl_tidybot.real_sim import build_planner_env_models


@hydra.main(
    config_name="config",
    config_path=str(Path(__file__).parent.parent / "conf"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Build the pipeline from `cfg`, run a rollout, print the final state."""
    # Kinder env registrations are imported lazily; bilevel-planning calls
    # this internally too but a duplicate call is harmless.
    kinder.register_all_environments()

    pipeline = cfg.env.pipelines[cfg.mode]
    real_env = hydra.utils.instantiate(pipeline.real_env)
    perceiver = hydra.utils.instantiate(pipeline.perceiver)
    grounder = hydra.utils.instantiate(pipeline.action_grounder)

    env_models = build_planner_env_models(
        cfg.env.env_name,
        cfg.env.make_kwargs,
        cfg.env.env_model_kwargs,
    )

    agent: BilevelPlanningAgent = BilevelPlanningAgent(
        env_models,
        cfg.seed,
        max_abstract_plans=cfg.agent.max_abstract_plans,
        samples_per_step=cfg.agent.samples_per_step,
        max_skill_horizon=cfg.agent.max_skill_horizon,
        heuristic_name=cfg.agent.heuristic_name,
        planning_timeout=cfg.agent.planning_timeout,
    )

    runner: Runner = Runner(
        real_env=real_env,
        perceiver=perceiver,
        agent=agent,
        action_grounder=grounder,
    )

    state = runner.reset(seed=cfg.seed)
    total_reward = 0.0
    finish_reason = "max_steps_reached"
    for _ in range(cfg.max_eval_steps):
        try:
            state, reward, terminated, truncated, _ = runner.step()
        except AgentFailure as e:
            # The bilevel planner produces a finite action sequence; once
            # it's exhausted the agent raises. For fake mode that's the
            # natural rollout end (the fake has no goal-detection to
            # terminate the env).
            finish_reason = f"agent_failure: {e}"
            break
        total_reward += float(reward)
        if terminated:
            finish_reason = "terminated"
            break
        if truncated:
            finish_reason = "truncated"
            break

    robot = state.get_object_from_name("robot")
    bx = state.get(robot, "pos_base_x")
    by = state.get(robot, "pos_base_y")
    bt = state.get(robot, "pos_base_rot")
    print(
        f"env={cfg.env.env_name}, mode={cfg.mode}, seed={cfg.seed}: "
        f"finish={finish_reason}, total_reward={total_reward:.3f}, "
        f"final base=({bx:.3f}, {by:.3f}, {bt:.3f})"
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
