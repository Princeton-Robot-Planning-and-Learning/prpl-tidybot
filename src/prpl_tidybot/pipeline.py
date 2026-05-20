"""Glue that wires a Hydra config into a Runner rollout.

The `scripts/run_planner.py` entry point delegates to `run_planner` here;
tests in `tests/` compose configs with `hydra.compose` and call
`run_planner` directly, without going through the Hydra `@main` decorator.
"""

from dataclasses import dataclass
from pathlib import Path

import hydra
import kinder
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.recording import Recorder, RecordingPerceiver


@dataclass(frozen=True)
class RolloutSummary:
    """Result of one rollout — handy for assertions in tests."""

    env_name: str
    mode: str
    seed: int
    steps: int
    finish_reason: str
    total_reward: float
    final_state: ObjectCentricState
    video_path: Path | None = None


def run_planner(cfg: DictConfig) -> RolloutSummary:
    """Build the pipeline from `cfg`, run a rollout, return a summary.

    Mode and env are picked from `cfg.mode` and `cfg.env.pipelines[mode]`
    respectively; there's no env-specific branching here.
    """
    # Kinder env registrations are imported lazily; bilevel-planning calls
    # this internally too but a duplicate call is harmless.
    kinder.register_all_environments()

    pipeline = cfg.env.pipelines[cfg.mode]
    real_env = hydra.utils.instantiate(pipeline.real_env)
    perceiver = hydra.utils.instantiate(pipeline.perceiver)
    plan_executor = hydra.utils.instantiate(pipeline.plan_executor)

    # Optional side-by-side recording. The shadow sim reuses the env's
    # own sim pipeline yaml (the same one sim mode uses for `real_env`),
    # which keeps kinder-specific construction out of this file.
    recorder: Recorder | None = None
    record_cfg = cfg.get("record")
    if record_cfg is not None and record_cfg.get("video_path"):
        shadow_sim = hydra.utils.instantiate(cfg.env.pipelines.sim.real_env)
        recorder = Recorder(
            real_env=real_env,
            shadow_sim=shadow_sim,
            video_path=record_cfg.video_path,
            fps=record_cfg.fps,
            seed=cfg.seed,
        )
        perceiver = RecordingPerceiver(perceiver, recorder)

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
        plan_executor=plan_executor,
    )

    state = runner.reset(seed=cfg.seed)
    total_reward = 0.0
    steps = 0
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
        steps += 1
        total_reward += float(reward)
        if terminated:
            finish_reason = "terminated"
            break
        if truncated:
            finish_reason = "truncated"
            break

    video_path = recorder.finish() if recorder is not None else None

    return RolloutSummary(
        env_name=cfg.env.env_name,
        mode=cfg.mode,
        seed=cfg.seed,
        steps=steps,
        finish_reason=finish_reason,
        total_reward=total_reward,
        final_state=state,
        video_path=video_path,
    )
