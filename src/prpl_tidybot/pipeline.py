"""Glue that wires a Hydra config into a Runner rollout.

The `scripts/run_planner.py` entry point delegates to `run_planner` here;
tests in `tests/` compose configs with `hydra.compose` and call
`run_planner` directly, without going through the Hydra `@main` decorator.
"""

from dataclasses import dataclass
from pathlib import Path

import hydra
import kinder
from hydra.core.hydra_config import HydraConfig
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

from prpl_tidybot.preview import planned_states_from_agent, preview_or_abort
from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.recording import RecordingPerceiver, TrajectoryRecorder


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
    trajectory_dir: Path | None = None
    video_path: Path | None = None


def run_planner(cfg: DictConfig, log_dir: Path | str | None = None) -> RolloutSummary:
    """Build the pipeline from `cfg`, run a rollout, return a summary.

    Mode and env are picked from `cfg.mode` and `cfg.env.pipelines[mode]`
    respectively; there's no env-specific branching here.

    `log_dir` is the trajectory output directory. When unset, falls back to
    the Hydra runtime output dir (set by ``@hydra.main``). When neither is
    available (e.g. test code composing the config without going through
    ``@hydra.main``), trajectory recording is skipped. ``cfg.record.video``
    additionally gates whether to compose ``video.mp4`` from the per-tick
    panels at the end of the rollout.
    """
    # Kinder env registrations are imported lazily; bilevel-planning calls
    # this internally too but a duplicate call is harmless.
    kinder.register_all_environments()

    pipeline = cfg.env.pipelines[cfg.mode]
    # `_convert_="all"` is needed wherever a nested `_target_: <EnvConfig>`
    # block may show up (sim's real_env wires the same custom config the
    # planner uses) — without it dataclass field defaults like
    # `Kinematic3DEnvConfig.robot_base_home_pose: SE2Pose` are wrapped as
    # OmegaConf structured configs and downstream `.to_se3(...)` accesses
    # fail. Applied uniformly to all three instantiations so future yamls
    # that nest configs don't trip over the same edge.
    real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
    perceiver = hydra.utils.instantiate(pipeline.perceiver, _convert_="all")
    plan_executor = hydra.utils.instantiate(pipeline.plan_executor, _convert_="all")

    # Trajectory recording is on whenever we have a place to write it. The
    # shadow sim reuses the env's own sim pipeline yaml (the same one sim
    # mode uses for `real_env`), which keeps kinder-specific construction
    # out of this file.
    resolved_log_dir = _resolve_log_dir(log_dir)
    recorder: TrajectoryRecorder | None = None
    record_cfg = cfg.get("record")
    if resolved_log_dir is not None:
        shadow_sim = hydra.utils.instantiate(
            cfg.env.pipelines.sim.real_env, _convert_="all"
        )
        recorder = TrajectoryRecorder(
            log_dir=resolved_log_dir,
            shadow_sim=shadow_sim,
            real_env=real_env,
            seed=cfg.seed,
            fps=record_cfg.fps if record_cfg is not None else 10,
            compose_video=(
                bool(record_cfg.get("video")) if record_cfg is not None else False
            ),
        )
        perceiver = RecordingPerceiver(perceiver, recorder)

    # `hydra.utils.instantiate(_recursive_=True, _convert_="all")` resolves
    # any `_target_:` blocks nested inside make_kwargs / env_model_kwargs
    # (e.g. a `Shelf3DEnvConfig` wrapping a `Pose` for a custom shelf_pose).
    # `_convert_="all"` returns plain Python objects rather than OmegaConf
    # wrappers — without it, dataclass field defaults like
    # `Kinematic3DEnvConfig.robot_base_home_pose: SE2Pose` get re-wrapped
    # as structured configs and downstream `.to_se3(...)` method calls fail
    # with "Key 'to_se3' not in 'SE2Pose'".
    env_models = build_planner_env_models(
        cfg.env.env_name,
        hydra.utils.instantiate(cfg.env.make_kwargs, _recursive_=True, _convert_="all"),
        hydra.utils.instantiate(
            cfg.env.env_model_kwargs, _recursive_=True, _convert_="all"
        ),
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

    # Wrap the rollout in try/finally so `real_env.close()` always runs —
    # otherwise Python process exit leaves the arm cyclic torque-control
    # loop running on the server (and the base RPC session open), and the
    # next rollout / hardware test inherits buffered state from this one
    # (see issue #54 for the base manifestation; the arm has an analogous
    # failure mode where Kortex's high-level controller appears to retain
    # the previous low-level trajectory state).
    try:
        state = runner.reset(seed=cfg.seed)
        total_reward = 0.0
        steps = 0
        finish_reason = "max_steps_reached"

        # Optional plan-preview gate. Render the agent's planned trajectory
        # through a shadow sim into preview.mp4 under the log dir and prompt
        # the operator before any real motion is commanded. Rejection raises
        # AgentFailure, which the main loop's existing handler treats as a
        # clean rollout end (the executor never gets a chance to step). Gated
        # on `cfg.mode == "real"` even when enabled — sim / fake / test runs
        # shouldn't block on stdin just because the global default is on.
        preview_cfg = cfg.get("preview")
        if (
            preview_cfg is not None
            and bool(preview_cfg.get("enabled"))
            and cfg.mode == "real"
            and resolved_log_dir is not None
        ):
            try:
                preview_or_abort(
                    planned_states=planned_states_from_agent(agent),
                    shadow_sim=hydra.utils.instantiate(
                        cfg.env.pipelines.sim.real_env, _convert_="all"
                    ),
                    log_dir=resolved_log_dir,
                    seed=cfg.seed,
                    fps=int(preview_cfg.get("fps", 10)),
                )
            except AgentFailure as e:
                finish_reason = f"agent_failure: {e}"
                return RolloutSummary(
                    env_name=cfg.env.env_name,
                    mode=cfg.mode,
                    seed=cfg.seed,
                    steps=0,
                    finish_reason=finish_reason,
                    total_reward=0.0,
                    final_state=state,
                    trajectory_dir=(
                        recorder.trajectory_dir if recorder is not None else None
                    ),
                    video_path=(recorder.finish() if recorder is not None else None),
                )

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

        trajectory_dir = recorder.trajectory_dir if recorder is not None else None
        video_path = recorder.finish() if recorder is not None else None

        return RolloutSummary(
            env_name=cfg.env.env_name,
            mode=cfg.mode,
            seed=cfg.seed,
            steps=steps,
            finish_reason=finish_reason,
            total_reward=total_reward,
            final_state=state,
            trajectory_dir=trajectory_dir,
            video_path=video_path,
        )
    finally:
        real_env.close()


def _resolve_log_dir(explicit: Path | str | None) -> Path | None:
    """Use the explicit path if given; else the Hydra runtime dir; else None."""
    if explicit is not None:
        return Path(explicit)
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except (ValueError, AttributeError):
        # HydraConfig.get() raises ValueError when no Hydra context is active
        # (i.e. tests composing configs via hydra.compose without going
        # through @hydra.main). Skip recording in that case.
        return None
