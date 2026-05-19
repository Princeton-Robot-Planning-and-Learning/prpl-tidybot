"""Smoke tests for `prpl_tidybot.pipeline.run_planner` across the env x mode matrix.

Each test composes a Hydra config (with a small `max_eval_steps` so the rollout finishes
in a few seconds) and invokes `run_planner` directly — no subprocess, no `@hydra.main`
decorator. We assert only that the rollout completes and the final state has the
expected robot object; the actual reward / number of steps depends on the env's task.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.pipeline import run_planner

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def _compose(env: str, mode: str, max_eval_steps: int = 30, seed: int = 0):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(
            config_name="config",
            overrides=[
                f"env={env}",
                f"mode={mode}",
                f"max_eval_steps={max_eval_steps}",
                f"seed={seed}",
            ],
        )


@pytest.mark.parametrize(
    "env,mode",
    [
        # base_motion3d's planner is reliable with default settings; the
        # PrplLab3D pick-and-place is more expensive and flakes on
        # slower CI runners ("No plan found" before the planning timeout
        # at default samples_per_step). Re-add `("prpl3d-o1", "sim")`
        # once we have a more reliable budget for it.
        ("base_motion3d", "sim"),
        ("base_motion3d", "fake"),
    ],
)
def test_run_planner_smoke(env: str, mode: str) -> None:
    """The pipeline composes from the env yaml and runs to completion without raising
    for the given (env, mode) pair."""
    cfg = _compose(env, mode, max_eval_steps=30)
    result = run_planner(cfg)
    assert result.env_name == cfg.env.env_name
    assert result.mode == mode
    # Some end condition fired (terminated, truncated, max_steps_reached,
    # or agent_failure for fake mode whose env never terminates).
    assert result.finish_reason
    # The final state always exposes a "robot" object.
    assert result.final_state.get_object_from_name("robot") is not None
    # Recording was not enabled, so no video should be produced.
    assert result.video_path is None


def test_run_planner_writes_video_when_record_video_path_set(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: with `record.video_path` set, the pipeline writes a side-by-
    side mp4 alongside the rollout."""
    video_path = tmp_path / "rollout.mp4"
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "env=base_motion3d",
                "mode=fake",
                "max_eval_steps=10",
                "seed=0",
                f"record.video_path={video_path}",
                "record.fps=5",
            ],
        )
    result = run_planner(cfg)
    assert result.video_path == video_path
    assert video_path.exists()
    assert video_path.stat().st_size > 0
