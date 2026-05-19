"""Run the bilevel planner agent through the prpl_tidybot pipeline.

Thin Hydra wrapper around `prpl_tidybot.pipeline.run_planner`. Each env
yaml under `conf/env/` declares all three pipelines
(`fake` / `sim` / `real`); pick one with `mode=...`. Adding a new env
means dropping a new yaml file — there's no env switch in code.

Examples:
    python scripts/run_planner.py env=base_motion3d mode=sim
    python scripts/run_planner.py env=base_motion3d mode=fake max_eval_steps=20
    python scripts/run_planner.py env=prpl3d-o1 mode=sim seed=42
    python scripts/run_planner.py env=base_motion3d mode=real    # raises
                                                                 # NotImplementedError
                                                                 # from the first
                                                                 # RealInterface
                                                                 # hardware read
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from prpl_tidybot.pipeline import run_planner


@hydra.main(
    config_name="config",
    config_path=str(Path(__file__).parent.parent / "conf"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Build the pipeline from `cfg`, run a rollout, print the final state."""
    result = run_planner(cfg)
    robot = result.final_state.get_object_from_name("robot")
    bx = result.final_state.get(robot, "pos_base_x")
    by = result.final_state.get(robot, "pos_base_y")
    bt = result.final_state.get(robot, "pos_base_rot")
    print(
        f"env={result.env_name}, mode={result.mode}, seed={result.seed}: "
        f"finish={result.finish_reason}, steps={result.steps}, "
        f"total_reward={result.total_reward:.3f}, "
        f"final base=({bx:.3f}, {by:.3f}, {bt:.3f})"
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
