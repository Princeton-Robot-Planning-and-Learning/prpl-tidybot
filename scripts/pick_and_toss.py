"""Pick the floor cube with the shelf3d PickController, then toss it.

Thin Hydra wrapper around `prpl_tidybot.pick_and_toss.run_pick_and_toss`; the
config is conf/pick_and_toss.yaml (which pulls in conf/config.yaml and the
kinematic-shelf3d-o1 env). No task planner: the pick is the PickController
alone and the toss is `ArmInterface.toss`.

Examples:
    python scripts/pick_and_toss.py mode=fake
    python scripts/pick_and_toss.py mode=sim                  # pick only
    python scripts/pick_and_toss.py mode=real                 # preview, confirm
    python scripts/pick_and_toss.py mode=real toss.max_vel=80 toss.heading=10
    python scripts/pick_and_toss.py mode=real skip_pick=true  # hand-in, then toss
    python scripts/pick_and_toss.py mode=real skip_toss=true  # pick and hold
    python scripts/pick_and_toss.py mode=real cycles=5        # pick, toss, repeat
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from prpl_tidybot.pick_and_toss import run_pick_and_toss


@hydra.main(
    config_name="pick_and_toss",
    config_path=str(Path(__file__).parent.parent / "conf"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run the pick and / or toss and print what happened."""
    result = run_pick_and_toss(cfg)
    line = (
        f"mode={result.mode}: {result.finish_reason}; picked={result.picked}, "
        f"tossed={result.tossed}"
    )
    if result.toss_outcome is not None:
        landing = result.toss_outcome["landing"]
        if landing is None:
            line += "; no cube marker visible after the toss"
        else:
            line += f"; landed at ({landing['x']:.3f}, {landing['y']:.3f})"
            if landing["distance_m"] is not None:
                line += (
                    f", {landing['distance_m']:.3f} m at "
                    f"{landing['bearing_deg']:+.1f} deg from the robot"
                )
    print(line)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
