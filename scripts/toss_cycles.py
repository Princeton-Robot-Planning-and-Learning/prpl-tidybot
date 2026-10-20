"""Pick the cube, drive to the toss pose, toss; repeat: the skill server's skills
(prpl_tidybot.skill_server.TossingSkills) called in a fixed order.

Config: conf/toss_cycles.yaml (or --config-name same_side_practice).

Every action asks for confirmation in mode=real, and whenever the run stops the arm
retracts and the base drives back to the toss pose (after one more confirmation).

Examples:
    python scripts/toss_cycles.py mode=fake
    python scripts/toss_cycles.py mode=real               # 2 cycles at 140 deg/s, 600 ms
    python scripts/toss_cycles.py mode=real cycles=1
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from prpl_tidybot.toss_cycles import run_toss_cycles


@hydra.main(
    config_name="toss_cycles",
    config_path=str(Path(__file__).parent.parent / "conf"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run the cycles and print what happened."""
    result = run_toss_cycles(cfg)
    print(f"mode={result.mode}: {result.finish_reason}")
    for record in result.cycles:
        cube = record.get("cube_before_pick")
        landing = record.get("landing") or {}
        line = f"  cycle {record['cycle']}: tossed={record['tossed']}"
        if cube:
            line += f", cube was at ({cube['x']:+.3f}, {cube['y']:+.3f})"
        if "distance_m" in landing:
            line += (
                f", landed {landing['distance_m']:.2f} m from the toss pose "
                f"({landing['off_heading_deg']:+.0f} deg off heading)"
            )
        print(line)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
