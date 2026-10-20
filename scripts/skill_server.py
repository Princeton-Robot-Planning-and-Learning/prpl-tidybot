"""Serve Tossing3D's skills (observe, can_pick, pick, toss, open_gripper, retract,
go_home, stop) over the LAN, one RPC call each.

Config: conf/skill_server.yaml. Run it ON the NUC, after the base and arm servers:

    python scripts/skill_server.py mode=real
    python scripts/skill_server.py mode=fake     # no robot
    python scripts/skill_server.py mode=sim      # the kinematic TidyBot (cfg.sim)

Connect from another machine with
``prpl_tidybot.skill_server.TossingRobotClient(<nuc ip>, 50002, RPC_AUTHKEY)``. Ctrl-C
here halts the arm and base before the server exits.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from prpl_tidybot.pipeline import _resolve_log_dir
from prpl_tidybot.skill_server import build_skills, serve
from prpl_tidybot.third_party.constants import RPC_AUTHKEY


@hydra.main(
    config_name="skill_server",
    config_path=str(Path(__file__).parent.parent / "conf"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Build the skills for cfg.mode and serve them until Ctrl-C."""
    logging.basicConfig(level=logging.INFO)
    log_dir = _resolve_log_dir(None)
    sim = None
    if cfg.mode == "sim":
        from prpl_tidybot.toss_sim import (  # pylint: disable=import-outside-toplevel
            TossSim,
        )

        sim = TossSim(cfg, log_dir, input, False)
    skills = build_skills(cfg, log_dir, sim=sim)
    try:
        serve(
            skills, str(cfg.skill_server.host), int(cfg.skill_server.port), RPC_AUTHKEY
        )
    finally:
        # The recordings, and the robot's clients (the arm's torque loop).
        skills.close()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
