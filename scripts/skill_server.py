"""Serve Tossing3D's skills (observe, can_pick, pick, toss, open_gripper, retract,
go_home, stop) over the LAN, one RPC call each.

Config: conf/skill_server.yaml. Run it ON the NUC, after the base and arm servers:

    python scripts/skill_server.py mode=real
    python scripts/skill_server.py mode=fake     # no robot

Connect from another machine with
``prpl_tidybot.skill_server.TossingRobotClient(<nuc ip>, 50002, RPC_AUTHKEY)``. Ctrl-C
here halts the arm and base before the server exits.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

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
    skills = build_skills(cfg)
    serve(skills, str(cfg.skill_server.host), int(cfg.skill_server.port), RPC_AUTHKEY)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
