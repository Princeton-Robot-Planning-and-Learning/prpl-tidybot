"""A real2sim benchmark of the pick out of the bin.

Each case is one real in-bin cube rest (``tests/data/in_bin_rests.json``): the bin as
the robot fitted it, the cube as it was localized, and the robot pose and arm joints the
pick was planned from, with what the real robot did. :func:`run_case` puts the kinematic
TidyBot (:mod:`prpl_tidybot.sim_robot`) at that pose, the bin's walls (the measured bin,
``cfg.bin``) at the fitted pose and the cube where it was seen, and runs the skill
server's own ``observe`` and ``pick`` in ``mode=sim``: plan, drive to the standoff,
waypoint above the rim, hover, descend, close, grasp check, lift above the rim. The
planner sees the fit the real robot saw.

The controller under test is the skill server's config (``conf/skill_server.yaml``) plus
a :class:`Controller`'s Hydra overrides, e.g.
``pick.top_down.in_bin.wall_margin=0.005``, so a candidate is config, not code.

What the sim does not model, so what the benchmark cannot see: the wrist camera's
lateral servo (sim mode builds none, so no servo step is checked against the walls), the
cube's contact with the fingers and the walls (the cube rides in a gripper that closes
within 2 cm of it; the walls stop nothing, so wall contact is measured, not felt), and a
second look at the standoff that differs from the first.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from prpl_tidybot import robot_skills
from prpl_tidybot.bin_localizer import ConstantBinSource
from prpl_tidybot.sim_video import SimMarkers, bin_wall_contacts
from prpl_tidybot.third_party import arm_server
from prpl_tidybot.top_down_pick import forward_kinematics

_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = _ROOT / "conf"
DEFAULT_FIXTURE = _ROOT / "tests" / "data" / "in_bin_rests.json"
_ARM_MOUNT_Z = 0.3948
_CUBE_HALF = 0.025

# The real verdict each sim verdict reproduces.
_REAL_FOR_SIM = {"refused": "refused", "held": "picked", "missed": "executed_missed"}


@dataclass(frozen=True)
class Controller:
    """A pick controller to benchmark: the skill server's config plus ``overrides``
    (Hydra override strings)."""

    name: str
    overrides: tuple[str, ...] = ()


CURRENT = Controller(name="current")

_IN_BIN = "pick.top_down.in_bin"
_MESH = f"{_IN_BIN}.gripper_model=mesh"
# A: the tilted, partly opened grasp (kinder-baselines #178's semantics) over the
# tilts the coverage analysis recommends; A178: #178's own tilts.
_A = (
    f"{_IN_BIN}.tilts_deg=[0,10,-10,15,-15,20,-20]",
    f"{_IN_BIN}.openings_mm=[85,70]",
    _MESH,
)
_A178 = (
    f"{_IN_BIN}.tilts_deg=[0,15,-15,30,-30]",
    f"{_IN_BIN}.openings_mm=[85,70]",
    _MESH,
)


def _b(cm: float) -> tuple[str, ...]:
    """B: the finger links may reach ``cm`` past the inner wall face."""
    return (_MESH, f"{_IN_BIN}.finger_allowance={cm / 100}")


# C: the gripper's yaw freed from the cube's faces, with A's tilts, openings and mesh:
# C1-<phi>, a yaw-relaxed face grasp within +-phi of face-aligned (5 deg steps); C2,
# an edge grasp across the diagonal (45 deg off).
def _yaws(phi: int) -> str:
    steps = [0] + [s * k for k in range(5, phi + 1, 5) for s in (1, -1)]
    return f"{_IN_BIN}.yaw_offsets_deg=[{','.join(str(v) for v in steps)}]"


_C = {
    "C1-15": (*_A, _yaws(15)),
    "C1-25": (*_A, _yaws(25)),
    "C2": (*_A, f"{_IN_BIN}.yaw_offsets_deg=[0,45]"),
}


# The candidate controllers, by name (scripts/in_bin_pick_benchmark.py --controller).
VARIANTS: dict[str, Controller] = {
    c.name: c
    for c in [
        CURRENT,
        Controller(name="mesh", overrides=(_MESH,)),
        Controller(name="A", overrides=_A),
        Controller(name="A178", overrides=_A178),
        *(Controller(name=f"B{cm}", overrides=_b(cm)) for cm in (0.5, 1.0, 2.0)),
        *(
            Controller(name=f"A+B{cm}", overrides=_A + _b(cm)[1:])
            for cm in (0.5, 1.0, 2.0)
        ),
        *(Controller(name=name, overrides=ov) for name, ov in _C.items()),
        *(
            Controller(name=f"{name}+B0.5", overrides=ov + _b(0.5)[1:])
            for name, ov in _C.items()
        ),
    ]
}


def load_cases(path: Path | str = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    """The fixture's cases."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = data["cases"]
    return cases


def compose_config(case: dict[str, Any], controller: Controller) -> DictConfig:
    """The skill server's config for ``mode=sim`` with the case's scene."""
    b, c, robot = case["bin"], case["cube"], case["robot"]
    overrides = [
        "mode=sim",
        f"sim.start_base_pose=[{robot[0]},{robot[1]},{robot[2]}]",
        f"sim.cube=[{c['x']},{c['y']},{c['yaw_deg']}]",
        f"sim.bin=[{b['x']},{b['y']},{math.degrees(b['yaw'])}]",
        "sim.video.enabled=false",
        "recording.motion_hz=0",
        *controller.overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="skill_server", overrides=overrides)


@contextlib.contextmanager
def _sim_clock() -> Iterator[None]:
    """The sim keeps its own clock: no real-time waits between commands."""
    period = robot_skills.POLICY_CONTROL_PERIOD
    clock = arm_server.time
    robot_skills.POLICY_CONTROL_PERIOD = 0.0  # type: ignore[misc]
    arm_server.time = SimpleNamespace(  # type: ignore[assignment]
        sleep=lambda _s: None, time=clock.time
    )
    try:
        yield
    finally:
        robot_skills.POLICY_CONTROL_PERIOD = period  # type: ignore[misc]
        arm_server.time = clock


def run_case(
    case: dict[str, Any],
    controller: Controller = CURRENT,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """The controller's pick of the case's cube in sim: ``verdict`` (``refused``,
    ``held`` or ``missed``), ``reason`` (the planner's refusal, or why a grasp missed),
    the arm's ``steps``, whether the held cube cleared the rim, and any
    ``wall_contacts`` (link and depth) of the TidyBot model with the walls."""
    # pylint: disable=import-outside-toplevel
    from prpl_tidybot.skill_server import build_skills
    from prpl_tidybot.toss_sim import TossSim

    cfg = compose_config(case, controller)
    with _sim_clock():
        sim = TossSim(cfg, log_dir, input, False)
        sim.world.q = np.array(case["arm_seed"], dtype=float)
        skills = build_skills(cfg, log_dir, sim=sim)
        b = case["bin"]
        # The planner sees the fit the robot saw (its size too); the walls are the
        # measured bin's.
        skills.bin_source = ConstantBinSource(
            b["x"], b["y"], b["yaw"], b["length"], b["width"], b["height"], b["wall"]
        )
        try:
            skills.observe()
            reply = skills.pick(max_attempts=1)
        finally:
            skills.close()
    world = sim.world
    attempts = reply.get("attempts") or []
    moves = attempts[-1].get("moves", []) if attempts else []
    if reply.get("refused"):
        verdict, reason = "refused", reply.get("reason")
    elif reply.get("held"):
        verdict, reason = "held", None
    else:
        verdict, reason = "missed", reply.get("note")
    walls = bin_wall_contacts(world, SimMarkers.from_config(cfg))
    real = case["real"]["verdict"]
    return {
        "id": case["id"],
        "controller": controller.name,
        "verdict": verdict,
        "reason": reason,
        "real_verdict": real,
        "agrees_with_real": None if real is None else _REAL_FOR_SIM[verdict] == real,
        "stand": reply.get("stand"),
        "closing_turn_deg": attempts[-1].get("closing_turn_deg") if attempts else None,
        "tilt_deg": attempts[-1].get("tilt_deg") if attempts else None,
        "open_position": attempts[-1].get("open_position") if attempts else None,
        "yaw_offset_deg": (
            attempts[-1].get("wrist_yaw_offset_deg") if attempts else None
        ),
        "gripper": reply.get("gripper"),
        "steps": [{"step": m["name"], "reached": m.get("reached")} for m in moves],
        "lifted_above_rim": verdict == "held"
        and _lifted_above(world, float(cfg.bin.height)),
        "wall_contacts": walls["contacts"],
        "min_wall_clearance_m": walls["min_clearance_m"],
    }


def _lifted_above(world: Any, rim: float) -> bool:
    """Whether the held cube's underside rose above the rim height."""
    geom = world._geom  # pylint: disable=protected-access
    for sample in world.samples:
        if sample[4] is not None:  # not held
            continue
        z = _ARM_MOUNT_Z + float(forward_kinematics(list(sample[2]), geom)[0][2])
        if z - _CUBE_HALF > rim:
            return True
    return False


def run_benchmark(
    cases: Sequence[dict[str, Any]], controller: Controller = CURRENT
) -> list[dict[str, Any]]:
    """:func:`run_case` on every case."""
    return [run_case(case, controller) for case in cases]


def summarize(
    results: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Counts as [x, y]: sim verdicts matching the real ones, real refusals refused in
    sim, real picks held in sim; and the cases that disagree."""
    by_id = {c["id"]: c for c in cases}
    judged = [r for r in results if r["real_verdict"] is not None]

    def count(real: str, sim: str) -> list[int]:
        rows = [r for r in judged if r["real_verdict"] == real]
        return [sum(r["verdict"] == sim for r in rows), len(rows)]

    return {
        "agree": [sum(bool(r["agrees_with_real"]) for r in judged), len(judged)],
        "real_refused_refused_in_sim": count("refused", "refused"),
        "real_picked_held_in_sim": count("picked", "held"),
        "real_refused_held_in_sim": count("refused", "held"),
        "real_missed_missed_in_sim": count("executed_missed", "missed"),
        "with_wall_contact": [
            sum(bool(r["wall_contacts"]) for r in results),
            len(results),
        ],
        "disagreements": [
            {
                "id": r["id"],
                "real": r["real_verdict"],
                "real_reason": by_id[r["id"]]["real"].get("reason"),
                "sim": r["verdict"],
                "sim_reason": r["reason"],
            }
            for r in judged
            if not r["agrees_with_real"]
        ],
    }


def format_table(
    results: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]
) -> str:
    """A Markdown table, one row per case."""
    by_id = {c["id"]: c for c in cases}
    lines = [
        "| case | kind | wall (cm) | cube yaw | real | sim | agree | wall contact "
        "| sim reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = by_id[r["id"]]
        contact = ", ".join(
            f"{w['link']} {100 * w['depth_m']:.1f} cm ({w['step']})"
            for w in r["wall_contacts"]
        )
        agree = "-" if r["agrees_with_real"] is None else str(r["agrees_with_real"])
        lines.append(
            f"| {r['id']} | {c['kind']} | {100 * c['wall_distance_m']:.1f} | "
            f"{c['cube']['yaw_deg']:.0f} | {r['real_verdict'] or '-'} | "
            f"{r['verdict']} | {agree} | {contact or 'none'} | {r['reason'] or ''} |"
        )
    return "\n".join(lines)
