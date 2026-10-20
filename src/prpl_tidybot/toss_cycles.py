"""Repeat pick -> toss, for scripts/toss_cycles.py: a short scripted client of the
skill server's skills (:class:`prpl_tidybot.skill_server.TossingSkills`), the one
implementation of the robot's skills, called in-process in a fixed order.

Each cycle calls ``pick`` (which localizes the bin and the cube, inside the bin first,
and picks it off the floor or out of the bin) and then ``toss`` (aimed at the
localized bin, ``toss.aimed``) or ``toss_at`` (from the fixed ``toss_pose``). The
person is asked for help where the skills cannot go on: no cube found, a pick refused
(past the barrier, against a bin wall, in the bin with the pick out of it off) or
missed. Actions are confirmed on stdin in mode=real (``confirm``: once, or before
every skill call). Whenever the run stops the arm retracts and the base drives back
to ``toss_pose`` (after one more confirmation). ``cycles.json`` in the log dir holds
each cycle's cube before the pick, its pick attempts, its toss and where it landed.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

import prpl_tidybot.config_resolvers  # noqa: F401  # pylint: disable=unused-import
from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.pipeline import _resolve_log_dir
from prpl_tidybot.robot_skills import IN_BIN_PICK_OFF
from prpl_tidybot.skill_server import SkillStopped, TossingSkills, build_skills
from prpl_tidybot.toss import describe_toss
from prpl_tidybot.visual_servo.image_sources import ImageSource

_logger = logging.getLogger(__name__)

PromptFn = Callable[[str], str]

_STRIP = "clear of the camera overlap strip (y -0.15..0.18)"
# A cube still seen in the bin after this many requests to put it back ends the run.
_MAX_PUT_BACKS = 3


class _Declined(Exception):
    """The operator said no to a step."""


@dataclass(frozen=True)
class TossCyclesSummary:
    """What happened, for the script's printout and for tests."""

    mode: str
    completed: int
    requested: int
    finish_reason: str
    cycles: list[dict[str, Any]] = field(default_factory=list)
    # Map-frame (x, y, theta) at the end, after the reset.
    final_base_pose: tuple[float, float, float] | None = None


def _landing(cube: dict[str, float], toss_pose: list[float]) -> dict[str, float]:
    """Where a throw ended relative to the toss pose: distance and bearing off its
    heading."""
    dx, dy = cube["x"] - toss_pose[0], cube["y"] - toss_pose[1]
    bearing = math.atan2(dy, dx) - toss_pose[2]
    return {
        "x": cube["x"],
        "y": cube["y"],
        "distance_m": math.hypot(dx, dy),
        "off_heading_deg": math.degrees((bearing + math.pi) % (2 * math.pi) - math.pi),
    }


def _scored_landing(reply: dict[str, Any], toss_pose: list[float]) -> dict[str, Any]:
    """Where the toss reply says the cube ended, and with a localized bin whether it
    is inside it (``in_bin``) and inside its scoring region (``scored``, as Tossing3D
    scores)."""
    if reply.get("landed") is None:
        return {"error": str(reply.get("note"))}
    x, y = reply["landed"]
    landing: dict[str, Any] = dict(_landing({"x": x, "y": y}, toss_pose))
    if reply.get("bin") is not None:
        landing["in_bin"] = bool(reply["inside_bin"])
        landing["scored"] = bool(reply["in_bin"])
    return landing


class _Operator:
    """The person at the keyboard: confirmations (``confirm``: once or each) and the
    requests to move the cube, at most ``asks`` of those per cycle."""

    def __init__(
        self,
        prompt_fn: PromptFn,
        interactive: bool,
        once: bool,
        requested: int,
        skills: TossingSkills,
    ) -> None:
        self.prompt_fn = prompt_fn
        self.interactive = interactive
        self.once = once
        self.requested = requested
        self.skills = skills
        self.confirmed = False

    def asking(self) -> bool:
        """Whether to prompt now: every step, or (confirm=once) only before the first."""
        return self.interactive and not (self.once and self.confirmed)

    def confirm(self, question: str) -> None:
        """Ask (or, when not asking, log) the step; raise ``_Declined`` on a no."""
        self.skills.label(question)
        if not self.asking():
            _logger.info("Step: %s", question.rstrip("?"))
            return
        if self.once:
            question = (
                f"Run all {self.requested} cycle(s) (pick, drive, toss, retract; asking "
                f"only if the cube must be moved) without asking again? First: "
                f"{question}"
            )
        answer = self.prompt_fn(f"\n{question} [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            raise _Declined(question)
        self.confirmed = True

    def ask_to_move(self, record: dict[str, Any], asks: int, question: str) -> bool:
        """Ask the person to move the cube (once per remaining ask); False when there
        is nobody to ask, no asks left, or they typed 'stop'."""
        if not self.interactive or record.get("cube_moved_by_person", 0) >= asks:
            return False
        if self.prompt_fn(question).strip().lower() == "stop":
            return False
        record["cube_moved_by_person"] = record.get("cube_moved_by_person", 0) + 1
        return True


def run_toss_cycles(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    cfg: DictConfig,
    log_dir: Path | str | None = None,
    prompt_fn: PromptFn = input,
    wrist_camera: Callable[[ArmInterface], ImageSource] | None = None,
) -> TossCyclesSummary:
    """Run ``cfg.cycles`` pick/toss cycles (conf/toss_cycles.yaml) through the skills.

    ``wrist_camera`` builds the wrist image source from the arm; by default the
    Kinova wrist camera in real mode and none otherwise (no servo).

    mode=sim runs the same skills on the kinematic TidyBot of
    :mod:`prpl_tidybot.sim_robot` (``cfg.sim``) and films it into ``sim.mp4``; its
    prompts are answered by a scripted operator unless ``interactive=true``."""
    aimed = cfg.toss.get("aimed")
    aiming = aimed is not None and bool(aimed.get("enabled", False))
    if cfg.toss.get("grid") and aiming:
        raise ValueError(
            "toss.grid (calibration, thrown from the fixed toss pose) and toss.aimed "
            "cannot both be enabled; turn one off."
        )
    interactive = (
        cfg.mode == "real" if cfg.get("interactive") is None else bool(cfg.interactive)
    )
    resolved_log_dir = _resolve_log_dir(log_dir)
    requested = int(cfg.cycles)
    toss_pose = [float(v) for v in cfg.toss_pose]

    sim: Any = None
    if cfg.mode == "sim":
        # pylint: disable=import-outside-toplevel
        import kinder

        from prpl_tidybot.toss_sim import TossSim

        kinder.register_all_environments()
        sim = TossSim(cfg, resolved_log_dir, prompt_fn, interactive)
        prompt_fn = sim.operator
    skills = build_skills(cfg, resolved_log_dir, wrist_camera=wrist_camera, sim=sim)
    operator = _Operator(
        prompt_fn,
        interactive,
        str(cfg.get("confirm", "once")) == "once",
        requested,
        skills,
    )

    records: list[dict[str, Any]] = []
    completed = 0
    finish_reason = f"completed {requested}/{requested} cycles"
    final_pose: tuple[float, float, float] | None = None
    aborted = False
    try:
        for cycle in range(1, requested + 1):
            record: dict[str, Any] = {"cycle": cycle, "tossed": False}
            records.append(record)
            _logger.info("Cycle %d/%d: pick.", cycle, requested)
            failure = _pick(cfg, skills, operator, cycle, record)
            if failure is not None:
                finish_reason = failure
                break
            record["picked"] = True
            if cfg.get("skip_toss"):
                finish_reason = "picked (toss skipped)"
                break
            if cfg.get("place_at") is not None:
                target = (float(cfg.place_at[0]), float(cfg.place_at[1]))
                operator.confirm(
                    f"Cycle {cycle}: Carry the cube to ({target[0]:+.3f}, "
                    f"{target[1]:+.3f}) and set it down there?"
                )
                placed = skills.place(*target)
                record["place"] = placed.get("place")
                finish_reason = placed["note"] or (
                    f"placed at ({target[0]:+.3f}, {target[1]:+.3f})"
                )
                break
            reply = _toss(cfg, skills, operator, cycle, record, toss_pose)
            if not reply["tossed"]:
                finish_reason = f"not tossed: {reply['note']}"
                break
            record["tossed"] = True
            record["toss"] = reply["swing"]
            record["landing"] = _scored_landing(
                reply, reply["pose"] if aiming else toss_pose
            )
            completed += 1
        else:
            finish_reason = f"completed {completed}/{requested} cycles"
    except _Declined as e:
        finish_reason = f"declined: {e}"
    except ValueError as e:
        # A toss or drive refused before anything moved (bounds, barrier, bin).
        finish_reason = f"refused: {e}"
    except SkillStopped as e:
        # The arm was stopped mid-move; command nothing more (the arm-server RPC is
        # mid-call and the operator wants motion to end). Rerun to reset.
        finish_reason = f"aborted: {e}"
        aborted = True
    finally:
        if not aborted:
            final_pose = _reset(skills, operator)
        if resolved_log_dir is not None:
            resolved_log_dir.mkdir(parents=True, exist_ok=True)
            (resolved_log_dir / "cycles.json").write_text(
                json.dumps(
                    {
                        "requested": requested,
                        "completed": completed,
                        "finish_reason": finish_reason,
                        "toss_pose": toss_pose,
                        "cycles": records,
                    },
                    indent=1,
                    default=repr,
                ),
                encoding="utf-8",
            )
        skills.close(release_robot=not aborted)
    _logger.info("Toss cycles: %s.", finish_reason)
    return TossCyclesSummary(
        mode=cfg.mode,
        completed=completed,
        requested=requested,
        finish_reason=finish_reason,
        cycles=records,
        final_base_pose=final_pose,
    )


def _pick(
    cfg: DictConfig,
    skills: TossingSkills,
    operator: _Operator,
    cycle: int,
    record: dict[str, Any],
) -> str | None:
    """Call the pick skill until it holds the cube: a refusal or a miss is handed to
    the person (``ask_to_move_after_miss`` times); with ``max_attempts`` > 1 a miss
    retracts and drives back to the toss pose, where the ceiling cameras see the cube
    without the robot over it, and tries again. Returns a stop reason, or None once
    held."""
    td = cfg.pick.top_down
    attempts = int(td.get("max_attempts", 3))
    asks = int(td.get("ask_to_move_after_miss", 3)) if operator.interactive else 0
    put_backs = 0
    attempt = 0
    while attempt < attempts + asks:
        attempt += 1
        operator.confirm(
            f"Cycle {cycle}: Pick attempt {attempt}/{attempts}: localize the cube, "
            "drive to the pick standoff, open, hover, align with the wrist camera, "
            "descend, close?"
        )
        reply = skills.pick()
        cube = reply.get("cube")
        if cube is not None:
            record.setdefault("cube_before_pick", cube)
        if reply.get("bin") is not None:
            record["bin"] = reply["bin"]
        for tried in reply["attempts"]:
            record.setdefault("pick_attempts", []).append(tried | {"attempt": attempt})
        if reply["held"]:
            return None
        if reply["refused"]:
            reason = str(reply["reason"])
            if reason.startswith("no cube found"):
                question = (
                    f"\nCycle {cycle}: I can't find the cube ({reason}). Please put it "
                    f"in view on the floor, clear of the robot and {_STRIP}, then press "
                    "Enter, or type 'stop' to end the run: "
                )
                if not operator.ask_to_move(record, asks, question):
                    return f"{reason}; move the cube into the drivable area and rerun"
                continue
            assert cube is not None
            if reason == IN_BIN_PICK_OFF and put_backs < _MAX_PUT_BACKS:
                # The pick's hover is below the rim: the person puts it back (not a
                # pick attempt).
                put_backs += 1
                attempt -= 1
                record["cube_in_bin"] = {"x": cube["x"], "y": cube["y"]}
                _logger.info(
                    "The cube is in the bin at (%+.3f, %+.3f).", cube["x"], cube["y"]
                )
                operator.prompt_fn(
                    f"\nThe cube is in the bin at ({cube['x']:+.3f}, {cube['y']:+.3f}): "
                    "put it on the floor clear of the bin, then press Enter. "
                )
                continue
            record.setdefault("handed_to_person", []).append(
                {"cube": cube, "problem": reason}
            )
            question = (
                f"\nCycle {cycle}: the cube at ({cube['x']:+.3f}, {cube['y']:+.3f}) "
                f"cannot be picked: {reason}. Please move it onto the floor on the "
                f"robot's side of the tape, clear of the bin and {_STRIP}, then press "
                "Enter, or type 'stop' to end the run: "
            )
            if not operator.ask_to_move(record, asks, question):
                return f"{reason}; move the cube and rerun"
            continue
        if operator.interactive and record.get("cube_moved_by_person", 0) < asks:
            skills.retract()
            question = (
                f"\nCycle {cycle}: the pick missed and the arm is retracted. Please "
                f"move the cube ({_STRIP}), then press Enter, or type 'stop' to end "
                "the run: "
            )
            if not operator.ask_to_move(record, asks, question):
                return "pick missed; the operator stopped the run"
            continue
        if attempt < attempts:
            operator.confirm(
                f"Cycle {cycle}: Missed. Retract the arm and drive back to the toss "
                "pose to re-localize the cube?"
            )
            skills.retract()
            skills.go_home()
    return (
        f"pick failed after {attempts} attempts (gripper never closed on the cube); "
        "move the cube and rerun"
    )


def _toss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    skills: TossingSkills,
    operator: _Operator,
    cycle: int,
    record: dict[str, Any],
    toss_pose: list[float],
) -> dict[str, Any]:
    """Confirm, then call the toss skill: aimed at the localized bin (``toss.aimed``)
    or from the fixed toss pose at the cycle's ``toss.grid`` setting."""
    aimed = cfg.toss.get("aimed")
    if aimed is not None and bool(aimed.get("enabled", False)):
        speed, release = float(aimed.speed), float(aimed.release_ms)
        direction = aimed.get("direction")
        side = (
            "on its spot's side"
            if direction is None
            else f"along {math.degrees(float(direction)):+.0f} deg"
        )
        operator.confirm(
            f"Cycle {cycle}: Drive to the aimed toss pose {float(aimed.standoff):.2f} m "
            f"from the bin, {side}, toss, then retract the arm after the toss?"
        )
        _confirm_throw(cfg, operator, speed, release)
        reply = skills.toss(
            float(aimed.standoff),
            speed,
            release,
            None if direction is None else float(direction),
        )
        record["aimed"] = {
            "bin": reply.get("bin"),
            "pose": reply.get("pose"),
            "standoff": float(aimed.standoff),
            "speed": speed,
            "release_ms": release,
        }
        return reply
    speed, release = float(cfg.toss.max_vel), float(cfg.toss.gripper_release_ms)
    grid = cfg.toss.get("grid")
    if grid:
        speed, release = (float(v) for v in grid[(cycle - 1) % len(grid)])
    operator.confirm(
        f"Cycle {cycle}: Drive to the toss pose ({toss_pose[0]:+.3f}, "
        f"{toss_pose[1]:+.3f}, {math.degrees(toss_pose[2]):+.0f} deg), toss, then "
        "retract the arm after the toss?"
    )
    _confirm_throw(cfg, operator, speed, release)
    return skills.toss_at(toss_pose, speed, release)


def _confirm_throw(
    cfg: DictConfig, operator: _Operator, speed: float, release: float
) -> None:
    """The throw itself, described, when every step asks or ``toss.confirm_each``."""
    if not (
        operator.asking()
        or (operator.interactive and bool(cfg.toss.get("confirm_each", False)))
    ):
        return
    info = describe_toss(
        heading=float(cfg.toss.get("heading", 0.0)),
        max_vel=speed,
        gripper_release_ms=int(round(release)),
    )
    answer = operator.prompt_fn(
        f"\nAbout to toss: peak {info['max_vel_deg_s']:.0f} deg/s, gripper opens "
        f"{info['gripper_release_ms']} ms into a {info['duration_s']:.2f} s swing "
        f"({100 * info['release_fraction']:.0f}% of the way).\n"
        "The arm will move to the wind-up pose and throw. Keep the throwing area "
        "clear. Toss? [y/N]: "
    )
    if answer.strip().lower() not in ("y", "yes"):
        raise _Declined(f"toss rejected by operator (answer={answer!r})")


def _reset(
    skills: TossingSkills, operator: _Operator
) -> tuple[float, float, float] | None:
    """Retract the arm and drive back to the toss pose (after a confirmation)."""
    if operator.asking():
        answer = operator.prompt_fn(
            "\nReset: retract the arm and drive back to the toss pose? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            _logger.info("Reset declined; leaving the robot where it is.")
            return None
    skills.label("Reset: retract the arm and drive back to the toss pose")
    try:
        reached = skills.go_home()
    except ValueError as e:
        _logger.warning("Reset drive refused, leaving the base where it is: %s", e)
        return None
    return (float(reached[0]), float(reached[1]), float(reached[2]))
