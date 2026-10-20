"""Tossing3D's skills on the real robot, one RPC call each: the only implementation of
the robot's skills.

The server runs ON the NUC beside the arm and base servers, so every time-critical
loop (the 10 Hz drive, the arm's streamed moves, the 1 kHz toss swing) stays local;
a planner elsewhere on the LAN (hitl-pmp, on agni) calls :class:`TossingRobotClient`.
``scripts/toss_cycles.py`` calls the same :class:`TossingSkills` in-process, in a fixed
order. The RPC is ``multiprocessing.managers.BaseManager``, as
``third_party/arm_server.py``.

The contract (names and fields are fixed; all values plain floats, lists, dicts; the
MAP frame, metres and radians; hitl converts frames on its side):

``observe() -> dict``
    ``robot`` [x, y, yaw]; ``cube`` [x, y, yaw] or None; ``bin`` {x, y, yaw, length,
    width, height} or None; ``holding``; ``gripper`` (measured) and
    ``gripper_command`` (0 = open, 1 = closed, as the sim's ``pos_gripper``);
    ``cube_upright``; ``cube_in_bin`` (scored); ``cube_inside_bin``; ``barrier`` {x,
    y, length, thickness, yaw, robot_side} (``length`` along ``yaw``; ``robot_side``
    the unit map-frame direction into the robot's side); ``barrier_x`` (the barrier's
    centre x); ``toss_direction`` (the azimuth ``toss`` uses for this bin when given
    none, or None); ``can_pick`` {feasible, note, refused, reason}: ``can_pick()``'s
    answer, decided from this same look. One camera round per call: the bin is
    localized, then the cube looked for inside it first.
``can_pick() -> {"feasible", "note", "refused", "reason"}``
    A dry run of the pick from a fresh look; moves nothing. ``observe()["can_pick"]``
    answers the same without a second camera round.
``pick() -> {"held", "gripper", "note", "refused", "reason", "cube", "bin",
"stand", "attempts"}``
    Plans from the latest ``observe``/``can_pick`` look when no motion skill came
    since (else looks afresh); the cube is re-localized at the standoff.
    ``refused``: the pick could not or may not start and NOTHING MOVED (no cube, no
    bin found while the config has one: ``"bin not localized"``, past the barrier, no
    standoff, too close to a bin wall, no IK), ``reason`` says why. ``held`` False and
    ``refused`` False is a real miss: the robot moved. After a miss the robot
    retracts, drives back to its home stand (where the ceiling cameras see the cube),
    looks again and retries, ``pick.max_attempts`` (2) in all; missing every attempt
    it ends there, not over the cube. ``attempts``: one entry (or one per camera
    estimate tried) per attempt, each with its ``attempt`` number.
``toss(standoff, speed, release_ms, direction=None) -> {"tossed", "landed",
"in_bin", "inside_bin", "note", "pose", "direction", "bin", "swing"}``
    ``direction`` None: the bin's spot's fixed direction (``toss.spot_directions``).
    Raises ``ValueError`` (and moves nothing) outside the real bounds or for a toss
    pose across the barrier, off the drivable box or on the bin.
``open_gripper()``, ``retract()``, ``go_home()``, ``stop()``.

Definitions the planner relies on:

* ``holding``: the gripper is commanded closed, its reading is in the held band
  (``pick.top_down.grasp_min`` .. ``grasp_max``) and the cube is not seen.
* ``cube_upright``: the cube was found and its flat-cube silhouette fit (a cube
  resting on a face, :func:`prpl_tidybot.cube_localizer.localize_cube`) scored an IoU
  of at least ``skill_server.upright_min_iou``; a cube on an edge or leaning on the
  bin's wall fits that model badly. The search inside the bin fits no silhouette, so a
  cube found there counts as upright.
* ``cube_in_bin``: the cube's centre is inside the bin's walls by ``bin.in_bin_margin``
  on every side (the server's config: 0, since a real cube cannot rest on the rim;
  Tossing3D's own scoring region is 5 cm inside); ``cube_inside_bin``: inside the walls
  at all (the pick goes into the bin).
* The bin's spot: ``far_side`` when its centre is across the barrier, else
  ``robot_side``.
* Over the RPC, a call made while another skill moves the robot, or one ended by
  ``stop()``, raises ``RuntimeError("SkillBusy: ...")`` / ``RuntimeError("SkillStopped:
  ...")``: builtin errors, which a client without prpl_tidybot can unpickle.

Every call is recorded in ``<log dir>/skill_calls.jsonl`` (arguments, reply or error,
times), the ceiling cameras film the session into ``ceiling.mp4`` labelled with the
call in progress, and the wrist camera films each pick (``pick_NNN_wrist.mp4``).
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from omegaconf import DictConfig

from prpl_tidybot.aimed_toss import (
    Barrier,
    RealTossBounds,
    Workspace,
    aimed_toss_pose,
    plan_aimed_toss,
)
from prpl_tidybot.bin_localizer import BinFit, ConstantBinSource, cube_in_bin
from prpl_tidybot.interfaces.arm_interface import ArmInterface, ArmMotionAborted
from prpl_tidybot.robot_skills import (
    InBinPick,
    _build_lateral_servo,
    _CeilingRecorder,
    _cube_source_kwargs,
    _drive_to,
    _fake_reading,
    _find_cube,
    _grasp_at,
    _place_top_down,
    _retract,
    _set_gripper,
    _WristRecorder,
    plan_pick,
)
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    geometry_from_config,
    grasp_succeeded,
)
from prpl_tidybot.toss import describe_toss

_logger = logging.getLogger(__name__)

# The cube's measured gripper reading counts as held inside this band; the gripper
# is commanded "closed" at or above this.
_CLOSED = 0.5

# What the RPC serves: the contract. Everything else (close, place, toss_at) is for
# in-process callers.
EXPOSED = (
    "observe",
    "can_pick",
    "pick",
    "toss",
    "open_gripper",
    "retract",
    "go_home",
    "stop",
)

_F = TypeVar("_F", bound=Callable[..., Any])

# Why can_pick/pick refuse when the config has a bin the cameras did not find: never
# plan as if there were none (that skips the arm's check against its walls).
BIN_NOT_LOCALIZED = "bin not localized"


class SkillBusy(RuntimeError):
    """Another skill is already moving the robot."""


class SkillStopped(RuntimeError):
    """``stop()`` (or an aborted arm move) ended the skill; the robot is halted."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


class _CallLog:
    """One JSON line per skill call in ``<log dir>/skill_calls.jsonl``: its index,
    name, arguments, reply or error, and start/end times (Unix s)."""

    def __init__(self, log_dir: Path | None) -> None:
        self._path = None if log_dir is None else Path(log_dir) / "skill_calls.jsonl"
        self._lock = threading.Lock()
        self._count = 0

    def next_index(self) -> int:
        """The next call's 1-based index."""
        with self._lock:
            self._count += 1
            return self._count

    def write(self, entry: dict[str, Any]) -> None:
        """Append one call."""
        if self._path is None:
            return
        line = json.dumps(entry, default=_jsonable)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _logged(method: _F) -> _F:
    """Record the call (skill_calls.jsonl) and label the ceiling video with it."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self: TossingSkills, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call_args = {k: v for k, v in bound.arguments.items() if k != "self"}
        index = self.calls.next_index()
        shown = ", ".join(f"{k}={v}" for k, v in call_args.items())
        self.label(f"#{index} {method.__name__}({shown})")
        entry: dict[str, Any] = {
            "i": index,
            "call": method.__name__,
            "args": call_args,
            "start": time.time(),
        }
        try:
            reply = method(self, *args, **kwargs)
        except BaseException as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        else:
            entry["reply"] = reply
            return reply
        finally:
            entry["end"] = time.time()
            entry["duration_s"] = entry["end"] - entry["start"]
            self.calls.write(entry)

    return wrapper  # type: ignore[return-value]


def bin_spot(bin_fit: BinFit, barrier: Barrier | None) -> str:
    """``far_side`` for a bin whose centre is across the barrier, else
    ``robot_side``."""
    if barrier is not None and barrier.depth((bin_fit.x, bin_fit.y)) < 0.0:
        return "far_side"
    return "robot_side"


def _spot_directions(cfg: Any) -> dict[str, float | None]:
    block = cfg.toss.get("spot_directions") if cfg is not None else None
    if block is None:
        return {"robot_side": None, "far_side": None}
    return {
        k: None if block.get(k) is None else float(block[k])
        for k in ("robot_side", "far_side")
    }


def _bin_source_kwargs(bin_cfg: Any, log_dir: Path | None) -> dict[str, Any]:
    """The ceiling bin source saves the frames of any missed localization beside the
    run's logs, as the cube's does."""
    if log_dir is None or not str(bin_cfg["_target_"]).endswith("CeilingBinSource"):
        return {}
    return {"failure_dir": str(Path(log_dir) / "localization_failures")}


@dataclass(frozen=True)
class _Look:
    """One camera round: the bin and the cube (or why not), and the motion count it was
    taken at (:attr:`TossingSkills.motions`; odd: during a motion)."""

    bin_fit: BinFit | None
    cube: dict[str, Any] | None
    missing: str
    motions: int


def _can_pick_reply(why: str | None) -> dict[str, Any]:
    return {
        "feasible": why is None,
        "note": why,
        "refused": why is not None,
        "reason": why,
    }


def _in_bin_pick(cfg: DictConfig) -> Any:
    """``pick.top_down.in_bin`` when the pick out of the bin is on, else None."""
    block = cfg.pick.top_down.get("in_bin")
    return block if block is not None and bool(block.get("enabled")) else None


class TossingSkills:  # pylint: disable=too-many-instance-attributes
    """The skills, on one robot. Built by :func:`build_skills` from the config; every
    collaborator is injected so fake mode, sim mode and the tests run the same code."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cfg: DictConfig,
        real_env: Any,
        arm: ArmInterface,
        cube_source: Any,
        bin_source: Any,
        *,
        halt_arm: Callable[[], None] = lambda: None,
        servo: Any = None,
        read: Callable[[float], float] = float,
        sleep: Callable[[float], None] = time.sleep,
        log_dir: Path | None = None,
        wrist_source: Any = None,
        ceiling: Any = None,
    ) -> None:
        self.cfg = cfg
        self.real_env = real_env
        self.arm = arm
        self.cube_source = cube_source
        self.bin_source = bin_source
        self.geom: TopDownPickGeometry = geometry_from_config(cfg.pick.get("top_down"))
        self.floor = tuple(float(v) for v in cfg.preview.floor_bounds)
        self.workspace = Workspace.from_config(cfg.get("barrier"), self.floor)
        self.bounds = RealTossBounds.from_config(cfg.real_toss_bounds)
        self.in_bin_margin = float(cfg.bin.in_bin_margin)
        server = cfg.get("skill_server")
        self.upright_min_iou = (
            0.75 if server is None else float(server.get("upright_min_iou", 0.75))
        )
        self.in_bin = _in_bin_pick(cfg)
        self.spot_directions = _spot_directions(cfg)
        box = cfg.get("bin_stop_box")
        self.bin_box = None if box is None else [float(v) for v in box]
        self.home_pose = [float(v) for v in cfg.toss_pose]
        self.servo = servo
        self._halt_arm = halt_arm
        self._read = read
        self._sleep = sleep
        self._motion = threading.Lock()
        self.stop_event = threading.Event()
        self.gripper_command = 1.0 if arm.get_gripper_state() >= _CLOSED else 0.0
        self.calls = _CallLog(log_dir)
        self._wrist_source = wrist_source
        record = bool(cfg.pick.top_down.get("record_wrist", True))
        self.wrist = _WristRecorder(wrist_source if record else None, log_dir)
        self.ceiling = (
            ceiling if ceiling is not None else _CeilingRecorder(cfg, log_dir)
        )
        self.max_attempts = int(cfg.pick.get("max_attempts", 2))
        self._picks = 0
        self._label = "start"
        # Each motion skill counts up on entry and exit; a look is reused only if no
        # motion started or ran since it began.
        self.motions = 0
        self._look: _Look | None = None

    # ----------------------------------------------------------------- perception

    def label(self, text: str) -> None:
        """Label the ceiling video (and the log) with what is happening now."""
        self._label = text
        self.ceiling.label = text

    def _step(self, text: str) -> None:
        _logger.info("Step: %s", text.rstrip("?"))
        self.ceiling.label = f"{self._label.split('(')[0]}: {text}"

    def _robot(self) -> tuple[float, float, float]:
        obs, _ = self.real_env.reset()
        pose = obs.map_base_pose
        return (float(pose.x), float(pose.y), float(pose.theta()))

    def _bin(self) -> BinFit | None:
        if self.bin_source is None:
            return None
        try:
            fit: BinFit = self.bin_source.localize()
            return fit
        except RuntimeError as e:
            _logger.info("No bin: %s", e)
            return None

    def _look_now(self) -> _Look:
        """Localize the bin, then the cube (inside the bin first), and keep the look for
        a pick that follows with no motion between."""
        motions = self.motions
        bin_fit = self._bin()
        cube, missing = self._find(bin_fit)
        self._look = _Look(bin_fit, cube, missing, motions)
        return self._look

    def _last_look(self) -> _Look | None:
        """The latest look, if it was taken before this motion and after the last one
        (called inside the motion: the count went up by one on entry)."""
        look = self._look
        if look is None or look.motions % 2 or look.motions != self.motions - 1:
            return None
        return look

    def _decide(
        self, look: _Look, robot: Sequence[float]
    ) -> tuple[tuple[float, float, float] | None, str | None]:
        """(standoff, why the pick cannot start) from ``look``: can_pick's decision,
        moves nothing."""
        if self._holding(look.cube):
            return None, "already holding the cube"
        if look.bin_fit is None and self.bin_source is not None:
            return None, BIN_NOT_LOCALIZED
        if look.cube is None:
            return None, f"no cube found ({look.missing})"
        return plan_pick(
            robot,
            look.cube,
            self.geom,
            self.floor,
            self.workspace,
            look.bin_fit,
            self.in_bin,
            self.arm.get_arm_state(),
        )

    def _find(self, bin_fit: BinFit | None) -> tuple[dict[str, Any] | None, str]:
        """The cube (inside the bin first; :func:`_find_cube`), or None and why."""
        try:
            return _find_cube(self.cube_source, bin_fit, self.bin_box), ""
        except RuntimeError as e:
            _logger.info("No cube: %s", e)
            return None, str(e)

    def _holding(self, cube: dict[str, Any] | None) -> bool:
        """``observe()["holding"]``, and can_pick's "already holding": the held
        reading, and the cube not seen (a person may take it out of the closed
        gripper; pilot run9's evaluation)."""
        return bool(self._held_reading() and cube is None)

    def _held_reading(self) -> bool:
        return self.gripper_command >= _CLOSED and grasp_succeeded(
            self._read(self.arm.get_gripper_state()), self.geom
        )

    def spot_direction(self, bin_fit: BinFit) -> float | None:
        """The fixed toss azimuth (bin -> robot) for the bin's spot, or None (the
        valid one nearest the robot's bearing)."""
        return self.spot_directions[bin_spot(bin_fit, self.workspace.barrier)]

    @_logged
    def observe(self) -> dict[str, Any]:
        """The state the planner abstracts, and whether the pick could start, from one
        look (see the module docstring)."""
        look = self._look_now()
        bin_fit, cube = look.bin_fit, look.cube
        robot = self._robot()
        _, why = self._decide(look, robot)
        barrier = self.workspace.barrier
        xy = None if cube is None else (cube["x"], cube["y"])
        iou = None if cube is None else cube.get("iou")
        return {
            "robot": list(robot),
            "cube": (
                None
                if cube is None
                else [cube["x"], cube["y"], math.radians(cube["yaw_deg"])]
            ),
            "bin": None if bin_fit is None else bin_fit.as_dict(),
            "holding": self._holding(cube),
            "gripper": float(self.arm.get_gripper_state()),
            "gripper_command": float(self.gripper_command),
            "cube_upright": bool(
                cube is not None and (iou is None or iou >= self.upright_min_iou)
            ),
            "cube_in_bin": bool(
                xy is not None
                and bin_fit is not None
                and cube_in_bin(xy, bin_fit, self.in_bin_margin)
            ),
            "cube_inside_bin": bool(cube is not None and cube.get("in_bin")),
            "barrier": None if barrier is None else barrier.as_dict(),
            "barrier_x": None if barrier is None else float(barrier.x),
            "toss_direction": None if bin_fit is None else self.spot_direction(bin_fit),
            "can_pick": _can_pick_reply(why),
        }

    # --------------------------------------------------------------------- skills

    @_logged
    def can_pick(self) -> dict[str, Any]:
        """Whether the pick could start now (see :func:`plan_pick`), from a fresh look;
        ``observe`` answers the same from its own.

        Moves nothing.
        """
        look = self._look_now()
        return _can_pick_reply(self._decide(look, self._robot())[1])

    @_logged
    def pick(self, max_attempts: int | None = None) -> dict[str, Any]:
        """Pick the cube: plan from the latest look (``observe``'s, when no motion came
        since; else localize the bin and the cube, inside the bin first) and refuse
        before moving if the pick cannot start; then up to ``max_attempts`` (None:
        ``pick.max_attempts``) attempts, each :meth:`_attempt`.

        After a miss the robot retracts and drives back to its home stand, where the
        ceiling cameras see the cube without the robot over it, looks again and plans
        again; missing every attempt, it ends there.
        """
        tries = self.max_attempts if max_attempts is None else int(max_attempts)
        with self._moving():
            look = self._last_look() or self._look_now()
            stand, why = self._decide(look, self._robot())
            reply: dict[str, Any] = {
                "held": False,
                "gripper": float(self.arm.get_gripper_state()),
                "note": why,
                "refused": why is not None,
                "reason": why,
                "cube": look.cube,
                "bin": None if look.bin_fit is None else look.bin_fit.as_dict(),
                "stand": None if stand is None else list(stand),
                "attempts": [],
            }
            if why is not None:
                return reply
            record: dict[str, Any] = {"pick_attempts": []}
            for attempt in range(1, max(tries, 1) + 1):
                if attempt > 1:
                    self._step(
                        f"Missed. Attempt {attempt}/{tries}: re-localize the bin "
                        "and the cube from the retreat"
                    )
                    look = self._look_now()
                    stand, why = self._decide(look, self._robot())
                    if why is not None:
                        skipped = f"attempt {attempt} not started: {why}"
                        record["pick_attempts"].append(
                            {"attempt": attempt, "held": False, "note": skipped}
                        )
                        reply |= {"note": skipped}
                        break
                assert stand is not None
                outcome = self._attempt(attempt, stand, look, record)
                if outcome.get("refused"):
                    if attempt == 1:
                        return reply | outcome
                    # The robot moved before (the first attempt and the retreat).
                    reply |= {
                        "note": f"attempt {attempt} not started: {outcome['note']}"
                    }
                    break
                reply |= outcome
                if reply["held"]:
                    break
                retreat = self._retreat(look.bin_fit)
                record["pick_attempts"][-1]["retreated_to"] = retreat
                if isinstance(retreat, str):
                    reply |= {"note": f"{reply['note']}; {retreat}"}
                    break
            return reply | {"attempts": record["pick_attempts"]}

    def _attempt(  # pylint: disable=too-many-locals
        self,
        attempt: int,
        stand: tuple[float, float, float],
        look: _Look,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """One attempt: drive to the standoff (clear of the bin and the barrier), look
        for the cube again there, then open, [above the rim], hover, align with the
        wrist servo, descend, close and read; lift, [above the rim], carry.

        A miss on the best camera's estimate tries the other cameras' (floor cubes
        only). Returns the reply's fields it decides (``refused``: the drive was refused
        before the base moved).
        """
        bin_fit = look.bin_fit
        try:
            base = _drive_to(
                self.real_env, list(stand), self.workspace, self.stop_event, bin_fit
            )
        except ValueError as e:  # checked before the base moved
            record["pick_attempts"].append(
                {"attempt": attempt, "held": False, "note": str(e)}
            )
            return {"note": str(e), "refused": True, "reason": str(e)}
        found, missing = self._find(bin_fit)
        if found is None:
            lost = f"{missing} (at the standoff)"
            record["pick_attempts"].append(
                {"attempt": attempt, "held": False, "note": lost, "base": list(base)}
            )
            return {"held": False, "note": lost, "stand": list(base)}
        from_bin = bin_fit if found.get("in_bin") else None
        targets = [(found, "best camera")]
        if from_bin is None and hasattr(self.cube_source, "alternates"):
            targets += [
                ({"x": x, "y": y, "yaw_deg": math.degrees(yaw)}, "other camera")
                for x, y, yaw in self.cube_source.alternates()
            ]
        self._picks += 1
        start = len(record["pick_attempts"])
        held, failure = False, None
        for k, (target, label) in enumerate(targets):
            if k > 0:
                self._step(
                    "Missed. Try the other ceiling camera's estimate of the cube, "
                    f"({target['x']:+.3f}, {target['y']:+.3f})"
                )
            held, failure = _grasp_at(
                self.cfg,
                target,
                label,
                base,
                self.arm,
                self.geom,
                self._step,
                attempt,
                record,
                self.wrist,
                self.servo,
                self._read,
                f"pick_{self._picks:03d}" + (f"_{k + 1}" if k else ""),
                self.stop_event,
                from_bin,
                self.in_bin,
            )
            if failure is not None or held:
                break
        self.wrist.stop()
        if len(record["pick_attempts"]) == start:  # no grasp was started
            record["pick_attempts"].append(
                {"attempt": attempt, "held": False, "note": failure}
            )
        last = record["pick_attempts"][-1]
        self.gripper_command = (
            float(self.cfg.pick.top_down.get("close_position", 1.0)) if held else 0.0
        )
        note: str | None = None
        if not held:
            reading = last.get("gripper")
            note = (
                failure
                or last.get("note")
                or (
                    "gripper closed on nothing"
                    + ("" if reading is None else f" (read {float(reading):.3f})")
                )
            )
        return {
            "held": held,
            "gripper": float(last.get("gripper") or self.arm.get_gripper_state()),
            "note": note,
            "cube": found,
            "stand": list(base),
        }

    def _retreat(self, bin_fit: BinFit | None) -> list[float] | str:
        """After a miss: retract and drive back to the home stand (``toss_pose``), where
        the ceiling cameras see the cube without the robot over it (as toss_cycles did
        between attempts).

        Returns the pose reached, or why the drive was refused (the base stays at the
        standoff).
        """
        self._step("Missed. Retract the arm and drive back to re-localize the cube")
        _retract(self.cfg, self.arm, self.stop_event)
        try:
            reached = _drive_to(
                self.real_env, self.home_pose, self.workspace, self.stop_event, bin_fit
            )
        except ValueError as e:
            return f"retreat refused: {e}"
        return [float(v) for v in reached]

    @_logged
    def toss(
        self,
        standoff: float,
        speed: float,
        release_ms: float,
        direction: float | None = None,
    ) -> dict[str, Any]:
        """Throw the held cube at the localized bin from ``standoff`` m along
        ``direction`` (map-frame azimuth, bin -> robot; None: the bin's spot's fixed
        direction), at ``speed`` deg/s, opening at ``release_ms``. Raises
        ``ValueError`` and moves nothing if the parameters are outside the real bounds
        or the pose is not allowed."""
        standoff, speed, release_ms = (
            _number(standoff, "standoff"),
            _number(speed, "speed"),
            _number(release_ms, "release_ms"),
        )
        if direction is not None:
            direction = _number(direction, "direction")
        self.bounds.validate(standoff, speed, release_ms)
        with self._moving():
            if not self._held_reading():
                return _toss_result(False, note="not holding the cube")
            bin_fit = self._bin()
            if bin_fit is None:
                return _toss_result(False, note="no bin found")
            if direction is None:
                direction = self.spot_direction(bin_fit)
            pose = plan_aimed_toss(
                bin_fit,
                standoff,
                speed,
                release_ms,
                direction,
                self._robot(),
                self.workspace,
                self.bounds,
            )
            used = math.atan2(pose[1] - bin_fit.y, pose[0] - bin_fit.x)
            return self._throw(pose, speed, release_ms, bin_fit, 0.0) | {
                "direction": used
            }

    def toss_at(
        self, pose: Sequence[float], speed: float, release_ms: float
    ) -> dict[str, Any]:
        """Throw the held cube from the fixed map pose ``pose`` (toss_cycles without
        an aimed toss; not served). Raises ``ValueError`` before moving for a swing
        outside the real bounds or a drive the workspace refuses."""
        self.bounds.validate(None, _number(speed, "speed"), _number(release_ms, "ms"))
        self.label(f"toss_at({list(pose)}, {speed}, {release_ms})")
        with self._moving():
            if not self._held_reading():
                return _toss_result(False, note="not holding the cube")
            heading = float(self.cfg.toss.get("heading", 0.0))
            return self._throw(
                tuple(float(v) for v in pose), speed, release_ms, self._bin(), heading
            )

    def _throw(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        pose: Sequence[float],
        speed: float,
        release_ms: float,
        bin_fit: BinFit | None,
        heading: float,
    ) -> dict[str, Any]:
        """Drive to ``pose``, throw with the arm server's own wind-up and 1 kHz swing,
        retract, let the cube settle and look for it."""
        _drive_to(self.real_env, list(pose), self.workspace, self.stop_event, bin_fit)
        self._check_stop()
        release = int(round(release_ms))
        swing = describe_toss(
            heading=heading, max_vel=speed, gripper_release_ms=release
        )
        swing["toss_call_time"] = time.time()
        self._step(f"Toss: {speed:.0f} deg/s, release {release} ms")
        timing = self.arm.toss(
            heading=heading, max_vel=speed, gripper_release_ms=release
        )
        # The call includes the wind-up; the arm reports when the swing began.
        if timing and "swing_start_time" in timing:
            swing["swing_start_time"] = float(timing["swing_start_time"])
            swing["release_time"] = swing["swing_start_time"] + float(
                swing["release_time_s"]
            )
        self.gripper_command = 0.0
        self._step("Retract the arm after the toss")
        _retract(self.cfg, self.arm, self.stop_event)
        settle = float(self.cfg.get("settle_after_toss_s", 5.0))
        if settle > 0:
            _logger.info("Waiting %.1f s for the cube to settle.", settle)
            self._sleep(settle)
        after = self._bin() or bin_fit
        cube, missing = self._find(after)
        base = {
            "pose": [float(v) for v in pose],
            "bin": None if after is None else after.as_dict(),
            "swing": swing,
        }
        if cube is None:
            return (
                _toss_result(True, note=f"cube not found after the toss ({missing})")
                | base
            )
        landed = [float(cube["x"]), float(cube["y"])]
        return (
            _toss_result(
                True,
                landed=landed,
                in_bin=after is not None
                and cube_in_bin(landed, after, self.in_bin_margin),
                inside=bool(cube.get("in_bin")),
            )
            | base
        )

    @_logged
    def open_gripper(self) -> None:
        """Open the gripper where the arm is."""
        with self._moving():
            _set_gripper(self.arm, 0.0, self.cfg.pick.top_down)
            self.gripper_command = 0.0

    @_logged
    def retract(self) -> None:
        """Fold the arm to its retract pose (the gripper as it is)."""
        with self._moving():
            _retract(self.cfg, self.arm, self.stop_event)

    @_logged
    def go_home(self) -> list[float]:
        """Retract, then drive (clear of the bin and the barrier) to the home pose
        (``toss_pose``); returns the pose reached. Not called before a human reset:
        the person works with the robot where it is."""
        with self._moving():
            _retract(self.cfg, self.arm, self.stop_event)
            reached = _drive_to(
                self.real_env,
                self.home_pose,
                self.workspace,
                self.stop_event,
                self._bin(),
            )
            return [float(v) for v in reached]

    def place(self, x: float, y: float) -> dict[str, Any]:
        """Carry the held cube to the map point (x, y) and set it down (toss_cycles'
        ``place_at``; not served): {"placed", "note", "place"}."""
        self.label(f"place({x:+.3f}, {y:+.3f})")
        with self._moving():
            record: dict[str, Any] = {}
            failure = _place_top_down(
                self.cfg,
                self.real_env,
                self.arm,
                self.geom,
                self._step,
                (float(x), float(y)),
                record,
                self.workspace,
                self.stop_event,
                self._bin(),
            )
            if failure is None:
                self.gripper_command = 0.0
            return {"placed": failure is None, "note": failure, **record}

    @_logged
    def stop(self) -> None:
        """Halt the arm (the onboard controller's Stop, as Ctrl-C does in
        ``toss_cycles``) and the base (its target set to where it is), and make the
        skill in progress stop commanding motion."""
        self.motions += 2
        self.stop_event.set()
        try:
            self._halt_arm()
        finally:
            base = self.real_env.interface.base_interface
            base.execute_action(base.get_base_state())

    def close(self, release_robot: bool = True) -> None:
        """Finish the recordings and release the sources; with ``release_robot`` also
        the robot's clients (the arm's torque loop). Not served."""
        self.wrist.close()
        if self._wrist_source is not None and hasattr(self._wrist_source, "close"):
            self._wrist_source.close()
        self.ceiling.close()
        self.cube_source.close()
        if self.bin_source is not None:
            self.bin_source.close()
        if release_robot:
            self.real_env.close()

    # ------------------------------------------------------------------- helpers

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise ArmMotionAborted("stop() during a skill")

    def _moving(self) -> "_Motion":
        return _Motion(self)

    def toss_pose_for(
        self, bin_fit: BinFit, standoff: float, direction: float
    ) -> tuple[float, float, float]:
        """Where ``toss`` would stand (for operators and tests); raises like it."""
        return aimed_toss_pose(bin_fit, standoff, direction, self.workspace)


class _Motion:
    """One skill moving the robot at a time; a fresh skill clears an old stop()."""

    def __init__(self, skills: TossingSkills) -> None:
        self._skills = skills

    def __enter__(self) -> None:
        if not self._skills._motion.acquire(  # pylint: disable=protected-access
            blocking=False
        ):
            raise SkillBusy("another skill is moving the robot")
        self._skills.motions += 1
        self._skills.stop_event.clear()

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self._skills.motions += 1
        self._skills._motion.release()  # pylint: disable=protected-access
        if kind is not None and issubclass(kind, ArmMotionAborted):
            # A KeyboardInterrupt would kill the RPC server's connection thread;
            # the caller gets an ordinary error instead.
            raise SkillStopped(str(value)) from value


def _toss_result(
    tossed: bool,
    landed: list[float] | None = None,
    in_bin: bool = False,
    note: str | None = None,
    inside: bool = False,
) -> dict[str, Any]:
    return {
        "tossed": tossed,
        "landed": landed,
        "in_bin": bool(in_bin),
        "inside_bin": bool(inside),
        "note": note,
        "pose": None,
        "direction": None,
        "bin": None,
        "swing": None,
    }


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}={value!r} is not a number.") from e
    if not math.isfinite(number):
        raise ValueError(f"{name}={value!r} is not a number.")
    return number


def build_skills(
    cfg: DictConfig,
    log_dir: Path | None = None,
    *,
    wrist_camera: Callable[[ArmInterface], Any] | None = None,
    sim: Any = None,
) -> TossingSkills:
    """The skills for ``cfg.mode`` (``fake``, ``real``, or ``sim`` with a
    :class:`prpl_tidybot.toss_sim.TossSim`) from the config (conf/skill_server.yaml,
    or toss_cycles' own). ``wrist_camera`` builds the wrist image source from the arm
    (default: the Kinova wrist camera in real mode, none otherwise: no servo)."""
    # pylint: disable=import-outside-toplevel
    import hydra
    import kinder

    kinder.register_all_environments()
    sleep: Callable[[float], None] = time.sleep
    ceiling: Any = None
    if sim is not None:
        real_env, cube_source = sim.real_env, sim.cube_source
        ceiling, sleep = sim.recorder, sim.world.idle
        bin_source: Any = None
        if cfg.sim.get("bin") is not None:
            x, y, yaw_deg = (float(v) for v in cfg.sim.bin)
            b = cfg.bin
            bin_source = ConstantBinSource(
                x, y, math.radians(yaw_deg), b.length, b.width, b.height, b.wall
            )
    else:
        pipeline = cfg.env.pipelines[cfg.mode]
        real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
        target_cfg = pipeline.perceiver.target_source
        cube_source = hydra.utils.instantiate(
            target_cfg, _convert_="all", **_cube_source_kwargs(cfg, target_cfg, log_dir)
        )
        bin_source = (
            hydra.utils.instantiate(
                cfg.bin.pipelines[cfg.mode],
                _convert_="all",
                **_bin_source_kwargs(cfg.bin.pipelines[cfg.mode], log_dir),
            )
            if cfg.mode in cfg.bin.pipelines
            else None
        )
    arm = real_env.interface.arm_interface
    halt: Callable[[], None] = lambda: None
    source: Any = None
    if wrist_camera is not None:
        source = wrist_camera(arm)
    elif cfg.mode == "real":
        from prpl_tidybot.visual_servo.image_sources import KinovaWristCameraSource

        source = KinovaWristCameraSource()
    if cfg.mode == "real":
        from prpl_tidybot.third_party.kinova import stop_arm_motion

        halt = stop_arm_motion
    servo = _build_lateral_servo(cfg, source, log_dir)
    read: Callable[[float], float] = float
    if cfg.mode == "fake":
        # The fake gripper closes on nothing; pretend it read fake_grasp_reading.
        fake = float(cfg.pick.top_down.get("fake_grasp_reading", 0.49))
        read = functools.partial(_fake_reading, fake)
    return TossingSkills(
        cfg,
        real_env,
        arm,
        cube_source,
        bin_source,
        halt_arm=halt,
        servo=servo,
        read=read,
        sleep=sleep,
        log_dir=log_dir,
        wrist_source=source,
        ceiling=ceiling,
    )


class SkillManager(BaseManager):
    """The skill server's RPC manager (``Skills`` is the one shared object)."""


class SkillClientManager(BaseManager):
    """The client side of :class:`SkillManager`. A separate class because
    ``register`` writes a class-wide registry: registering ``Skills`` without a
    callable for the client must not replace the server's."""


SkillClientManager.register("Skills")


class _RpcSkills:
    """The skills as the RPC serves them. SkillBusy and SkillStopped leave as a
    builtin RuntimeError naming them: the client unpickles every error, and hitl-pmp
    has no prpl_tidybot to unpickle these classes with. ValueError (a refusal) passes
    through unchanged."""

    def __init__(self, skills: TossingSkills) -> None:
        self._skills = skills

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._skills, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except (SkillBusy, SkillStopped) as e:
                raise RuntimeError(f"{type(e).__name__}: {e}") from None

        return call


def serve(skills: TossingSkills, host: str, port: int, authkey: bytes) -> None:
    """Serve ``skills`` (the contract's calls only) until interrupted; Ctrl-C halts
    the robot first."""
    rpc = _RpcSkills(skills)
    SkillManager.register("Skills", callable=lambda: rpc, exposed=EXPOSED)
    manager = SkillManager(address=(host, port), authkey=authkey)
    server = manager.get_server()
    _logger.info("Tossing skill server on %s:%d", host, port)
    try:
        server.serve_forever()  # type: ignore[no-untyped-call]
    except KeyboardInterrupt:
        skills.stop()
        raise


class TossingRobotClient:
    """The skills from another machine (agni): the contract in the module docstring.

    Ctrl-C during a call sends ``stop()`` on a fresh connection (the first is busy
    with the call), then re-raises."""

    def __init__(self, host: str, port: int, authkey: bytes) -> None:
        self._address = (host, int(port))
        self._authkey = authkey
        self._skills = self._connect()

    def _connect(self) -> Any:
        manager = SkillClientManager(address=self._address, authkey=self._authkey)
        manager.connect()
        return manager.Skills()  # type: ignore[attr-defined]  # pylint: disable=no-member

    def _call(self, name: str, *args: Any) -> Any:
        try:
            return getattr(self._skills, name)(*args)
        except KeyboardInterrupt:
            self.stop()
            raise

    def observe(self) -> dict[str, Any]:
        """See the module docstring."""
        return dict(self._call("observe"))

    def can_pick(self) -> dict[str, Any]:
        """See the module docstring."""
        return dict(self._call("can_pick"))

    def pick(self) -> dict[str, Any]:
        """See the module docstring."""
        return dict(self._call("pick"))

    def toss(
        self,
        standoff: float,
        speed: float,
        release_ms: float,
        direction: float | None = None,
    ) -> dict[str, Any]:
        """See the module docstring."""
        return dict(
            self._call(
                "toss",
                float(standoff),
                float(speed),
                float(release_ms),
                None if direction is None else float(direction),
            )
        )

    def open_gripper(self) -> None:
        """See the module docstring."""
        self._call("open_gripper")

    def retract(self) -> None:
        """See the module docstring."""
        self._call("retract")

    def go_home(self) -> None:
        """See the module docstring."""
        self._call("go_home")

    def stop(self) -> None:
        """Halt arm and base now, on a fresh connection."""
        self._connect().stop()


# The real room (conf/skill_server.yaml): FakeTossingRobot's defaults.
REAL_BARRIER = Barrier(
    x=0.0,
    y=0.374,
    length=6.0,
    thickness=0.05,
    yaw=0.026694,
    robot_side=(0.026690, -0.999644),
)
REAL_WORKSPACE = Workspace(
    floor_bounds=(-1.5, 1.7, -1.5, 1.5),
    barrier=REAL_BARRIER,
    clearance=0.05,
    cube_half=0.025,
)
CLOSE_SPOT_BIN = BinFit(-0.29, -0.77, math.pi / 2, 0.39, 0.29, 0.21, 0.005)
REAL_GEOMETRY = TopDownPickGeometry(
    reach=0.55,
    floor_z=-0.395,
    pad_clearance=0.022,
    hover_height=0.09,
    grasp_min=0.2,
    grasp_max=0.9,
)
SPOT_DIRECTIONS: dict[str, float | None] = {
    "robot_side": 0.0,
    "far_side": -math.pi / 2,
}
HOME_POSE = (1.10, -0.78, math.pi)


class FakeTossingRobot:  # pylint: disable=too-many-instance-attributes
    """The client's interface with no robot, in the real room: deterministic, for
    planner tests.

    The same checks as the real skills (bounds, barrier, drivable box, bin, the pick out
    of the bin's wall check; :func:`prpl_tidybot.robot_skills.plan_pick`) decide what is
    allowed; ``pick_outcomes`` (cycled) says whether each pick that starts holds (each
    attempt of a pick takes the next; up to ``max_attempts`` per pick, a miss retreating
    to ``home`` first), ``bin_localized`` False is a bin the cameras miss (can_pick and
    pick refuse), and ``landing`` maps a toss ``(pose, speed, release_ms)`` to where the
    cube comes to rest (default: straight ahead of the base, ``0.9 + 0.55 *`` the
    parameters' position in their box, so the standoff that lands in the bin depends on
    them).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        robot: Sequence[float] = HOME_POSE,
        cube: Sequence[float] | None = (0.6, -0.5, 0.0),
        bin_fit: BinFit | None = CLOSE_SPOT_BIN,
        workspace: Workspace = REAL_WORKSPACE,
        bounds: RealTossBounds = RealTossBounds(),
        geom: TopDownPickGeometry = REAL_GEOMETRY,
        in_bin: InBinPick | None = InBinPick(),
        spot_directions: dict[str, float | None] | None = None,
        pick_outcomes: Sequence[bool] = (True,),
        landing: (
            Callable[[Sequence[float], float, float], Sequence[float]] | None
        ) = None,
        in_bin_margin: float = 0.0,
        home: Sequence[float] = HOME_POSE,
        max_attempts: int = 2,
        bin_localized: bool = True,
    ) -> None:
        self.robot = [float(v) for v in robot]
        self.cube = None if cube is None else [float(v) for v in cube]
        self.bin_fit = bin_fit
        self.workspace = workspace
        self.bounds = bounds
        self.geom = geom
        self.in_bin = in_bin
        self.spot_directions = dict(
            SPOT_DIRECTIONS if spot_directions is None else spot_directions
        )
        self._outcomes = list(pick_outcomes) or [True]
        self._picks = 0
        self._landing = landing or _default_landing(bounds)
        self.in_bin_margin = in_bin_margin
        self.home = [float(v) for v in home]
        self.max_attempts = int(max_attempts)
        self.bin_localized = bool(bin_localized)
        self.holding = False
        self.gripper_command = 0.0
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _cube(self) -> dict[str, Any] | None:
        if self.holding or self.cube is None:
            return None
        inside = self.bin_fit is not None and cube_in_bin(
            self.cube[:2], self.bin_fit, 0.0
        )
        return {
            "x": self.cube[0],
            "y": self.cube[1],
            "yaw_deg": math.degrees(self.cube[2]),
            "in_bin": bool(inside),
        }

    def _plan(self) -> tuple[tuple[float, float, float] | None, str | None]:
        from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
            RETRACT_ARM_CONF,
        )

        if self.holding:
            return None, "already holding the cube"
        if not self.bin_localized:
            return None, BIN_NOT_LOCALIZED
        cube = self._cube()
        if cube is None:
            return None, "no cube found"
        return plan_pick(
            self.robot,
            cube,
            self.geom,
            self.workspace.floor_bounds,
            self.workspace,
            self.bin_fit,
            self.in_bin,
            [float(v) for v in RETRACT_ARM_CONF],
        )

    def observe(self) -> dict[str, Any]:
        """As the real ``observe``."""
        self.calls.append(("observe", ()))
        barrier = self.workspace.barrier
        cube = self._cube()
        seen = self.bin_fit if self.bin_localized else None
        return {
            "robot": list(self.robot),
            "cube": None if cube is None else list(self.cube or []),
            "bin": None if seen is None else seen.as_dict(),
            "holding": self.holding,
            "gripper": 0.49 if self.holding else (1.0 if self.gripper_command else 0.0),
            "gripper_command": float(self.gripper_command),
            "cube_upright": cube is not None,
            "cube_in_bin": bool(
                cube is not None
                and seen is not None
                and cube_in_bin((cube["x"], cube["y"]), seen, self.in_bin_margin)
            ),
            "cube_inside_bin": bool(cube is not None and cube["in_bin"]),
            "barrier": None if barrier is None else barrier.as_dict(),
            "barrier_x": None if barrier is None else float(barrier.x),
            "toss_direction": None if seen is None else self._spot_direction(seen),
            "can_pick": _can_pick_reply(self._plan()[1]),
        }

    def _spot_direction(self, bin_fit: BinFit) -> float | None:
        return self.spot_directions[bin_spot(bin_fit, self.workspace.barrier)]

    def can_pick(self) -> dict[str, Any]:
        """As the real ``can_pick``."""
        self.calls.append(("can_pick", ()))
        return _can_pick_reply(self._plan()[1])

    def pick(self) -> dict[str, Any]:
        """Refuse as the real pick does, else drive to the standoff and grasp, held per
        ``pick_outcomes``; a miss retreats to ``home`` and tries again, up to
        ``max_attempts``."""
        self.calls.append(("pick", ()))
        cube = self._cube()
        stand, why = self._plan()
        reply: dict[str, Any] = {
            "held": False,
            "gripper": 0.0,
            "note": why,
            "refused": why is not None,
            "reason": why,
            "cube": cube,
            "bin": None if self.bin_fit is None else self.bin_fit.as_dict(),
            "stand": None if stand is None else list(stand),
            "attempts": [],
        }
        if why is not None:
            return reply
        attempts: list[dict[str, Any]] = []
        held = False
        for attempt in range(1, max(self.max_attempts, 1) + 1):
            if attempt > 1:
                stand, why = self._plan()
                if why is not None:
                    attempts.append({"attempt": attempt, "held": False, "note": why})
                    break
            assert stand is not None
            self.robot = list(stand)
            held = self._outcomes[self._picks % len(self._outcomes)]
            self._picks += 1
            attempts.append({"attempt": attempt, "held": held, "base": list(stand)})
            if held:
                break
            self.robot = list(self.home)
        self.holding = held
        self.gripper_command = 1.0 if held else 0.0
        return reply | {
            "held": held,
            "gripper": 0.49 if held else 0.0,
            "note": None if held else "gripper closed on nothing",
            "stand": attempts[-1].get("base", reply["stand"]),
            "attempts": attempts,
        }

    def toss(
        self,
        standoff: float,
        speed: float,
        release_ms: float,
        direction: float | None = None,
    ) -> dict[str, Any]:
        """As the real ``toss``: the same ``ValueError``s and default direction; the
        cube lands per ``landing``."""
        self.calls.append(("toss", (standoff, speed, release_ms, direction)))
        standoff, speed, release_ms = (
            _number(standoff, "standoff"),
            _number(speed, "speed"),
            _number(release_ms, "release_ms"),
        )
        if direction is not None:
            direction = _number(direction, "direction")
        self.bounds.validate(standoff, speed, release_ms)
        if not self.holding:
            return _toss_result(False, note="not holding the cube")
        if self.bin_fit is None or not self.bin_localized:
            return _toss_result(False, note="no bin found")
        if direction is None:
            direction = self._spot_direction(self.bin_fit)
        pose = plan_aimed_toss(
            self.bin_fit,
            standoff,
            speed,
            release_ms,
            direction,
            self.robot,
            self.workspace,
            self.bounds,
        )
        self.robot = list(pose)
        landed = [float(v) for v in self._landing(pose, speed, release_ms)][:2]
        self.cube = [landed[0], landed[1], 0.0]
        self.holding = False
        self.gripper_command = 0.0
        return _toss_result(
            True,
            landed=landed,
            in_bin=cube_in_bin(landed, self.bin_fit, self.in_bin_margin),
            inside=cube_in_bin(landed, self.bin_fit, 0.0),
        ) | {
            "pose": list(pose),
            "direction": math.atan2(pose[1] - self.bin_fit.y, pose[0] - self.bin_fit.x),
            "bin": self.bin_fit.as_dict(),
        }

    def open_gripper(self) -> None:
        """Open: a held cube drops where the gripper is (in front of the base)."""
        self.calls.append(("open_gripper", ()))
        if self.holding:
            ahead = self.geom.standoff
            self.cube = [
                self.robot[0] + ahead * math.cos(self.robot[2]),
                self.robot[1] + ahead * math.sin(self.robot[2]),
                0.0,
            ]
        self.holding = False
        self.gripper_command = 0.0

    def retract(self) -> None:
        """Nothing moves on the floor."""
        self.calls.append(("retract", ()))

    def go_home(self) -> None:
        """Back to the home pose."""
        self.calls.append(("go_home", ()))
        self.robot = list(self.home)

    def stop(self) -> None:
        """Nothing to halt."""
        self.calls.append(("stop", ()))


def _default_landing(
    bounds: RealTossBounds,
) -> Callable[[Sequence[float], float, float], Sequence[float]]:
    def landing(pose: Sequence[float], speed: float, release_ms: float) -> list[float]:
        s = (speed - bounds.speed[0]) / (bounds.speed[1] - bounds.speed[0])
        r = (release_ms - bounds.release_ms[0]) / (
            bounds.release_ms[1] - bounds.release_ms[0]
        )
        reach = 0.9 + 0.55 * float(np.clip(0.5 * (s + r), 0.0, 1.0))
        return [
            pose[0] + reach * math.cos(pose[2]),
            pose[1] + reach * math.sin(pose[2]),
        ]

    return landing
