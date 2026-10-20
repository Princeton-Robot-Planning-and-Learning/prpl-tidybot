"""Tossing3D's skills on the real robot, one RPC call each.

The server runs ON the NUC beside the arm and base servers, so every time-critical
loop (the 10 Hz drive, the arm's streamed moves, the 1 kHz toss swing) stays local;
a planner elsewhere on the LAN (hitl-pmp, on agni) calls :class:`TossingRobotClient`.
The RPC is ``multiprocessing.managers.BaseManager``, as ``third_party/arm_server.py``.

The contract (names and fields are fixed; all values plain floats, lists, dicts):

``observe() -> dict``
    ``robot`` [x, y, yaw]; ``cube`` [x, y, yaw] or None; ``bin`` {x, y, yaw, length,
    width, height} or None; ``holding``; ``gripper`` (measured) and
    ``gripper_command`` (0 = open, 1 = closed, as the sim's ``pos_gripper``);
    ``cube_upright``; ``cube_in_bin``; ``barrier`` {x, y, length, thickness, yaw,
    robot_side} (``length`` along ``yaw``; ``robot_side`` the unit map-frame direction
    into the robot's side); ``barrier_x`` (the barrier's centre x). Map frame, metres
    and radians.
``can_pick() -> {"feasible", "note"}``
    A dry run of the top-down pick; moves nothing.
``pick() -> {"held", "gripper", "note"}``
``toss(standoff, speed, release_ms, direction) -> {"tossed", "landed", "in_bin",
"note"}``
    Raises ``ValueError`` (and moves nothing) outside the real bounds or for a toss
    pose across the barrier, off the drivable box or on the bin.
``open_gripper()``, ``retract()``, ``go_home()``, ``stop()``.

Definitions the planner relies on:

* ``holding``: the gripper is commanded closed, its reading is in the held band
  (``pick.top_down.grasp_min`` .. ``grasp_max``) and the cube is not seen on the floor.
* ``cube_upright``: the cube was found and its flat-cube silhouette fit (a cube
  resting on a face, :func:`prpl_tidybot.cube_localizer.localize_cube`) scored an IoU
  of at least ``skill_server.upright_min_iou``; a cube on an edge or leaning on the
  bin's wall fits that model badly.
* ``cube_in_bin``: the cube's centre is inside the bin's walls by ``bin.in_bin_margin``
  on every side.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from multiprocessing.managers import BaseManager
from typing import Any

import numpy as np
from omegaconf import DictConfig

from prpl_tidybot.aimed_toss import (
    RealTossBounds,
    Workspace,
    aimed_toss_pose,
    plan_aimed_toss,
)
from prpl_tidybot.bin_localizer import BinFit, cube_in_bin
from prpl_tidybot.interfaces.arm_interface import ArmInterface, ArmMotionAborted
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    arm_frame_target,
    geometry_from_config,
    grasp_succeeded,
    standoff_pose,
    top_down_joint_targets,
)

_logger = logging.getLogger(__name__)

# The cube's measured gripper reading counts as held inside this band; the gripper
# is commanded "closed" at or above this.
_CLOSED = 0.5


class SkillBusy(RuntimeError):
    """Another skill is already moving the robot."""


class SkillStopped(RuntimeError):
    """``stop()`` (or an aborted arm move) ended the skill; the robot is halted."""


class TossingSkills:  # pylint: disable=too-many-instance-attributes
    """The skills, on one robot. Built by :func:`build_skills` from the
    ``skill_server`` config; every collaborator is injected so fake mode and the
    tests run the same code."""

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
    ) -> None:
        self.cfg = cfg
        self.real_env = real_env
        self.arm = arm
        self.cube_source = cube_source
        self.bin_source = bin_source
        self.geom: TopDownPickGeometry = geometry_from_config(cfg.pick.get("top_down"))
        self.floor = tuple(float(v) for v in cfg.preview.floor_bounds)
        self.workspace = Workspace.from_config(cfg.barrier, self.floor)
        self.bounds = RealTossBounds.from_config(cfg.real_toss_bounds)
        self.in_bin_margin = float(cfg.bin.in_bin_margin)
        self.upright_min_iou = float(cfg.skill_server.upright_min_iou)
        self.home_pose = [float(v) for v in cfg.toss_pose]
        self._halt_arm = halt_arm
        self._servo = servo
        self._read = read
        self._sleep = sleep
        self._motion = threading.Lock()
        self.stop_event = threading.Event()
        self.gripper_command = 1.0 if arm.get_gripper_state() >= _CLOSED else 0.0

    # ----------------------------------------------------------------- perception

    def _robot(self) -> tuple[float, float, float]:
        obs, _ = self.real_env.reset()
        pose = obs.map_base_pose
        return (float(pose.x), float(pose.y), float(pose.theta()))

    def _cube(self) -> tuple[list[float], float] | None:
        """The cube's [x, y, yaw] and its flat-cube fit IoU, or None."""
        try:
            if hasattr(self.cube_source, "localize"):
                fit = self.cube_source.localize()
                return [float(fit.x), float(fit.y), float(fit.yaw)], float(fit.iou)
            x, y, _ = self.cube_source.get_target()
            yaw = (
                float(self.cube_source.get_yaw())
                if hasattr(self.cube_source, "get_yaw")
                else 0.0
            )
            return [float(x), float(y), yaw], 1.0
        except RuntimeError as e:
            _logger.info("No cube: %s", e)
            return None

    def _bin(self) -> BinFit | None:
        try:
            return self.bin_source.localize()
        except RuntimeError as e:
            _logger.info("No bin: %s", e)
            return None

    def _held_reading(self) -> bool:
        return self.gripper_command >= _CLOSED and grasp_succeeded(
            self._read(self.arm.get_gripper_state()), self.geom
        )

    def observe(self) -> dict[str, Any]:
        """The state the planner abstracts (see the module docstring)."""
        cube = self._cube()
        bin_fit = self._bin()
        barrier = self.workspace.barrier
        return {
            "robot": list(self._robot()),
            "cube": None if cube is None else cube[0],
            "bin": None if bin_fit is None else bin_fit.as_dict(),
            "holding": bool(self._held_reading() and cube is None),
            "gripper": float(self.arm.get_gripper_state()),
            "gripper_command": float(self.gripper_command),
            "cube_upright": bool(cube is not None and cube[1] >= self.upright_min_iou),
            "cube_in_bin": bool(
                cube is not None
                and bin_fit is not None
                and cube_in_bin(cube[0][:2], bin_fit, self.in_bin_margin)
            ),
            "barrier": None if barrier is None else barrier.as_dict(),
            "barrier_x": None if barrier is None else float(barrier.x),
        }

    # --------------------------------------------------------------------- skills

    def can_pick(self) -> dict[str, Any]:
        """Whether the top-down pick could start now: cube found, on the robot's side,
        a standoff in the drivable box on the robot's side with a straight drive
        there, and IK for the hover and the grasp. Moves nothing."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import pick_problem

        if self._held_reading():
            return {"feasible": False, "note": "already holding the cube"}
        cube = self._cube()
        if cube is None:
            return {"feasible": False, "note": "no cube found"}
        robot = self._robot()
        xy = {"x": cube[0][0], "y": cube[0][1], "yaw": cube[0][2]}
        problem = pick_problem(robot, xy, self.geom, self.floor, self.workspace)
        if problem is not None:
            return {"feasible": False, "note": problem}
        stand = standoff_pose(
            robot, cube[0][:2], self.geom, self.floor, self.workspace.valid()
        )
        ax, ay = arm_frame_target(stand, cube[0][:2], self.geom)
        try:
            top_down_joint_targets(
                ax, ay, cube[0][2] - stand[2], self.geom, seed=self.arm.get_arm_state()
            )
        except ValueError as e:
            return {"feasible": False, "note": f"pick IK: {e}"}
        return {"feasible": True, "note": None}

    def pick(self) -> dict[str, Any]:
        """The top-down pick (``toss_cycles``' own: drive to the standoff, re-localize,
        open, hover, wrist servo per config, descend, close, read)."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import _pick_top_down, _WristRecorder

        with self._moving():
            record: dict[str, Any] = {}
            failure = _pick_top_down(
                _with_one_attempt(self.cfg),
                self.real_env,
                self.cube_source,
                self.arm,
                self.geom,
                _log_step,
                1,
                record,
                _WristRecorder(None, None),
                self.home_pose,
                self._servo,
                workspace=self.workspace,
                stop=self.stop_event,
            )
            attempts = record.get("pick_attempts") or [{}]
            held = failure is None
            self.gripper_command = (
                float(self.cfg.pick.top_down.get("close_position", 1.0))
                if held
                else 0.0
            )
            note = failure if failure is not None else attempts[-1].get("note")
            return {
                "held": held,
                "gripper": float(
                    attempts[-1].get("gripper", self.arm.get_gripper_state())
                ),
                "note": note,
            }

    def toss(
        self, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        """Throw the held cube at the bin from ``standoff`` m along ``direction``
        (map-frame azimuth, bin -> robot), at ``speed`` deg/s, opening at
        ``release_ms``. Raises ``ValueError`` and moves nothing if the parameters are
        outside the real bounds or the pose is not allowed."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import _drive_to, _retract

        standoff, speed, release_ms = (
            _number(standoff, "standoff"),
            _number(speed, "speed"),
            _number(release_ms, "release_ms"),
        )
        direction = _number(direction, "direction")
        self.bounds.validate(standoff, speed, release_ms)
        with self._moving():
            if not self._held_reading():
                return _toss_result(False, None, False, "not holding the cube")
            bin_fit = self._bin()
            if bin_fit is None:
                return _toss_result(False, None, False, "no bin found")
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
            _drive_to(self.real_env, list(pose), self.workspace, self.stop_event)
            self._check_stop()
            self.arm.toss(
                heading=0.0, max_vel=speed, gripper_release_ms=int(round(release_ms))
            )
            self.gripper_command = 0.0
            _retract(self.cfg, self.arm, self.stop_event)
            settle = float(self.cfg.get("settle_after_toss_s", 5.0))
            if settle > 0:
                self._sleep(settle)
            cube = self._cube()
            after = self._bin() or bin_fit
            if cube is None:
                return _toss_result(True, None, False, "cube not found after the toss")
            landed = [float(cube[0][0]), float(cube[0][1])]
            return _toss_result(
                True, landed, cube_in_bin(landed, after, self.in_bin_margin), None
            )

    def open_gripper(self) -> None:
        """Open the gripper where the arm is."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import _set_gripper

        with self._moving():
            _set_gripper(self.arm, 0.0, self.cfg.pick.top_down)
            self.gripper_command = 0.0

    def retract(self) -> None:
        """Fold the arm to its retract pose (the gripper as it is)."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import _retract

        with self._moving():
            _retract(self.cfg, self.arm, self.stop_event)

    def go_home(self) -> None:
        """Retract, then drive to the home pose (``toss_pose``)."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import _drive_to, _retract

        with self._moving():
            _retract(self.cfg, self.arm, self.stop_event)
            _drive_to(self.real_env, self.home_pose, self.workspace, self.stop_event)

    def stop(self) -> None:
        """Halt the arm (the onboard controller's Stop, as Ctrl-C does in
        ``toss_cycles``) and the base (its target set to where it is), and make the
        skill in progress stop commanding motion."""
        self.stop_event.set()
        try:
            self._halt_arm()
        finally:
            base = self.real_env.interface.base_interface
            base.execute_action(base.get_base_state())

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
        self._skills.stop_event.clear()

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self._skills._motion.release()  # pylint: disable=protected-access
        if kind is not None and issubclass(kind, ArmMotionAborted):
            # A KeyboardInterrupt would kill the RPC server's connection thread;
            # the caller gets an ordinary error instead.
            raise SkillStopped(str(value)) from value


def _toss_result(
    tossed: bool, landed: list[float] | None, in_bin: bool, note: str | None
) -> dict[str, Any]:
    return {"tossed": tossed, "landed": landed, "in_bin": bool(in_bin), "note": note}


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}={value!r} is not a number.") from e
    if not math.isfinite(number):
        raise ValueError(f"{name}={value!r} is not a number.")
    return number


def _log_step(question: str) -> None:
    _logger.info("Step: %s", question.rstrip("?"))


def _with_one_attempt(cfg: DictConfig) -> DictConfig:
    """The skill is one pick attempt: a miss is the planner's to handle."""
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

    merged = OmegaConf.merge(cfg, {"pick": {"top_down": {"max_attempts": 1}}})
    assert isinstance(merged, DictConfig)
    return merged


def build_skills(cfg: DictConfig) -> TossingSkills:
    """The skills for ``cfg.mode`` (``fake`` or ``real``) from conf/skill_server.yaml."""
    # pylint: disable=import-outside-toplevel
    import functools

    import hydra
    import kinder

    from prpl_tidybot.toss_cycles import _build_servo, _fake_reading

    kinder.register_all_environments()
    pipeline = cfg.env.pipelines[cfg.mode]
    real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
    cube_source = hydra.utils.instantiate(
        pipeline.perceiver.target_source, _convert_="all"
    )
    bin_source = hydra.utils.instantiate(cfg.bin.pipelines[cfg.mode], _convert_="all")
    arm = real_env.interface.arm_interface
    halt: Callable[[], None] = lambda: None
    servo = None
    read: Callable[[float], float] = float
    if cfg.mode == "real":
        from prpl_tidybot.third_party.kinova import stop_arm_motion
        from prpl_tidybot.visual_servo.image_sources import KinovaWristCameraSource

        halt = stop_arm_motion
        servo = _build_servo(cfg, KinovaWristCameraSource(), None)
    else:
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
    )


class SkillManager(BaseManager):
    """The skill server's RPC manager (``Skills`` is the one shared object)."""


class SkillClientManager(BaseManager):
    """The client side of :class:`SkillManager`. A separate class because
    ``register`` writes a class-wide registry: registering ``Skills`` without a
    callable for the client must not replace the server's."""


SkillClientManager.register("Skills")


def serve(skills: TossingSkills, host: str, port: int, authkey: bytes) -> None:
    """Serve ``skills`` until interrupted; Ctrl-C halts the robot first."""
    SkillManager.register("Skills", callable=lambda: skills)
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
        self, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        """See the module docstring."""
        return dict(
            self._call(
                "toss",
                float(standoff),
                float(speed),
                float(release_ms),
                float(direction),
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


class FakeTossingRobot:  # pylint: disable=too-many-instance-attributes
    """The client's interface with no robot: deterministic, for planner tests.

    The same checks as the real skills (bounds, barrier, drivable box, bin) decide
    what is allowed; ``pick_outcomes`` (cycled) says whether each pick holds, and
    ``landing`` maps a toss ``(pose, speed, release_ms)`` to where the cube comes to
    rest (default: straight ahead of the base, ``0.9 + 0.55 *`` the parameters'
    position in their box, so the standoff that lands in the bin depends on them).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        robot: Sequence[float] = (1.326, -0.493, -3.132),
        cube: Sequence[float] | None = (0.8, 0.8, 0.0),
        bin_fit: BinFit | None = BinFit(-0.45, 0.4, 0.0, 0.40, 0.32, 0.19),
        workspace: Workspace | None = None,
        bounds: RealTossBounds = RealTossBounds(),
        geom: TopDownPickGeometry = TopDownPickGeometry(),
        pick_outcomes: Sequence[bool] = (True,),
        landing: (
            Callable[[Sequence[float], float, float], Sequence[float]] | None
        ) = None,
        in_bin_margin: float = 0.05,
        home: Sequence[float] = (1.326, -0.493, -3.132),
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.aimed_toss import Barrier

        self.robot = [float(v) for v in robot]
        self.cube = None if cube is None else [float(v) for v in cube]
        self.bin_fit = bin_fit
        self.workspace = workspace or Workspace(
            floor_bounds=(-1.5, 1.7, -1.5, 1.5), barrier=Barrier(), clearance=0.05
        )
        self.bounds = bounds
        self.geom = geom
        self._outcomes = list(pick_outcomes) or [True]
        self._picks = 0
        self._landing = landing or _default_landing(bounds)
        self.in_bin_margin = in_bin_margin
        self.home = [float(v) for v in home]
        self.holding = False
        self.gripper_command = 0.0
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def observe(self) -> dict[str, Any]:
        """As the real ``observe``."""
        self.calls.append(("observe", ()))
        barrier = self.workspace.barrier
        cube = None if self.holding or self.cube is None else list(self.cube)
        return {
            "robot": list(self.robot),
            "cube": cube,
            "bin": None if self.bin_fit is None else self.bin_fit.as_dict(),
            "holding": self.holding,
            "gripper": 0.49 if self.holding else (1.0 if self.gripper_command else 0.0),
            "gripper_command": float(self.gripper_command),
            "cube_upright": cube is not None,
            "cube_in_bin": bool(
                cube is not None
                and self.bin_fit is not None
                and cube_in_bin(cube[:2], self.bin_fit, self.in_bin_margin)
            ),
            "barrier": None if barrier is None else barrier.as_dict(),
            "barrier_x": None if barrier is None else float(barrier.x),
        }

    def can_pick(self) -> dict[str, Any]:
        """As the real ``can_pick`` (without IK: the fake arm reaches anything the
        standoff allows)."""
        # pylint: disable=import-outside-toplevel
        from prpl_tidybot.toss_cycles import pick_problem

        self.calls.append(("can_pick", ()))
        if self.holding:
            return {"feasible": False, "note": "already holding the cube"}
        if self.cube is None:
            return {"feasible": False, "note": "no cube found"}
        problem = pick_problem(
            self.robot,
            {"x": self.cube[0], "y": self.cube[1]},
            self.geom,
            self.workspace.floor_bounds,
            self.workspace,
        )
        return {"feasible": problem is None, "note": problem}

    def pick(self) -> dict[str, Any]:
        """Drive to the standoff and grasp; held per ``pick_outcomes``."""
        self.calls.append(("pick", ()))
        check = self.can_pick()
        self.calls.pop()
        if not check["feasible"]:
            return {"held": False, "gripper": 0.0, "note": check["note"]}
        assert self.cube is not None
        self.robot = list(
            standoff_pose(
                self.robot,
                self.cube[:2],
                self.geom,
                self.workspace.floor_bounds,
                self.workspace.valid(),
            )
        )
        held = self._outcomes[self._picks % len(self._outcomes)]
        self._picks += 1
        self.holding = held
        self.gripper_command = 1.0 if held else 0.0
        return {
            "held": held,
            "gripper": 0.49 if held else 0.0,
            "note": None if held else "gripper closed on nothing",
        }

    def toss(
        self, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        """As the real ``toss``: the same ``ValueError``s; the cube lands per
        ``landing``."""
        self.calls.append(("toss", (standoff, speed, release_ms, direction)))
        standoff, speed, release_ms = (
            _number(standoff, "standoff"),
            _number(speed, "speed"),
            _number(release_ms, "release_ms"),
        )
        direction = _number(direction, "direction")
        self.bounds.validate(standoff, speed, release_ms)
        if not self.holding:
            return _toss_result(False, None, False, "not holding the cube")
        if self.bin_fit is None:
            return _toss_result(False, None, False, "no bin found")
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
            landed,
            cube_in_bin(landed, self.bin_fit, self.in_bin_margin),
            None,
        )

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
