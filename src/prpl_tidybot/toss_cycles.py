"""Repeat pick -> arm home -> drive to the toss pose -> toss, for scripts/toss_cycles.py.

Each cycle localizes the cube (``floor-cube-o1``'s ceiling localizer in real mode),
picks it with the shelf3d PickController (:class:`PickSkillAgent`, executed through the
usual ``Runner``), moves the arm to its home pose still holding the cube, drives the base
to ``toss_pose``, and throws. Every action waits for a stdin confirmation (the pick's
through the plan preview in real mode). Whenever the run stops - all cycles done, a step
declined, no cube found, a failure - the robot is reset: after one more confirmation the
arm retracts and the base drives back to ``toss_pose``, and the cube is localized once
more so the last throw's landing is recorded. ``cycles.json`` in the log dir holds each
cycle's cube pose before the pick and where its throw landed relative to ``toss_pose``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import kinder
import numpy as np
from kinder_bilevel_planning.agent import AgentFailure
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

import prpl_tidybot.config_resolvers  # noqa: F401  # pylint: disable=unused-import
from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.pick_and_toss import PickSkillAgent, PromptFn, _toss
from prpl_tidybot.pipeline import _resolve_log_dir
from prpl_tidybot.preview import planned_trajectory_from_agent, preview_or_abort
from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD, RETRACT_ARM_CONF
from prpl_tidybot.third_party.ik_solver import IKSolver
from prpl_tidybot.top_down_pick import (
    CARRY_ARM_CONF,
    TopDownPickGeometry,
    arm_frame_target,
    forward_kinematics,
    geometry_from_config,
    grasp_succeeded,
    standoff_pose,
    top_down_joint_targets,
)

_logger = logging.getLogger(__name__)

_MAX_DRIVE_TICKS = 600


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


def cube_pose(state: ObjectCentricState) -> dict[str, float]:
    """The perceived cube's map-frame x, y and yaw."""
    cube = state.get_object_from_name("cube0")
    qz, qw = state.get(cube, "pose_qz"), state.get(cube, "pose_qw")
    return {
        "x": float(state.get(cube, "pose_x")),
        "y": float(state.get(cube, "pose_y")),
        "yaw_deg": math.degrees(2 * math.atan2(qz, qw)),
    }


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


def run_toss_cycles(
    cfg: DictConfig,
    log_dir: Path | str | None = None,
    prompt_fn: PromptFn = input,
) -> TossCyclesSummary:
    """Run ``cfg.cycles`` pick/home/drive/toss cycles (conf/toss_cycles.yaml)."""
    if cfg.mode == "sim":
        raise ValueError(
            "toss_cycles needs a toss, which the kinematic sim does not have."
        )
    interactive = (
        cfg.mode == "real" if cfg.get("interactive") is None else bool(cfg.interactive)
    )
    resolved_log_dir = _resolve_log_dir(log_dir)
    requested = int(cfg.cycles)
    toss_pose = [float(v) for v in cfg.toss_pose]

    kinder.register_all_environments()
    pipeline = cfg.env.pipelines[cfg.mode]
    real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
    perceiver = hydra.utils.instantiate(pipeline.perceiver, _convert_="all")
    plan_executor = hydra.utils.instantiate(pipeline.plan_executor, _convert_="all")
    env_models = build_planner_env_models(
        cfg.env.env_name,
        hydra.utils.instantiate(cfg.env.make_kwargs, _recursive_=True, _convert_="all"),
        hydra.utils.instantiate(
            cfg.env.env_model_kwargs, _recursive_=True, _convert_="all"
        ),
    )
    home_joints = (
        IKSolver()  # type: ignore[no-untyped-call]
        .solve(  # type: ignore[no-untyped-call]
            np.array(cfg.arm_home.position),
            np.array(cfg.arm_home.quat_xyzw),
            RETRACT_ARM_CONF,
        )
        .tolist()
    )
    arm: ArmInterface = real_env.interface.arm_interface
    cube_source = hydra.utils.instantiate(
        pipeline.perceiver.target_source, _convert_="all"
    )
    geom = geometry_from_config(cfg.pick.get("top_down"))
    wrist = _WristRecorder(
        cfg.mode == "real" and bool(cfg.pick.top_down.get("record_wrist", True)),
        resolved_log_dir,
    )

    def confirm(question: str) -> None:
        if not interactive:
            return
        answer = prompt_fn(f"\n{question} [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            raise _Declined(question)

    records: list[dict[str, Any]] = []
    completed = 0
    finish_reason = f"completed {requested}/{requested} cycles"
    final_pose: tuple[float, float, float] | None = None
    try:
        for cycle in range(1, requested + 1):
            record: dict[str, Any] = {"cycle": cycle, "tossed": False}
            records.append(record)
            _logger.info("Cycle %d/%d: localizing the cube.", cycle, requested)
            if cfg.pick.method == "top_down":
                failure = _pick_top_down(
                    cfg, real_env, cube_source, arm, geom, confirm, cycle, record, wrist
                )
            else:
                failure = _pick_kinder(
                    cfg,
                    real_env,
                    perceiver,
                    plan_executor,
                    env_models,
                    resolved_log_dir,
                    interactive,
                    prompt_fn,
                    confirm,
                    cycle,
                    record,
                )
            if cycle > 1 and "cube_before_pick" in record:
                records[-2]["landing"] = _landing(record["cube_before_pick"], toss_pose)
            if failure is not None:
                finish_reason = failure
                break
            record["picked"] = True
            if cfg.get("skip_toss"):
                finish_reason = "picked (toss skipped)"
                break

            confirm(f"Cycle {cycle}: move the arm to home, holding the cube?")
            _move_arm(arm, home_joints, arm.get_gripper_state(), cfg.arm_home)

            confirm(
                f"Cycle {cycle}: Drive to the toss pose ({toss_pose[0]:+.3f}, "
                f"{toss_pose[1]:+.3f}, {math.degrees(toss_pose[2]):+.0f} deg)?"
            )
            _drive_to(real_env, toss_pose)

            tossed, reason, info, _ = _toss(
                cfg, arm, resolved_log_dir, interactive, prompt_fn
            )
            record["toss"] = info
            if not tossed:
                raise _Declined(reason)
            record["tossed"] = True
            completed += 1
        else:
            finish_reason = f"completed {completed}/{requested} cycles"
    except _Declined as e:
        finish_reason = f"declined: {e}"
    finally:
        final_pose = _reset(
            cfg, real_env, perceiver, arm, toss_pose, records, interactive, prompt_fn
        )
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
                ),
                encoding="utf-8",
            )
        wrist.close()
        cube_source.close()
        real_env.close()
    _logger.info("Toss cycles: %s.", finish_reason)
    return TossCyclesSummary(
        mode=cfg.mode,
        completed=completed,
        requested=requested,
        finish_reason=finish_reason,
        cycles=records,
        final_base_pose=final_pose,
    )


def _confirm_pick(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    agent: PickSkillAgent,
    state: ObjectCentricState,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
    confirm: Any,
    cycle: int,
) -> None:
    """In real mode the plan preview (video + prompt) is the confirmation; otherwise a
    plain prompt."""
    preview_cfg = cfg.get("preview")
    if (
        cfg.mode == "real"
        and preview_cfg is not None
        and bool(preview_cfg.get("enabled"))
        and log_dir is not None
    ):
        shadow_sim = hydra.utils.instantiate(
            cfg.env.pipelines.sim.real_env, _convert_="all"
        )
        planned_states, planned_actions = planned_trajectory_from_agent(agent)
        try:
            preview_or_abort(
                planned_states=planned_states,
                planned_actions=planned_actions,
                shadow_sim=shadow_sim,
                log_dir=log_dir / f"cycle_{cycle:02d}",
                seed=int(cfg.seed),
                fps=int(preview_cfg.get("fps", 10)),
                max_frames=preview_cfg.get("max_frames", 50),
                prompt_fn=prompt_fn,
                floor_bounds=(
                    tuple(preview_cfg["floor_bounds"])
                    if preview_cfg.get("floor_bounds") is not None
                    else None
                ),
                base_margin=float(preview_cfg.get("base_margin", 0.0)),
            )
        except AgentFailure as e:
            raise _Declined(f"cycle {cycle} pick preview: {e}") from e
        finally:
            shadow_sim.close()
        return
    cube = cube_pose(state)
    del interactive
    confirm(
        f"Cycle {cycle}: execute the pick of the cube at ({cube['x']:+.3f}, "
        f"{cube['y']:+.3f}, yaw {cube['yaw_deg']:+.0f} deg)?"
    )


def _move_arm(
    arm: ArmInterface, goal: list[float], gripper: float, arm_cfg: DictConfig
) -> bool:
    """Stream ``goal`` until every joint is within tolerance (wrapped) or time runs
    out; returns whether it got there."""
    for _ in range(int(arm_cfg.max_ticks)):
        arm.execute_action(goal, gripper)
        err = np.array(arm.get_arm_state()) - np.array(goal)
        if np.all(
            np.abs((err + np.pi) % (2 * np.pi) - np.pi) < float(arm_cfg.tolerance)
        ):
            return True
        time.sleep(POLICY_CONTROL_PERIOD)
    _logger.warning(
        "Arm did not reach its goal within %d ticks.", int(arm_cfg.max_ticks)
    )
    return False


def _logged_move(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    arm: ArmInterface,
    name: str,
    goal: list[float],
    gripper: float,
    arm_cfg: DictConfig,
    geom: TopDownPickGeometry,
    moves: list[dict[str, Any]],
) -> None:
    """`_move_arm`, then record planned vs measured joints and pinch-point heights:
    the arm's torque loop is compliant, so where it settles (sag, or stopped by the
    floor) can differ from the target."""
    reached = _move_arm(arm, goal, gripper, arm_cfg)
    measured = list(arm.get_arm_state())
    move = {
        "name": name,
        "reached": reached,
        "planned_deg": np.round(np.degrees(goal), 2).tolist(),
        "measured_deg": np.round(np.degrees(measured), 2).tolist(),
        "planned_pinch_z": float(forward_kinematics(goal, geom)[0][2]),
        "measured_pinch_z": float(forward_kinematics(measured, geom)[0][2]),
    }
    moves.append(move)
    _logger.info(
        "Move %s: pinch z planned %+.4f measured %+.4f (floor %+.3f); max joint error "
        "%.2f deg.",
        name,
        move["planned_pinch_z"],
        move["measured_pinch_z"],
        geom.floor_z,
        float(
            np.max(
                np.abs(
                    (np.array(move["measured_deg"]) - move["planned_deg"] + 180) % 360
                    - 180
                )
            )
        ),
    )


def _drive_to(real_env: Any, pose: list[float]) -> tuple[float, float, float]:
    """Drive the base to the map-frame pose with the kinematic3d executor (as
    hardware_tests/test_base_map_target.py does); returns the reached pose.

    Uses a robot-only perceiver: driving needs no cube, and must still work when the
    cube cannot be found (the reset after a failed localization).
    """
    perceiver = PrplLab3DPerceiver()
    obs, info = real_env.reset()
    state = perceiver.step(obs, info)
    robot = state.get_object_from_name("robot")
    delta = np.zeros(11)
    delta[0] = pose[0] - state.get(robot, "pos_base_x")
    delta[1] = pose[1] - state.get(robot, "pos_base_y")
    dtheta = pose[2] - state.get(robot, "pos_base_rot")
    delta[2] = (dtheta + math.pi) % (2 * math.pi) - math.pi
    if math.hypot(delta[0], delta[1]) < 0.01 and abs(delta[2]) < 0.02:
        reached = (
            obs.map_base_pose
        )  # already there; an empty move is not a base segment
        return (float(reached.x), float(reached.y), float(reached.theta()))
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, delta)])
    for _ in range(_MAX_DRIVE_TICKS):
        if executor.done(state):
            break
        real_action, _ = executor.step(state)
        obs, _, _, _, info = real_env.step(real_action)
        state = perceiver.step(obs, info)
    reached = obs.map_base_pose
    return (float(reached.x), float(reached.y), float(reached.theta()))


def _reset(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    real_env: Any,
    perceiver: Any,
    arm: ArmInterface,
    toss_pose: list[float],
    records: list[dict[str, Any]],
    interactive: bool,
    prompt_fn: PromptFn,
) -> tuple[float, float, float] | None:
    """Retract the arm and drive back to the toss pose (after a confirmation), then
    localize the cube once more to record where the last throw landed."""
    if interactive:
        answer = prompt_fn(
            "\nReset: retract the arm and drive back to the toss pose? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            _logger.info("Reset declined; leaving the robot where it is.")
            return None
    _move_arm(arm, list(RETRACT_ARM_CONF), arm.get_gripper_state(), cfg.arm_home)
    reached = _drive_to(real_env, toss_pose)
    if records and records[-1].get("tossed") and "landing" not in records[-1]:
        try:
            obs, info = real_env.reset()
            records[-1]["landing"] = _landing(
                cube_pose(perceiver.reset(obs, info)), toss_pose
            )
        except RuntimeError as e:
            records[-1]["landing"] = {"error": str(e)}
    return reached


def _pick_kinder(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    real_env: Any,
    perceiver: Any,
    plan_executor: Any,
    env_models: Any,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
    confirm: Any,
    cycle: int,
    record: dict[str, Any],
) -> str | None:
    """The kinder shelf3d PickController, planned and executed through the Runner.

    Returns a stop reason, or None once the cube is held.
    """
    agent = PickSkillAgent(
        env_models,
        int(cfg.seed) + cycle - 1,
        max_skill_horizon=cfg.agent.max_skill_horizon,
        max_attempts=cfg.pick.max_attempts,
    )
    runner: Runner = Runner(
        real_env=real_env, perceiver=perceiver, agent=agent, plan_executor=plan_executor
    )
    try:
        state = runner.reset(seed=int(cfg.seed) + cycle - 1)
    except RuntimeError as e:
        return f"{e}; move the cube into the drivable area and rerun"
    except AgentFailure as e:
        return f"pick_failure: {e}; move the cube and rerun"
    record["cube_before_pick"] = cube_pose(state)
    _confirm_pick(cfg, agent, state, log_dir, interactive, prompt_fn, confirm, cycle)
    try:
        runner.step()
    except ExecutionFailure as e:
        return f"executor_failure during the pick: {e}"
    return None


def _localize(source: Any) -> dict[str, float]:
    x, y, _ = source.get_target()
    return {"x": float(x), "y": float(y), "yaw_deg": math.degrees(source.get_yaw())}


def _pick_top_down(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    real_env: Any,
    cube_source: Any,
    arm: ArmInterface,
    geom: TopDownPickGeometry,
    confirm: Any,
    cycle: int,
    record: dict[str, Any],
    wrist: "_WristRecorder",
) -> str | None:
    """Jimmy Wu's top-down pick (prpl_tidybot.top_down_pick) on the localized cube.

    Drive to a standoff facing the cube, re-localize, then for up to ``max_attempts``:
    open, hover 10 cm above, descend, close, and read the gripper. Returns a stop
    reason, or None once the cube is held (arm at its carry pose).
    """
    td = cfg.pick.top_down
    try:
        cube = _localize(cube_source)
    except RuntimeError as e:
        return f"{e}; move the cube into the drivable area and rerun"
    record["cube_before_pick"] = cube
    obs, _ = real_env.reset()
    robot = (obs.map_base_pose.x, obs.map_base_pose.y, obs.map_base_pose.theta())
    floor = tuple(cfg.preview.floor_bounds)
    try:
        stand = standoff_pose(robot, (cube["x"], cube["y"]), geom, floor)
    except ValueError as e:
        return f"{e}; move the cube and rerun"
    confirm(
        f"Cycle {cycle}: Drive to the pick standoff ({stand[0]:+.3f}, {stand[1]:+.3f}, "
        f"{math.degrees(stand[2]):+.0f} deg), {geom.standoff:.2f} m from the cube at "
        f"({cube['x']:+.3f}, {cube['y']:+.3f})?"
    )
    base = _drive_to(real_env, list(stand))
    attempts = int(td.get("max_attempts", 3))
    for attempt in range(1, attempts + 1):
        try:
            cube = _localize(cube_source)  # the cube may have been nudged
        except RuntimeError as e:
            return f"{e}; move the cube and rerun"
        ax, ay = arm_frame_target(base, (cube["x"], cube["y"]), geom)
        yaw_rel = math.radians(cube["yaw_deg"]) - base[2]
        try:
            hover, grasp = top_down_joint_targets(
                ax, ay, yaw_rel, geom, seed=arm.get_arm_state()
            )
        except ValueError as e:
            return f"pick_failure: {e}"
        moves: list[dict[str, Any]] = []
        record.setdefault("pick_attempts", []).append(
            {
                "attempt": attempt,
                "cube": cube,
                "arm_target_xy": [ax, ay],
                "moves": moves,
            }
        )
        confirm(
            f"Cycle {cycle}: Pick attempt {attempt}/{attempts}: open, arm to 10 cm "
            f"above the cube (arm frame {ax:+.3f}, {ay:+.3f}), descend, close?"
        )
        wrist.start(f"cycle_{cycle:02d}_pick_{attempt}")
        _set_gripper(arm, 0.0, td)
        _logged_move(arm, "hover", hover, 0.0, cfg.arm_home, geom, moves)
        _logged_move(arm, "grasp", grasp, 0.0, cfg.arm_home, geom, moves)
        _set_gripper(arm, float(td.get("close_position", 1.0)), td)
        reading = arm.get_gripper_state()
        if cfg.mode == "fake":  # the fake gripper closes on nothing; pretend it holds
            reading = float(td.get("fake_grasp_reading", 0.49))
        held = grasp_succeeded(reading, geom)
        record["pick_attempts"][-1].update({"gripper": reading, "held": held})
        _logger.info(
            "Pick attempt %d: gripper %.3f -> %s.",
            attempt,
            reading,
            "held" if held else "missed",
        )
        if held:
            grip = arm.get_gripper_state()
            _logged_move(arm, "lift", hover, grip, cfg.arm_home, geom, moves)
            carry = list(CARRY_ARM_CONF)
            _logged_move(arm, "carry", carry, grip, cfg.arm_home, geom, moves)
            wrist.stop()
            return None
        _set_gripper(arm, 0.0, td)
        _logged_move(arm, "retreat", hover, 0.0, cfg.arm_home, geom, moves)
        wrist.stop()
    return (
        f"pick failed after {attempts} attempts (gripper never closed on the cube); "
        "move the cube and rerun"
    )


def _set_gripper(arm: ArmInterface, position: float, td: Any) -> None:
    """Command the gripper and wait until it stops moving."""
    arm.execute_action(arm.get_arm_state(), position)
    deadline = time.monotonic() + float(td.get("gripper_timeout_s", 4.0))
    last = arm.get_gripper_state()
    while time.monotonic() < deadline:
        time.sleep(0.15)
        arm.execute_action(arm.get_arm_state(), position)
        now = arm.get_gripper_state()
        if abs(now - last) < 0.005:
            return
        last = now


class _WristRecorder:
    """Records the Kinova wrist camera during each pick attempt (real mode only)."""

    def __init__(self, enabled: bool, log_dir: Path | None) -> None:
        self._enabled = enabled and log_dir is not None
        self._log_dir = log_dir
        self._source: Any = None
        self._thread: Any = None
        self._frames: list[Any] = []
        self._name = ""
        self._running = False

    def start(self, name: str) -> None:
        """Begin recording a clip called `name`."""
        if not self._enabled:
            return
        import threading  # pylint: disable=import-outside-toplevel

        from prpl_tidybot.visual_servo.image_sources import (  # pylint: disable=import-outside-toplevel
            KinovaWristCameraSource,
        )

        if self._source is None:
            self._source = KinovaWristCameraSource()
        self._frames, self._name, self._running = [], name, True

        def grab() -> None:
            while self._running:
                img = self._source.get_image()
                if img is not None:
                    self._frames.append(img)
                time.sleep(0.1)

        self._thread = threading.Thread(target=grab, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and write the clip to <log_dir>/<name>_wrist.mp4."""
        if not self._enabled or not self._running:
            return
        self._running = False
        self._thread.join(timeout=2)
        if self._frames and self._log_dir is not None:
            from prpl_tidybot.video import (  # pylint: disable=import-outside-toplevel
                write_mp4,
            )

            write_mp4(self._frames, self._log_dir / f"{self._name}_wrist.mp4", fps=10)

    def close(self) -> None:
        """Release the camera."""
        self.stop()
        if self._source is not None:
            self._source.close()
