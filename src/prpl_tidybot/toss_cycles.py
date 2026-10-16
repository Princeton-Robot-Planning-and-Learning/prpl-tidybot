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

import dataclasses
import functools
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import hydra
import kinder
import numpy as np
from kinder_bilevel_planning.agent import AgentFailure
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

import prpl_tidybot.config_resolvers  # noqa: F401  # pylint: disable=unused-import
from prpl_tidybot.interfaces.arm_interface import ArmInterface, ArmMotionAborted
from prpl_tidybot.pick_and_toss import PickSkillAgent, PromptFn, _toss
from prpl_tidybot.pipeline import _resolve_log_dir
from prpl_tidybot.preview import planned_trajectory_from_agent, preview_or_abort
from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD, RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    CARRY_ARM_CONF,
    TopDownPickGeometry,
    arm_frame_target,
    cube_closing_yaw,
    forward_kinematics,
    geometry_from_config,
    grasp_search_bounds,
    grasp_succeeded,
    pointing_down_joints,
    standoff_pose,
    top_down_joint_targets,
)
from prpl_tidybot.visual_servo.image_sources import ImageSource, KinovaWristCameraSource
from prpl_tidybot.visual_servo.top_down_cube import (
    TopDownCubeServo,
    TopDownServoParams,
    TopDownServoResult,
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


def _asked_to_move(
    ask_to_move: PromptFn | None, asks: int, record: dict[str, Any], question: str
) -> bool:
    """Ask the person to move the cube (once per remaining ask); False when there is
    nobody to ask, no asks left, or they typed 'stop'."""
    if ask_to_move is None or record.get("cube_moved_by_person", 0) >= asks:
        return False
    if ask_to_move(question).strip().lower() == "stop":
        return False
    record["cube_moved_by_person"] = record.get("cube_moved_by_person", 0) + 1
    return True


def _with_grid_setting(cfg: DictConfig, cycle: int) -> DictConfig:
    """The run's config with cycle `cycle`'s toss setting from ``toss.grid``, a list of
    ``[max_vel, gripper_release_ms]`` thrown in turn (a calibration sweep); unchanged
    without a grid."""
    grid = cfg.toss.get("grid")
    if not grid:
        return cfg
    speed, release = grid[(cycle - 1) % len(grid)]
    setting = {"toss": {"max_vel": float(speed), "gripper_release_ms": int(release)}}
    merged = OmegaConf.merge(cfg, setting)
    assert isinstance(merged, DictConfig)
    return merged


def _put_back_if_in_bin(
    cfg: DictConfig, cube_source: Any, record: dict[str, Any], prompt_fn: PromptFn
) -> None:
    """A thrown cube that ends up in (or against) the bin is not picked: the hover is
    below the rim. The person puts it back on the floor (``bin_stop_box`` is the bin's
    footprint plus a margin, [x_min, x_max, y_min, y_max])."""
    box = cfg.get("bin_stop_box")
    if box is None:
        return
    try:
        cube = _localize(cube_source)
    except RuntimeError:
        return  # the pick's own localization reports a missing cube
    if not (box[0] <= cube["x"] <= box[1] and box[2] <= cube["y"] <= box[3]):
        return
    record["cube_in_bin"] = {"x": cube["x"], "y": cube["y"]}
    _logger.info("The cube is in the bin at (%+.3f, %+.3f).", cube["x"], cube["y"])
    prompt_fn(
        f"\nThe cube is in the bin at ({cube['x']:+.3f}, {cube['y']:+.3f}): put it on "
        "the floor clear of the bin, then press Enter. "
    )


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


def run_toss_cycles(  # pylint: disable=too-many-locals
    cfg: DictConfig,
    log_dir: Path | str | None = None,
    prompt_fn: PromptFn = input,
    wrist_camera: Callable[[ArmInterface], ImageSource] | None = None,
) -> TossCyclesSummary:
    """Run ``cfg.cycles`` pick/drive/toss/retract cycles (conf/toss_cycles.yaml).

    ``wrist_camera`` builds the wrist image source from the arm; by default the
    Kinova wrist camera in real mode and none otherwise (no servo)."""
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
    # Only the kinder pick plans; its env models are large and never freed, so a
    # top-down run does not build them.
    plan_executor: Any = None
    env_models: Any = None
    if cfg.pick.method != "top_down":
        plan_executor = hydra.utils.instantiate(pipeline.plan_executor, _convert_="all")
        env_models = build_planner_env_models(
            cfg.env.env_name,
            hydra.utils.instantiate(
                cfg.env.make_kwargs, _recursive_=True, _convert_="all"
            ),
            hydra.utils.instantiate(
                cfg.env.env_model_kwargs, _recursive_=True, _convert_="all"
            ),
        )
    arm: ArmInterface = real_env.interface.arm_interface
    geom = geometry_from_config(cfg.pick.get("top_down"))
    cube_source = hydra.utils.instantiate(
        pipeline.perceiver.target_source,
        _convert_="all",
        **_cube_source_kwargs(cfg, pipeline.perceiver.target_source, resolved_log_dir),
    )
    source: ImageSource | None = None
    if wrist_camera is not None:
        source = wrist_camera(arm)
    elif cfg.mode == "real":
        source = KinovaWristCameraSource()
    wrist = _WristRecorder(
        source if bool(cfg.pick.top_down.get("record_wrist", True)) else None,
        resolved_log_dir,
    )
    servo = _build_servo(cfg, source, resolved_log_dir)
    ceiling = _CeilingRecorder(cfg, resolved_log_dir)

    once = str(cfg.get("confirm", "once")) == "once"
    confirmed = {"done": False}

    def asking() -> bool:
        """Whether to prompt now: every step, or (confirm=once) only before the first."""
        return interactive and not (once and confirmed["done"])

    def confirm(question: str) -> None:
        ceiling.label = question
        if not asking():
            _logger.info("Step: %s", question.rstrip("?"))
            return
        if once:
            question = (
                f"Run all {requested} cycle(s) (pick, drive, toss, retract; asking only "
                f"if the cube must be moved) without asking again? First: "
                f"{question}"
            )
        answer = prompt_fn(f"\n{question} [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            raise _Declined(question)
        confirmed["done"] = True

    records: list[dict[str, Any]] = []
    completed = 0
    finish_reason = f"completed {requested}/{requested} cycles"
    final_pose: tuple[float, float, float] | None = None
    aborted = False
    try:
        for cycle in range(1, requested + 1):
            record: dict[str, Any] = {"cycle": cycle, "tossed": False}
            records.append(record)
            _logger.info("Cycle %d/%d: localizing the cube.", cycle, requested)
            if cycle > 1:
                _put_back_if_in_bin(cfg, cube_source, record, prompt_fn)
            if cfg.pick.method == "top_down":
                failure = _pick_top_down(
                    cfg,
                    real_env,
                    cube_source,
                    arm,
                    geom,
                    confirm,
                    cycle,
                    record,
                    wrist,
                    toss_pose,
                    servo,
                    prompt_fn if interactive else None,
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

            confirm(
                f"Cycle {cycle}: Drive to the toss pose ({toss_pose[0]:+.3f}, "
                f"{toss_pose[1]:+.3f}, {math.degrees(toss_pose[2]):+.0f} deg)?"
            )
            _drive_to(real_env, toss_pose)

            cycle_cfg = _with_grid_setting(cfg, cycle)
            ask_toss = interactive and bool(cfg.toss.get("confirm_each", False))
            tossed, reason, info, _ = _toss(
                cycle_cfg, arm, resolved_log_dir, asking() or ask_toss, prompt_fn
            )
            record["toss"] = info
            if not tossed:
                raise _Declined(reason)
            record["tossed"] = True
            completed += 1
            # The toss leaves the arm raised in its release pose.
            confirm(f"Cycle {cycle}: retract the arm after the toss?")
            _retract(cfg, arm)
            settle = float(cfg.get("settle_after_toss_s", 5.0))
            if settle > 0:
                _logger.info("Waiting %.1f s for the cube to settle.", settle)
                time.sleep(settle)
        else:
            finish_reason = f"completed {completed}/{requested} cycles"
    except _Declined as e:
        finish_reason = f"declined: {e}"
    except ArmMotionAborted as e:
        # The arm was stopped mid-move; command nothing more (the arm-server RPC is
        # mid-call and the operator wants motion to end). Rerun to reset.
        finish_reason = f"aborted: {e}"
        aborted = True
    finally:
        if not aborted:
            final_pose = _reset(
                cfg,
                real_env,
                perceiver,
                arm,
                toss_pose,
                records,
                asking(),
                prompt_fn,
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
        if source is not None and hasattr(source, "close"):
            source.close()
        ceiling.close()
        cube_source.close()
        if not aborted:
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


def _cube_source_kwargs(
    cfg: DictConfig, target_cfg: Any, log_dir: Path | None = None
) -> dict[str, Any]:
    """The ceiling localizer searches everywhere the arm can grasp from inside the
    drivable box, not just the box itself (a bin just past its edge), and saves the
    frames of any missed localization beside the run's logs."""
    if not str(target_cfg["_target_"]).endswith("CeilingCubeTargetSource"):
        return {}
    bounds = grasp_search_bounds(
        tuple(cfg.preview.floor_bounds), geometry_from_config(cfg.pick.get("top_down"))
    )
    kwargs: dict[str, Any] = {"floor_bounds": list(bounds)}
    if log_dir is not None:
        kwargs["failure_dir"] = str(Path(log_dir) / "localization_failures")
    return kwargs


def _retract(cfg: DictConfig, arm: ArmInterface) -> None:
    """Fold the arm to its retract pose under the compliant loop: it needs no
    precision, and only the grasp itself runs under the stiff onboard controller."""
    _move_arm(arm, list(RETRACT_ARM_CONF), arm.get_gripper_state(), cfg.arm_home)


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
    _retract(cfg, arm)
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
    toss_pose: list[float],
    servo: TopDownCubeServo | None = None,
    ask_to_move: PromptFn | None = None,
) -> str | None:
    """Jimmy Wu's top-down pick (prpl_tidybot.top_down_pick) on the localized cube.

    For up to ``max_attempts``: localize the cube, drive to a standoff facing it,
    re-localize, then open, hover 10 cm above, descend, close, and read the gripper.
    After a miss the arm retracts and the base returns to the toss pose, where the
    ceiling cameras see the (possibly nudged) cube without the robot over it. With
    ``ask_to_move``, a miss instead retracts and asks the person to move the cube (up to
    ``ask_to_move_after_miss`` times), then tries again. Returns a stop reason, or None
    once the cube is held (arm at its carry pose).
    """
    td = cfg.pick.top_down
    floor = tuple(cfg.preview.floor_bounds)
    attempts = int(td.get("max_attempts", 3))
    asks = int(td.get("ask_to_move_after_miss", 3)) if ask_to_move is not None else 0
    for attempt in range(1, attempts + asks + 1):
        try:
            cube = _localize(cube_source)
        except RuntimeError as e:
            if not _asked_to_move(
                ask_to_move,
                asks,
                record,
                f"\nCycle {cycle}: I can't find the cube ({e}). Please put it in view "
                "on the floor, clear of the robot and of the camera overlap strip "
                "(y -0.15..0.18), then press Enter, or type 'stop' to end the run: ",
            ):
                return f"{e}; move the cube into the drivable area and rerun"
            continue
        record.setdefault("cube_before_pick", cube)
        obs, _ = real_env.reset()
        robot = (obs.map_base_pose.x, obs.map_base_pose.y, obs.map_base_pose.theta())
        try:
            stand = standoff_pose(robot, (cube["x"], cube["y"]), geom, floor)
        except ValueError as e:
            return f"{e}; move the cube and rerun"
        confirm(
            f"Cycle {cycle}: Drive to the pick standoff ({stand[0]:+.3f}, "
            f"{stand[1]:+.3f}, {math.degrees(stand[2]):+.0f} deg), "
            f"{geom.standoff:.2f} m from the cube at ({cube['x']:+.3f}, "
            f"{cube['y']:+.3f})?"
        )
        base = _drive_to(real_env, list(stand))
        try:
            cube = _localize(cube_source)  # corrects for where the base parked
        except RuntimeError as e:
            return f"{e}; move the cube and rerun"
        # The more confident camera's estimate first; on a miss, the other cameras'
        # (they disagree by ~3 cm where they overlap).
        targets = [(cube, "best camera")] + [
            ({"x": x, "y": y, "yaw_deg": math.degrees(yaw)}, "other camera")
            for x, y, yaw in cube_source.alternates()
        ]
        for k, (cube, label) in enumerate(targets):
            if k > 0:
                confirm(
                    f"Cycle {cycle}: Missed. Try the other ceiling camera's estimate "
                    f"of the cube, ({cube['x']:+.3f}, {cube['y']:+.3f})?"
                )
            held, failure = _grasp_at(
                cfg,
                cube,
                label,
                base,
                arm,
                geom,
                confirm,
                cycle,
                attempt,
                record,
                wrist,
                servo,
                attempts,
                f"{attempt}" + (f"_{k + 1}" if k else ""),
            )
            if failure is not None:
                return failure
            if held:
                wrist.stop()
                return None
        wrist.stop()
        if ask_to_move is not None and record.get("cube_moved_by_person", 0) < asks:
            _retract(cfg, arm)
            if not _asked_to_move(
                ask_to_move,
                asks,
                record,
                f"\nCycle {cycle}: the pick missed and the arm is retracted. Please "
                "move the cube (clear of the camera overlap strip, y -0.15..0.18), then "
                "press Enter, or type 'stop' to end the run: ",
            ):
                return "pick missed; the operator stopped the run"
            continue
        if attempt < attempts:
            confirm(
                f"Cycle {cycle}: Missed. Retract the arm and drive back to the toss "
                "pose to re-localize the cube?"
            )
            _retract(cfg, arm)
            _drive_to(real_env, list(toss_pose))
    return (
        f"pick failed after {attempts} attempts (gripper never closed on the cube); "
        "move the cube and rerun"
    )


# An onboard-controller move that ends further than this from its goal has hit
# something (the floor, the cube) or faulted; the toss uses the same check.
_ANGULAR_ARRIVAL_DEG = 1.0


class _Mover:
    """How the grasp sequence moves the arm and gripper: under the onboard position
    controller (`_AngularMover`) or the compliant torque loop (`_CompliantMover`).
    One controller is used for a whole run; nothing switches mid-episode."""

    name = ""

    def __init__(
        self,
        arm: ArmInterface,
        td: Any,
        geom: TopDownPickGeometry,
        moves: list[dict[str, Any]],
    ) -> None:
        self.arm = arm
        self.td = td
        self.geom = geom
        self.moves = moves

    def joints(self) -> list[float]:
        """Where the arm is now (IK seeds, the servo's jump check and position)."""
        return list(self.arm.get_arm_state())

    def begin(self) -> None:
        """Take whatever control the controller needs."""

    def end(self) -> None:
        """Hand the arm back to the compliant loop, holding where it is."""

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        """Move to `goal`; record it; return whether it arrived. `correct` asks a
        controller that can settle short of its goal to correct onto it."""
        raise NotImplementedError

    def gripper(self, position: float) -> float:
        """Move the gripper; return where it stopped."""
        raise NotImplementedError

    def _record(
        self, name: str, goal: list[float], measured: list[float], **extra: Any
    ) -> dict[str, Any]:
        err = np.degrees(
            np.abs((np.array(measured) - np.array(goal) + np.pi) % (2 * np.pi) - np.pi)
        )
        move = {
            "name": name,
            "controller": self.name,
            "reached": bool(np.all(err < _ANGULAR_ARRIVAL_DEG)),
            "planned_deg": np.round(np.degrees(goal), 2).tolist(),
            "measured_deg": np.round(np.degrees(measured), 2).tolist(),
            "planned_pinch_z": float(forward_kinematics(goal, self.geom)[0][2]),
            "measured_pinch_z": float(forward_kinematics(measured, self.geom)[0][2]),
        } | extra
        self.moves.append(move)
        _logger.info(
            "Move %s (%s): pinch z planned %+.4f measured %+.4f (floor %+.3f); max "
            "joint error %.2f deg.",
            name,
            self.name,
            move["planned_pinch_z"],
            move["measured_pinch_z"],
            self.geom.floor_z,
            float(err.max()),
        )
        return move


class _AngularMover(_Mover):
    """Jimmy Wu's pick_object moves: the onboard controller, torque loop released."""

    name = "angular"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._aborted = False
        self._joints = list(self.arm.get_arm_state())

    def joints(self) -> list[float]:
        # With the torque loop released the arm server's joint reading is frozen at
        # the release (run 10: an IK seeded from it flipped branch); each onboard
        # move reports where the arm stopped, so track that instead.
        return list(self._joints)

    def begin(self) -> None:
        self._joints = list(self.arm.get_arm_state())
        self.arm.release()

    def end(self) -> None:
        if not self._aborted:
            self.arm.resume()

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        try:
            measured = self.arm.move_angular(goal)
        except ArmMotionAborted:
            self._aborted = True
            raise
        self._joints = list(measured)
        return bool(self._record(name, goal, list(measured))["reached"])

    def gripper(self, position: float) -> float:
        try:
            return self.arm.set_gripper_angular(
                position, float(self.td.get("gripper_timeout_s", 4.0))
            )
        except ArmMotionAborted:
            self._aborted = True
            raise


class _CompliantMover(_Mover):
    """The compliant torque loop.

    As in Tom Silver's cylinder servo, a move streams its joint target and waits for
    the arm to arrive (or stop): the loop settles a few degrees short wherever its
    gravity and friction models miss (24 mm at the gripper in run 3), and for the
    hover and the servo's nudges that does not matter, because the wrist camera
    measures the cube from wherever the gripper really is. Only the descent, which
    the camera cannot watch, is corrected (`correct=True`): after the arm settles the
    pinch point is re-aimed past the goal by the measured error, never below the
    grasp height, so an arm held up by the floor or the cube is not pushed into it.
    """

    name = "compliant"

    def __init__(
        self,
        arm: ArmInterface,
        td: Any,
        geom: TopDownPickGeometry,
        moves: list[dict[str, Any]],
        arm_cfg: DictConfig,
    ) -> None:
        super().__init__(arm, td, geom, moves)
        self.arm_cfg = arm_cfg
        self.command = list(arm.get_arm_state())
        self.grip = arm.get_gripper_state()

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        goal_pos, approach, closing = forward_kinematics(goal, self.geom)
        if correct and approach[2] < -0.99:
            return self._move_down(name, goal, goal_pos, closing)
        if correct:
            return self._move_joints(name, goal)
        settled = self._settle(list(goal))
        self._record(name, goal, list(self.arm.get_arm_state()), settled=settled)
        return settled

    def _move_down(
        self,
        name: str,
        goal: list[float],
        goal_pos: NDArray[np.float64],
        closing: NDArray[np.float64],
    ) -> bool:
        """A pointing-down move (hover, grasp, lift, servo nudges), corrected in
        Cartesian space: re-aim the pinch point at goal + (goal - measured), keeping
        the tool pointing down, until within `correction_tol_mm`. Near the floor the
        aim never goes below the goal's own height, so an arm held up by the floor
        or the cube is not pushed into it."""
        tol = float(self.td.get("correction_tol_mm", 2.0)) / 1000
        rounds = int(self.td.get("correction_rounds", 3))
        max_push = float(self.td.get("correction_max_mm", 60.0)) / 1000
        near_floor = goal_pos[2] < self.geom.floor_z + self.geom.grasp_height + 0.03
        yaw = math.atan2(closing[1], closing[0])
        aim = np.array(goal_pos, float)
        command = list(goal)
        used = 0
        for used in range(rounds + 1):
            self._settle(command)
            measured_pos = forward_kinematics(self.arm.get_arm_state(), self.geom)[0]
            err = measured_pos - goal_pos
            if np.linalg.norm(err) < tol or used == rounds:
                break
            aim = aim - err
            push = aim - goal_pos
            if np.linalg.norm(push) > max_push:
                aim = goal_pos + push * (max_push / np.linalg.norm(push))
            if near_floor and aim[2] < goal_pos[2]:
                aim[2] = goal_pos[2]
            command = pointing_down_joints(
                aim, yaw, self.geom, seed=self.arm.get_arm_state()
            )
        measured = list(self.arm.get_arm_state())
        off = float(
            np.linalg.norm(forward_kinematics(measured, self.geom)[0] - goal_pos)
        )
        return bool(
            self._record(
                name,
                goal,
                measured,
                reached=off < 1.5 * tol,
                pinch_error_mm=round(off * 1000, 1),
                commanded_deg=np.round(np.degrees(command), 2).tolist(),
                correction_rounds=used,
            )["reached"]
        )

    def _move_joints(self, name: str, goal: list[float]) -> bool:
        """Any other move (the carry pose): corrected in joint space."""
        tol = math.radians(float(self.td.get("correction_tol_deg", 0.3)))
        rounds = int(self.td.get("correction_rounds", 3))
        max_push = math.radians(float(self.td.get("correction_max_deg", 8.0)))
        command = np.array(goal, float)
        used = 0
        for used in range(rounds + 1):
            self._settle(list(command))
            err = _wrap(np.array(self.arm.get_arm_state()) - np.array(goal))
            if np.all(np.abs(err) < tol) or used == rounds:
                break
            push = _wrap(command - err - np.array(goal))
            command = np.array(goal) + np.clip(push, -max_push, max_push)
        return bool(
            self._record(
                name,
                goal,
                list(self.arm.get_arm_state()),
                commanded_deg=np.round(np.degrees(command), 2).tolist(),
                correction_rounds=used,
            )["reached"]
        )

    def _settle(self, command: list[float]) -> bool:
        """Stream `command` until the arm is within `arrive_deg` of it, or has moved
        and stopped (it can settle short of the command), or `max_ticks` pass. The
        first quiet ticks after a new command are before the arm starts to move, so
        stillness only counts once it has moved or been streamed for a while."""
        self.command = command
        arrive = math.radians(float(self.td.get("arrive_deg", 1.0)))
        start = np.array(self.arm.get_arm_state())
        last = start
        still = 0
        for tick in range(int(self.arm_cfg.max_ticks)):
            self.arm.execute_action(command, self.grip)
            now = np.array(self.arm.get_arm_state())
            if np.all(np.abs(_wrap(now - np.array(command))) < arrive):
                return True
            moving = np.any(np.abs(_wrap(now - last)) > math.radians(0.05))
            still = 0 if moving else still + 1
            last = now
            started = np.any(np.abs(_wrap(now - start)) > math.radians(0.5))
            if still >= 5 and (started or tick >= 15):
                return True
            time.sleep(POLICY_CONTROL_PERIOD)
        return False

    def gripper(self, position: float) -> float:
        self.grip = position
        _set_gripper(self.arm, position, self.td, hold=self.command)
        self.grip = self.arm.get_gripper_state()
        return self.grip


def _grasp_at(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    cfg: DictConfig,
    cube: dict[str, float],
    label: str,
    base: tuple[float, float, float],
    arm: ArmInterface,
    geom: TopDownPickGeometry,
    confirm: Any,
    cycle: int,
    attempt: int,
    record: dict[str, Any],
    wrist: "_WristRecorder",
    servo: Any,
    attempts: int,
    clip: str,
) -> tuple[bool, str | None]:
    """One grasp from where the base parked at `cube`: (held, stop reason or None).
    The wrist clip is left recording when the cube is held."""
    td = cfg.pick.top_down
    ax, ay = arm_frame_target(base, (cube["x"], cube["y"]), geom)
    yaw_rel = math.radians(cube["yaw_deg"]) - base[2]
    try:
        top_down_joint_targets(ax, ay, yaw_rel, geom, seed=arm.get_arm_state())
    except ValueError as e:
        return False, f"pick_failure: {e}"
    moves: list[dict[str, Any]] = []
    record.setdefault("pick_attempts", []).append(
        {
            "attempt": attempt,
            "cube": cube,
            "target": label,
            "arm_target_xy": [ax, ay],
            "moves": moves,
        }
    )
    confirm(
        f"Cycle {cycle}: Pick attempt {attempt}/{attempts}: open, arm to 10 cm "
        f"above the cube (arm frame {ax:+.3f}, {ay:+.3f}), "
        + ("align with the wrist camera, " if servo is not None else "")
        + "descend, close?"
    )
    wrist.start(f"cycle_{cycle:02d}_pick_{clip}")
    # The fake gripper closes on nothing; pretend it read fake_grasp_reading.
    fake = float(td.get("fake_grasp_reading", 0.49))
    read = functools.partial(_fake_reading, fake) if cfg.mode == "fake" else float
    mover: _Mover = (
        _AngularMover(arm, td, geom, moves)
        if td.get("arm_controller", "compliant") == "angular"
        else _CompliantMover(arm, td, geom, moves, cfg.arm_home)
    )
    held, reading, note, servo_log = _grasp(
        mover,
        geom,
        (ax, ay, yaw_rel),
        read,
        servo,
        f"c{cycle}a{clip}",
    )
    if servo_log is not None:
        record["pick_attempts"][-1]["servo"] = servo_log
    record["pick_attempts"][-1].update({"gripper": reading, "held": held})
    if note is not None:
        record["pick_attempts"][-1]["note"] = note
    _logger.info(
        "Pick attempt %d: gripper %.3f -> %s%s.",
        attempt,
        reading,
        "held" if held else "missed",
        f" ({note})" if note else "",
    )
    if not held:
        wrist.stop()
    return held, None


def _wrap(angles: Any) -> Any:
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def _tool_axes(
    geom: TopDownPickGeometry, yaw_rel: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The top-down tool's x and closing (y) axes in the arm frame for a cube yaw
    `yaw_rel` (the closing yaw is that, modulo the cube's 90-degree symmetry)."""
    closing = cube_closing_yaw(yaw_rel, geom)
    c, s = math.cos(closing), math.sin(closing)
    return np.array([-s, c]), np.array([c, s])


def _grasp(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    mover: _Mover,
    geom: TopDownPickGeometry,
    start: tuple[float, float, float],
    read: Any,
    servo: TopDownCubeServo | None,
    name: str,
) -> tuple[bool, float, str | None, dict[str, Any] | None]:
    """Jimmy Wu's pick_object sequence: open, hover, (align with the wrist camera),
    descend, close, read; then lift and carry, or open and back off.

    Returns (held, gripper reading, note, servo log); the note says why the gripper
    was not closed, if it was not.
    """
    td = mover.td
    ax, ay, yaw = start
    hover, grasp = top_down_joint_targets(ax, ay, yaw, geom, seed=mover.joints())
    servo_log: dict[str, Any] | None = None
    mover.begin()
    try:
        mover.gripper(0.0)
        if not mover.move("hover", hover):
            mover.move("retreat", hover)
            return False, mover.arm.get_gripper_state(), "hover did not arrive", None
        if servo is not None:

            def move_hover(label: str, x: float, y: float, w: float) -> bool:
                here = mover.joints()
                joints, _ = top_down_joint_targets(x, y, w, geom, seed=here)
                if not _servo_step_is_small(here, joints, td):
                    return False
                return mover.move(f"servo_{label}", joints)

            def measured_xy() -> tuple[float, float]:
                pos = forward_kinematics(mover.joints(), geom)[0]
                return float(pos[0]), float(pos[1])

            result = servo.align(
                (ax, ay, yaw),
                move_hover,
                functools.partial(_tool_axes, geom),
                name=name,
                measured_xy=measured_xy,
            )
            servo_log = _servo_log(result)
            ax, ay, yaw = result.ax, result.ay, result.yaw
            hover, grasp = top_down_joint_targets(
                ax, ay, yaw, geom, seed=mover.joints()
            )
            if not result.aligned:
                mover.move("retreat", hover)
                return False, mover.arm.get_gripper_state(), result.note, servo_log
        if not mover.move("grasp", grasp, correct=True):
            mover.move("retreat", hover)
            return (
                False,
                mover.arm.get_gripper_state(),
                "grasp did not arrive",
                servo_log,
            )
        reading = read(mover.gripper(float(td.get("close_position", 1.0))))
        held = grasp_succeeded(reading, geom)
        if held:
            mover.move("lift", hover)
            mover.move("carry", list(CARRY_ARM_CONF))
        else:
            mover.gripper(0.0)
            mover.move("retreat", hover)
        return held, reading, None, servo_log
    finally:
        mover.end()


def _servo_step_is_small(
    here: Sequence[float], target: Sequence[float], td: Any
) -> bool:
    """As in the cylinder servo, refuse a nudge whose IK jumps an arm joint by more
    than `max_joint_step` (a branch flip near a singularity); the wrist (joint 7) may
    turn further, up to `max_wrist_step`, to square the fingers to the cube."""
    jump = np.abs(_wrap(np.asarray(target) - np.asarray(here)))
    arm_limit = float(td.get("max_joint_step", 0.2))
    wrist_limit = float(td.get("max_wrist_step", 0.9))
    if jump[:6].max() > arm_limit or jump[6] > wrist_limit:
        _logger.warning(
            "Servo nudge refused: joint change %s rad (limits %.2f arm, %.2f wrist); "
            "likely an IK branch flip.",
            np.round(jump, 3).tolist(),
            arm_limit,
            wrist_limit,
        )
        return False
    return True


def _servo_log(result: TopDownServoResult) -> dict[str, Any]:
    return {
        "aligned": result.aligned,
        "note": result.note,
        "final": [result.ax, result.ay, math.degrees(result.yaw)],
        "jacobian_px_per_m": result.jacobian,
        "looks": [
            {
                "label": t.label,
                "pinch_xy": list(t.pinch_xy),
                "cube_uv": None if t.cube is None else [t.cube.u, t.cube.v],
                "error_px": t.error_px,
                "delta_tool_m": None if t.delta_tool is None else list(t.delta_tool),
            }
            for t in result.trace
        ],
    }


def _build_servo(
    cfg: DictConfig, source: ImageSource | None, log_dir: Path | None
) -> TopDownCubeServo | None:
    """The wrist servo from ``pick.top_down.wrist_servo``, or None when it is off or
    there is no wrist camera."""
    servo_cfg = cfg.pick.top_down.get("wrist_servo")
    if source is None or servo_cfg is None or not servo_cfg.get("enabled", False):
        return None
    names = {f.name for f in dataclasses.fields(TopDownServoParams)}
    kwargs = {k: servo_cfg[k] for k in names if k in servo_cfg}
    if "target_uv" in kwargs:
        kwargs["target_uv"] = tuple(float(v) for v in kwargs["target_uv"])
    return TopDownCubeServo(
        _SettledImages(source, float(servo_cfg.get("settle_s", 0.3))),
        TopDownServoParams(**kwargs),
        None if log_dir is None else log_dir / "wrist_servo",
    )


class _SettledImages:
    """Wait `settle_s` after a move before taking the wrist image, so the frame shows
    where the arm stopped, not where it was moving."""

    def __init__(self, source: ImageSource, settle_s: float) -> None:
        self._source = source
        self._settle_s = settle_s

    def get_image(self) -> Any:
        """The newest frame after the settle delay."""
        if self._settle_s > 0:
            time.sleep(self._settle_s)
        return self._source.get_image()


def _fake_reading(value: float, _reading: float) -> float:
    """The fake gripper closes on nothing; report `value` instead."""
    return value


def _set_gripper(
    arm: ArmInterface, position: float, td: Any, hold: list[float] | None = None
) -> None:
    """Command the gripper and wait until it stops moving, holding the arm at `hold`
    (the last command: re-sending the measured joints would let a sagging arm sag
    further) or, without it, where it is."""
    target = list(hold) if hold is not None else arm.get_arm_state()
    arm.execute_action(target, position)
    deadline = time.monotonic() + float(td.get("gripper_timeout_s", 4.0))
    last = arm.get_gripper_state()
    while time.monotonic() < deadline:
        time.sleep(0.15)
        arm.execute_action(target, position)
        now = arm.get_gripper_state()
        if abs(now - last) < 0.005:
            return
        last = now


class _CeilingRecorder:
    """Films the whole run from the ceiling cameras into <log_dir>/ceiling.mp4,
    stamping each frame with the time and the step last asked about. Frames are
    encoded as they arrive (H.264), so a long run costs no memory."""

    def __init__(self, cfg: DictConfig, log_dir: Path | None) -> None:
        video_cfg = cfg.get("ceiling_video")
        self.label = "start"
        self._running = False
        if log_dir is None or video_cfg is None or not video_cfg.enabled:
            return
        import threading  # pylint: disable=import-outside-toplevel

        from prpl_tidybot.marker_detector.ceiling_image_client import (  # pylint: disable=import-outside-toplevel
            CeilingImageClient,
            FakeCeilingImageClient,
        )

        self._client: Any = (
            CeilingImageClient(str(video_cfg.host))
            if cfg.mode == "real"
            else FakeCeilingImageClient()
        )
        self._fps = int(video_cfg.fps)
        self._scale = float(video_cfg.scale)
        self._path = log_dir / "ceiling.mp4"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import imageio_ffmpeg  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        period = 1.0 / self._fps
        while self._running:
            start = time.monotonic()
            try:
                img = self._client.get_image()
            except Exception:  # pylint: disable=broad-except
                break  # the publisher went away; keep what was recorded
            if img is not None:
                frame = cv.resize(img, None, fx=self._scale, fy=self._scale)
                frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
                text = f"{start - self._t0:6.1f}s  {self.label.strip()[:90]}"
                cv.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
                cv.putText(frame, text, (8, 22), 0, 0.6, (255, 255, 255), 1)
                if self._writer is None:
                    self._writer = imageio_ffmpeg.write_frames(
                        str(self._path),
                        (frame.shape[1], frame.shape[0]),
                        fps=self._fps,
                        codec="libx264",
                        pix_fmt_out="yuv420p",
                        output_params=["-movflags", "+faststart"],
                        ffmpeg_log_level="error",
                    )
                    self._writer.send(None)
                self._writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    def close(self) -> None:
        """Stop recording and finish the file."""
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=5)
        if self._writer is not None:
            self._writer.close()
        self._client.close()


class _WristRecorder:
    """Records the wrist camera during each pick attempt (when it has a source)."""

    def __init__(self, source: ImageSource | None, log_dir: Path | None) -> None:
        self._enabled = source is not None and log_dir is not None
        self._log_dir = log_dir
        self._source = source
        self._thread: Any = None
        self._frames: list[Any] = []
        self._name = ""
        self._running = False

    def start(self, name: str) -> None:
        """Begin recording a clip called `name`."""
        if not self._enabled:
            return
        import threading  # pylint: disable=import-outside-toplevel

        self._frames, self._name, self._running = [], name, True

        def grab() -> None:
            assert self._source is not None
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
        """Stop any clip in progress (the source is closed by its owner)."""
        self.stop()
