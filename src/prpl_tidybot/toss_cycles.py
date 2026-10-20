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
import threading
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
from prpl_tidybot.aimed_toss import RealTossBounds, Workspace, plan_aimed_toss
from prpl_tidybot.bin_localizer import BinFit, ConstantBinSource, cube_in_bin
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
    arm_link_points,
    cube_closing_yaw,
    forward_kinematics,
    geometry_from_config,
    grasp_search_bounds,
    grasp_succeeded,
    pointing_down_joints,
    standoff_pose,
    top_down_joint_targets,
)
from prpl_tidybot.toss_sim import TossSim
from prpl_tidybot.visual_servo.image_sources import ImageSource, KinovaWristCameraSource
from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownCubeServo,
    TopDownLateralServo,
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
    # Inside the bin the cube is in shadow and beyond the floor search (run 31): look
    # inside the box first when the source can.
    in_box = getattr(cube_source, "find_in_box", None)
    found = in_box(list(box)) if callable(in_box) else None
    if found is not None:
        cube = {"x": float(found[0]), "y": float(found[1])}
    else:
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


def _bin_box(bin_fit: BinFit) -> list[float]:
    """The axis-aligned floor box (x_min, x_max, y_min, y_max) around the bin."""
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    hx = 0.5 * (abs(bin_fit.length * c) + abs(bin_fit.width * s))
    hy = 0.5 * (abs(bin_fit.length * s) + abs(bin_fit.width * c))
    return [bin_fit.x - hx, bin_fit.x + hx, bin_fit.y - hy, bin_fit.y + hy]


def _find_cube(source: Any, bin_fit: BinFit | None) -> dict[str, Any]:
    """The cube's pose, and with a bin whether it lies inside it (``in_bin``): inside
    the bin it is in shadow and beyond the floor search, so the bin is looked in
    first when the source can (run 31)."""
    if bin_fit is not None:
        finder = getattr(source, "find_pose_in_box", None)
        found = finder(_bin_box(bin_fit)) if callable(finder) else None
        if found is not None and cube_in_bin(found[:2], bin_fit, margin=0.0):
            return {
                "x": float(found[0]),
                "y": float(found[1]),
                "yaw_deg": math.degrees(found[2]),
                "in_bin": True,
            }
    cube: dict[str, Any] = dict(_localize(source))
    if bin_fit is not None:
        cube["in_bin"] = cube_in_bin((cube["x"], cube["y"]), bin_fit, margin=0.0)
    return cube


def _overlaps(poly: NDArray[np.float64], half: tuple[float, float]) -> bool:
    """Whether the convex polygon ``poly`` (N x 2) overlaps the rectangle centred on
    the origin with half extents ``half`` (separating axes)."""
    rect = np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ]
    )
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for k in range(len(poly)):
        edge = poly[(k + 1) % len(poly)] - poly[k]
        axes.append(np.array([-edge[1], edge[0]]))
    for axis in axes:
        a, b = poly @ axis, rect @ axis
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def arm_bin_collision(
    joints: Sequence[float],
    geom: TopDownPickGeometry,
    base: Sequence[float],
    bin_fit: BinFit,
    in_bin: Any,
) -> str | None:
    """Why the arm at ``joints``, its base at map pose ``base``, would touch the bin's
    walls, or None.

    Each arm link's collision mesh, and the 2F-85 as a box (the IK model has no
    gripper mesh), is cut below the rim (plus ``wall_margin``); a part touches a wall
    unless that cut lies wholly inside the bin's inner footprint or wholly outside its
    outer one, both by ``wall_margin``. The box is long along the fingers' travel,
    which is the tool's x axis (the IK site's 'closing' axis is across the fingers).
    """
    margin = float(in_bin.wall_margin)
    pos, approach, closing = forward_kinematics(joints, geom)
    fingers = np.cross(closing, approach)
    corners = [
        pos
        + a * approach
        + b * float(in_bin.finger_half_span) * fingers
        + c * float(in_bin.half_width) * closing
        for a in (float(in_bin.fingertip_drop), -float(in_bin.gripper_length))
        for b in (-1.0, 1.0)
        for c in (-1.0, 1.0)
    ]
    parts = dict(arm_link_points(joints, geom))
    parts["gripper"] = np.asarray(corners)
    # Arm frame -> the bin's own frame (its x along its length).
    x, y, theta = base
    mx = x + geom.mounting_offset * math.cos(theta)
    my = y + geom.mounting_offset * math.sin(theta)
    rim = geom.floor_z + bin_fit.height + margin
    outer = (0.5 * bin_fit.length + margin, 0.5 * bin_fit.width + margin)
    inner = (
        0.5 * bin_fit.length - bin_fit.wall - margin,
        0.5 * bin_fit.width - bin_fit.wall - margin,
    )
    ct, st = math.cos(theta), math.sin(theta)
    cb, sb = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    for name, points in parts.items():
        low = points[points[:, 2] < rim]
        if len(low) == 0:
            continue
        wx = mx + ct * low[:, 0] - st * low[:, 1] - bin_fit.x
        wy = my + st * low[:, 0] + ct * low[:, 1] - bin_fit.y
        local = np.c_[cb * wx + sb * wy, -sb * wx + cb * wy]
        if np.all(np.abs(local[:, 0]) <= inner[0]) and np.all(
            np.abs(local[:, 1]) <= inner[1]
        ):
            continue
        hull = cv.convexHull(local.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if _overlaps(hull, outer):
            depth = (bin_fit.height + geom.floor_z) - float(low[:, 2].min())
            return f"the {name} would touch the bin wall ({depth:.3f} m below the rim)"
    return None


def _bin_pick_targets(
    geom: TopDownPickGeometry,
    target: tuple[float, float, float],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> list[float]:
    """The waypoint above the cube with the fingertips ``rim_clearance`` above the
    bin's rim: where the pick into the bin starts and ends."""
    above_hover = (
        bin_fit.height
        + float(in_bin.rim_clearance)
        + float(in_bin.fingertip_drop)
        - geom.grasp_height
    )
    high = dataclasses.replace(geom, hover_height=max(above_hover, geom.hover_height))
    return top_down_joint_targets(*target, high, seed=seed)[0]


def _bin_grasp_geometry(
    geom: TopDownPickGeometry,
    base: Sequence[float],
    cube: dict[str, Any],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> tuple[TopDownPickGeometry | None, str | None]:
    """The pick geometry for a cube inside the bin: of the grasps on its two pairs of
    faces (``closing_turn`` 0 and pi/2), the one that touches no wall, the usual one
    first; else None and why the usual one would touch."""
    first: str | None = None
    for turn in (0.0, math.pi / 2):
        g = dataclasses.replace(geom, closing_turn=turn)
        problem = _bin_pick_problem(g, base, cube, bin_fit, in_bin, seed)
        if problem is None:
            return g, None
        first = first or problem
    return None, first


def _bin_pick_problem(
    geom: TopDownPickGeometry,
    base: Sequence[float],
    cube: dict[str, Any],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> str | None:
    """Why the pick of the cube inside the bin, from ``base``, would touch a wall at
    the waypoint above the rim, the hover or the grasp, or None."""
    ax, ay = arm_frame_target(base, (cube["x"], cube["y"]), geom)
    yaw = math.radians(cube["yaw_deg"]) - base[2]
    try:
        above = _bin_pick_targets(geom, (ax, ay, yaw), bin_fit, in_bin, seed)
        hover, grasp = top_down_joint_targets(ax, ay, yaw, geom, seed=above)
    except ValueError as e:
        return f"no top-down pick into the bin: {e}"
    for name, joints in (("above", above), ("hover", hover), ("grasp", grasp)):
        problem = arm_bin_collision(joints, geom, base, bin_fit, in_bin)
        if problem is not None:
            return f"{name}: {problem}"
    return None


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


def _scored_landing(
    cube: dict[str, Any], toss_pose: list[float], bin_fit: BinFit | None, margin: float
) -> dict[str, Any]:
    """Where a throw ended, and with a localized bin whether the cube is inside it
    (``in_bin``) and inside its scoring region, ``margin`` inside the walls
    (``scored``, as Tossing3D scores)."""
    landing: dict[str, Any] = dict(_landing(cube, toss_pose))
    if bin_fit is not None:
        xy = (cube["x"], cube["y"])
        landing["in_bin"] = bool(cube.get("in_bin", cube_in_bin(xy, bin_fit, 0.0)))
        landing["scored"] = cube_in_bin(xy, bin_fit, margin)
    return landing


def run_toss_cycles(  # pylint: disable=too-many-locals
    cfg: DictConfig,
    log_dir: Path | str | None = None,
    prompt_fn: PromptFn = input,
    wrist_camera: Callable[[ArmInterface], ImageSource] | None = None,
) -> TossCyclesSummary:
    """Run ``cfg.cycles`` pick/drive/toss/retract cycles (conf/toss_cycles.yaml).

    ``wrist_camera`` builds the wrist image source from the arm; by default the
    Kinova wrist camera in real mode and none otherwise (no servo).

    mode=sim runs the same commands on the kinematic TidyBot of
    :mod:`prpl_tidybot.sim_robot` (``cfg.sim``) and films it into ``sim.mp4``; its
    prompts are answered by a scripted operator unless ``interactive=true``."""
    if cfg.mode == "sim" and cfg.pick.method != "top_down":
        raise ValueError("toss_cycles mode=sim simulates the top-down pick only.")
    aimed_cfg = cfg.toss.get("aimed")
    if cfg.toss.get("grid") and aimed_cfg is not None and aimed_cfg.get("enabled"):
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

    kinder.register_all_environments()
    sim: TossSim | None = None
    if cfg.mode == "sim":
        sim = TossSim(cfg, resolved_log_dir, prompt_fn, interactive)
        prompt_fn = sim.operator
        real_env, perceiver = sim.real_env, sim.perceiver
    else:
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
    if sim is not None:
        cube_source: Any = sim.cube_source
    else:
        cube_source = hydra.utils.instantiate(
            pipeline.perceiver.target_source,
            _convert_="all",
            **_cube_source_kwargs(
                cfg, pipeline.perceiver.target_source, resolved_log_dir
            ),
        )
    workspace = Workspace.from_config(cfg.get("barrier"), cfg.preview.floor_bounds)
    aimed = cfg.toss.get("aimed")
    aiming = aimed is not None and bool(aimed.get("enabled", False))
    # Same-side practice: the bin is localized every cycle and a cube inside it picked.
    in_bin_cfg = cfg.pick.get("top_down", {}).get("in_bin")
    bin_pick = aiming and in_bin_cfg is not None and bool(in_bin_cfg.get("enabled"))
    score_margin = float(cfg.bin.get("in_bin_margin", 0.05))
    bin_source: Any = None
    if aiming and sim is not None and cfg.sim.get("bin") is not None:
        x, y, yaw_deg = (float(v) for v in cfg.sim.bin)
        bin_source = ConstantBinSource(
            x,
            y,
            math.radians(yaw_deg),
            float(cfg.bin.length),
            float(cfg.bin.width),
            float(cfg.bin.height),
            float(cfg.bin.wall),
        )
    elif aiming:
        bin_source = hydra.utils.instantiate(
            cfg.bin.pipelines[cfg.mode], _convert_="all"
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
    lateral = _build_lateral_servo(cfg, source, resolved_log_dir)
    servo: TopDownCubeServo | TopDownLateralServo | None = (
        _build_servo(cfg, source, resolved_log_dir) or lateral
    )
    ceiling: Any = (
        sim.recorder if sim is not None else _CeilingRecorder(cfg, resolved_log_dir)
    )

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
            bin_fit: BinFit | None = None
            if bin_pick:
                assert bin_source is not None
                _logger.info("Cycle %d/%d: localizing the bin.", cycle, requested)
                bin_fit = bin_source.localize()
                record["bin"] = bin_fit.as_dict()
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
                    workspace,
                    bin_fit=bin_fit,
                    in_bin=in_bin_cfg if bin_pick else None,
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
                records[-2]["landing"] = _scored_landing(
                    record["cube_before_pick"],
                    records[-2].get("aimed", {}).get("pose", toss_pose),
                    bin_fit,
                    score_margin,
                )
            if failure is not None:
                finish_reason = failure
                break
            record["picked"] = True
            if cfg.get("skip_toss"):
                finish_reason = "picked (toss skipped)"
                break
            if cfg.get("place_at") is not None:
                target = (float(cfg.place_at[0]), float(cfg.place_at[1]))
                failure = _place_top_down(
                    cfg, real_env, arm, geom, confirm, cycle, target, record, workspace
                )
                finish_reason = (
                    failure or f"placed at ({target[0]:+.3f}, {target[1]:+.3f})"
                )
                break

            toss_cfg = cfg
            if aiming:
                assert bin_source is not None and aimed is not None
                throw_from, toss_cfg = _aim(cfg, real_env, bin_source, workspace)
                record["aimed"] = {
                    "bin": bin_source.last_fit.as_dict(),
                    "pose": list(throw_from),
                    "standoff": float(aimed.standoff),
                    "speed": float(aimed.speed),
                    "release_ms": float(aimed.release_ms),
                }
                confirm(
                    f"Cycle {cycle}: Drive to the aimed toss pose "
                    f"({throw_from[0]:+.3f}, {throw_from[1]:+.3f}, "
                    f"{math.degrees(throw_from[2]):+.0f} deg), "
                    f"{float(aimed.standoff):.2f} m from the bin?"
                )
                _drive_to(
                    real_env, list(throw_from), workspace, None, bin_source.last_fit
                )
            else:
                confirm(
                    f"Cycle {cycle}: Drive to the toss pose ({toss_pose[0]:+.3f}, "
                    f"{toss_pose[1]:+.3f}, {math.degrees(toss_pose[2]):+.0f} deg)?"
                )
                _drive_to(real_env, toss_pose, workspace)

            # toss.grid and toss.aimed are exclusive (checked at startup).
            cycle_cfg = _with_grid_setting(toss_cfg, cycle)
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
            if settle > 0 and sim is not None:
                sim.world.idle(settle)
            elif settle > 0:
                _logger.info("Waiting %.1f s for the cube to settle.", settle)
                time.sleep(settle)
        else:
            finish_reason = f"completed {completed}/{requested} cycles"
            if bin_pick and records and records[-1].get("tossed"):
                assert bin_source is not None
                try:
                    last_fit = bin_source.localize()
                    records[-1]["landing"] = _scored_landing(
                        _find_cube(cube_source, last_fit),
                        records[-1].get("aimed", {}).get("pose", toss_pose),
                        last_fit,
                        score_margin,
                    )
                except RuntimeError as e:
                    records[-1]["landing"] = {"error": str(e)}
    except _Declined as e:
        finish_reason = f"declined: {e}"
    except ValueError as e:
        # A toss or drive refused before anything moved (bounds, barrier, bin).
        finish_reason = f"refused: {e}"
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
        if bin_source is not None:
            bin_source.close()
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
    arm: ArmInterface,
    goal: list[float],
    gripper: float,
    arm_cfg: DictConfig,
    stop: threading.Event | None = None,
) -> bool:
    """Stream ``goal`` until every joint is within tolerance (wrapped) or time runs
    out; returns whether it got there. A set ``stop`` ends the move at once
    (``ArmMotionAborted``), the arm held where it is."""
    for _ in range(int(arm_cfg.max_ticks)):
        _check_stop(stop, arm)
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


def _retract(
    cfg: DictConfig, arm: ArmInterface, stop: threading.Event | None = None
) -> None:
    """Fold the arm to its retract pose under the compliant loop: it needs no
    precision, and only the grasp itself runs under the stiff onboard controller."""
    arm.note_step("retract")
    _move_arm(arm, list(RETRACT_ARM_CONF), arm.get_gripper_state(), cfg.arm_home, stop)


def _check_stop(stop: threading.Event | None, arm: ArmInterface | None = None) -> None:
    """Raise ``ArmMotionAborted`` once ``stop`` is set (the skill server's stop()),
    holding the arm where it is first."""
    if stop is None or not stop.is_set():
        return
    if arm is not None:
        arm.execute_action(list(arm.get_arm_state()), arm.get_gripper_state())
    raise ArmMotionAborted("stopped")


def _drive_to(
    real_env: Any,
    pose: list[float],
    workspace: Workspace | None = None,
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
) -> tuple[float, float, float]:
    """Drive the base to the map-frame pose with the kinematic3d executor (as
    hardware_tests/test_base_map_target.py does); returns the reached pose.

    Uses a robot-only perceiver: driving needs no cube, and must still work when the
    cube cannot be found (the reset after a failed localization). With a
    ``workspace`` the straight drive is checked first (drivable box, barrier) and a
    drive that would leave it raises ``ValueError`` before the base moves. A set
    ``stop`` ends the drive (``ArmMotionAborted``) without another command.
    """
    perceiver = PrplLab3DPerceiver()
    obs, info = real_env.reset()
    state = perceiver.step(obs, info)
    robot = state.get_object_from_name("robot")
    if workspace is not None:
        here = obs.map_base_pose
        problem = workspace.path_problem(
            (float(here.x), float(here.y), float(here.theta())), pose, bin_fit
        )
        if problem is not None:
            raise ValueError(f"Drive refused: {problem}.")
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
        _check_stop(stop)
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
    try:
        reached = _drive_to(
            real_env,
            toss_pose,
            Workspace.from_config(cfg.get("barrier"), cfg.preview.floor_bounds),
        )
    except ValueError as e:
        _logger.warning("Reset drive refused, leaving the base where it is: %s", e)
        return None
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


def pick_problem(
    robot: Sequence[float],
    cube: dict[str, float],
    geom: TopDownPickGeometry,
    floor: Sequence[float],
    workspace: Workspace | None,
    bin_fit: BinFit | None = None,
) -> str | None:
    """Why the top-down pick of ``cube`` cannot even start from ``robot``, or None:

    the cube past or on the barrier (the robot never crosses it), no standoff that keeps
    the base in the drivable box and on its side, or no straight drive there.
    """
    if workspace is not None:
        yaw = cube.get("yaw")
        if yaw is None and cube.get("yaw_deg") is not None:
            yaw = math.radians(cube["yaw_deg"])
        problem = workspace.cube_problem((cube["x"], cube["y"]), yaw)
        if problem is not None:
            return problem
    try:
        stand = standoff_pose(
            robot,
            (cube["x"], cube["y"]),
            geom,
            floor,
            None if workspace is None else workspace.valid(bin_fit),
        )
    except ValueError as e:
        return str(e)
    if workspace is not None:
        problem = workspace.path_problem(robot, stand, bin_fit)
        if problem is not None:
            return f"pick standoff unreachable: {problem}"
    return None


def _aim(
    cfg: DictConfig, real_env: Any, bin_source: Any, workspace: Workspace
) -> tuple[tuple[float, float, float], DictConfig]:
    """Localize the bin and check the configured aimed toss (``toss.aimed``) before
    anything moves; returns the pose to throw from and the config to toss with
    (``toss.max_vel`` / ``gripper_release_ms`` from the aimed speed and release, the arm
    straight ahead)."""
    aimed = cfg.toss.aimed
    bin_fit = bin_source.localize()
    obs, _ = real_env.reset()
    here = obs.map_base_pose
    direction = aimed.get("direction")
    pose = plan_aimed_toss(
        bin_fit,
        float(aimed.standoff),
        float(aimed.speed),
        float(aimed.release_ms),
        None if direction is None else float(direction),
        (float(here.x), float(here.y), float(here.theta())),
        workspace,
        RealTossBounds.from_config(cfg.real_toss_bounds),
    )
    toss_cfg = OmegaConf.merge(
        cfg,
        {
            "toss": {
                "heading": 0.0,
                "max_vel": float(aimed.speed),
                "gripper_release_ms": int(round(float(aimed.release_ms))),
            }
        },
    )
    assert isinstance(toss_cfg, DictConfig)
    return pose, toss_cfg


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
    servo: TopDownCubeServo | TopDownLateralServo | None = None,
    ask_to_move: PromptFn | None = None,
    workspace: Workspace | None = None,
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
    in_bin: Any = None,
) -> str | None:
    """Jimmy Wu's top-down pick (prpl_tidybot.top_down_pick) on the localized cube.

    For up to ``max_attempts``: localize the cube, drive to a standoff facing it,
    re-localize, then open, hover 10 cm above, descend, close, and read the gripper.
    After a miss the arm retracts and the base returns to the toss pose, where the
    ceiling cameras see the (possibly nudged) cube without the robot over it. With
    ``ask_to_move``, a miss instead retracts and asks the person to move the cube (up to
    ``ask_to_move_after_miss`` times), then tries again. Returns a stop reason, or None
    once the cube is held (arm at its carry pose).

    With a localized ``bin_fit`` and ``in_bin`` (same-side practice) the cube is looked
    for inside the bin first and picked from there through a waypoint above the rim;
    every standoff and drive keeps the base clear of the bin; and a cube the robot may
    not or cannot pick (past the barrier, or too close to a bin wall to grasp without
    touching it) is handed to the person instead of ending the run.
    """
    td = cfg.pick.top_down
    floor = tuple(cfg.preview.floor_bounds)
    attempts = int(td.get("max_attempts", 3))
    asks = int(td.get("ask_to_move_after_miss", 3)) if ask_to_move is not None else 0
    for attempt in range(1, attempts + asks + 1):
        try:
            cube = _find_cube(cube_source, bin_fit)
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
        problem = pick_problem(robot, cube, geom, floor, workspace, bin_fit)
        stand: tuple[float, float, float] | None = None
        if problem is None:
            stand = standoff_pose(
                robot,
                (cube["x"], cube["y"]),
                geom,
                floor,
                None if workspace is None else workspace.valid(bin_fit),
            )
            if cube.get("in_bin") and bin_fit is not None and in_bin is not None:
                chosen, wall = _bin_grasp_geometry(
                    geom, stand, cube, bin_fit, in_bin, arm.get_arm_state()
                )
                if chosen is None:
                    problem = f"cube too close to the bin wall to grasp ({wall})"
        if problem is not None:
            if in_bin is None or ask_to_move is None:
                return f"{problem}; move the cube and rerun"
            record.setdefault("handed_to_person", []).append(
                {"cube": cube, "problem": problem}
            )
            if not _asked_to_move(
                ask_to_move,
                asks,
                record,
                f"\nCycle {cycle}: the cube at ({cube['x']:+.3f}, {cube['y']:+.3f}) "
                f"cannot be picked: {problem}. Please move it onto the floor on the "
                "robot's side of the tape, clear of the bin and of the camera overlap "
                "strip (y -0.15..0.18), then press Enter, or type 'stop' to end the "
                "run: ",
            ):
                return f"{problem}; the operator stopped the run"
            continue
        assert stand is not None
        confirm(
            f"Cycle {cycle}: Drive to the pick standoff ({stand[0]:+.3f}, "
            f"{stand[1]:+.3f}, {math.degrees(stand[2]):+.0f} deg), "
            f"{geom.standoff:.2f} m from the cube at ({cube['x']:+.3f}, "
            f"{cube['y']:+.3f})?"
        )
        base = _drive_to(real_env, list(stand), workspace, stop, bin_fit)
        try:
            # Corrects for where the base parked.
            cube = _find_cube(cube_source, bin_fit)
        except RuntimeError as e:
            return f"{e}; move the cube and rerun"
        # The more confident camera's estimate first; on a miss, the other cameras'
        # (they disagree by ~3 cm where they overlap).
        # Inside the bin there is one estimate: the floor localizer's others are stale.
        from_bin = bin_fit if cube.get("in_bin") else None
        targets = [(cube, "best camera")] + (
            []
            if from_bin is not None
            else [
                ({"x": x, "y": y, "yaw_deg": math.degrees(yaw)}, "other camera")
                for x, y, yaw in cube_source.alternates()
            ]
        )
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
                stop,
                from_bin,
                in_bin,
            )
            if failure is not None:
                return failure
            if held:
                wrist.stop()
                return None
        wrist.stop()
        if ask_to_move is not None and record.get("cube_moved_by_person", 0) < asks:
            _retract(cfg, arm, stop)
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
            _retract(cfg, arm, stop)
            _drive_to(real_env, list(toss_pose), workspace, stop, bin_fit)
    return (
        f"pick failed after {attempts} attempts (gripper never closed on the cube); "
        "move the cube and rerun"
    )


def _place_top_down(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    real_env: Any,
    arm: ArmInterface,
    geom: TopDownPickGeometry,
    confirm: Any,
    cycle: int,
    target: tuple[float, float],
    record: dict[str, Any],
    workspace: Workspace | None = None,
    stop: threading.Event | None = None,
) -> str | None:
    """The top-down pick in reverse: carry the held cube to the map point `target` and
    set it down. The closed gripper is lowered to the grasp height, where it lifted the
    cube off the floor, so the cube rests on the floor before the gripper opens (the
    foam cube bounces if dropped). Returns a stop reason, or None once placed."""
    obs, _ = real_env.reset()
    robot = (obs.map_base_pose.x, obs.map_base_pose.y, obs.map_base_pose.theta())
    stand = standoff_pose(
        robot,
        target,
        geom,
        tuple(cfg.preview.floor_bounds),
        None if workspace is None else workspace.valid(),
    )
    confirm(
        f"Cycle {cycle}: Carry the cube to the standoff ({stand[0]:+.3f}, "
        f"{stand[1]:+.3f}, {math.degrees(stand[2]):+.0f} deg) and set it down at "
        f"({target[0]:+.3f}, {target[1]:+.3f})?"
    )
    base = _drive_to(real_env, list(stand), workspace, stop)
    ax, ay = arm_frame_target(base, target, geom)
    moves: list[dict[str, Any]] = []
    record["place"] = {
        "target": list(target),
        "base": list(base),
        "arm_target_xy": [ax, ay],
        "moves": moves,
    }
    try:
        hover, down = top_down_joint_targets(
            ax, ay, 0.0, geom, seed=arm.get_arm_state()
        )
    except ValueError as e:
        return f"place_failure: {e}"
    td = cfg.pick.top_down
    mover: _Mover = (
        _AngularMover(arm, td, geom, moves)
        if td.get("arm_controller", "compliant") == "angular"
        else _CompliantMover(arm, td, geom, moves, cfg.arm_home)
    )
    mover.stop = stop
    mover.begin()
    try:
        if not mover.move("place_hover", hover):
            return "place: the hover did not arrive"
        if not mover.move("place_down", down, correct=True):
            mover.move("place_up", hover)
            return "place: the set-down did not arrive (the cube is still held)"
        mover.gripper(0.0)
        mover.move("place_up", hover)
    finally:
        mover.end()
    return None


# An onboard-controller move that ends further than this from its goal has hit
# something (the floor, the cube) or faulted; the toss uses the same check.
_ANGULAR_ARRIVAL_DEG = 1.0


class _Mover:
    """How the grasp sequence moves the arm and gripper: under the onboard position
    controller (`_AngularMover`) or the compliant torque loop (`_CompliantMover`).
    One controller is used for a whole run; nothing switches mid-episode."""

    name = ""
    # Set by the skill server: once set, the next move or gripper command raises
    # ArmMotionAborted instead of moving.
    stop: threading.Event | None = None

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

    def check_stop(self) -> None:
        """Raise ``ArmMotionAborted`` if ``stop`` is set (nothing more is commanded)."""
        if self.stop is not None and self.stop.is_set():
            raise ArmMotionAborted("stopped")

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
        self.arm.note_step(name)
        try:
            self.check_stop()
            measured = self.arm.move_angular(goal)
        except ArmMotionAborted:
            self._aborted = True
            raise
        self._joints = list(measured)
        return bool(self._record(name, goal, list(measured))["reached"])

    def gripper(self, position: float) -> float:
        self.arm.note_step(f"gripper to {position:.2f}")
        try:
            self.check_stop()
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
        self.arm.note_step(name)
        self.check_stop()
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
        err = forward_kinematics(measured, self.geom)[0] - goal_pos
        off = float(np.linalg.norm(err))
        # The error in the gripper's frame: along its closing axis, across it, up.
        across = np.cross(closing, np.array([0.0, 0.0, -1.0]))
        tool = (float(err @ closing), float(err @ across), float(err[2]))
        return bool(
            self._record(
                name,
                goal,
                measured,
                reached=self._arrived(tool, off, tol),
                pinch_error_mm=round(off * 1000, 1),
                pinch_error_tool_mm=[round(v * 1000, 1) for v in tool],
                commanded_deg=np.round(np.degrees(command), 2).tolist(),
                correction_rounds=used,
            )["reached"]
        )

    def _arrived(
        self, tool: tuple[float, float, float], off: float, tol: float
    ) -> bool:
        """Whether the descent ended close enough to close the gripper: within each
        axis's limit in ``correction_arrive_mm`` (closing, across, down, up; mm), or
        without them within 1.5 x the correction tolerance."""
        limits = self.td.get("correction_arrive_mm")
        if limits is None:
            return off < 1.5 * tol
        closing, across, up = (v * 1000 for v in tool)
        return (
            abs(closing) <= float(limits.closing)
            and abs(across) <= float(limits.across)
            and -float(limits.down) <= up <= float(limits.up)
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
        self.arm.note_step(f"gripper to {position:.2f}")
        self.check_stop()
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
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
    in_bin: Any = None,
) -> tuple[bool, str | None]:
    """One grasp from where the base parked at `cube`: (held, stop reason or None).
    The wrist clip is left recording when the cube is held. With ``bin_fit`` the cube
    is inside the bin: the arm goes through the waypoint above the rim, and no arm
    pose touching a wall is commanded."""
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
            "base": list(base),
            "moves": moves,
        }
    )
    above: list[float] | None = None
    clear: Callable[[Sequence[float]], str | None] | None = None
    if bin_fit is not None and in_bin is not None:
        chosen, wall = _bin_grasp_geometry(
            geom, base, cube, bin_fit, in_bin, arm.get_arm_state()
        )
        if chosen is None:
            record["pick_attempts"][-1].update(
                {"gripper": None, "held": False, "note": f"bin wall: {wall}"}
            )
            _logger.warning("Pick attempt %d not started: bin wall: %s.", attempt, wall)
            return False, None
        geom = chosen
        record["pick_attempts"][-1]["closing_turn_deg"] = math.degrees(
            geom.closing_turn
        )
        above = _bin_pick_targets(
            geom, (ax, ay, yaw_rel), bin_fit, in_bin, arm.get_arm_state()
        )
        fit, cfg_in_bin = bin_fit, in_bin

        def clear(joints: Sequence[float]) -> str | None:
            return arm_bin_collision(joints, geom, base, fit, cfg_in_bin)

    confirm(
        f"Cycle {cycle}: Pick attempt {attempt}/{attempts}: open, "
        + ("arm above the bin's rim, " if above is not None else "")
        + f"arm to 10 cm above the cube (arm frame {ax:+.3f}, {ay:+.3f}), "
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
    mover.stop = stop
    held, reading, note, servo_log = _grasp(
        mover,
        geom,
        (ax, ay, yaw_rel),
        read,
        servo,
        f"c{cycle}a{clip}",
        above,
        clear,
    )
    if servo_log is not None:
        key = "lateral_servo" if isinstance(servo, TopDownLateralServo) else "servo"
        record["pick_attempts"][-1][key] = servo_log
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
    servo: TopDownCubeServo | TopDownLateralServo | None,
    name: str,
    above: list[float] | None = None,
    clear: Callable[[Sequence[float]], str | None] | None = None,
) -> tuple[bool, float, str | None, dict[str, Any] | None]:
    """Jimmy Wu's pick_object sequence: open, hover, (align with the wrist camera),
    descend, close, read; then lift and carry, or open and back off.

    With ``above`` (a cube inside the bin) the arm enters and leaves through that
    waypoint over the rim, and ``clear`` (why a pose would touch the bin, or None)
    refuses any servo step or grasp that would.

    Returns (held, gripper reading, note, servo log); the note says why the gripper
    was not closed, if it was not.
    """
    td = mover.td
    ax, ay, yaw = start
    hover, grasp = top_down_joint_targets(
        ax, ay, yaw, geom, seed=above if above is not None else mover.joints()
    )
    servo_log: dict[str, Any] | None = None

    def retreat(to: list[float]) -> None:
        mover.move("retreat", to)
        if above is not None:
            mover.move("above", above)

    mover.begin()
    try:
        mover.gripper(0.0)
        if above is not None and not mover.move("above", above):
            mover.move("above", above)
            return False, mover.arm.get_gripper_state(), "above did not arrive", None
        if not mover.move("hover", hover):
            retreat(hover)
            return False, mover.arm.get_gripper_state(), "hover did not arrive", None
        if servo is not None:

            def move_hover(label: str, x: float, y: float, w: float) -> bool:
                here = mover.joints()
                joints, _ = top_down_joint_targets(x, y, w, geom, seed=here)
                if not _servo_step_is_small(here, joints, td):
                    return False
                if clear is not None and clear(joints) is not None:
                    _logger.warning("Servo step refused: %s.", clear(joints))
                    return False
                prefix = (
                    "lateral" if isinstance(servo, TopDownLateralServo) else "servo"
                )
                return mover.move(f"{prefix}_{label}", joints)

            def measured_xy() -> tuple[float, float]:
                pos = forward_kinematics(mover.joints(), geom)[0]
                return float(pos[0]), float(pos[1])

            if isinstance(servo, TopDownLateralServo):
                result = servo.align(
                    (ax, ay, yaw), move_hover, functools.partial(_tool_axes, geom), name
                )
            else:
                result = servo.align(
                    (ax, ay, yaw),
                    move_hover,
                    functools.partial(_tool_axes, geom),
                    name=name,
                    measured_xy=measured_xy,
                )
            servo_log = _servo_log(result)
            ax, ay, yaw = result.ax, result.ay, result.yaw
            if result.aligned and isinstance(mover, _CompliantMover):
                # The servo centred the gripper where it really is; the compliant arm
                # lags its command (run 36: 42 mm), so descend from there.
                ax, ay = (float(v) for v in measured_xy())
                servo_log["descend_from_measured"] = [ax, ay]
            hover, grasp = top_down_joint_targets(
                ax, ay, yaw, geom, seed=mover.joints()
            )
            if not result.aligned:
                retreat(hover)
                return False, mover.arm.get_gripper_state(), result.note, servo_log
        if clear is not None and clear(grasp) is not None:
            retreat(hover)
            return (
                False,
                mover.arm.get_gripper_state(),
                f"grasp refused: {clear(grasp)}",
                servo_log,
            )
        if not mover.move("grasp", grasp, correct=True):
            retreat(hover)
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
            if above is not None:
                mover.move("above", above)
            mover.move("carry", list(CARRY_ARM_CONF))
        else:
            mover.gripper(0.0)
            retreat(hover)
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


def _build_lateral_servo(
    cfg: DictConfig, source: ImageSource | None, log_dir: Path | None
) -> TopDownLateralServo | None:
    """Tom Silver's align loop from ``pick.top_down.lateral_servo``, or None when it
    is off or there is no wrist camera."""
    td = cfg.pick.top_down
    lateral_cfg = td.get("lateral_servo")
    if lateral_cfg is None or not lateral_cfg.get("enabled", False):
        return None
    if td.get("wrist_servo") is not None and td.wrist_servo.get("enabled", False):
        raise ValueError(
            "pick.top_down.wrist_servo and pick.top_down.lateral_servo are both "
            "enabled; enable at most one."
        )
    if source is None:
        return None
    names = {f.name for f in dataclasses.fields(LateralServoParams)}
    kwargs = {k: lateral_cfg[k] for k in names if k in lateral_cfg}
    for key in ("target_uv", "lateral_gain", "lateral_sign"):
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])
    return TopDownLateralServo(
        _SettledImages(source, float(lateral_cfg.get("settle_s", 0.3))),
        LateralServoParams(**kwargs),
        None if log_dir is None else log_dir / "lateral_servo",
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
