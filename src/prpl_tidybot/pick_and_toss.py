"""Pick the floor cube with the shelf3d PickController, then toss it.

The logic behind ``scripts/pick_and_toss.py``: a two-step validation of two
real-robot controllers without a task planner.

1. **Pick.** The env's own pipeline (``cfg.env.pipelines[mode]``: real env,
   perceiver, plan executor) is built exactly as ``run_planner`` builds it.
   Instead of the bilevel planner, :class:`PickSkillAgent` runs only the
   env model's Pick skill: it samples PickController parameters and rolls the
   controller out in the planner's own simulator model (the same trajectory
   sampler SeSamE uses) until the Pick operator's effects hold. The resulting
   (state, action) trajectory goes through the plan preview gate in real mode
   and is then executed by the configured ``Kinematic3DPlanExecutor`` via the
   usual ``Runner``. The shelf Place is never planned.
2. **Toss.** ``ArmInterface.toss`` with the configured heading, speed and
   release cycle, after a stdin confirmation in real mode. With capture
   enabled, the toss is filmed from the ceiling and the landing read from the
   marker detector (``prpl_tidybot.toss_outcome``).

``skip_pick`` hands the object in instead of picking it (only the arm
interface is built), and ``skip_toss`` stops after the pick.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import kinder
import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractState, SesameModels
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from bilevel_planning.utils import RelationalControllerGenerator
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

import prpl_tidybot.config_resolvers  # noqa: F401  # pylint: disable=unused-import
from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.pipeline import _resolve_log_dir
from prpl_tidybot.preview import (
    find_floor_violation,
    planned_trajectory_from_agent,
    preview_or_abort,
)
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.sim_env import KinderSimEnv
from prpl_tidybot.toss import describe_toss
from prpl_tidybot.toss_outcome import TossCapture

_logger = logging.getLogger(__name__)

PromptFn = Callable[[str], str]

PICK_SKILL_NAME = "Pick"
_CUBE_NAME = "cube0"


@dataclass(frozen=True)
class PickAndTossSummary:
    """What happened, for the script's printout and for tests."""

    mode: str
    # True once the pick trajectory has been executed to completion; None when
    # the pick was skipped.
    picked: bool | None
    pick_steps: int
    tossed: bool
    # "tossed", "picked (toss skipped)", "toss rejected by operator", ...
    finish_reason: str
    toss: dict[str, Any] | None = None
    toss_outcome: dict[str, Any] | None = None
    final_state: ObjectCentricState | None = None
    # One record per cycle (also written to cycles.json in the log dir).
    cycles: list[dict[str, Any]] = field(default_factory=list)


class PickSkillAgent(BilevelPlanningAgent):
    """A one-skill "planner": roll out the Pick skill's controller until it works.

    Same contract as :class:`BilevelPlanningAgent` (the trajectory is produced in
    ``reset`` and handed out once by ``plan``; ``AgentFailure`` when there is none), so
    the Runner, the plan preview and the executors treat it like the bilevel planner.
    Each attempt samples controller parameters and rolls the controller out with the env
    model's transition function for at most ``max_skill_horizon`` steps, using SeSamE's
    own trajectory sampler; it succeeds when the abstract state afterwards is exactly
    the Pick operator's predicted successor.
    """

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_skill_horizon: int = 100,
        max_attempts: int = 20,
    ) -> None:
        super().__init__(env_models, seed, max_skill_horizon=max_skill_horizon)
        self._max_attempts = max_attempts
        self.attempts_used = 0
        # One stream across resets, so each cycle of a loop samples fresh
        # parameters rather than replaying the first cycle's.
        self._rng = np.random.default_rng(seed)

    def _run_planning(self) -> tuple[list, list]:
        models = self._env_models
        state = models.observation_to_state(self._last_observation)
        skill = next(
            (s for s in models.skills if s.operator.name == PICK_SKILL_NAME), None
        )
        if skill is None:
            raise AgentFailure(f"The env models have no {PICK_SKILL_NAME} skill.")
        objects = []
        for variable in skill.operator.parameters:
            match = next((o for o in state if o.is_instance(variable.type)), None)
            if match is None:
                raise AgentFailure(f"No object of type {variable.type} to pick.")
            objects.append(match)
        operator = skill.operator.ground(tuple(objects))
        abstract = models.state_abstractor(state)
        if not operator.preconditions <= abstract.atoms:
            raise AgentFailure(
                f"{operator.short_str} is not applicable: preconditions "
                f"{sorted(map(str, operator.preconditions))} vs abstract state "
                f"{sorted(map(str, abstract.atoms))}."
            )
        successor = RelationalAbstractState(
            (abstract.atoms - operator.delete_effects) | operator.add_effects,
            abstract.objects,
        )
        sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )
        for attempt in range(1, self._max_attempts + 1):
            self.attempts_used = attempt
            try:
                states, actions = sampler(
                    state,
                    abstract,
                    operator,
                    successor,
                    BilevelPlanningGraph(),
                    self._rng,
                )
            except TrajectorySamplingFailure as e:
                _logger.info("Pick attempt %d failed: %s", attempt, e or "no effect")
                continue
            _logger.info(
                "Pick attempt %d succeeded: %d actions.", attempt, len(actions)
            )
            return states, actions
        raise AgentFailure(
            f"No successful {PICK_SKILL_NAME} rollout in {self._max_attempts} attempts."
        )


def run_pick_and_toss(
    cfg: DictConfig,
    log_dir: Path | str | None = None,
    prompt_fn: PromptFn = input,
) -> PickAndTossSummary:
    """Run the pick and/or the toss described by ``cfg`` (conf/pick_and_toss.yaml).

    ``log_dir`` falls back to the Hydra runtime dir as in ``run_planner``; the plan
    preview and the toss capture need one and are skipped without it.
    """
    if cfg.skip_pick and cfg.skip_toss:
        raise ValueError("skip_pick and skip_toss together leave nothing to do.")
    if cfg.skip_pick and int(cfg.get("cycles", 1)) > 1:
        raise ValueError("cycles > 1 needs the pick; drop skip_pick.")
    interactive = (
        cfg.mode == "real" if cfg.get("interactive") is None else bool(cfg.interactive)
    )
    resolved_log_dir = _resolve_log_dir(log_dir)
    if cfg.skip_pick:
        return _run_toss_only(cfg, resolved_log_dir, interactive, prompt_fn)
    return _run_pick(cfg, resolved_log_dir, interactive, prompt_fn)


def _run_pick(
    cfg: DictConfig,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
) -> PickAndTossSummary:
    cycles = int(cfg.get("cycles", 1))
    if cycles < 1:
        raise ValueError(f"cycles must be at least 1, got {cycles}.")
    if cycles > 1:
        if cfg.skip_toss:
            raise ValueError("cycles > 1 needs the toss; drop skip_toss.")
        if cfg.mode == "sim":
            raise ValueError("cycles > 1 needs a toss, which the sim does not have.")
        if not cfg.capture.enabled or log_dir is None:
            raise ValueError(
                "cycles > 1 needs the landing capture (capture.enabled and a log "
                "dir): the landing decides whether the next pick is attempted."
            )
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
    agent = PickSkillAgent(
        env_models,
        cfg.seed,
        max_skill_horizon=cfg.agent.max_skill_horizon,
        max_attempts=cfg.pick.max_attempts,
    )
    runner: Runner = Runner(
        real_env=real_env, perceiver=perceiver, agent=agent, plan_executor=plan_executor
    )
    shadow_sim = None
    records: list[dict[str, Any]] = []
    last: _CycleResult | None = None
    stop_reason: str | None = None

    # As in run_planner: always close the env, so the arm's torque loop and
    # the base session do not outlive the process.
    try:
        for cycle in range(1, cycles + 1):
            if last is not None:
                stop_reason = _landing_problem(cfg, last.toss_outcome)
                if stop_reason is not None:
                    break
            if (
                shadow_sim is None
                and cfg.mode == "real"
                and log_dir is not None
                and bool(cfg.preview.get("enabled"))
            ):
                shadow_sim = hydra.utils.instantiate(
                    cfg.env.pipelines.sim.real_env, _convert_="all"
                )
            cycle_dir = log_dir
            if log_dir is not None and cycles > 1:
                cycle_dir = log_dir / f"cycle_{cycle:02d}"
                cycle_dir.mkdir(parents=True, exist_ok=True)
            _logger.info("Cycle %d/%d.", cycle, cycles)
            result = _run_cycle(
                cfg,
                cycle,
                runner,
                real_env,
                agent,
                shadow_sim,
                cycle_dir,
                interactive,
                prompt_fn,
                previous_landing=(
                    None
                    if last is None or last.toss_outcome is None
                    else last.toss_outcome["landing"]
                ),
            )
            records.append(result.record)
            last = result
            if result.stop_reason is not None:
                stop_reason = result.stop_reason
                break
    finally:
        if shadow_sim is not None:
            shadow_sim.close()
        real_env.close()

    assert last is not None
    if stop_reason is not None:
        _logger.warning("Stopping after cycle %d: %s", len(records), stop_reason)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "cycles.json").write_text(
            json.dumps(
                {
                    "cycles_requested": cycles,
                    "stop_reason": stop_reason,
                    "cycles": records,
                },
                indent=2,
            )
        )
    if cycles == 1:
        finish_reason = last.finish_reason
    elif stop_reason is None:
        finish_reason = f"completed {len(records)}/{cycles} cycles"
    else:
        finish_reason = f"stopped after {len(records)}/{cycles} cycles: {stop_reason}"
    return PickAndTossSummary(
        mode=cfg.mode,
        picked=last.picked,
        pick_steps=last.pick_steps,
        tossed=last.tossed,
        finish_reason=finish_reason,
        toss=last.toss,
        toss_outcome=last.toss_outcome,
        final_state=last.final_state,
        cycles=records,
    )


@dataclass
class _CycleResult:
    """One pick (and toss) of the loop."""

    record: dict[str, Any]
    picked: bool
    pick_steps: int
    tossed: bool
    finish_reason: str
    # Why the loop must stop after this cycle, or None to go on.
    stop_reason: str | None = None
    toss: dict[str, Any] | None = None
    toss_outcome: dict[str, Any] | None = None
    final_state: ObjectCentricState | None = None


def preview_prompts(cfg: DictConfig, cycle: int) -> bool:
    """Whether cycle ``cycle``'s plan preview asks the operator (real mode only)."""
    if cfg.mode != "real" or not bool(cfg.preview.get("enabled")):
        return False
    return cycle == 1 or not bool(cfg.loop.confirm_first_cycle_only)


def cube_pose_from_state(state: ObjectCentricState) -> dict[str, float]:
    """The perceived cube's map-frame position and yaw (from its quaternion)."""
    cube = state.get_object_from_name(_CUBE_NAME)
    qz = float(state.get(cube, "pose_qz"))
    qw = float(state.get(cube, "pose_qw"))
    qx = float(state.get(cube, "pose_qx"))
    qy = float(state.get(cube, "pose_qy"))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return {
        "x": float(state.get(cube, "pose_x")),
        "y": float(state.get(cube, "pose_y")),
        "z": float(state.get(cube, "pose_z")),
        "yaw": yaw,
    }


def _inside(x: float, y: float, box: Any) -> bool:
    x_min, x_max, y_min, y_max = (float(v) for v in box)
    return x_min <= x <= x_max and y_min <= y <= y_max


def _landing_problem(cfg: DictConfig, outcome: dict[str, Any] | None) -> str | None:
    """Why the cube's landing rules out another pick, or None."""
    if outcome is None or outcome["landing"] is None:
        return "cube not visible after the toss"
    x, y = outcome["landing"]["x"], outcome["landing"]["y"]
    floor_bounds = cfg.preview.get("floor_bounds")
    if floor_bounds is not None and not _inside(x, y, floor_bounds):
        return (
            f"cube at ({x:.2f}, {y:.2f}) is outside the base pose bounds "
            f"{list(floor_bounds)}"
        )
    region = cfg.loop.get("pickable_region")
    if region is not None and not _inside(x, y, region):
        return (
            f"cube at ({x:.2f}, {y:.2f}) is outside the pickable region {list(region)}"
        )
    return None


def _run_cycle(
    cfg: DictConfig,
    cycle: int,
    runner: Runner,
    real_env: RealTidyBotEnv | KinderSimEnv,
    agent: PickSkillAgent,
    shadow_sim: Any,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
    previous_landing: dict[str, Any] | None,
) -> _CycleResult:
    record: dict[str, Any] = {
        "cycle": cycle,
        "cube_perceived": None,
        "picked": False,
        "pick_attempts": None,
        "pick_steps": 0,
        "tossed": False,
        "landing": None,
        "outcome": None,
    }

    def done(reason: str, stop: str | None = None, **kwargs: Any) -> _CycleResult:
        record["outcome"] = reason
        return _CycleResult(
            record=record,
            picked=record["picked"],
            pick_steps=record["pick_steps"],
            tossed=record["tossed"],
            finish_reason=reason,
            stop_reason=stop,
            **kwargs,
        )

    _logger.info("Perceiving and rolling out the PickController...")
    start = time.monotonic()
    try:
        state = runner.reset(seed=cfg.seed)
    except AgentFailure as e:
        return done(f"pick_failure: {e}", stop=f"pick failed: {e}")
    cube = cube_pose_from_state(state)
    record["cube_perceived"] = cube
    record["pick_attempts"] = agent.attempts_used
    _logger.info(
        "Cube perceived at (%.3f, %.3f), yaw %.1f deg; pick trajectory found in "
        "%.1fs (%d attempt(s)).",
        cube["x"],
        cube["y"],
        math.degrees(cube["yaw"]),
        time.monotonic() - start,
        agent.attempts_used,
    )
    if previous_landing is not None:
        gap = math.dist(
            (cube["x"], cube["y"]), (previous_landing["x"], previous_landing["y"])
        )
        if gap > float(cfg.loop.max_landing_disagreement_m):
            reason = (
                f"cube perceived at ({cube['x']:.2f}, {cube['y']:.2f}), {gap:.2f} m "
                f"from where it landed ({previous_landing['x']:.2f}, "
                f"{previous_landing['y']:.2f})"
            )
            return done(f"pick_failure: {reason}", stop=reason, final_state=state)

    planned_states, planned_actions = planned_trajectory_from_agent(agent)
    steps = len(planned_actions)
    record["pick_steps"] = steps
    if shadow_sim is not None and log_dir is not None:
        preview_cfg = cfg.preview
        floor_bounds = (
            tuple(preview_cfg["floor_bounds"])
            if preview_cfg.get("floor_bounds") is not None
            else None
        )
        base_margin = float(preview_cfg.get("base_margin", 0.0))
        try:
            if preview_prompts(cfg, cycle):
                preview_or_abort(
                    planned_states=planned_states,
                    planned_actions=planned_actions,
                    shadow_sim=shadow_sim,
                    log_dir=log_dir,
                    seed=cfg.seed,
                    fps=int(preview_cfg.get("fps", 10)),
                    max_frames=preview_cfg.get("max_frames", 50),
                    prompt_fn=prompt_fn,
                    floor_bounds=floor_bounds,
                    base_margin=base_margin,
                )
            elif floor_bounds is not None:
                violation = find_floor_violation(
                    planned_states, floor_bounds, base_margin
                )
                if violation is not None:
                    index, x, y = violation
                    raise AgentFailure(
                        f"Plan leaves the floor: planned state {index} puts the "
                        f"base at ({x:.2f}, {y:.2f})"
                    )
        except AgentFailure as e:
            return done(
                f"pick_failure: {e}", stop=f"pick refused: {e}", final_state=state
            )

    try:
        state, _, _, _, _ = runner.step()
    except ExecutionFailure as e:
        _logger.warning("Pick stopped: %s", e)
        return done(
            f"executor_failure: {e}", stop=f"pick failed: {e}", final_state=state
        )
    except Exception as e:  # pylint: disable=broad-except
        # An RPC error out of the arm or base server mid-pick.
        return done(
            f"fault: {e}", stop=f"arm fault during the pick: {e}", final_state=state
        )
    record["picked"] = True
    _logger.info("Pick executed (%d planned steps).", steps)

    if cfg.skip_toss:
        return done("picked (toss skipped)", final_state=state)
    if cfg.mode == "sim":
        _logger.info("The kinematic sim has no toss; stopping after the pick.")
        return done("picked (toss not simulated in sim mode)", final_state=state)

    assert isinstance(real_env, RealTidyBotEnv)
    arm = real_env.interface.arm_interface
    confirm = interactive and (
        cycle == 1 or not bool(cfg.loop.confirm_first_cycle_only)
    )
    try:
        tossed, reason, toss_info, outcome = _toss(
            cfg, arm, log_dir, confirm, prompt_fn
        )
    except Exception as e:  # pylint: disable=broad-except
        # Includes the arm server's refusal to swing from a missed wind-up pose.
        return done(
            f"fault: {e}", stop=f"arm fault during the toss: {e}", final_state=state
        )
    record["tossed"] = tossed
    record["toss"] = toss_info
    if outcome is not None:
        record["landing"] = outcome["landing"]
        record["landing_status"] = outcome["status"]
        record["robot_map_pose_at_throw"] = outcome["robot_map_pose_at_throw"]
    return done(
        reason,
        stop=None if tossed else reason,
        toss=toss_info,
        toss_outcome=outcome,
        final_state=state,
    )


def _run_toss_only(
    cfg: DictConfig,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
) -> PickAndTossSummary:
    if cfg.mode not in cfg.toss_only.arm_interface:
        raise ValueError(
            f"Toss-only needs an arm that can toss; mode={cfg.mode} has none "
            f"(choose from {sorted(cfg.toss_only.arm_interface)})."
        )
    arm: ArmInterface = hydra.utils.instantiate(
        cfg.toss_only.arm_interface[cfg.mode], _convert_="all"
    )
    try:
        _hand_object_in(cfg, arm, interactive, prompt_fn)
        tossed, reason, toss_info, outcome = _toss(
            cfg, arm, log_dir, interactive, prompt_fn
        )
        return PickAndTossSummary(
            mode=cfg.mode,
            picked=None,
            pick_steps=0,
            tossed=tossed,
            finish_reason=reason,
            toss=toss_info,
            toss_outcome=outcome,
        )
    finally:
        arm.close()


def _hand_object_in(
    cfg: DictConfig, arm: ArmInterface, interactive: bool, prompt_fn: PromptFn
) -> None:
    """Open the gripper, wait for the operator, close it on whatever is there."""
    arm.execute_action(arm.get_arm_state(), 0.0)
    _wait_for_gripper(arm, cfg.toss.gripper_settle_timeout_s)
    if interactive:
        prompt_fn(
            "\nGripper open. Hand the object into the gripper (or leave it empty), "
            "keep clear, and press Enter to close it: "
        )
    arm.execute_action(arm.get_arm_state(), float(cfg.toss.gripper_close_position))
    _wait_for_gripper(arm, cfg.toss.gripper_settle_timeout_s)
    _logger.info("Gripper closed at %.2f.", arm.get_gripper_state())


def _wait_for_gripper(arm: ArmInterface, timeout_s: float) -> None:
    """Return once two gripper readings 0.1 s apart agree, or after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    last = arm.get_gripper_state()
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = arm.get_gripper_state()
        if abs(current - last) < 0.01:
            return
        last = current


def _toss(
    cfg: DictConfig,
    arm: ArmInterface,
    log_dir: Path | None,
    interactive: bool,
    prompt_fn: PromptFn,
) -> tuple[bool, str, dict[str, Any], dict[str, Any] | None]:
    """Confirm, then toss (filmed when capture is on).

    Returns ``(tossed, finish_reason, toss description, landing outcome or None)``.
    """
    toss_cfg = cfg.toss
    info = describe_toss(
        heading=float(toss_cfg.heading),
        max_vel=float(toss_cfg.max_vel),
        gripper_release_ms=int(toss_cfg.gripper_release_ms),
    )
    description = (
        f"heading {info['heading_deg']:.1f} deg, peak {info['max_vel_deg_s']:.0f} "
        f"deg/s, gripper opens {info['gripper_release_ms']} ms into a "
        f"{info['duration_s']:.2f} s swing ({100 * info['release_fraction']:.0f}% "
        "of the way)"
    )
    _logger.info("Toss: %s.", description)
    if interactive:
        answer = prompt_fn(
            f"\nAbout to toss: {description}.\n"
            "The arm will move to the wind-up pose and throw. Keep the throwing "
            "area clear. Toss? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            _logger.info("Toss rejected by operator (answer=%r).", answer)
            return False, f"toss rejected by operator (answer={answer!r})", info, None

    capture = _build_capture(cfg, log_dir)
    try:
        if capture is not None:
            capture.start()
        arm.toss(
            heading=float(toss_cfg.heading),
            max_vel=float(toss_cfg.max_vel),
            gripper_release_ms=int(toss_cfg.gripper_release_ms),
        )
        outcome = capture.finish(toss=info) if capture is not None else None
    finally:
        if capture is not None:
            capture.close()
    return True, "tossed", info, outcome


def _build_capture(cfg: DictConfig, log_dir: Path | None) -> TossCapture | None:
    capture_cfg = cfg.get("capture")
    if capture_cfg is None or not capture_cfg.enabled:
        return None
    if log_dir is None:
        _logger.info("No log dir; the toss is not filmed or measured.")
        return None
    if cfg.mode not in capture_cfg.pipelines:
        _logger.info("No toss capture for mode=%s.", cfg.mode)
        return None
    pipeline = capture_cfg.pipelines[cfg.mode]
    return TossCapture(
        renderer=hydra.utils.instantiate(pipeline.renderer, _convert_="all"),
        marker_client=hydra.utils.instantiate(pipeline.marker_client, _convert_="all"),
        log_dir=log_dir,
        cube_marker_ids=list(capture_cfg.cube_marker_ids),
        fps=int(capture_cfg.fps),
        post_toss_record_s=float(capture_cfg.post_toss_record_s),
        settle_s=float(capture_cfg.settle_s),
    )
