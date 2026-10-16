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

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
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
from prpl_tidybot.preview import planned_trajectory_from_agent, preview_or_abort
from prpl_tidybot.real_sim import build_planner_env_models
from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.toss import describe_toss
from prpl_tidybot.toss_outcome import TossCapture

_logger = logging.getLogger(__name__)

PromptFn = Callable[[str], str]

PICK_SKILL_NAME = "Pick"


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
        rng = np.random.default_rng(self._seed)
        for attempt in range(1, self._max_attempts + 1):
            self.attempts_used = attempt
            try:
                states, actions = sampler(
                    state, abstract, operator, successor, BilevelPlanningGraph(), rng
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

    def summary(**kwargs: Any) -> PickAndTossSummary:
        return PickAndTossSummary(mode=cfg.mode, **kwargs)

    # As in run_planner: always close the env, so the arm's torque loop and
    # the base session do not outlive the process.
    try:
        _logger.info("Perceiving and rolling out the PickController...")
        start = time.monotonic()
        try:
            state = runner.reset(seed=cfg.seed)
        except AgentFailure as e:
            return summary(
                picked=False,
                pick_steps=0,
                tossed=False,
                finish_reason=f"pick_failure: {e}",
            )
        _logger.info(
            "Pick trajectory found in %.1fs (%d attempt(s)).",
            time.monotonic() - start,
            agent.attempts_used,
        )

        preview_cfg = cfg.get("preview")
        if (
            preview_cfg is not None
            and bool(preview_cfg.get("enabled"))
            and cfg.mode == "real"
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
                    log_dir=log_dir,
                    seed=cfg.seed,
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
                return summary(
                    picked=False,
                    pick_steps=0,
                    tossed=False,
                    finish_reason=f"pick_failure: {e}",
                    final_state=state,
                )
            finally:
                shadow_sim.close()

        steps = len(planned_trajectory_from_agent(agent)[1])
        try:
            state, _, _, _, _ = runner.step()
        except ExecutionFailure as e:
            _logger.warning("Pick stopped: %s", e)
            return summary(
                picked=False,
                pick_steps=steps,
                tossed=False,
                finish_reason=f"executor_failure: {e}",
                final_state=state,
            )
        _logger.info("Pick executed (%d planned steps).", steps)

        if cfg.skip_toss:
            return summary(
                picked=True,
                pick_steps=steps,
                tossed=False,
                finish_reason="picked (toss skipped)",
                final_state=state,
            )
        if cfg.mode == "sim":
            _logger.info("The kinematic sim has no toss; stopping after the pick.")
            return summary(
                picked=True,
                pick_steps=steps,
                tossed=False,
                finish_reason="picked (toss not simulated in sim mode)",
                final_state=state,
            )
        arm = real_env.interface.arm_interface
        tossed, reason, toss_info, outcome = _toss(
            cfg, arm, log_dir, interactive, prompt_fn
        )
        return summary(
            picked=True,
            pick_steps=steps,
            tossed=tossed,
            finish_reason=reason,
            toss=toss_info,
            toss_outcome=outcome,
            final_state=state,
        )
    finally:
        real_env.close()


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
        info["swing_start_time"] = time.time()
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
