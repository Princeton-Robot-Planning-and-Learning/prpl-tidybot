"""Tests for `prpl_tidybot.ir_agent.InjectedPlanAgent`.

The fixture (`tests/fixtures/restock3d_plan_ir.json`) is a genuine export from
alphatamp's restock3d deploy kit on the measured lab scene. The unit test refines it
once and checks the served trajectory's invariants; the pipeline test runs the same IR
end to end through the executor stack in fake mode.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DEnvTypeFeatures,
    Kinematic3DRobotType,
)
from kinder_bilevel_planning.agent import AgentFailure
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.ir_agent import InjectedPlanAgent
from prpl_tidybot.pipeline import run_planner

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "restock3d_plan_ir.json"


def _make_obs(
    base: tuple[float, float, float], joints: list[float]
) -> ObjectCentricState:
    robot = Object("robot", Kinematic3DRobotType)
    features = {
        "pos_base_x": base[0],
        "pos_base_y": base[1],
        "pos_base_rot": base[2],
        **{f"joint_{j + 1}": joints[j] for j in range(7)},
        "finger_state": 0.0,
        "grasp_active": 0.0,
        "grasp_tf_x": 0.0,
        "grasp_tf_y": 0.0,
        "grasp_tf_z": 0.0,
        "grasp_tf_qx": 0.0,
        "grasp_tf_qy": 0.0,
        "grasp_tf_qz": 0.0,
        "grasp_tf_qw": 1.0,
    }
    return create_state_from_dict({robot: features}, Kinematic3DEnvTypeFeatures)


@pytest.fixture(scope="module", name="planned_agent")
def _planned_agent() -> InjectedPlanAgent:
    """One refined agent shared by the unit tests (planning takes ~10-60 s)."""
    agent = InjectedPlanAgent(_FIXTURE, seed=0)
    agent.reset(_make_obs((1.48, 0.67, 1.54), [0.0] * 7), {})
    return agent


def test_served_trajectory_invariants(planned_agent: InjectedPlanAgent) -> None:
    """The refined trajectory is non-trivial, 11-D, and every action moves either the
    base or the arm/gripper — never both (the executor dispatcher enforces this)."""
    pairs = planned_agent.plan()
    assert len(pairs) > 100
    for _, action in pairs:
        assert action.shape == (11,)
        base_moves = bool(np.abs(action[:3]).max() > 1e-4)
        arm_moves = bool(np.abs(action[3:]).max() > 1e-4)
        assert not (base_moves and arm_moves)
    # Six picks and six places: six close events and six open events.
    gripper = [float(a[10]) for _, a in pairs]
    assert sum(1 for g in gripper if g < -0.5) == 6
    assert sum(1 for g in gripper if g > 0.5) == 6


def test_plan_is_one_shot(planned_agent: InjectedPlanAgent) -> None:
    """A second plan() call within an episode raises AgentFailure, matching the
    bilevel planner's one-shot contract."""
    with pytest.raises(AgentFailure):
        planned_agent.plan()


def test_mismatched_ir_rejected(tmp_path: Path) -> None:
    """An IR whose object table disagrees with the scene fails at construction."""
    ir = json.loads(_FIXTURE.read_text())
    ir["objects"][0]["height"] = 0.5
    bad = tmp_path / "bad_ir.json"
    bad.write_text(json.dumps(ir))
    with pytest.raises(ValueError, match="height"):
        InjectedPlanAgent(bad)


def test_settle_states_render_in_shadow_sim() -> None:
    """Every planned state — the settle pairs included — can be set_state into the
    shadow sim, which is what the real-mode preview does. (The settle states must be
    built on the plan's start state: the perceiver's robot-only observation has no
    scene objects and the wrong state class.)"""
    agent = InjectedPlanAgent(_FIXTURE, seed=0)
    # An obs whose base AND arm differ from the plan start, so both settle
    # pairs are generated.
    agent.reset(_make_obs((1.2, 0.4, 1.0), [0.0] * 7), {})
    overrides = ["env=restock3d-ir", "mode=real", f"env.ir_path={_FIXTURE}"]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    shadow_sim = hydra.utils.instantiate(
        cfg.env.pipelines.sim.real_env, _convert_="all"
    )
    shadow_sim.reset(seed=0)
    # pylint: disable=protected-access
    assert len(agent._planned_states) == len(agent._planned_actions) + 1
    # Both settle pairs were generated: arm-only, then base-only.
    assert np.abs(agent._planned_actions[0][3:10]).max() > 1e-4
    assert np.abs(agent._planned_actions[1][:3]).max() > 1e-4
    for state in agent._planned_states[:3]:
        shadow_sim.set_state(state)


def test_ir_pipeline_fake_mode() -> None:
    """The IR refines and executes end to end through the executor stack in fake
    mode, finishing at the plan's final base pose."""
    overrides = [
        "env=restock3d-ir",
        "mode=fake",
        "max_eval_steps=3",
        "seed=0",
        f"env.ir_path={_FIXTURE}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    result = run_planner(cfg)
    # One full trajectory executes; the second Runner.step exhausts the
    # one-shot plan, which is the natural rollout end in fake mode.
    assert result.steps == 1
    assert result.finish_reason.startswith("agent_failure")
