"""Tests for `prpl_tidybot.replay_agent.NpzPlanAgent`.

Unit tests exercise the npz→trajectory conversion (frame transform, settle pairs, one-
shot plan contract) on a tiny synthetic plan; the pipeline test replays a truncated real
export (the bundled demo6 scene's first pick-and-place,
`tests/fixtures/restock3d_demo6_first_pick.npz`) end to end in fake mode.
"""

import math
from pathlib import Path

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

from prpl_tidybot.pipeline import run_planner
from prpl_tidybot.replay_agent import NpzPlanAgent

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / ("restock3d_demo6_first_pick.npz")
)

_HOME_JOINTS = [0.0, 0.3, 0.0, 1.2, 0.0, -0.6, 0.0]


def _make_state(
    base: tuple[float, float, float],
    joints: list[float],
    finger: float = 0.0,
) -> ObjectCentricState:
    robot = Object("robot", Kinematic3DRobotType)
    features = {
        "pos_base_x": base[0],
        "pos_base_y": base[1],
        "pos_base_rot": base[2],
        **{f"joint_{j + 1}": joints[j] for j in range(7)},
        "finger_state": finger,
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


def _write_plan(path: Path) -> None:
    """A 4-step plan: one base move, one arm move, one gripper close."""
    joints_moved = list(_HOME_JOINTS)
    joints_moved[3] += 0.5
    base = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.1], [0.1, 0.2, 0.1], [0.1, 0.2, 0.1]]
    )
    joints = np.array([_HOME_JOINTS, _HOME_JOINTS, joints_moved, joints_moved])
    gripper = np.array([0.31, 0.31, 0.31, 0.0])
    actions = np.zeros((3, 11))
    actions[0, :3] = [0.1, 0.2, 0.1]
    actions[1, 3:10] = np.array(joints_moved) - np.array(_HOME_JOINTS)
    actions[2, 10] = -1.0
    np.savez(path, base=base, joints=joints, gripper=gripper, actions=actions)


def test_plan_identity_frame(tmp_path: Path) -> None:
    """With the identity transform and the robot already at the start, the plan is
    served verbatim with no settle pairs."""
    plan_path = tmp_path / "plan.npz"
    _write_plan(plan_path)
    agent = NpzPlanAgent(plan_path)
    agent.reset(_make_state((0.0, 0.0, 0.0), list(_HOME_JOINTS)), {})
    pairs = agent.plan()
    assert len(pairs) == 3
    state0, action0 = pairs[0]
    robot = state0.get_object_from_name("robot")
    assert state0.get(robot, "pos_base_x") == pytest.approx(0.0)
    np.testing.assert_allclose(action0[:3], [0.1, 0.2, 0.1])
    # The last pair carries the gripper close.
    assert pairs[2][1][10] == pytest.approx(-1.0)


def test_plan_map_frame_transform(tmp_path: Path) -> None:
    """Base poses are rotated+translated and (dx, dy) deltas rotated; headings shift by
    the yaw and joints are untouched."""
    plan_path = tmp_path / "plan.npz"
    _write_plan(plan_path)
    yaw = math.pi / 2
    agent = NpzPlanAgent(
        plan_path, home_origin_map=(1.0, 2.0), home_yaw_map=yaw, settle_first=False
    )
    agent.reset(_make_state((1.0, 2.0, yaw), list(_HOME_JOINTS)), {})
    pairs = agent.plan()
    state0, action0 = pairs[0]
    robot = state0.get_object_from_name("robot")
    assert state0.get(robot, "pos_base_x") == pytest.approx(1.0)
    assert state0.get(robot, "pos_base_y") == pytest.approx(2.0)
    assert state0.get(robot, "pos_base_rot") == pytest.approx(yaw)
    # R(pi/2) @ (0.1, 0.2) = (-0.2, 0.1); dtheta unchanged.
    np.testing.assert_allclose(action0[:3], [-0.2, 0.1, 0.1], atol=1e-12)
    # State + delta lands on the transformed second pose.
    state1, _ = pairs[1]
    assert state1.get(robot, "pos_base_x") == pytest.approx(1.0 - 0.2)
    assert state1.get(robot, "pos_base_y") == pytest.approx(2.0 + 0.1)
    # Joints are frame-independent.
    assert state0.get(robot, "joint_4") == pytest.approx(_HOME_JOINTS[3])


def test_settle_pairs(tmp_path: Path) -> None:
    """A perceived start away from the export's start prepends an arm-only pair then a
    base-only pair, with the base heading delta wrapped."""
    plan_path = tmp_path / "plan.npz"
    _write_plan(plan_path)
    agent = NpzPlanAgent(plan_path)
    perceived_joints = [0.0] * 7
    agent.reset(_make_state((0.05, -0.02, -3.0), perceived_joints), {})
    pairs = agent.plan()
    assert len(pairs) == 5
    arm_action = pairs[0][1]
    np.testing.assert_allclose(arm_action[:3], 0.0)
    np.testing.assert_allclose(arm_action[3:10], _HOME_JOINTS)
    base_state, base_action = pairs[1]
    robot = base_state.get_object_from_name("robot")
    # The base pair holds the arm at the export's start configuration.
    assert base_state.get(robot, "joint_2") == pytest.approx(_HOME_JOINTS[1])
    np.testing.assert_allclose(base_action[3:11], 0.0)
    np.testing.assert_allclose(base_action[:2], [-0.05, 0.02])
    # 0.0 - (-3.0) = 3.0 wraps to itself; use the short way around instead:
    assert base_action[2] == pytest.approx(math.atan2(math.sin(3.0), math.cos(3.0)))


def test_repeated_gripper_commands_collapse(tmp_path: Path) -> None:
    """A run of repeated close commands (the export repeats ±1 while the sim's fingers
    move) is collapsed to its first step, so the executor dwells once per event instead
    of once per repeat."""
    joints = np.array([_HOME_JOINTS] * 5)
    base = np.zeros((5, 3))
    gripper = np.array([0.31, 0.2, 0.1, 0.0, 0.31])
    actions = np.zeros((4, 11))
    actions[0:3, 10] = -1.0  # one close event, repeated over three steps
    actions[3, 10] = 1.0  # a distinct open event right after stays intact
    plan_path = tmp_path / "plan.npz"
    np.savez(plan_path, base=base, joints=joints, gripper=gripper, actions=actions)
    agent = NpzPlanAgent(plan_path)
    agent.reset(_make_state((0.0, 0.0, 0.0), list(_HOME_JOINTS)), {})
    commands = [pair[1][10] for pair in agent.plan()]
    assert commands == [-1.0, 0.0, 0.0, 1.0]


def test_plan_is_one_shot(tmp_path: Path) -> None:
    """A second plan() call within an episode raises AgentFailure, matching the bilevel
    planner's one-shot contract."""
    plan_path = tmp_path / "plan.npz"
    _write_plan(plan_path)
    agent = NpzPlanAgent(plan_path)
    agent.reset(_make_state((0.0, 0.0, 0.0), list(_HOME_JOINTS)), {})
    agent.plan()
    with pytest.raises(AgentFailure):
        agent.plan()


def test_malformed_npz_rejected(tmp_path: Path) -> None:
    """An npz without the required arrays is rejected at construction."""
    plan_path = tmp_path / "bad.npz"
    np.savez(plan_path, base=np.zeros((4, 3)))
    with pytest.raises(ValueError, match="missing arrays"):
        NpzPlanAgent(plan_path)


def test_replay_pipeline_fake_mode() -> None:
    """The truncated demo6 export replays end to end through the executor stack in fake
    mode and the base finishes at the export's final pose."""
    overrides = [
        "env=restock3d-replay",
        "mode=fake",
        "max_eval_steps=3",
        "seed=0",
        f"env.plan_path={_FIXTURE}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    result = run_planner(cfg)
    # One full trajectory executes; the second Runner.step exhausts the
    # one-shot plan, which is the natural rollout end in fake mode.
    assert result.steps == 1
    assert result.finish_reason.startswith("agent_failure")
    robot = result.final_state.get_object_from_name("robot")
    with np.load(_FIXTURE) as plan:
        final_base = plan["base"][-1]
    assert result.final_state.get(robot, "pos_base_x") == pytest.approx(
        final_base[0], abs=0.05
    )
    assert result.final_state.get(robot, "pos_base_y") == pytest.approx(
        final_base[1], abs=0.05
    )
