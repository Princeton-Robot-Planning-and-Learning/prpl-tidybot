"""A PlanningAgent that plans locally from an alphatamp plan IR.

`InjectedPlanAgent` is the second-generation seam to alphatamp's restock3d
deploy kit. Where `plan_level_b.npz` replay shipped a full joint trajectory,
the IR (``plan_ir.json``, format ``restock3d-ir-v1``) ships only the
planner's decisions — the skill skeleton (which can to pick next, which
board it goes to) and the continuous placement positions. This agent refines
those decisions into motion HERE, with the kinder cylinder-shelf skills and
this robot's own calibration, via
``kinder_bilevel_planning.injection.run_injected_sesame`` (no abstract
search, one sample per step; a few tens of seconds).

The planning scene is `kinder_bilevel_planning.restock_scene`: the measured
boxed lab layout in the map frame, with the per-cylinder grasp/staging
calibration. Because the scene is already map-frame, the planned trajectory
needs no frame conversion; the physical staging must match the scene (cans
at their spots inside the two boxes, shelf at its surveyed pose).

Planning runs once, lazily, on the first :meth:`reset`; the plan is then
served through the one-shot :meth:`plan` contract and the standard executor
stack (`Kinematic3DPlanExecutor`) drives the robot along it. Like the npz
replay agent, :meth:`reset` prepends unplanned settle pairs (arm first, then
base) from the perceived start configuration to the plan's first
configuration, so staging error is taken up by an explicit move.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from kinder.envs.kinematic3d.cylinder_shelf3d import CylinderShelf3DEnv
from kinder_bilevel_planning import restock_scene
from kinder_bilevel_planning.agent import AgentFailure
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from kinder_bilevel_planning.injection import (
    place_params_from_ir,
    run_injected_sesame,
    skeleton_from_ir,
)
from numpy.typing import NDArray
from prpl_utils.planning_agent import PlanningAgent
from relational_structs import ObjectCentricState

# Same threshold the plan executor uses to classify a pair as base or arm
# motion; settle pairs below it would be no-ops.
_MOTION_EPS = 1e-4


class InjectedPlanAgent(
    PlanningAgent[ObjectCentricState, NDArray[np.floating], ObjectCentricState]
):
    """Refine an alphatamp plan IR into a trajectory and serve it once."""

    def __init__(
        self,
        ir_path: str | Path,
        seed: int = 0,
        samples_per_step: int = 1,
        planning_timeout: float = 300.0,
        robot_name: str = "robot",
        settle_first: bool = True,
    ) -> None:
        super().__init__(seed)
        self._plan_seed = seed
        self._ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
        self._samples_per_step = samples_per_step
        self._planning_timeout = planning_timeout
        self._robot_name = robot_name
        self._settle_first = settle_first
        self._config = restock_scene.real_restock_config()
        self._check_ir_objects()
        # Filled by the lazy planning pass on the first reset.
        self._plan_states: list[ObjectCentricState] | None = None
        self._plan_actions: list[NDArray[np.floating]] = []
        # Exposed under the same private names BilevelPlanningAgent uses so
        # `preview.planned_trajectory_from_agent` finds them.
        self._planned_states: list[ObjectCentricState] = []
        self._planned_actions: list[NDArray[np.floating]] = []
        self._plan_consumed = True  # nothing to serve before the first reset

    def reset(self, obs: ObjectCentricState, info: dict[str, Any]) -> None:
        super().reset(obs, info)
        if self._plan_states is None:
            self._plan_states, self._plan_actions = self._compute_plan()
        settle = self._settle_pairs(obs) if self._settle_first else []
        self._planned_states = [s for s, _ in settle] + self._plan_states
        self._planned_actions = [a for _, a in settle] + list(self._plan_actions)
        self._plan_consumed = False

    def plan(self) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        """Return the refined trajectory; a second call raises AgentFailure.

        The one-shot contract matches `BilevelPlanningAgent.plan`: the IR
        holds one skeleton, so exhausting the plan ends the rollout.
        """
        if self._plan_consumed:
            raise AgentFailure("Injected plan already consumed")
        self._plan_consumed = True
        return list(zip(self._planned_states[:-1], self._planned_actions))

    def _get_action(self) -> NDArray[np.floating]:
        raise RuntimeError("InjectedPlanAgent is driven via plan(), not step()")

    # ------------------------------------------------------------- Internals

    def _check_ir_objects(self) -> None:
        """The IR must describe the same cans, in the same order, as the scene."""
        heights = self._config.cylinder_heights
        objects = self._ir.get("objects", [])
        if len(objects) != len(heights):
            raise ValueError(
                f"IR has {len(objects)} objects; the scene has {len(heights)}"
            )
        for i, obj in enumerate(objects):
            if obj["cylinder"] != f"cylinder{i}":
                raise ValueError(
                    f"IR object #{i} maps to {obj['cylinder']!r}, expected cylinder{i}"
                )
            if not math.isclose(obj["height"], heights[i], abs_tol=1e-6):
                raise ValueError(
                    f"IR object {obj['name']} height {obj['height']} != scene "
                    f"cylinder{i} height {heights[i]}"
                )

    def _compute_plan(
        self,
    ) -> tuple[list[ObjectCentricState], list[NDArray[np.floating]]]:
        num = len(self._config.cylinder_heights)
        env = CylinderShelf3DEnv(
            num_cylinders=num, config=self._config, allow_state_access=True
        )
        try:
            env_models = create_bilevel_planning_models(
                "cylinder_shelf3d",
                env.observation_space,
                env.action_space,
                num_objects=num,
                config=self._config,
                place_params=place_params_from_ir(
                    self._ir,
                    y_offset=restock_scene.PLACE_Y_OFFSET,
                    base_distance=restock_scene.PLACE_BASE_DISTANCE,
                ),
                grasp_params=restock_scene.real_restock_grasp_params(),
                move_params=restock_scene.real_restock_move_params(),
                carry_lift_z=restock_scene.CARRY_LIFT_Z,
            )
            obs, _ = env.reset(seed=self._plan_seed)
            x0 = env_models.observation_to_state(obs)
            plan, _ = run_injected_sesame(
                env_models,
                x0,
                skeleton_from_ir(self._ir),
                seed=self._plan_seed,
                samples_per_step=self._samples_per_step,
                timeout=self._planning_timeout,
            )
        finally:
            env.close()
        if plan is None:
            raise AgentFailure("Failed to refine the injected IR skeleton")
        actions = _collapse_gripper_runs(
            [np.asarray(a, dtype=float).ravel() for a in plan.actions]
        )
        return list(plan.states), actions

    def _settle_pairs(
        self, obs: ObjectCentricState
    ) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        """Unplanned pairs from the perceived start to the plan's start."""
        assert self._plan_states is not None
        robot = obs.get_object_from_name(self._robot_name)
        start = self._plan_states[0]
        start_robot = start.get_object_from_name(self._robot_name)
        start_joints = [
            float(start.get(start_robot, f"joint_{j + 1}")) for j in range(7)
        ]
        pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []

        perceived_joints = [float(obs.get(robot, f"joint_{j + 1}")) for j in range(7)]
        arm_delta = np.asarray(start_joints) - np.asarray(perceived_joints)
        if np.abs(arm_delta).max() > _MOTION_EPS:
            action = np.zeros(11)
            action[3:10] = arm_delta
            pairs.append((obs, action))

        base_delta = np.array(
            [
                float(start.get(start_robot, "pos_base_x"))
                - float(obs.get(robot, "pos_base_x")),
                float(start.get(start_robot, "pos_base_y"))
                - float(obs.get(robot, "pos_base_y")),
                _wrap_angle(
                    float(start.get(start_robot, "pos_base_rot"))
                    - float(obs.get(robot, "pos_base_rot"))
                ),
            ]
        )
        if np.abs(base_delta).max() > _MOTION_EPS:
            # The pair's state carries the post-arm-settle joints so the base
            # executor holds the arm there while driving.
            state = obs.copy()
            for j in range(7):
                state.set(robot, f"joint_{j + 1}", start_joints[j])
            action = np.zeros(11)
            action[:3] = base_delta
            pairs.append((state, action))
        return pairs


def _collapse_gripper_runs(
    actions: Sequence[NDArray[np.floating]],
) -> list[NDArray[np.floating]]:
    """Keep only the first command of each run of repeated gripper commands.

    The sim's fingers take a few steps to move, so the plan repeats its ±1
    gripper command, but the arm executor treats every gripper pair as its
    own event and dwells `gripper_dwell_ticks` at each; the repeats would
    multiply the dwell. One command per event is sufficient: the executor
    latches the last explicit open/close and re-issues it on hold ticks.
    """
    out = [a.copy() for a in actions]
    previous_cmd = 0.0
    for row in out:
        cmd = float(row[10])
        if abs(cmd) > 0.5 and abs(previous_cmd) > 0.5 and cmd * previous_cmd > 0:
            row[10] = 0.0
        else:
            previous_cmd = cmd
    return out


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
