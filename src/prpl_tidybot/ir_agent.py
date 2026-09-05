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
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DEnv,
    ObjectCentricCylinderShelf3DEnv,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
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
from pybullet_helpers.geometry import SE2Pose
from relational_structs import ObjectCentricState

from prpl_tidybot.place_from_demo import load_demo, rewrite_places_with_demo

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
        only_pick_index: int | None = None,
        place_demo_path: str | None = None,
        place_demo_bottom_path: str | None = None,
    ) -> None:
        super().__init__(seed)
        self._plan_seed = seed
        self._ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
        self._samples_per_step = samples_per_step
        self._planning_timeout = planning_timeout
        self._robot_name = robot_name
        self._settle_first = settle_first
        # Execute only the N-th pick-and-place of the plan (0-based, plan
        # order): the full plan is still refined, then sliced to that
        # pick's segment; the settle pairs drive the robot from wherever
        # it stands to the slice's start. For testing one can at a time.
        self._only_pick_index = only_pick_index
        # A teleoperated place demonstration (see place_from_demo): when set,
        # every top-board (layer-1) placement is replaced by the demonstrated
        # motion, base-shifted laterally per can. Bottom-board placements keep
        # the refined plan.
        self._place_demo = (
            load_demo(place_demo_path) if place_demo_path is not None else None
        )
        # The bottom-board (layer-0) demonstration for the talls; the top-board
        # demo above covers the layer-1 shorts.
        self._place_demo_bottom = (
            load_demo(place_demo_bottom_path)
            if place_demo_bottom_path is not None
            else None
        )
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
                place_release_heights=restock_scene.PLACE_RELEASE_HEIGHTS,
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
            if plan is None:
                raise AgentFailure("Failed to refine the injected IR skeleton")
            actions = _collapse_gripper_runs(
                [np.asarray(a, dtype=float).ravel() for a in plan.actions]
            )
            states = list(plan.states)
            if self._place_demo is not None or self._place_demo_bottom is not None:
                states, actions = self._apply_place_demo(states, actions)
        finally:
            env.close()
        if self._only_pick_index is not None:
            states, actions = _slice_pick(states, actions, self._only_pick_index)
        return states, actions

    def _apply_place_demo(
        self,
        states: list[ObjectCentricState],
        actions: list[NDArray[np.floating]],
    ) -> tuple[list[ObjectCentricState], list[NDArray[np.floating]]]:
        """Replace each board's placements with its demonstration.

        The top-board demo covers layer-1 (shorts) and the bottom-board demo
        covers layer-0 (talls); either may be absent, leaving that board's
        placements on the model insertion.
        """
        shelf_x = float(self._config.shelf_pose.position[0])
        place_cylinders = [
            args[1] for op, args in self._ir["skeleton"] if op == "Place"
        ]
        fk_env = ObjectCentricCylinderShelf3DEnv(
            num_cylinders=len(self._config.cylinder_heights),
            config=self._config,
            allow_state_access=True,
        )
        try:

            def fk(joints, base):
                fk_env.robot.set_base(SE2Pose(*base))
                fk_env.robot.arm.set_joints(
                    extend_joints_to_include_fingers(list(joints[:7]))
                )
                return list(fk_env.robot.arm.get_end_effector_pose().position)

            for demo, layer in ((self._place_demo, 1), (self._place_demo_bottom, 0)):
                if demo is None:
                    continue
                targets_x: list[float | None] = [
                    (
                        shelf_x + self._ir["placements"][cyl]["x_offset"]
                        if self._ir["placements"][cyl]["layer"] == layer
                        else None
                    )
                    for cyl in place_cylinders
                ]
                states, actions = rewrite_places_with_demo(
                    states, actions, targets_x, demo, fk, self._robot_name
                )
            return states, actions
        finally:
            fk_env.close()

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

        # Settle states are built on the plan's start state, not the perceived
        # observation: the perceiver reports a robot-only ObjectCentricState,
        # which the preview's shadow sim cannot set_state (no scene objects,
        # not the env-specific state class). The executors read only the
        # robot features either way.
        perceived = start.copy()
        for feat in ("pos_base_x", "pos_base_y", "pos_base_rot", "finger_state"):
            perceived.set(start_robot, feat, float(obs.get(robot, feat)))
        for j in range(7):
            perceived.set(
                start_robot, f"joint_{j + 1}", float(obs.get(robot, f"joint_{j + 1}"))
            )

        perceived_joints = [float(obs.get(robot, f"joint_{j + 1}")) for j in range(7)]
        arm_delta = np.asarray(start_joints) - np.asarray(perceived_joints)
        if np.abs(arm_delta).max() > _MOTION_EPS:
            action = np.zeros(11)
            action[3:10] = arm_delta
            pairs.append((perceived, action))

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
            state = perceived.copy()
            for j in range(7):
                state.set(start_robot, f"joint_{j + 1}", start_joints[j])
            action = np.zeros(11)
            action[:3] = base_delta
            pairs.append((state, action))
        return pairs


def _slice_pick(
    states: list[ObjectCentricState],
    actions: list[NDArray[np.floating]],
    pick_index: int,
) -> tuple[list[ObjectCentricState], list[NDArray[np.floating]]]:
    """The sub-trajectory of the ``pick_index``-th pick-and-place (plan order).

    A pick-and-place runs from its staging base motion to the end of its
    release's retract: each gripper OPEN is followed by arm-only retract
    actions, and the next base motion begins the next pick.
    """
    starts = [0]
    for i, action in enumerate(actions):
        if float(action[10]) > 0.5:  # a release
            for j in range(i + 1, len(actions)):
                if np.abs(actions[j][:3]).max() > 1e-9:  # next base motion
                    starts.append(j)
                    break
    if not 0 <= pick_index < len(starts):
        raise ValueError(
            f"only_pick_index {pick_index} out of range: the plan has "
            f"{len(starts)} pick-and-places"
        )
    start = starts[pick_index]
    stop = starts[pick_index + 1] if pick_index + 1 < len(starts) else len(actions)
    return states[start : stop + 1], actions[start:stop]


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
