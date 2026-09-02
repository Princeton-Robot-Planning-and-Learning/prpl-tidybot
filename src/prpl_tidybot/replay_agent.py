"""A PlanningAgent that replays an externally planned kinder trajectory.

`NpzPlanAgent` loads a dense absolute base+arm trajectory from an npz file —
the "Level B" export of alphatamp's `restock3d_deploy` kit — and serves it
through the :meth:`plan` contract, so the existing executor stack
(`Kinematic3DPlanExecutor` and its base/arm sub-executors) drives the robot
along it. No planning happens here; the npz was produced by a planner running
in its own repo and venv, and this file is the seam between the two stacks.

Expected npz arrays, one row per timestep ``t`` (``T`` timesteps total):

* ``base`` ``(T, 3)`` — base pose ``(x, y, theta)``, home frame, m/rad.
* ``joints`` ``(T, 7)`` — the 7 Kinova joint angles (rad).
* ``gripper`` ``(T,)`` — finger opening; only recorded into ``finger_state``.
* ``actions`` ``(T-1, 11)`` — per-step deltas
  ``[base_dx, base_dy, base_dtheta, dj1..dj7, gripper]`` with the kinder
  bipolar gripper convention (``< -0.5`` close, ``> 0.5`` open, else hold).
  Every action must move either the base or the arm/gripper, never both —
  the dispatcher enforces this at ``set_trajectory`` time.

The export's home frame has its origin at the robot's planned start pose;
``home_origin_map`` / ``home_yaw_map`` place that frame in the map frame
(base poses are rotated+translated, per-step ``(dx, dy)`` deltas are rotated,
headings and joint angles shift/copy unchanged).

The trajectory assumes the robot starts at the export's first configuration.
With ``settle_first`` (the default), :meth:`reset` prepends up to two
unplanned settle pairs built from the perceived initial state — an arm pair
to the export's first joint configuration, then a base pair to its first base
pose (arm before base, mirroring `SettleGapExecutor`) — so small staging
error is taken up by an explicit move instead of being smeared into the first
planned segment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DEnvTypeFeatures,
    Kinematic3DRobotType,
)
from kinder_bilevel_planning.agent import AgentFailure
from numpy.typing import NDArray
from prpl_utils.planning_agent import PlanningAgent
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

# Same thresholds the plan executor uses to classify a pair as base or arm
# motion; settle pairs below these would be no-ops.
_MOTION_EPS = 1e-4


class NpzPlanAgent(
    PlanningAgent[ObjectCentricState, NDArray[np.floating], ObjectCentricState]
):
    """Serve a Level-B npz trajectory through the PlanningAgent contract."""

    def __init__(
        self,
        plan_path: str | Path,
        seed: int = 0,
        home_origin_map: Sequence[float] = (0.0, 0.0),
        home_yaw_map: float = 0.0,
        robot_name: str = "robot",
        settle_first: bool = True,
    ) -> None:
        super().__init__(seed)
        self._robot_name = robot_name
        self._settle_first = settle_first
        base, joints, gripper, actions = _load_plan(Path(plan_path))
        actions = _collapse_gripper_runs(actions)
        self._base, self._actions = _to_map_frame(
            base,
            actions,
            (float(home_origin_map[0]), float(home_origin_map[1])),
            home_yaw_map,
        )
        self._joints = joints
        self._gripper = gripper
        # The full planned trajectory (settle pairs included), rebuilt on every
        # reset. Exposed under the same private names BilevelPlanningAgent
        # uses so `preview.planned_trajectory_from_agent` finds them.
        self._planned_states: list[ObjectCentricState] = []
        self._planned_actions: list[NDArray[np.floating]] = []
        self._plan_consumed = True  # no plan to serve before the first reset

    def reset(self, obs: ObjectCentricState, info: dict[str, Any]) -> None:
        super().reset(obs, info)
        states = [self._make_state(t) for t in range(len(self._base))]
        settle = self._settle_pairs(obs) if self._settle_first else []
        self._planned_states = [s for s, _ in settle] + states
        self._planned_actions = [a for _, a in settle] + list(self._actions)
        self._plan_consumed = False

    def plan(self) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        """Return the replay trajectory; a second call raises AgentFailure.

        The one-shot contract matches `BilevelPlanningAgent.plan`: there is
        nothing to replan from, so exhausting the plan ends the rollout.
        """
        if self._plan_consumed:
            raise AgentFailure("Replay trajectory already consumed")
        self._plan_consumed = True
        return list(zip(self._planned_states[:-1], self._planned_actions))

    def _get_action(self) -> NDArray[np.floating]:
        raise RuntimeError("NpzPlanAgent is driven via plan(), not step()")

    # ------------------------------------------------------------- Internals

    def _make_state(self, t: int) -> ObjectCentricState:
        return self._state_from_features(
            base=(
                float(self._base[t, 0]),
                float(self._base[t, 1]),
                float(self._base[t, 2]),
            ),
            joints=[float(j) for j in self._joints[t]],
            finger=float(self._gripper[t]),
        )

    def _state_from_features(
        self,
        base: tuple[float, float, float],
        joints: Sequence[float],
        finger: float,
    ) -> ObjectCentricState:
        robot = Object(self._robot_name, Kinematic3DRobotType)
        features = {
            "pos_base_x": base[0],
            "pos_base_y": base[1],
            "pos_base_rot": base[2],
            **{f"joint_{j + 1}": joints[j] for j in range(7)},
            "finger_state": finger,
            # Grasp bookkeeping is not tracked on the real side; the
            # executors never read these.
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

    def _settle_pairs(
        self, obs: ObjectCentricState
    ) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        """Unplanned pairs from the perceived start to the export's start."""
        robot = obs.get_object_from_name(self._robot_name)
        perceived_joints = [float(obs.get(robot, f"joint_{j + 1}")) for j in range(7)]
        perceived_base = (
            float(obs.get(robot, "pos_base_x")),
            float(obs.get(robot, "pos_base_y")),
            float(obs.get(robot, "pos_base_rot")),
        )
        pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []

        arm_delta = np.asarray(self._joints[0], dtype=float) - np.asarray(
            perceived_joints, dtype=float
        )
        if np.abs(arm_delta).max() > _MOTION_EPS:
            action = np.zeros(11)
            action[3:10] = arm_delta
            pairs.append((obs, action))

        start = self._base[0]
        base_delta = np.array(
            [
                float(start[0]) - perceived_base[0],
                float(start[1]) - perceived_base[1],
                _wrap_angle(float(start[2]) - perceived_base[2]),
            ]
        )
        if np.abs(base_delta).max() > _MOTION_EPS:
            # The pair's state carries the post-arm-settle joints so the base
            # executor holds the arm there while driving.
            state = self._state_from_features(
                base=perceived_base,
                joints=[float(j) for j in self._joints[0]],
                finger=float(obs.get(robot, "finger_state")),
            )
            action = np.zeros(11)
            action[:3] = base_delta
            pairs.append((state, action))
        return pairs


def _load_plan(
    path: Path,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")
    with np.load(path) as data:
        missing = {"base", "joints", "gripper", "actions"} - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        base = np.asarray(data["base"], dtype=float)
        joints = np.asarray(data["joints"], dtype=float)
        gripper = np.asarray(data["gripper"], dtype=float)
        actions = np.asarray(data["actions"], dtype=float)
    num_steps = base.shape[0]
    if (
        base.shape != (num_steps, 3)
        or joints.shape != (num_steps, 7)
        or gripper.shape != (num_steps,)
        or actions.shape != (num_steps - 1, 11)
    ):
        raise ValueError(
            f"{path} has inconsistent shapes: base {base.shape}, joints "
            f"{joints.shape}, gripper {gripper.shape}, actions {actions.shape}"
        )
    return base, joints, gripper, actions


def _collapse_gripper_runs(
    actions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Keep only the first command of each run of repeated gripper commands.

    The export repeats its ±1 gripper command for the 2-3 steps the sim's
    fingers take to move, but the arm executor treats every gripper pair as
    its own event and dwells `gripper_dwell_ticks` at each, so the repeats
    would multiply the dwell (~6 s of extra hold per pick on the real robot).
    One command per event is sufficient: the executor latches the last
    explicit open/close and re-issues it on hold ticks.
    """
    actions = actions.copy()
    previous_cmd = 0.0
    for row in actions:
        cmd = float(row[10])
        if abs(cmd) > 0.5 and abs(previous_cmd) > 0.5 and cmd * previous_cmd > 0:
            row[10] = 0.0
        else:
            previous_cmd = cmd
    return actions


def _to_map_frame(
    base: NDArray[np.floating],
    actions: NDArray[np.floating],
    origin: tuple[float, float],
    yaw: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Rotate+translate home-frame base poses (and rotate deltas) into the map frame.

    Joint columns and the gripper command are frame-independent.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    base = base.copy()
    base[:, :2] = base[:, :2] @ rotation.T + np.asarray(origin)
    base[:, 2] += yaw
    actions = actions.copy()
    actions[:, :2] = actions[:, :2] @ rotation.T
    return base, actions


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
