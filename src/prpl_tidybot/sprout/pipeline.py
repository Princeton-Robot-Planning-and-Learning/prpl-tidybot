"""Sprout recipe adapter for the original PRPL Runner/env/executor pipeline.

Only perception and the predefined trajectory are Sprout-specific. All motion
goes through RealTidyBotEnv and Kinematic3DPlanExecutor, as in main. Simulation
uses the same Runner with FakeInterface and the original Kinder shadow env.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from prpl_utils.planning_agent import PlanningAgent
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.interfaces.interface import FakeInterface, Interface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    CarrotArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.distance_factories import (
    create_kinova_distance_fn,
)
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.sprout.sequence import Stopped

Pair = tuple[ObjectCentricState, NDArray[np.floating]]


class RecipeAgent(PlanningAgent[ObjectCentricState, NDArray, ObjectCentricState]):
    """Supply one predefined phase to the existing trajectory-mode Runner."""

    def __init__(self) -> None:
        super().__init__(0)
        self.trajectory: list[Pair] = []

    def plan(self) -> list[Pair]:
        return self.trajectory

    def _get_action(self) -> NDArray:
        raise NotImplementedError("Use Runner trajectory mode")


def arm_pairs(state: ObjectCentricState, frames: list[dict]) -> list[Pair]:
    """Translate saved joint targets/open-close events to main's 11D actions."""
    current = state.copy()
    robot = current.get_object_from_name("robot")
    pairs = []
    last_grip = None
    for frame in frames:
        q = np.asarray(frame["arm_joints_rad"], dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected seven finite planned joints")
        action = np.zeros(11)
        previous = np.array([current.get(robot, f"joint_{j+1}") for j in range(7)])
        # Keep the continuous planned branch rather than introducing 2*pi jumps.
        action[3:10] = (q - previous + np.pi) % (2 * np.pi) - np.pi
        desired_grip = 1.0 if frame.get("gripper") == "close" else 0.0
        if frame.get("gripper") != last_grip:
            # A gripper event is its own waypoint, after reaching the arm target.
            if np.any(np.abs(action[3:10]) > 1e-9):
                pairs.append((current.copy(), action.copy()))
                for j, value in enumerate(previous + action[3:10]):
                    current.set(robot, f"joint_{j+1}", float(value))
                action[:] = 0
            action[10] = -1.0 if desired_grip else 1.0
        pairs.append((current.copy(), action))
        for j, value in enumerate(
            previous + (q - previous + np.pi) % (2 * np.pi) - np.pi
        ):
            current.set(robot, f"joint_{j+1}", float(value))
        current.set(robot, "finger_state", desired_grip)
        last_grip = frame.get("gripper")
    # Preserve the existing executor's gripper event/dwell handling even for
    # gripper-only phases whose joint target is already reached.
    if pairs and abs(pairs[-1][1][10]) > 0.5:
        pairs.append((current.copy(), np.zeros(11)))
    return pairs


def base_pairs(state: ObjectCentricState, path: list[list[float]]) -> list[Pair]:
    """Use the original base-only delta convention and Pure Pursuit executor."""
    current = state.copy()
    robot = current.get_object_from_name("robot")
    pairs = []
    for target in path[1:]:
        action = np.zeros(11)
        for i, key in enumerate(("pos_base_x", "pos_base_y", "pos_base_rot")):
            action[i] = target[i] - current.get(robot, key)
        action[2] = math.remainder(action[2], 2 * math.pi)
        pairs.append((current.copy(), action))
        for i, key in enumerate(("pos_base_x", "pos_base_y", "pos_base_rot")):
            current.set(robot, key, current.get(robot, key) + float(action[i]))
    return pairs


def recipe_phases(trajectory: list[dict]) -> list[tuple[str, list[dict]]]:
    """Keep phase boundaries so close/open complete before the next motion."""
    phases: list[tuple[str, list[dict]]] = []
    for frame in trajectory:
        name = frame["phase"]
        if not phases or phases[-1][0] != name:
            phases.append((name, []))
        phases[-1][1].append(frame)
    return phases


@lru_cache(maxsize=1)
def _distance_fn():
    """Reuse main's weighted joint metric and one model per process."""
    return create_kinova_distance_fn()


def make_executor(config: dict) -> Kinematic3DPlanExecutor:
    """Use the same executors as main's cylinder-shelf real pipeline."""
    return Kinematic3DPlanExecutor(
        base_executor=PurePursuitBaseMotion3DPlanExecutor(
            position_tolerance=config["navigation"]["position_tolerance_m"],
            angle_tolerance=config["navigation"]["yaw_tolerance_rad"],
        ),
        arm_executor=CarrotArmMotion3DPlanExecutor(
            distance_fn=_distance_fn(),
            advance_radius=0.03,
            arrival_tolerance=0.03,
            lookahead=0.05,
            max_iter_total=20000,
            gripper_dwell_ticks=20,
            gripper_close_position=config["motion"]["close_position"],
        ),
    )


class ObservedRunner(Runner):
    """Use the original Runner loop, adding only its public logging/stop hook."""

    def __init__(self, *args: Any, callback: Callable[..., None]) -> None:
        super().__init__(*args)
        self.callback = callback

    def on_step(
        self,
        state: Any,
        sim_action: Any,
        real_action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self.callback(
            state, sim_action, real_action, reward, terminated, truncated, info
        )


class SproutPipeline:
    """Thin phase orchestration around the original Runner, not a controller."""

    def __init__(
        self,
        interface: Interface,
        config: dict,
        stop: threading.Event,
        *,
        control_period: float = 0.1,
        on_step: Callable[..., None] | None = None,
    ) -> None:
        self.interface, self.config, self.stop_event = interface, config, stop
        self.env = RealTidyBotEnv(interface, control_period=control_period)
        self.agent = RecipeAgent()
        self.executor = make_executor(config)

        def checked_step(*args: Any) -> None:
            if self.stop_event.is_set():
                raise Stopped("Operator stopped the pipeline")
            if on_step is not None:
                on_step(*args)

        self.runner = ObservedRunner(
            self.env,
            PrplLab3DPerceiver(num_cubes=0),
            self.agent,
            self.executor,
            callback=checked_step,
        )
        self.state = self.runner.reset()

    def run(self, name: str, trajectory: list[Pair]) -> ObjectCentricState:
        """Execute one phase using Runner's unmodified inner feedback loop."""
        if self.stop_event.is_set():
            raise Stopped("Operator stopped the pipeline")
        print(f"PRPL {name}: {len(trajectory)} planned actions", flush=True)
        self.agent.trajectory = trajectory
        self.state, _, _, _, _ = self.runner.step()
        if self.stop_event.is_set():
            raise Stopped("Operator stopped the pipeline")
        return self.state

    def close(self) -> None:
        """Use main's env.close() cleanup for arm/base RPC loops."""
        self.env.close()


def fake_interface(observed: dict) -> FakeInterface:
    """Initialize the original FakeInterface from the measured preview state."""
    interface = FakeInterface()
    interface.arm_interface.execute_action(observed["arm_joints_rad"], 0.0)
    interface.base_interface.execute_action(SE2(*observed["base_xyyaw"]))
    return interface
