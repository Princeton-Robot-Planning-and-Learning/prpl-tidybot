"""End-to-end test for the real-to-sim-to-real pipeline.

Wires:

    FakeInterface -> RealTidyBotEnv
                     ^                          PrplLab3DPerceiver
                     |                                  |
                     |                                  v
            PurePursuitKinematic3DPlanExecutor <- PlanningAgent <- ObjectCentricState

through `prpl_utils.real_sim.Runner` and asserts that the FakeInterface
ends up at the cumulative absolute target after several Runner steps of
canned sim actions. Proves the env, perceiver, plan executor, and runner
compose end-to-end without real hardware.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from prpl_utils.planning_agent import PlanningAgent
from prpl_utils.real_sim import Runner
from relational_structs import ObjectCentricState

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim import PrplLab3DPerceiver, PurePursuitKinematic3DPlanExecutor


class _OneActionPerPlanAgent(
    PlanningAgent[ObjectCentricState, NDArray[np.floating], ObjectCentricState]
):
    """Yields one (state, action) pair per `plan()` call from a canned list.

    Pairs the planner's "trajectory" with the most recent perceived state, so each outer
    `Runner.step` runs exactly one inner real-env tick — matching the granularity of the
    old per-action Runner.
    """

    def __init__(
        self,
        actions: list[NDArray[np.floating]],
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self._actions = actions
        self._idx = 0

    def plan(self) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        if self._idx >= len(self._actions):
            return []
        action = self._actions[self._idx]
        self._idx += 1
        assert self._last_observation is not None
        return [(self._last_observation, action)]

    def _get_action(self) -> NDArray[np.floating]:  # pragma: no cover
        raise NotImplementedError("Not used in trajectory mode")

    def update(
        self,
        obs: ObjectCentricState,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        # Skip the base Agent's assertion that _last_action was set; the
        # Runner sets it via record_trajectory_step before update fires.
        self._last_observation = obs
        self._last_info = info


def _build_runner(
    actions: list[NDArray[np.floating]],
) -> tuple[Runner, FakeInterface]:
    interface = FakeInterface()
    env = RealTidyBotEnv(interface, control_period=0.0)
    runner: Runner = Runner(
        real_env=env,
        perceiver=PrplLab3DPerceiver(),
        agent=_OneActionPerPlanAgent(actions),
        plan_executor=PurePursuitKinematic3DPlanExecutor(),
    )
    return runner, interface


def test_constant_delta_accumulates_over_steps():
    """Five steps of the same 11-d delta should leave the FakeInterface at exactly 5 *
    delta in base and arm; gripper stays at the initial 0.0 because the gripper command
    is in the no-change band."""
    delta = np.zeros(11)
    delta[0] = 0.01  # base dx
    delta[1] = 0.02  # base dy
    delta[2] = 0.03  # base drot
    delta[3:10] = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    delta[10] = 0.0
    runner, interface = _build_runner([delta.copy() for _ in range(5)])

    runner.reset()
    for _ in range(5):
        runner.step()

    base = interface.get_base_state()
    assert base.x == pytest.approx(0.05)
    assert base.y == pytest.approx(0.10)
    assert base.theta() == pytest.approx(0.15)
    assert interface.get_arm_state() == pytest.approx(
        [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
    )
    assert interface.get_gripper_state() == 0.0


def test_gripper_close_then_open():
    """Gripper command <-0.5 closes the gripper (TidyBot 1.0); a subsequent >0.5 command
    opens it (TidyBot 0.0)."""
    close = np.zeros(11)
    close[10] = -1.0
    do_open = np.zeros(11)
    do_open[10] = 1.0
    runner, interface = _build_runner([close, do_open])

    runner.reset()
    runner.step()
    assert interface.get_gripper_state() == 1.0
    runner.step()
    assert interface.get_gripper_state() == 0.0


def test_single_outer_step_drives_multiple_inner_ticks():
    """Pure pursuit caps each commanded target to `lookahead_distance` ahead of the
    cursor along the path, so a single outer Runner.step against a path longer than
    `lookahead_distance` yields multiple inner env.step calls and arrives at the final
    waypoint."""
    delta = np.zeros(11)
    delta[0] = 1.0  # 1 m delta, much longer than the lookahead below
    interface = FakeInterface()
    env = RealTidyBotEnv(interface, control_period=0.0)
    runner: Runner = Runner(
        real_env=env,
        perceiver=PrplLab3DPerceiver(),
        agent=_OneActionPerPlanAgent([delta]),
        plan_executor=PurePursuitKinematic3DPlanExecutor(lookahead_distance=0.2),
    )

    runner.reset()
    runner.step()
    # FakeInterface instantly settles to whatever target it's commanded, so the
    # robot advances lookahead_distance per tick — multiple ticks are needed to
    # cover the 1 m path. The final position matches the final waypoint.
    assert interface.get_base_state().x == pytest.approx(1.0)
