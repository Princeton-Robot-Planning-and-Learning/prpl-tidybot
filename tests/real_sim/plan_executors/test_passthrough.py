"""Tests for real_sim/plan_executors/passthrough.py."""

from prpl_tidybot.real_sim.plan_executors.passthrough import (
    PassThroughPlanExecutor,
)


def test_passthrough_walks_trajectory_one_action_per_tick():
    """One real-env tick per (state, action) pair; real action == sim action."""
    executor = PassThroughPlanExecutor[str]()
    executor.set_trajectory([("s0", "a"), ("s1", "b"), ("s2", "c")])
    assert executor.done("ignored") is False
    assert executor.step("ignored") == ("a", "a")
    assert executor.step("ignored") == ("b", "b")
    assert executor.done("ignored") is False
    assert executor.step("ignored") == ("c", "c")
    assert executor.done("ignored") is True


def test_passthrough_empty_trajectory_is_immediately_done():
    """An empty trajectory yields no ticks."""
    executor = PassThroughPlanExecutor[str]()
    executor.set_trajectory([])
    assert executor.done("ignored") is True


def test_passthrough_set_trajectory_resets_progress():
    """set_trajectory restarts the executor from the beginning."""
    executor = PassThroughPlanExecutor[str]()
    executor.set_trajectory([("s0", "a"), ("s1", "b")])
    executor.step("ignored")
    executor.step("ignored")
    assert executor.done("ignored") is True
    executor.set_trajectory([("s0", "x"), ("s1", "y"), ("s2", "z")])
    assert executor.done("ignored") is False
    assert executor.step("ignored") == ("x", "x")
