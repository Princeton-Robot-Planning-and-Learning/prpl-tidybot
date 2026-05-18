"""Tests for real_sim/perceivers/passthrough.py."""

from prpl_tidybot.real_sim.perceivers.passthrough import PassThroughPerceiver


def test_reset_and_step_return_obs_unchanged():
    """Identity over any type."""
    perceiver = PassThroughPerceiver[str]()
    sentinel = "the-obs"
    assert perceiver.reset(sentinel, {}) is sentinel
    assert perceiver.step(sentinel, {}) is sentinel
