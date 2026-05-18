"""Tests for real_sim/action_grounders/passthrough.py."""

from prpl_tidybot.real_sim.action_grounders.passthrough import (
    PassThroughActionGrounder,
)


def test_call_returns_action_unchanged():
    """Identity over any action type; sim_state is ignored."""
    grounder = PassThroughActionGrounder[str]()
    assert grounder("the-action", "irrelevant-state") == "the-action"
