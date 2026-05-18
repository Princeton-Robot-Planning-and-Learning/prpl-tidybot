"""Tests for base_interface.py."""

import numpy as np
from spatialmath import SE2

from prpl_tidybot.interfaces.base_interface import FakeBaseInterface


def test_fake_base_interface_defaults():
    """FakeBaseInterface() starts at the origin in both frames."""
    base = FakeBaseInterface()
    identity = SE2(x=0, y=0, theta=0)
    assert np.allclose(base.get_base_state().A, identity.A)
    assert np.allclose(base.get_map_base_state().A, identity.A)


def test_fake_base_interface_execute_action():
    """execute_action() stores the commanded pose in both frames."""
    base = FakeBaseInterface()
    target = SE2(x=1.5, y=-0.5, theta=0.3)
    base.execute_action(target)
    assert np.allclose(base.get_base_state().A, target.A)
    assert np.allclose(base.get_map_base_state().A, target.A)
