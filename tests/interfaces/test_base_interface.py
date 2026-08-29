"""Tests for base_interface.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.interfaces.base_interface import FakeBaseInterface, unwrap_heading


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


@pytest.mark.parametrize(
    "target, reference, expected",
    [
        (0.3, 0.0, 0.3),
        (-3.0, 3.1, 3.2831853071795862),  # just past +pi: stay on that branch
        (3.0, -3.1, -3.2831853071795862),  # just past -pi
        (1.76, 1.87 + 2 * np.pi, 1.76 + 2 * np.pi),  # controller two turns in
        (-1.0, 12.0, 11.566370614359172),
    ],
)
def test_unwrap_heading_picks_the_branch_nearest_the_reference(
    target, reference, expected
):
    """The unwrapped heading is equivalent to the target and within pi of the
    reference, so the controller turns the short way instead of a full circle."""
    unwrapped = unwrap_heading(target, reference)
    assert abs(unwrapped - expected) < 1e-9
    assert abs(unwrapped - reference) <= np.pi + 1e-9
    assert (
        abs(np.arctan2(np.sin(unwrapped - target), np.cos(unwrapped - target))) < 1e-9
    )
