"""Tests for interface.py."""

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.interfaces.arm_interface import (
    FakeArmInterface,
    RealArmInterface,
)
from prpl_tidybot.interfaces.interface import FakeInterface, RealInterface
from prpl_tidybot.structs import TidyBotAction


def test_fake_interface_get_observation():
    """get_observation() pulls from each component interface."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.arm_interface.gripper_state = 0.7
    interface.base_interface.base_state = SE2(x=1.0, y=2.0, theta=0.0)
    interface.base_interface.map_base_state = SE2(x=3.0, y=4.0, theta=0.5)

    obs = interface.get_observation()

    assert np.allclose(obs.arm_conf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert obs.gripper == 0.7
    assert np.allclose(obs.base_pose.A, SE2(x=1.0, y=2.0, theta=0.0).A)
    assert np.allclose(obs.map_base_pose.A, SE2(x=3.0, y=4.0, theta=0.5).A)


def test_fake_interface_execute_action():
    """execute_action() routes each TidyBotAction component to the right fake."""
    interface = FakeInterface()
    arm_goal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    base_goal = SE2(x=2.0, y=-1.0, theta=0.4)
    action = TidyBotAction(
        arm_goal=arm_goal,
        base_local_goal=base_goal,
        gripper_goal=0.8,
    )

    interface.execute_action(action)

    assert np.allclose(interface.get_arm_state(), arm_goal)
    assert np.allclose(interface.get_base_state().A, base_goal.A)
    assert interface.get_gripper_state() == 0.8


def test_real_interface_default_components_all_raise():
    """No ctor args → every component is a Real stub that raises."""
    interface = RealInterface()
    assert isinstance(interface.arm_interface, RealArmInterface)
    with pytest.raises(NotImplementedError):
        interface.get_arm_state()


def test_real_interface_arm_swap_routes_to_fake():
    """Passing FakeArmInterface to RealInterface swaps just the arm; base and camera
    remain Real stubs that raise."""
    interface = RealInterface(arm_interface=FakeArmInterface())
    # Arm reads now use the fake (returns zeros, no raise).
    assert interface.get_arm_state() == [0.0] * 7
    assert interface.get_gripper_state() == 0.0
    # Base / camera still raise from their Real stubs.
    with pytest.raises(NotImplementedError):
        interface.get_base_state()
    with pytest.raises(NotImplementedError):
        interface.get_base_image()
