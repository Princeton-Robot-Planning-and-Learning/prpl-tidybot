"""Tests for tool-frame stepping with the standalone Kinova model."""

import numpy as np
import pytest
from pybullet_helpers.geometry import matrix_from_quat

from prpl_tidybot.visual_servo.tool_frame import (
    ToolFrameStepError,
    ToolFrameStepper,
    tool_delta,
)

# A planar carrying-pose configuration from the cylinder-shelf skills.
_JOINTS = [0.0, 0.628, 3.142, -2.495, 0.0, 0.291, 1.571]


@pytest.fixture(name="stepper", scope="module")
def _stepper() -> ToolFrameStepper:
    return ToolFrameStepper()


def test_tool_delta_builds_axis_vectors():
    """tool_delta places the distance on the requested axis only."""
    assert tool_delta("x", 0.02) == [0.02, 0.0, 0.0]
    assert tool_delta("z", -0.01) == [0.0, 0.0, -0.01]
    with pytest.raises(KeyError):
        tool_delta("w", 1.0)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_step_moves_the_tool_along_its_own_axis(stepper: ToolFrameStepper, axis: str):
    """Stepping 2 cm along a tool axis moves the end effector 2 cm along that axis's
    world direction and leaves the orientation unchanged."""
    before = stepper.end_effector_pose(_JOINTS)
    rotation = matrix_from_quat(before.orientation)
    target = stepper.step(_JOINTS, tool_delta(axis, 0.02))
    after = stepper.end_effector_pose(target)
    moved = np.array(after.position) - np.array(before.position)
    expected = 0.02 * rotation[:, "xyz".index(axis)]
    assert np.allclose(moved, expected, atol=2e-3)
    assert np.allclose(
        matrix_from_quat(after.orientation), rotation, atol=1e-2
    ), "orientation should be preserved"
    assert len(target) == 7
    # A 2 cm move is a small joint-space change from a nearby seed.
    assert np.max(np.abs(np.array(target) - np.array(_JOINTS))) < 0.2


def test_step_is_relative_to_the_given_joints(stepper: ToolFrameStepper):
    """Two half-steps land where one full step does."""
    full = stepper.step(_JOINTS, tool_delta("z", 0.04))
    half = stepper.step(
        stepper.step(_JOINTS, tool_delta("z", 0.02)), tool_delta("z", 0.02)
    )
    assert np.allclose(
        stepper.end_effector_pose(full).position,
        stepper.end_effector_pose(half).position,
        atol=2e-3,
    )


def test_unreachable_step_raises(stepper: ToolFrameStepper):
    """A move far outside the workspace has no IK solution."""
    with pytest.raises(ToolFrameStepError):
        stepper.step(_JOINTS, tool_delta("z", 5.0))
