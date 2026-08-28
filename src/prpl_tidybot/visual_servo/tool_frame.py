"""Small end-effector moves expressed in the tool frame.

:class:`ToolFrameStepper` converts "move the gripper 1 cm along its own
axis" into a joint target with the standalone pybullet Kinova model that
the arm executors already use for their joint metric. Forward kinematics
gives the current tool pose, the translation is composed in the tool
frame, and inverse kinematics (seeded at the current joints, so the
nearest solution is returned) gives the target.

Tool axes of the Kinova model at the planar grasp configurations: ``z`` is
the approach axis (out of the gripper, toward the object), ``x`` is the
lateral axis (the gripper's open/close direction, which is also the image's
horizontal axis for the wrist camera), ``y`` is up/down in the tool frame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

JointPositions = list[float]

_AXES = {"x": 0, "y": 1, "z": 2}


class ToolFrameStepError(RuntimeError):
    """Raised when a requested tool-frame move has no inverse-kinematics solution."""


class ToolFrameStepper:
    """FK / IK helper for tool-frame translations of the 7-DOF Kinova arm."""

    def __init__(self, robot: SingleArmPyBulletRobot | None = None) -> None:
        if robot is None:
            # Imported here: the plan-executor package imports this module, so a
            # module-level import would be circular.
            from prpl_tidybot.real_sim.plan_executors.distance_factories import (  # pylint: disable=import-outside-toplevel
                create_kinova_robot,
            )

            robot = create_kinova_robot()
        self._robot = robot

    def end_effector_pose(self, joints: Sequence[float]) -> Pose:
        """Tool pose (arm-base frame) at `joints`."""
        return self._robot.forward_kinematics(list(joints[:7]))

    def step(
        self, joints: Sequence[float], delta_tool: Sequence[float]
    ) -> JointPositions:
        """Joint target that translates the tool by `delta_tool` (metres, tool frame)
        from the pose at `joints`, keeping its orientation."""
        current = list(joints[:7])
        pose = self.end_effector_pose(current)
        dx, dy, dz = (float(d) for d in delta_tool)
        target = multiply_poses(pose, Pose((dx, dy, dz)))
        self._robot.set_joints(current)
        try:
            solution = inverse_kinematics(self._robot, target, validate=True)
        except InverseKinematicsError as e:
            raise ToolFrameStepError(
                f"No IK solution for a tool-frame move of {np.round(delta_tool, 4)} m "
                f"from joints {np.round(current, 3)}: {e}"
            ) from e
        return [float(q) for q in solution[:7]]


def tool_delta(axis: str, distance: float) -> list[float]:
    """A tool-frame translation of `distance` metres along `axis` ('x', 'y' or 'z')."""
    delta = [0.0, 0.0, 0.0]
    delta[_AXES[axis]] = distance
    return delta
