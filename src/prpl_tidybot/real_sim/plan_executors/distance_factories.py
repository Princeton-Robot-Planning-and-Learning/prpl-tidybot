"""Factory functions that build joint-space distance metrics for arm executors.

Wired into Hydra plan-executor configs via ``_target_``; the returned callable
is passed to ``StreamingArmMotion3DPlanExecutor.distance_fn``. Kept in its own
module so callers of ``arm_motion3d.py`` that already have a distance function
(unit tests with a hand-built L1, etc.) don't pull in pybullet just by
importing the executor.
"""

import pybullet as p
from pybullet_helpers.motion_planning import create_joint_distance_fn
from pybullet_helpers.robots.kinova import KinovaGen3NoGripperPyBulletRobot

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import JointDistanceFn


def create_kinova_distance_fn(
    metric: str = "weighted_joints",
    weight_base: float = 0.9,
) -> JointDistanceFn:
    """Build a weighted-L1 joint distance over the 7-DOF Kinova arm.

    Loads a standalone Kinova arm (no Robotiq gripper, so ``arm_joints`` is
    exactly the 7 joints we get from ``obs.arm_conf``) into a fresh pybullet
    DIRECT client and returns the distance function. The pybullet client is
    held alive by the closure returned from ``create_joint_distance_fn``.
    Intended to be called once at pipeline construction time.
    """
    client_id = p.connect(p.DIRECT)
    robot = KinovaGen3NoGripperPyBulletRobot(physics_client_id=client_id)
    return create_joint_distance_fn(robot, metric=metric, weight_base=weight_base)
