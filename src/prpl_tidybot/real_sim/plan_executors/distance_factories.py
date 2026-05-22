"""Factory functions that build pybullet Kinova arms + joint-space distance
metrics for the streaming arm executor.

The distance-fn factory is wired into Hydra plan-executor configs via
``_target_``; the returned callable is passed to
``StreamingArmMotion3DPlanExecutor.distance_fn``. The robot factory is the
shared building block — direct callers (hardware tests, scripts) that need
both the distance function and the joint metadata from the same arm can call
``create_kinova_robot`` once and derive both from it.

Kept in its own module so callers of ``arm_motion3d.py`` that already have a
distance function (unit tests with a hand-built L1, etc.) don't pull in
pybullet just by importing the executor.
"""

import pybullet as p
from pybullet_helpers.motion_planning import create_joint_distance_fn
from pybullet_helpers.robots.kinova import KinovaGen3NoGripperPyBulletRobot

from prpl_tidybot.real_sim.plan_executors.arm_motion3d import JointDistanceFn


def create_kinova_robot() -> KinovaGen3NoGripperPyBulletRobot:
    """Build a standalone 7-DOF Kinova arm in a fresh pybullet DIRECT client.

    No-gripper variant on purpose: its ``arm_joints`` is exactly the 7-joint
    kinematic chain to the end effector, matching the 7-element configs we get
    from ``obs.arm_conf``. ``KinovaGen3RobotiqGripperPyBulletRobot`` (the
    "with gripper" version used by TidyBotKinova) would extend ``arm_joints``
    with 6 Robotiq finger ids, breaking length checks in
    ``get_jointwise_difference`` and the distance fn.

    The pybullet client is held alive by the returned robot; callers don't
    need to manage it explicitly.
    """
    client_id = p.connect(p.DIRECT)
    return KinovaGen3NoGripperPyBulletRobot(physics_client_id=client_id)


def create_kinova_distance_fn(
    metric: str = "weighted_joints",
    weight_base: float = 0.9,
) -> JointDistanceFn:
    """Build a weighted-L1 joint distance over the 7-DOF Kinova arm.

    Hydra-friendly factory: ``_target_:
    prpl_tidybot.real_sim.plan_executors.distance_factories.create_kinova_distance_fn``
    in a plan-executor yaml wires this in as
    ``StreamingArmMotion3DPlanExecutor.distance_fn``. Intended to be called
    once at pipeline construction time.
    """
    return create_joint_distance_fn(
        create_kinova_robot(), metric=metric, weight_base=weight_base
    )
