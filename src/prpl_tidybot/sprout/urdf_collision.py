# Room assets are loaded only when a scene needs them.
# pylint: disable=import-outside-toplevel
"""Use the existing PRPL real_sim robot, IK and PyBullet collision machinery.

Robot geometry comes from TidyBotKinova's URDF collision elements, never from
Sprout-specific bounding boxes. Only perceived environmental objects are boxes.
MuJoCo is retained by the caller for rendering and checked tool-frame FK.
"""

from __future__ import annotations

import builtins
import hashlib
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import cast

import mujoco
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    InverseKinematicsHyperparameters,
    check_body_collisions,
    check_collisions_with_held_object,
    inverse_kinematics,
    pybullet_inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
)
from pybullet_helpers.robots import create_pybullet_mobile_robot
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.utils import get_assets_path


@lru_cache(maxsize=1)
def room_digest() -> str:
    """Bind main's static PRPL room collision asset, not only the robot."""
    from kinder.envs.kinematic3d.prpl3d import _PRPL_LAB_URDF, PrplLab3DEnvConfig

    root = _PRPL_LAB_URDF.parent.parent.resolve()
    files = {_PRPL_LAB_URDF.resolve()}
    for mesh in ET.parse(_PRPL_LAB_URDF).getroot().findall(".//mesh"):
        files.add((_PRPL_LAB_URDF.parent / mesh.attrib["filename"]).resolve())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    pose = PrplLab3DEnvConfig().lab_pose
    digest.update(
        np.asarray([*pose.position, *pose.orientation], dtype="<f8").tobytes()
    )
    return digest.hexdigest()


@lru_cache(maxsize=1)
def urdf_digest() -> str:
    """Bind the same public URDFs and their referenced meshes on both machines."""
    # Deployment links .venv to the machine's existing environment. Resolve
    # the root as well as its meshes so relative names stay installation-independent.
    root = (get_assets_path() / "urdf").resolve()
    files = set()
    for name in ("tidybot/tidybot_base.urdf", "kortex_description/gen3_7dof.urdf"):
        path = root / name
        files.add(path)
        for mesh in ET.parse(path).getroot().findall(".//mesh"):
            filename = mesh.attrib["filename"]
            dependency = (
                (root / filename[len("package://") :])
                if filename.startswith("package://")
                else path.parent / filename
            )
            files.add(dependency.resolve())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(room_digest().encode())
    return digest.hexdigest()


class PRPLCollisionScene:
    """An isolated DIRECT client using the same robot as the shelf pipeline."""

    def __init__(self, bounds=((-2, -2), (2, 2))) -> None:
        self.client = p.connect(p.DIRECT)
        self.closed = False
        self._room: int | None = None
        try:
            self.robot = create_pybullet_mobile_robot(
                "tidybot-kinova",
                self.client,
                base_z=0.0,
                base_pose_lower_bound=SE2Pose(bounds[0][0], bounds[0][1], -math.pi),
                base_pose_upper_bound=SE2Pose(bounds[1][0], bounds[1][1], math.pi),
            )
        except BaseException:
            self.close()
            raise
        self.arm = cast(KinovaGen3RobotiqGripperPyBulletRobot, self.robot.arm)
        self.fingers = tuple(np.linspace(0, 0.8, 9))

    def __enter__(self) -> PRPLCollisionScene:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Disconnect this client's simulation, never a hardware controller."""
        if not self.closed:
            p.disconnect(self.client)
            self.closed = True

    def __del__(self) -> None:
        if hasattr(self, "closed") and not self.closed:
            self.close()

    def set(self, base, joints, finger: float = 0.0) -> None:
        """Set measured map/base and arm state with the URDF finger linkage."""
        self.robot.set_base(
            SE2Pose(base[0], base[1], math.remainder(base[2], 2 * math.pi))
        )
        self.robot.arm.set_joints(
            list(joints) + self.arm.finger_state_to_joints(finger)
        )

    def box(self, center, half, *, rgba=(0.8, 0.4, 0.1, 0.3)) -> int:
        """Add a perceived object box without approximating the robot itself."""
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half, physicsClientId=self.client
        )
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.client
        )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=center,
            physicsClientId=self.client,
        )

    @property
    def room(self) -> int:
        """Main's room asset and unchanged configured placement."""
        if self._room is None:
            from kinder.envs.kinematic3d.prpl3d import (
                _PRPL_LAB_URDF,
                PrplLab3DEnvConfig,
                _suppress_urdf_warnings,
            )

            pose = PrplLab3DEnvConfig().lab_pose
            with _suppress_urdf_warnings():
                self._room = p.loadURDF(
                    str(_PRPL_LAB_URDF),
                    basePosition=pose.position,
                    baseOrientation=pose.orientation,
                    useFixedBase=True,
                    physicsClientId=self.client,
                )
        return self._room

    def collision(
        self,
        bodies: builtins.set[int],
        clearance: float = 1e-6,
        *,
        base_only: bool = False,
        arm_only: bool = False,
    ) -> bool:
        """Run PRPL's model collision helpers, including base/arm interference."""
        bodies = set(bodies) | {self.room}
        robot = self.robot
        if not base_only and check_collisions_with_held_object(
            robot.arm,
            bodies,
            self.client,
            None,
            None,
            robot.arm.get_joint_positions(),
            distance_threshold=clearance,
        ):
            return True
        if not arm_only and any(
            check_body_collisions(
                robot.base.robot_id, body, self.client, distance_threshold=clearance
            )
            for body in bodies
        ):
            return True
        return check_body_collisions(
            robot.base.robot_id,
            robot.arm.robot_id,
            self.client,
            distance_threshold=1e-6,
        )

    def aabbs(self) -> list[tuple]:
        """Broad-phase bounds used only for map limits and motion-distance budgets."""
        result = []
        for body in (self.robot.base.robot_id, self.robot.arm.robot_id):
            for link in range(-1, p.getNumJoints(body, physicsClientId=self.client)):
                if p.getCollisionShapeData(body, link, physicsClientId=self.client):
                    result.append(p.getAABB(body, link, physicsClientId=self.client))
        return result

    def solve(self, xyz, rotation, seed, *, local: bool = False) -> list[float]:
        """Use PRPL IKFast for goals and its seeded PyBullet IK for servo steps."""
        # Pose conversion uses the URDF tool_frame, which is checked against
        # MuJoCo pinch_site by the caller before accepting the solution.
        arm = self.arm
        arm.set_joints(list(seed) + arm.finger_state_to_joints(0.0))
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rotation).ravel())
        target = Pose(tuple(xyz), tuple(quat[[1, 2, 3, 0]]))
        try:
            result = (
                pybullet_inverse_kinematics(
                    arm,
                    target,
                    hyperparameters=InverseKinematicsHyperparameters(
                        convergence_atol=1e-4, max_iters=500
                    ),
                )
                if local
                else inverse_kinematics(arm, target)
            )
        except (InverseKinematicsError, ValueError) as error:
            raise ValueError("PRPL URDF IK failed: " + str(error)) from error
        return list(result[:7])

    def arm_path(
        self, start, goal, bodies: builtins.set[int], *, clearance: float = 0.005
    ) -> list[list[float]]:
        """Use PRPL's joint-space BiRRT instead of assuming a straight approach."""
        bodies = set(bodies) | {self.room}
        arm = self.arm
        fingers = arm.finger_state_to_joints(0.0)
        path = run_motion_planning(
            arm,
            list(start) + fingers,
            list(goal) + fingers,
            bodies,
            seed=0,
            physics_client_id=self.client,
            distance_threshold=clearance,
            hyperparameters=MotionPlanningHyperparameters(birrt_extend_num_interp=50),
            additional_state_constraint_fn=lambda _q: not check_body_collisions(
                self.robot.base.robot_id,
                arm.robot_id,
                self.client,
            ),
        )
        if path is None:
            raise ValueError("PRPL URDF pregrasp motion planning failed")
        return [list(q[:7]) for q in path]

    def base_path(self, start, goal, bodies: builtins.set[int]) -> list[list[float]]:
        """Call the existing mobile-manipulator RRT/BiRRT entry point directly."""
        bodies = set(bodies) | {self.room}
        self.robot.set_base(
            SE2Pose(start[0], start[1], math.remainder(start[2], 2 * math.pi))
        )
        path = run_single_arm_mobile_base_motion_planning(
            self.robot,
            SE2Pose(start[0], start[1], math.remainder(start[2], 2 * math.pi)),
            SE2Pose(goal[0], goal[1], math.remainder(goal[2], 2 * math.pi)),
            bodies,
            seed=0,
            hyperparameters=MotionPlanningHyperparameters(birrt_extend_num_interp=100),
        )
        if path is None:
            raise ValueError("PRPL URDF base motion planning failed")
        return [[pose.x, pose.y, pose.rot] for pose in path]
