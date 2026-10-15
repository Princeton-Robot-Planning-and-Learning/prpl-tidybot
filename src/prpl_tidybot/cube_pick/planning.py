"""The pick's planning scene: KINDER's PyBullet model of the robot and its world.

This is the planner the old controller already uses (`kinder_models`' `PyBulletSim`,
`run_base_motion_planning`, `run_motion_planning`), wrapped so the pick policy can ask
the questions an analytic grasp needs answered:

* is there an inverse-kinematics solution for this tool pose from that base pose;
* does the arm, with the fingers at *this* opening, keep clear of the container, the
  floor and every other collider there (not just "is the path collision-free with the
  fingers fully open", which is all the old controller asks);
* plan a base path, an arm path, a straight-line Cartesian approach.

The scene is built from an `ObjectCentricState` describing what the robot believes:
the robot's own configuration, the static scene and containers (on the real robot:
from the map and the ceiling marker detector), and the cube at its *perceived* pose.
The simulator's ground-truth cube pose never enters it; see
`PlanningScene.set_cube_estimate`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
)
from scipy.spatial.transform import Rotation
from spatialmath import SE2

FloatArray = NDArray[np.float64]
JointPositions = list[float]

# The Robotiq 2F-85 fingers, measured in KINDER's MuJoCo model: gripper command (0 =
# open, 1 = closed), the driver-joint angle it settles at with nothing in the hand
# (the same joint PyBullet's model calls `finger_joint`), and the inner gap between
# the pads in metres.
GRIPPER_TABLE = np.array(
    [
        # command, driver angle, inner gap
        [0.00, 0.003, 0.0851],
        [0.20, 0.154, 0.0706],
        [0.30, 0.230, 0.0628],
        [0.35, 0.268, 0.0588],
        [0.40, 0.306, 0.0547],
        [0.45, 0.343, 0.0506],
        [0.50, 0.381, 0.0464],
        [0.60, 0.456, 0.0378],
        [0.70, 0.532, 0.0292],
        [1.00, 0.757, 0.0026],
    ]
)


def opening_to_driver(opening: float) -> float:
    """Driver-joint angle for an inner pad gap of `opening` metres."""
    gaps = GRIPPER_TABLE[::-1, 2]
    return float(np.interp(opening, gaps, GRIPPER_TABLE[::-1, 1]))


def driver_to_opening(driver: float) -> float:
    """Inner pad gap (metres) at a driver-joint angle."""
    return float(np.interp(driver, GRIPPER_TABLE[:, 1], GRIPPER_TABLE[:, 2]))


def opening_to_command(opening: float) -> float:
    """MuJoCo gripper command that settles the empty gripper at `opening` metres."""
    gaps = GRIPPER_TABLE[::-1, 2]
    return float(np.interp(opening, gaps, GRIPPER_TABLE[::-1, 0]))


def matrix_to_pose(matrix: FloatArray) -> Pose:
    """4x4 transform to a pybullet_helpers Pose."""
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return Pose(tuple(float(v) for v in matrix[:3, 3]), tuple(float(v) for v in quat))


def pose_to_matrix(pose: Pose) -> FloatArray:
    """pybullet_helpers Pose to a 4x4 transform."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(pose.orientation).as_matrix()
    out[:3, 3] = pose.position
    return out


PHANTOM_CAMERA_BOX = """    <!-- Bounding box for wrist-mounted RealSense camera -->
    <collision>
      <origin rpy="0 0 0" xyz="0.05 0 0.02"/>
      <geometry>
        <box size="0.025 0.09 0.035"/>
      </geometry>
    </collision>
"""


def gripper_urdf_without_phantom_camera() -> Path:
    """pybullet_helpers' Kinova Gen3 + 2F-85 model, minus its wrist-RealSense box.

    The model KINDER plans with puts a 2.5 x 9 x 3.5 cm collision box beside the
    Robotiq base for "a wrist-mounted RealSense camera". Our TidyBot has no such camera
    (its camera is inside the Kinova bracelet), and neither KINDER's MuJoCo model nor
    the robot's own URDF has the box; near a container wall it is often the only thing
    in collision. The copy is written next to a link to the original package so its
    `package://` mesh paths still resolve.
    """
    import pybullet_helpers  # pylint: disable=import-outside-toplevel

    source_dir = Path(pybullet_helpers.__file__).parent / "assets" / "urdf"
    text = (source_dir / "kortex_description" / "gen3_7dof.urdf").read_text()
    if PHANTOM_CAMERA_BOX not in text:
        raise RuntimeError("pybullet_helpers' Kinova URDF changed: phantom camera box not found")
    out_dir = Path(tempfile.gettempdir()) / f"prpl_tidybot_urdf_{os.getuid()}"
    out_dir.mkdir(exist_ok=True)
    link = out_dir / "kortex_description"
    if not link.exists():
        link.symlink_to(source_dir / "kortex_description")
    target = out_dir / "gen3_7dof_no_phantom_camera.urdf"
    stripped = text.replace(PHANTOM_CAMERA_BOX, "")
    if not target.exists() or target.read_text() != stripped:
        target.write_text(stripped)
    return target


@dataclass(frozen=True)
class ContainerSpec:
    """An open box the cube may sit in (KINDER's Bin primitive: origin at the base)."""

    name: str
    pose: FloatArray
    length: float
    width: float
    height: float
    wall_thickness: float


@dataclass(frozen=True)
class ClearanceReport:
    """Result of a configuration check."""

    ok: bool
    closest: float
    culprit: str | None


class PlanningScene:
    """KINDER's PyBullet planning scene, built from a believed state."""

    FLOOR_NAME = "floor"

    def __init__(
        self,
        state: Any,
        *,
        cube_name: str,
        containers: Sequence[ContainerSpec] = (),
        extra_obstacle_names: Sequence[str] = (),
        phantom_camera_box: bool = False,
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
        from kinder_models.dynamic3d.utils import PyBulletSim

        self._state = state.copy()
        self._cube_name = cube_name
        self._sim = PyBulletSim(self._state)
        if not phantom_camera_box:
            self._swap_robot_model()
        for spec in containers:
            self._sim.add_bin(
                name=spec.name,
                pose=matrix_to_pose(spec.pose),
                length=spec.length,
                width=spec.width,
                height=spec.height,
                wall_thickness=spec.wall_thickness,
            )
        (self._robot_obj,) = self._state.get_objects(MujocoTidyBotRobotObjectType)
        self._client = self._sim.physics_client_id
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[20, 20, 0.05], physicsClientId=self._client
        )
        self._floor = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_shape,
            basePosition=[0, 0, -0.05],
            physicsClientId=self._client,
        )
        self._names: dict[int, str] = {self._floor: self.FLOOR_NAME}
        # pylint: disable=protected-access
        for name, body in self._sim._bins.items():
            self._names[body] = name
        for name, body in self._sim._static_colliders.items():
            self._names[body] = name
        for name, body in self._sim._cubes.items():
            self._names[body] = name
        self._cube_body = self._sim._cubes[cube_name]
        self._extra = list(extra_obstacle_names)
        self._sim.set_state(self._state)

    def _swap_robot_model(self) -> None:
        """Replace KINDER's arm model with the same one minus the phantom camera box."""
        # pylint: disable=import-outside-toplevel,protected-access
        from pybullet_helpers.motion_planning import create_joint_distance_fn
        from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot

        sim = self._sim
        old = sim._robot
        p.removeBody(old.robot_id, physicsClientId=sim.physics_client_id)
        sim._robot = KinovaGen3RobotiqGripperPyBulletRobot(
            sim.physics_client_id,
            fixed_base=False,
            control_mode="reset",
            custom_urdf_path=gripper_urdf_without_phantom_camera(),
        )
        sim._joint_distance_fn = create_joint_distance_fn(sim._robot)

    # --- the believed world -------------------------------------------------------------

    @property
    def robot(self) -> Any:
        """The PyBullet Kinova model."""
        return self._sim.robot

    @property
    def client(self) -> int:
        """PyBullet client id."""
        return self._client

    @property
    def state(self) -> Any:
        """The believed state the scene was last synchronized to."""
        return self._state

    def set_robot(self, base: SE2Pose | tuple[float, float, float], arm: Sequence[float]) -> None:
        """Move the model robot to a base pose and arm configuration."""
        x, y, rot = (base.x, base.y, base.rot) if isinstance(base, SE2Pose) else base
        self._state.set(self._robot_obj, "pos_base_x", float(x))
        self._state.set(self._robot_obj, "pos_base_y", float(y))
        self._state.set(self._robot_obj, "pos_base_rot", float(rot))
        for i in range(7):
            self._state.set(self._robot_obj, f"pos_arm_joint{i + 1}", float(arm[i]))
        self._sim.set_state(self._state)

    def set_cube_estimate(self, world_from_cube: FloatArray) -> None:
        """Place the model cube at the *perceived* pose."""
        cube = self._state.get_object_from_name(self._cube_name)
        quat = Rotation.from_matrix(world_from_cube[:3, :3]).as_quat()
        for key, value in zip(("x", "y", "z"), world_from_cube[:3, 3]):
            self._state.set(cube, key, float(value))
        for key, value in zip(("qx", "qy", "qz", "qw"), quat):
            self._state.set(cube, key, float(value))
        self._sim.set_state(self._state)

    def base_pose(self) -> tuple[float, float, float]:
        """Current model base pose."""
        s, r = self._state, self._robot_obj
        return (s.get(r, "pos_base_x"), s.get(r, "pos_base_y"), s.get(r, "pos_base_rot"))

    def arm_joints(self) -> JointPositions:
        """Current model arm joints (7)."""
        return [self._state.get(self._robot_obj, f"pos_arm_joint{i + 1}") for i in range(7)]

    def obstacle_bodies(self, *, include_cube: bool) -> list[int]:
        """Collision bodies: containers, static colliders, the floor, optionally the cube."""
        bodies = set(self._sim.get_collision_bodies())
        bodies.add(self._floor)
        if not include_cube:
            bodies.discard(self._cube_body)
        return sorted(bodies)

    # --- kinematics -----------------------------------------------------------------------

    def tool_pose(self, arm: Sequence[float]) -> FloatArray:
        """Forward kinematics of the tool frame at the current base."""
        return pose_to_matrix(self.robot.forward_kinematics(list(arm[:7]) + [0.0] * 6))

    def ik(self, tool_pose: FloatArray, seed: Sequence[float] | None = None) -> JointPositions | None:
        """Arm joints reaching `tool_pose` from the current base, nearest `seed`."""
        robot = self.robot
        if seed is not None:
            robot.set_joints(list(seed[:7]) + [0.0] * 6)
        try:
            joints = inverse_kinematics(
                robot, matrix_to_pose(tool_pose), validate=True, set_joints=False
            )
        except (InverseKinematicsError, ValueError, AssertionError):
            return None
        finally:
            self._sim.set_state(self._state)
        return [float(q) for q in joints[:7]]

    def check(
        self,
        arm: Sequence[float],
        *,
        opening: float,
        clearance: float,
        include_cube: bool,
        cube_clearance: float = 0.0,
    ) -> ClearanceReport:
        """Is the arm at `arm`, fingers at `opening`, at least `clearance` from every
        obstacle (and `cube_clearance` from the cube, when included)?"""
        robot = self.robot
        driver = opening_to_driver(opening)
        robot.set_joints(list(arm[:7]) + robot.finger_state_to_joints(driver))
        closest = np.inf
        culprit = None
        try:
            for body in self.obstacle_bodies(include_cube=include_cube):
                threshold = self._threshold(body, clearance, cube_clearance)
                points = p.getClosestPoints(
                    robot.robot_id,
                    body,
                    distance=threshold + 0.02,
                    physicsClientId=self._client,
                )
                for point in points:
                    distance = point[8]
                    margin = distance - threshold
                    if margin < closest:
                        closest = margin
                        culprit = f"{self._names.get(body, str(body))} ({self._link_name(point[3])})"
        finally:
            self._sim.set_state(self._state)
        return ClearanceReport(ok=bool(closest >= 0), closest=float(closest), culprit=culprit)

    # Links rigidly attached beyond the last arm joint: the hand (bracelet, Robotiq base,
    # fingers). Where these are, given the tool pose, does not depend on the arm.
    HAND_LINK_PREFIXES = ("bracelet", "robotiq", "left_", "right_")

    def hand_check(
        self,
        tool_pose: FloatArray,
        *,
        opening: float,
        clearance: float,
        include_cube: bool,
        cube_clearance: float = 0.0,
    ) -> ClearanceReport:
        """Clearance of the hand alone with its tool frame at `tool_pose`.

        A second copy of the arm is kept at a fixed configuration and its base moved so
        its tool frame lands on `tool_pose`; only its hand links are queried. No
        inverse kinematics is needed, so a grasp the hand cannot fit is rejected before
        any arm configuration is searched -- and that rejection holds for every base
        placement.
        """
        if not hasattr(self, "_hand"):
            self._hand = type(self.robot)(
                self._client,
                fixed_base=False,
                control_mode="reset",
                custom_urdf_path=getattr(self.robot, "_custom_urdf_path", None),
            )
            self._hand_q0 = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
            self._hand.set_joints(self._hand_q0 + [0.0] * 6)
            base_pose = pose_to_matrix(self._hand.get_base_pose()) if hasattr(self._hand, "get_base_pose") else np.eye(4)
            tool = pose_to_matrix(self._hand.forward_kinematics(self._hand_q0 + [0.0] * 6))
            self._base_from_tool = np.linalg.inv(base_pose) @ tool
            names = {
                i: p.getJointInfo(self._hand.robot_id, i, physicsClientId=self._client)[12].decode()
                for i in range(p.getNumJoints(self._hand.robot_id, physicsClientId=self._client))
            }
            self._hand_links = [i for i, n in names.items() if n.startswith(self.HAND_LINK_PREFIXES)]
            self._hand_names = names
            # Park every link of the copy out of the way of the planning robot.
            for body in (self.robot.robot_id,):
                for link in range(-1, p.getNumJoints(self._hand.robot_id, physicsClientId=self._client)):
                    p.setCollisionFilterPair(
                        self._hand.robot_id, body, link, -1, 0, physicsClientId=self._client
                    )
        hand = self._hand
        hand.set_base(matrix_to_pose(tool_pose @ np.linalg.inv(self._base_from_tool)))
        hand.set_joints(self._hand_q0 + hand.finger_state_to_joints(opening_to_driver(opening)))
        closest = np.inf
        culprit = None
        for body in self.obstacle_bodies(include_cube=include_cube):
            threshold = self._threshold(body, clearance, cube_clearance)
            for link in self._hand_links:
                for point in p.getClosestPoints(
                    hand.robot_id,
                    body,
                    distance=threshold + 0.02,
                    linkIndexA=link,
                    physicsClientId=self._client,
                ):
                    margin = point[8] - threshold
                    if margin < closest:
                        closest = margin
                        culprit = f"{self._names.get(body, str(body))} ({self._hand_names[link]})"
        return ClearanceReport(ok=bool(closest >= 0), closest=float(closest), culprit=culprit)

    # The fingertips reach below the pads; KINDER's MuJoCo fingers sit a few millimetres
    # lower than the planning model's, so a grasp planned 1 mm off the floor touches it.
    FLOOR_CLEARANCE = 0.004

    def _threshold(self, body: int, clearance: float, cube_clearance: float) -> float:
        if body == self._cube_body:
            return cube_clearance
        if body == self._floor:
            return max(clearance, self.FLOOR_CLEARANCE)
        return clearance

    def _link_name(self, link: int) -> str:
        if not hasattr(self, "_link_names"):
            robot_id = self.robot.robot_id
            self._link_names = {
                i: p.getJointInfo(robot_id, i, physicsClientId=self._client)[12].decode()
                for i in range(p.getNumJoints(robot_id, physicsClientId=self._client))
            }
        return self._link_names.get(link, "base")

    # --- motion ---------------------------------------------------------------------------

    def plan_arm(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        opening: float,
        include_cube: bool,
        clearance: float = 0.0,
        seed: int = 0,
    ) -> list[JointPositions] | None:
        """A collision-free arm path at the current base, fingers at `opening`."""
        robot = self.robot
        fingers = robot.finger_state_to_joints(opening_to_driver(opening))
        bodies = self.obstacle_bodies(include_cube=include_cube)

        def clear(joints: JointPositions) -> bool:
            return self.check(
                joints, opening=opening, clearance=clearance, include_cube=include_cube
            ).ok

        robot.set_joints(list(start[:7]) + fingers)
        try:
            plan = run_motion_planning(
                robot,
                list(start[:7]) + fingers,
                list(goal[:7]) + fingers,
                collision_bodies=bodies,
                seed=seed,
                physics_client_id=self._client,
                additional_state_constraint_fn=clear if clearance > 0 else None,
            )
        except (AssertionError, ValueError):
            plan = None
        finally:
            self._sim.set_state(self._state)
        if plan is None:
            return None
        plan = remap_joint_position_plan_to_constant_distance(plan, robot, max_distance=0.2)
        return [[float(q) for q in conf[:7]] for conf in plan]

    def plan_cartesian(
        self,
        start: Sequence[float],
        target_pose: FloatArray,
        *,
        step: float = 0.01,
        max_joint_step: float = 0.25,
    ) -> list[JointPositions] | None:
        """Straight-line tool motion from the pose at `start` to `target_pose`, by IK
        seeded at each previous step (orientation interpolated by slerp)."""
        start_pose = self.tool_pose(start)
        distance = float(np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3]))
        angle = float(
            (Rotation.from_matrix(target_pose[:3, :3]) * Rotation.from_matrix(start_pose[:3, :3]).inv()).magnitude()
        )
        steps = max(1, int(np.ceil(max(distance / step, angle / np.radians(3)))))
        from scipy.spatial.transform import Slerp  # pylint: disable=import-outside-toplevel

        slerp = Slerp([0, 1], Rotation.from_matrix([start_pose[:3, :3], target_pose[:3, :3]]))
        path = [list(start[:7])]
        for i in range(1, steps + 1):
            t = i / steps
            pose = np.eye(4)
            pose[:3, :3] = slerp(t).as_matrix()
            pose[:3, 3] = (1 - t) * start_pose[:3, 3] + t * target_pose[:3, 3]
            joints = self.ik(pose, seed=path[-1])
            if joints is None:
                return None
            if np.max(np.abs(_wrap(np.subtract(joints, path[-1])))) > max_joint_step:
                return None
            path.append(joints)
        return path

    def plan_base(
        self, target: tuple[float, float, float], *, seed: int = 0
    ) -> list[SE2] | None:
        """KINDER's base RRT from the current base pose (cube excluded, as upstream)."""
        # pylint: disable=import-outside-toplevel
        from kinder_models.dynamic3d.utils import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            run_base_motion_planning,
        )

        return run_base_motion_planning(
            state=self._state,
            target_base_pose=SE2(*target),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=seed,
            disable_collision_objects=[self._cube_name, *self._extra],
        )

    def base_pose_collides(self, target: tuple[float, float, float]) -> bool:
        """Would the base footprint at `target` intersect an obstacle (overhead check)?"""
        # pylint: disable=import-outside-toplevel
        from kinder.envs.dynamic3d.object_types import (
            MujocoDrawerObjectType,
            MujocoObjectType,
        )
        from kinder_models.dynamic3d.utils import (
            get_bounding_box,
            get_overhead_kinematic2ds,
        )
        from tomsgeoms2d.structs import Rectangle
        from tomsgeoms2d.utils import geom2ds_intersect

        geoms = get_overhead_kinematic2ds(self._state)
        width, height, _ = get_bounding_box(self._state, self._robot_obj)
        robot_geom = Rectangle.from_center(
            target[0], target[1], width, height, rotation_about_center=target[2]
        )
        for obj in self._state.get_objects(MujocoObjectType):
            if obj.name in (self._cube_name, *self._extra) or obj.is_instance(
                MujocoDrawerObjectType
            ):
                continue
            if obj.name in geoms and geom2ds_intersect(robot_geom, geoms[obj.name]):
                return True
        return False

    def close(self) -> None:
        """Release the PyBullet client."""
        self._sim.close()


def _wrap(diff: Sequence[float]) -> FloatArray:
    out = np.asarray(diff, dtype=float).copy()
    for i in (0, 2, 4, 6):
        if i < len(out):
            out[i] = (out[i] + np.pi) % (2 * np.pi) - np.pi
    return out
