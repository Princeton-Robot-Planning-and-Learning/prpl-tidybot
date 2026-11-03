"""PRPL URDF robot collision/IK, verified render FK and object-box checks.

Optional obstacle boxes bound configured non-target body parts; an optional
held-forearm box bounds the moving limb in the pinch frame. If omitted, no
arm-to-body or held-limb collision claim is made. Self/floor checks remain active.
Checks sample the joint path at an explicit resolution with a clearance margin;
this is not a continuous collision proof or a prediction of contact forces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pybullet as bullet

from prpl_tidybot.sprout.live_config import collision_boxes, model_digest
from prpl_tidybot.sprout.sequence import GraspPose
from prpl_tidybot.sprout.urdf_collision import PRPLCollisionScene


def top_down(yaw: float) -> np.ndarray:
    """Construct the downward pinch orientation for a map-frame jaw yaw."""
    return np.array(
        [
            [math.cos(yaw), math.sin(yaw), 0],
            [math.sin(yaw), -math.cos(yaw), 0],
            [0, 0, -1],
        ]
    )


def rotation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the axis-angle correction from actual to target orientation."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (target @ actual.T).ravel())
    if quat[0] < 0:
        quat *= -1
    result = np.zeros(3)
    mujoco.mju_quat2Vel(result, quat, 1)
    return result


class LiveKinematics:
    """Model-based pinch kinematics and conservative collision guards."""

    def __init__(self, model_path: Path, base_xyyaw: list[float], config: dict) -> None:
        if model_digest(model_path) != config["model_sha256"]:
            raise ValueError("NUC TidyBot model/assets differ from the reviewed model")
        self.config = config
        self.base_xyyaw = base_xyyaw
        self._urdf_scene: PRPLCollisionScene | None = None
        self._obstacle_signature: str | None = None
        self._urdf_obstacles: set[int] = set()
        self._urdf_floor: int | None = None
        self._urdf_body: int | None = None
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("pinch_site").id
        self.joints = [self.model.joint(f"joint_{i}").id for i in range(1, 8)]
        self.addresses = [self.model.jnt_qposadr[j] for j in self.joints]
        for name, value in zip(("joint_x", "joint_y", "joint_th"), base_xyyaw):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value

    def set(self, q, gripper: float = 0.0) -> None:
        """Set model joints and gripper linkage before forward kinematics."""
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Invalid arm joints")
        self.data.qpos[self.addresses] = (q + np.pi) % (2 * np.pi) - np.pi
        # Check the full open/closed linkage range later, rather than treating
        # encoder percentage as a calibrated physical jaw width.
        for side in ("left", "right"):
            for name, sign in (
                ("driver", 1),
                ("spring_link", 1),
                ("coupler", -1),
                ("follower", -1),
            ):
                self.data.qpos[self.model.joint(f"{side}_{name}_joint").qposadr[0]] = (
                    sign * gripper
                )
        mujoco.mj_forward(self.model, self.data)

    def full_pose(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Return pinch translation and orientation for the supplied joints."""
        self.set(q)
        return (
            self.data.site_xpos[self.site].copy(),
            self.data.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def pose(self, q) -> GraspPose:
        """Return the measured pinch pose in the calibrated map frame."""
        xyz, rotation = self.full_pose(q)
        return GraspPose(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            math.atan2(rotation[1, 0], rotation[0, 0]),
        )

    @property
    def urdf(self) -> PRPLCollisionScene:
        """Lazy shared PRPL URDF scene for IK and live collision guards."""
        if self._urdf_scene is None:
            self._urdf_scene = PRPLCollisionScene()
        return self._urdf_scene

    def solve(self, xyz: np.ndarray, rotation: np.ndarray, seed) -> tuple[float, ...]:
        """Reuse PRPL seeded IK for short servo steps and verify tool frames."""
        self.urdf.set(self.base_xyyaw, seed)
        try:
            q = self.urdf.solve(xyz, rotation, seed, local=True)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        actual, orientation = self.full_pose(q)
        if (
            np.linalg.norm(actual - xyz) > 0.002
            or np.linalg.norm(rotation_error(rotation, orientation)) > 0.025
        ):
            raise RuntimeError("URDF/MuJoCo tool frames disagree; no command issued")
        return tuple(q)

    def check_state(self, q, *, held: bool = False, on_support: bool = False) -> None:
        """Check joint limits, link collisions, and floor clearance."""
        # Support does not relax the conservative floor clearance in this model.
        del on_support
        p = self.config["servo"]
        h = self.config["hardware"]
        q = (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi
        if np.any(q < h["joint_min_rad"]) or np.any(q > h["joint_max_rad"]):
            raise RuntimeError("Joint target outside configured range")
        for joint, value in zip(self.joints, q):
            if (
                self.model.jnt_limited[joint]
                and not self.model.jnt_range[joint, 0]
                <= value
                <= self.model.jnt_range[joint, 1]
            ):
                raise RuntimeError("Joint target outside model range")
        scene = self.urdf
        obstacles, held_box = collision_boxes(self.config)
        signature = json.dumps(
            [obstacles, self.config.get("detection", {})],
            sort_keys=True,
        )
        if signature != self._obstacle_signature:
            for body_id in self._urdf_obstacles:
                bullet.removeBody(body_id, physicsClientId=scene.client)
            for name in ("_urdf_floor", "_urdf_body"):
                optional_body_id = getattr(self, name, None)
                if optional_body_id is not None:
                    bullet.removeBody(optional_body_id, physicsClientId=scene.client)
            self._obstacle_signature = signature
            self._urdf_obstacles = {
                scene.box(box["center"], box["half_size"]) for box in obstacles
            }
            self._urdf_floor = scene.box([0, 0, -0.1], [20, 20, 0.1])
            body_box = (
                self.config.get("detection", {})
                .get("body_obstacle", {})
                .get("reviewed_box")
            )
            self._urdf_body = (
                None
                if body_box is None
                else scene.box(body_box["center"], body_box["half_size"])
            )
        from pybullet_helpers.inverse_kinematics import (  # pylint: disable=import-outside-toplevel
            check_body_collisions,
        )

        assert self._urdf_floor is not None
        for closure in scene.fingers:
            self.set(q, float(closure))
            scene.set(self.base_xyyaw, q, float(closure))
            if scene.collision(self._urdf_obstacles, p["collision_margin_m"]):
                raise RuntimeError(
                    "TidyBot URDF intersects collision envelope or itself"
                )
            if check_body_collisions(
                scene.robot.arm.robot_id,
                self._urdf_floor,
                scene.client,
                distance_threshold=p["collision_margin_m"],
            ):
                raise RuntimeError("Floor clearance violation: TidyBot URDF")
            if self._urdf_body is not None and check_body_collisions(
                scene.robot.base.robot_id,
                self._urdf_body,
                scene.client,
                distance_threshold=p["collision_margin_m"],
            ):
                raise RuntimeError("TidyBot base overlaps Sprout whole-body keep-out")
            if held and held_box is not None:
                box = held_box
                rotation = self.data.site_xmat[self.site].reshape(3, 3)
                center = self.data.site_xpos[self.site] + rotation @ box["center"]
                half = np.abs(rotation) @ box["half_size"]
                if center[2] - half[2] < -p["collision_margin_m"]:
                    raise RuntimeError("Held forearm below floor")
                for obstacle in obstacles:
                    if np.all(
                        np.abs(center - obstacle["center"])
                        < half
                        + np.asarray(obstacle["half_size"])
                        + p["collision_margin_m"]
                    ):
                        raise RuntimeError("Held forearm intersects obstacle envelope")

    def check_path(
        self, start, goal, *, held: bool = False, on_support: bool = False
    ) -> None:
        """Sample the joint segment and check every intermediate state."""
        difference = (np.asarray(goal) - start + np.pi) % (2 * np.pi) - np.pi
        if max(abs(difference)) > self.config["servo"]["max_joint_step_rad"]:
            raise RuntimeError("IK branch jump / excessive joint step")
        count = max(
            2,
            math.ceil(
                max(abs(difference)) / self.config["servo"]["path_resolution_rad"]
            )
            + 1,
        )
        for fraction in np.linspace(0, 1, count):
            self.check_state(
                np.asarray(start) + fraction * difference,
                held=held,
                on_support=on_support,
            )

    def validate_trajectory(self, trajectory: list[dict], start) -> None:
        """Check every preview segment, including map/tool model consistency."""
        if not trajectory:
            raise ValueError("Empty preview trajectory")
        previous = start
        for frame in trajectory:
            q = frame["arm_joints_rad"]
            xyz, _ = self.full_pose(q)
            if np.linalg.norm(xyz - frame["pinch_xyz"]) > 1e-5:
                raise RuntimeError("Preview/runtime map or tool frames differ")
            self.check_path(
                previous,
                q,
                held=frame["phase"] in ("lift", "lower"),
                on_support=frame["phase"] == "lower",
            )
            previous = q
