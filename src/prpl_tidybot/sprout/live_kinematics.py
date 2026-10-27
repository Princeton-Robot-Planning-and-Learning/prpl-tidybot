"""Full TidyBot model FK/IK, floor clearance and conservative box checks.

Obstacle boxes bound all non-target body parts; the configured held-forearm box
bounds the moving limb in the pinch frame. They must be verified in the lab.
Checks sample the joint path at an explicit resolution with a clearance margin;
this is not a continuous collision proof or a prediction of contact forces.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from prpl_tidybot.sprout.live_config import model_digest
from prpl_tidybot.sprout.sequence import GraspPose


def top_down(yaw: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(yaw), math.sin(yaw), 0],
            [math.sin(yaw), -math.cos(yaw), 0],
            [0, 0, -1],
        ]
    )


def rotation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (target @ actual.T).ravel())
    if quat[0] < 0:
        quat *= -1
    result = np.zeros(3)
    mujoco.mju_quat2Vel(result, quat, 1)
    return result


class LiveKinematics:
    def __init__(self, model_path: Path, base_xyyaw: list[float], config: dict) -> None:
        if model_digest(model_path) != config["model_sha256"]:
            raise ValueError("NUC TidyBot model/assets differ from the reviewed model")
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("pinch_site").id
        self.joints = [self.model.joint(f"joint_{i}").id for i in range(1, 8)]
        self.addresses = [self.model.jnt_qposadr[j] for j in self.joints]
        self.dofs = [self.model.jnt_dofadr[j] for j in self.joints]
        for name, value in zip(("joint_x", "joint_y", "joint_th"), base_xyyaw):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.collision_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_contype[i] or self.model.geom_conaffinity[i]
        ]
        self.mesh_bounds = {}
        for g in self.collision_geoms:
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[g]
                start, count = (
                    self.model.mesh_vertadr[mesh],
                    self.model.mesh_vertnum[mesh],
                )
                points = self.model.mesh_vert[start : start + count]
                self.mesh_bounds[g] = (
                    (points.min(0) + points.max(0)) / 2,
                    (points.max(0) - points.min(0)) / 2,
                )

    def set(self, q, gripper: float = 0.0) -> None:
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
        self.set(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[
            self.site
        ].reshape(3, 3).copy()

    def pose(self, q) -> GraspPose:
        xyz, rotation = self.full_pose(q)
        return GraspPose(*map(float, xyz), math.atan2(rotation[1, 0], rotation[0, 0]))

    def solve(self, xyz: np.ndarray, rotation: np.ndarray, seed) -> tuple[float, ...]:
        self.set(seed)
        p = self.config["servo"]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(160):
            mujoco.mj_forward(self.model, self.data)
            dp = xyz - self.data.site_xpos[self.site]
            dr = rotation_error(rotation, self.data.site_xmat[self.site].reshape(3, 3))
            if (
                np.linalg.norm(dp) <= min(0.001, p["tolerance_m"] / 2)
                and np.linalg.norm(dr) <= p["orientation_tolerance_rad"] / 2
            ):
                q = self.data.qpos[self.addresses].copy()
                q = (q + np.pi) % (2 * np.pi) - np.pi
                return tuple(float(v) for v in q)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.dofs], jr[:, self.dofs]])
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + np.eye(6) * 0.001, np.r_[dp, dr]
            )
            delta *= min(1.0, 0.05 / max(float(np.max(np.abs(delta))), 1e-9))
            self.data.qpos[self.addresses] += delta
            for joint, address in zip(self.joints, self.addresses):
                if self.model.jnt_limited[joint]:
                    self.data.qpos[address] = np.clip(
                        self.data.qpos[address], *self.model.jnt_range[joint]
                    )
        raise RuntimeError("Live IK did not converge; no command issued")

    def bounds(self, geom: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.model.geom_size[geom].copy()
        center = np.zeros(3)
        kind = self.model.geom_type[geom]
        if geom in self.mesh_bounds:
            center, size = self.mesh_bounds[geom]
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            size[:] = size[0]
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            size = np.array(
                [
                    size[0],
                    size[0],
                    size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0),
                ]
            )
        elif kind != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Unsupported collision primitive")
        rotation = self.data.geom_xmat[geom].reshape(3, 3)
        return self.data.geom_xpos[geom] + rotation @ center, np.abs(rotation) @ size

    def check_state(self, q, *, held: bool = False, on_support: bool = False) -> None:
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
        # Conservative envelope across the entire finger linkage travel.
        for closure in (0.0, 0.4, 0.8):
            self.set(q, closure)
            for contact in self.data.contact:
                bodies = {
                    self.model.body(self.model.geom_bodyid[g]).name
                    for g in (contact.geom1, contact.geom2)
                }
                # Opposing pads intentionally meet at empty full closure.
                if contact.dist < -1e-5 and bodies != {"left_pad", "right_pad"}:
                    raise RuntimeError("TidyBot model self-collision")
            for geom in self.collision_geoms:
                center, half = self.bounds(geom)
                body = self.model.body(self.model.geom_bodyid[geom]).name
                # Mobile-base shell/wheels normally touch the ground; arm and
                # fingers may not. Determine membership by arm joint subtree.
                ancestor = int(self.model.geom_bodyid[geom])
                arm_body = int(self.model.jnt_bodyid[self.joints[0]])
                in_arm = False
                while ancestor:
                    if ancestor == arm_body:
                        in_arm = True
                    ancestor = int(self.model.body_parentid[ancestor])
                if in_arm and center[2] - half[2] < p["collision_margin_m"]:
                    raise RuntimeError(f"Floor clearance violation: {body}")
                body_box = (
                    self.config.get("detection", {})
                    .get("body_obstacle", {})
                    .get("reviewed_box")
                )
                if (
                    not in_arm
                    and body_box is not None
                    and np.all(
                        np.abs(center[:2] - np.asarray(body_box["center"])[:2])
                        < half[:2] + np.asarray(body_box["half_size"])[:2]
                    )
                ):
                    raise RuntimeError(
                        "TidyBot base overlaps Sprout whole-body keep-out"
                    )
                for obstacle in self.config["obstacles"]:
                    if np.all(
                        np.abs(center - obstacle["center"])
                        < half
                        + np.asarray(obstacle["half_size"])
                        + p["collision_margin_m"]
                    ):
                        raise RuntimeError(
                            f"Robot collision envelope intersects {obstacle.get('name', 'obstacle')}: {body}"
                        )
            if held:
                box = self.config["held_forearm"]
                rotation = self.data.site_xmat[self.site].reshape(3, 3)
                center = self.data.site_xpos[self.site] + rotation @ box["center"]
                half = np.abs(rotation) @ box["half_size"]
                if center[2] - half[2] < -p["collision_margin_m"]:
                    raise RuntimeError("Held forearm below floor")
                for obstacle in self.config["obstacles"]:
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
