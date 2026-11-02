# Hardware and optional model imports stay lazy for offline use.
# pylint: disable=import-outside-toplevel
"""Local MuJoCo command preview using full TidyBot and private Sprout models.

This is a kinematic rehearsal (like the repo's shadow-sim preview), not a
contact-dynamics or compliant-force validation. Models stay at their original
local paths; only images, joint trajectories and review metadata are written.
"""

from __future__ import annotations

import itertools
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import cv2
import mujoco  # type: ignore[import-untyped]
import numpy as np

from prpl_tidybot.sprout.live_kinematics import top_down
from prpl_tidybot.sprout.sequence import GraspPose, Recipe, make_plan
from prpl_tidybot.sprout.urdf_collision import PRPLCollisionScene
from prpl_tidybot.video import write_mp4


class Rehearsal:
    """Preview the planned sequence with the TidyBot MuJoCo model."""

    def __init__(
        self, tidybot_model: Path, sprout_model: Path | None, scene: dict[str, Any]
    ) -> None:
        self.box_mode = scene.get("representation") == "forearm_box"
        required = ["tidybot_base_xyyaw", "tidybot_arm_joints_rad"]
        required += (
            ["forearm_box", "grasp"]
            if self.box_mode
            else ["sprout_root_xyz_wxyz", "forearm_point_local_m"]
        )
        for key in required:
            if scene.get(key) is None:
                raise ValueError(f"Scene calibration missing: {key}")
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = 960
        spec.visual.global_.offheight = 720
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[4, 4, 0.05],
            rgba=[0.88, 0.90, 0.92, 1],
        )
        if scene.get("body_obstacle"):
            box = scene["body_obstacle"]
            spec.worldbody.add_geom(
                name="sprout_body_base_keepout",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box["center"],
                size=box["half_size"],
                rgba=[1, 0.45, 0.1, 0.18],
                contype=0,
                conaffinity=0,
            )
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        models = [("tidy/", tidybot_model)]
        if not self.box_mode:
            if sprout_model is None:
                raise ValueError("Full-body mode requires Sprout model")
            models.append(("sprout/", sprout_model))
        for prefix, path in models:
            child = mujoco.MjSpec.from_file(str(path.resolve(strict=True)))
            if prefix == "sprout/":
                child.body("left_radius_link").add_site(
                    name="grasp_marker",
                    pos=scene["forearm_point_local_m"],
                    size=[0.012, 0.012, 0.012],
                    rgba=[1, 0.8, 0, 1],
                    group=0,
                )
            spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
        if self.box_mode:
            box = scene["forearm_box"]
            xyz = [*box["center_xyz"][:2], box["height_m"] / 2]
            yaw = box["short_axis_yaw_rad"]
            body = spec.worldbody.add_body(
                name="sprout/forearm_box",
                mocap=True,
                pos=xyz,
                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
            )
            body.add_geom(
                name="forearm_box_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[box["width_m"] / 2, box["length_m"] / 2, box["height_m"] / 2],
                rgba=[0.65, 0.72, 0.8, 1],
            )
            body.add_site(
                name="sprout/grasp_marker",
                pos=[0, 0, scene["grasp"]["z"] - xyz[2]],
                size=[0.008, 0.008, 0.008],
                rgba=[1, 0.8, 0, 1],
            )
        spec.worldbody.add_body(name="goal_marker", mocap=True).add_site(
            name="goal",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.018, 0.018, 0.018],
            rgba=[0.9, 0.05, 0.1, 0.8],
            group=0,
        )
        tape = scene.get("white_tape_xyz")
        if tape is not None:
            tape = np.asarray(tape, dtype=float)
            if tape.shape != (3,) or not np.all(np.isfinite(tape)):
                raise ValueError("Expected finite measured white-tape xyz")
            spec.worldbody.add_body(name="white_tape_marker", pos=tape).add_site(
                name="white_tape",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.015, 0.010, 0.001],
                rgba=[1, 1, 1, 1],
                group=0,
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.arm_joints = [self.model.joint(f"tidy/joint_{i}").id for i in range(1, 8)]
        self.limb_joints = (
            []
            if self.box_mode
            else [
                self.model.joint("sprout/" + name).id
                for name in (
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                )
            ]
        )
        self.pinch = self.model.site("tidy/pinch_site").id
        self.forearm = self.model.site("sprout/grasp_marker").id
        self.goal_site = self.model.site("goal").id
        if not self.box_mode:
            for name, value in scene.get("sprout_joints_rad", {}).items():
                self.data.qpos[self.model.joint("sprout/" + name).qposadr[0]] = value
            root = self.model.joint("sprout/floating_base").qposadr[0]
            root_pose = np.asarray(scene["sprout_root_xyz_wxyz"], dtype=float)
            if (
                root_pose.shape != (7,)
                or not np.all(np.isfinite(root_pose))
                or not np.isclose(np.linalg.norm(root_pose[3:]), 1)
            ):
                raise ValueError("Expected finite root xyz and unit wxyz quaternion")
            self.data.qpos[root : root + 7] = root_pose
        for name, value in zip(
            ("joint_x", "joint_y", "joint_th"), scene["tidybot_base_xyyaw"]
        ):
            self.data.qpos[self.model.joint("tidy/" + name).qposadr[0]] = value
        joints = scene["tidybot_arm_joints_rad"]
        if len(joints) != 7 or not np.all(np.isfinite(joints)):
            raise ValueError("Expected measured seven-joint TidyBot configuration")
        self.arm_qpos = [self.model.jnt_qposadr[i] for i in self.arm_joints]
        self.data.qpos[self.arm_qpos] = joints
        mujoco.mj_forward(self.model, self.data)
        self.scene = scene
        self.urdf = PRPLCollisionScene()
        self.arm_obstacles = {self.urdf.box([0, 0, -0.1], [20, 20, 0.1])}
        self.arm_obstacles.update(
            self.urdf.box(box["center"], box["half_size"])
            for box in scene.get("arm_obstacles", [])
        )
        self.pregrasp_path: list[list[float]] = []

    def ik(
        self,
        site: int,
        joints: list[int],
        target: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> float:
        """Use shared PRPL robot IK, with optional private-limb visualization IK."""
        if site == self.pinch:
            self.urdf.set(self.base_pose(), self.data.qpos[self.arm_qpos])
            if rotation is None:
                rotation = self.data.site_xmat[site].reshape(3, 3).copy()
            q = self.urdf.solve(
                target, rotation, self.data.qpos[self.arm_qpos], local=True
            )
            self.data.qpos[self.arm_qpos] = q
            mujoco.mj_forward(self.model, self.data)
            error = float(np.linalg.norm(self.data.site_xpos[site] - target))
            if error > 0.002:
                raise ValueError("URDF/MuJoCo tool frames disagree")
            self.urdf.set(self.base_pose(), q)
            if self.urdf.collision(self.arm_obstacles, 0.005, arm_only=True):
                raise ValueError("PRPL URDF arm/floor/obstacle collision")
            return error
        m, d = self.model, self.data
        dofs = [m.jnt_dofadr[i] for i in joints]
        qpos = [m.jnt_qposadr[i] for i in joints]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        error = math.inf
        for _ in range(180):
            mujoco.mj_forward(m, d)
            delta = target - d.site_xpos[site]
            error = float(np.linalg.norm(delta))
            mujoco.mj_jacSite(m, d, jp, jr, site)
            jac, rhs = jp[:, dofs], delta
            orientation_error = 0.0
            if rotation is not None:
                current = d.site_xmat[site].reshape(3, 3)
                relative = rotation @ current.T
                # Quaternion logarithm remains valid for a 180-degree error.
                quat, omega = np.zeros(4), np.zeros(3)
                mujoco.mju_mat2Quat(quat, relative.reshape(9))
                mujoco.mju_quat2Vel(omega, quat, 1)
                orientation_error = float(np.linalg.norm(omega))
                jac = np.vstack((jac, jr[:, dofs]))
                rhs = np.r_[delta, omega]
            if error < 0.002 and orientation_error < 0.025:
                return error
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.001 * np.eye(len(rhs)), rhs)
            step *= min(1.0, 0.08 / max(float(np.max(np.abs(step))), 1e-9))
            d.qpos[qpos] += step
            for joint, address in zip(joints, qpos):
                if m.jnt_limited[joint]:
                    d.qpos[address] = np.clip(d.qpos[address], *m.jnt_range[joint])
        raise ValueError(
            f"IK failed: position error {error:.4f} m; scene/goal must be corrected"
        )

    def joint_delta(self, start, goal) -> np.ndarray:
        """Compute joint displacements respecting limited and continuous joints."""
        delta = (goal - start + np.pi) % (2 * np.pi) - np.pi
        for i, joint in enumerate(self.arm_joints):
            if self.model.jnt_limited[joint]:
                delta[i] = goal[i] - start[i]
        return delta

    def validate_initial_joints(self) -> None:
        """Reject a snapshot outside the model joint limits."""
        for joint, value in zip(self.arm_joints, self.scene["tidybot_arm_joints_rad"]):
            low, high = self.model.jnt_range[joint]
            if self.model.jnt_limited[joint] and not low <= value <= high:
                name = self.model.joint(joint).name
                raise ValueError(
                    f"Measured {name}={value:.4f} rad outside model "
                    f"[{low:.4f}, {high:.4f}]; "
                    "reconcile feedback/model before motion"
                )

    def base_pose(self) -> list[float]:
        """Read the current preview base, including candidate poses during planning."""
        return [
            float(self.data.qpos[self.model.joint("tidy/" + name).qposadr[0]])
            for name in ("joint_x", "joint_y", "joint_th")
        ]

    def pregrasp_joints(self, pose: GraspPose) -> np.ndarray:
        """Use PRPL Kinova IKFast and joint BiRRT for the complete approach path."""
        self.validate_initial_joints()
        origin = self.data.qpos[self.arm_qpos].copy()
        self.urdf.set(self.base_pose(), origin)
        goal = self.urdf.solve([pose.x, pose.y, pose.z], top_down(pose.yaw), origin)
        self.pregrasp_path = self.urdf.arm_path(origin, goal, self.arm_obstacles)
        return np.asarray(goal)

    def grip(self, closed: bool) -> None:
        """Set the preview gripper linkage configuration."""
        # Model-only linkage configuration. Not a hardware gripper calibration.
        angle = 0.45 if closed else 0.0
        for side in ("left", "right"):
            for name, value in (
                ("driver", angle),
                ("spring_link", angle),
                ("coupler", -angle),
                ("follower", -angle),
            ):
                self.data.qpos[
                    self.model.joint(f"tidy/{side}_{name}_joint").qposadr[0]
                ] = value

    def render_plan(
        self,
        grasp: GraspPose,
        recipe: Recipe,
        output: Path,
        *,
        fps: int = 12,
        steps: int = 20,
        blocked_reason: str | None = None,
    ) -> dict[str, Any]:
        """Render the proposed sequence and return its feasibility report."""
        if fps < 1 or steps < 2:
            raise ValueError(
                "Positive fps and at least two interpolation steps required"
            )
        output.mkdir(parents=True, exist_ok=True)
        plan = make_plan(grasp, recipe)
        goal = np.array([grasp.x, grasp.y, grasp.z])
        self.data.mocap_pos[self.model.body("goal_marker").mocapid[0]] = goal
        yaw = grasp.yaw
        rotation = np.array(
            [
                [math.cos(yaw), math.sin(yaw), 0],
                [math.sin(yaw), -math.cos(yaw), 0],
                [0, 0, -1],
            ]
        )
        stages = [
            ("pregrasp", plan.pregrasp, False),
            ("align", plan.pregrasp, False),
            ("descend", plan.grasp, False),
            ("close", plan.grasp, True),
            ("lift", plan.lifted, True),
            ("lower", plan.put_down, True),
            ("open", plan.put_down, False),
            ("retreat", plan.retreat, False),
        ]
        marker_start = self.data.site_xpos[self.forearm].copy()
        box_mocap = (
            self.model.body("sprout/forearm_box").mocapid[0] if self.box_mode else None
        )
        box_start = self.data.mocap_pos[box_mocap].copy() if self.box_mode else None
        if np.linalg.norm(marker_start - goal) > 0.025:
            raise ValueError(
                "Model forearm point disagrees with detected grasp by >2.5 cm; "
                "align the body model first"
            )
        frames, trajectory, contacts = [], [], set()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (goal[0], goal[1], 0.35)
        camera.distance, camera.azimuth, camera.elevation = 2.3, 135, -45
        option = mujoco.MjvOption()
        option.geomgroup[1] = option.geomgroup[3] = 0
        renderer = mujoco.Renderer(self.model, height=720, width=960)
        failure = blocked_reason
        try:
            base_plan = self.scene.get("base_plan")
            if not failure and base_plan and base_plan["required"]:
                from prpl_tidybot.sprout.base_planning import validate_base_plan

                validate_base_plan(base_plan)
                addresses = [
                    self.model.joint("tidy/" + n).qposadr[0]
                    for n in ("joint_x", "joint_y", "joint_th")
                ]
                path = np.asarray(base_plan["path_xyyaw"])
                for a, b in itertools.pairwise(path):
                    delta = b - a
                    delta[2] = math.remainder(delta[2], 2 * math.pi)
                    count = max(
                        2,
                        int(
                            np.ceil(
                                max(
                                    np.linalg.norm(delta[:2]) / 0.04,
                                    abs(delta[2]) / 0.1,
                                )
                            )
                        ),
                    )
                    for f in np.linspace(0, 1, count):
                        self.data.qpos[addresses] = a + f * delta
                        mujoco.mj_forward(self.model, self.data)
                        renderer.update_scene(self.data, camera, scene_option=option)
                        frame = renderer.render().copy()
                        cv2.putText(
                            frame,
                            "BASE REPOSITION | orange = body keep-out",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 180, 60),
                            2,
                        )
                        frames.append(frame)
                print("SIM base reposition complete; starting arm cycle", flush=True)
            for phase, pose, closed in stages:
                xyz = np.array([pose.x, pose.y, pose.z])
                start = self.data.site_xpos[self.pinch].copy()
                start_rotation = self.data.site_xmat[self.pinch].copy()
                start_quat, final_quat = np.zeros(4), np.zeros(4)
                mujoco.mju_mat2Quat(start_quat, start_rotation)
                mujoco.mju_mat2Quat(final_quat, rotation.reshape(9))
                if start_quat @ final_quat < 0:
                    final_quat *= -1
                joint_origin = self.data.qpos[self.arm_qpos].copy()
                joint_frames = None
                if phase == "pregrasp" and not failure:
                    try:
                        self.pregrasp_joints(pose)
                        joint_frames = [joint_origin]
                        for end in self.pregrasp_path:
                            start_q = np.asarray(joint_frames[-1])
                            delta = self.joint_delta(start_q, np.asarray(end))
                            count = max(2, int(np.ceil(max(abs(delta)) / 0.02)) + 1)
                            joint_frames.extend(
                                start_q + f * delta
                                for f in np.linspace(0, 1, count)[1:]
                            )
                    except ValueError as error:
                        failure = f"pregrasp: {error}"
                phase_steps = steps if joint_frames is None else len(joint_frames)
                for frame_index, fraction in enumerate(np.linspace(0, 1, phase_steps)):
                    aim = start + fraction * (xyz - start)
                    quat = start_quat * (1 - fraction) + final_quat * fraction
                    quat /= np.linalg.norm(quat)
                    interpolated = np.zeros(9)
                    mujoco.mju_quat2Mat(interpolated, quat)
                    previous = self.data.qpos.copy()
                    try:
                        if failure:
                            raise ValueError(failure)
                        if joint_frames is not None:
                            self.data.qpos[self.arm_qpos] = joint_frames[frame_index]
                        else:
                            self.ik(
                                self.pinch,
                                self.arm_joints,
                                aim,
                                interpolated.reshape(3, 3),
                            )
                        self.grip(closed)
                        if phase in ("lift", "lower"):
                            if self.box_mode:
                                self.data.mocap_pos[box_mocap] = box_start + aim - goal
                            else:
                                self.ik(
                                    self.forearm,
                                    self.limb_joints,
                                    marker_start + aim - goal,
                                )
                    except ValueError as error:
                        self.data.qpos[:] = previous
                        failure = failure or f"{phase}: {error}"
                    mujoco.mj_forward(self.model, self.data)
                    for contact in self.data.contact:
                        a, b = (
                            self.model.body(self.model.geom_bodyid[g]).name
                            for g in (contact.geom1, contact.geom2)
                        )
                        if (a.startswith("tidy/") and b.startswith("sprout/")) or (
                            b.startswith("tidy/") and a.startswith("sprout/")
                        ):
                            contacts.add((a, b))
                    renderer.update_scene(self.data, camera, scene_option=option)
                    frame = renderer.render().copy()
                    # Draw an on-image grasp crosshair so the goal stays visible
                    # even when the gripper/forearm occludes the 3D goal sphere.
                    glcam = renderer.scene.camera[0]
                    markers = [(goal, "GRASP TARGET", (255, 65, 45), 32)]
                    if self.scene.get("white_tape_xyz") is not None:
                        markers.append(
                            (
                                np.asarray(self.scene["white_tape_xyz"]),
                                (
                                    "WHITE TAPE (demo)"
                                    if "synthetic"
                                    in self.scene.get("white_tape_source", "")
                                    else "WHITE TAPE (captured)"
                                ),
                                (255, 255, 255),
                                -24,
                            )
                        )
                    for point, label, color, label_offset in markers:
                        delta_camera = point - glcam.pos
                        depth = float(delta_camera @ glcam.forward)
                        if depth <= 0:
                            continue
                        focal = 360 * glcam.frustum_near / glcam.frustum_top
                        right = np.cross(glcam.forward, glcam.up)
                        u = int(480 + focal * (delta_camera @ right) / depth)
                        v = int(360 - focal * (delta_camera @ glcam.up) / depth)
                        if 0 <= u < 960 and 80 <= v < 720:
                            cv2.drawMarker(
                                frame, (u, v), (0, 0, 0), cv2.MARKER_CROSS, 28, 5
                            )
                            cv2.drawMarker(
                                frame, (u, v), color, cv2.MARKER_CROSS, 26, 2
                            )
                            location = (max(8, min(u - 90, 700)), v + label_offset)
                            for stroke, width in (((0, 0, 0), 4), (color, 1)):
                                cv2.putText(
                                    frame,
                                    label,
                                    location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    stroke,
                                    width,
                                    cv2.LINE_AA,
                                )
                    cv2.rectangle(frame, (0, 0), (960, 76), (20, 25, 35), -1)
                    cv2.putText(
                        frame,
                        f"{phase.upper()}  |  "
                        f"goal [{pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}] m",
                        (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        (
                            "SYNTHETIC INITIAL STATE - not hardware authorization"
                            if "synthetic"
                            in self.scene.get("body_pose_source", "").lower()
                            else "KINEMATIC PREVIEW - no contact-force validation"
                        ),
                        (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 205, 80),
                        1,
                        cv2.LINE_AA,
                    )
                    if failure:
                        cv2.rectangle(frame, (0, 625), (960, 720), (140, 0, 0), -1)
                        for row, line in enumerate(
                            textwrap.wrap("PLAN BLOCKED: " + failure, 110)[:2]
                        ):
                            cv2.putText(
                                frame,
                                line,
                                (12, 660 + row * 26),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    frames.append(frame)
                    if failure:
                        frames.extend([frame] * fps * 2)
                        break
                    trajectory.append(
                        {
                            "phase": phase,
                            "arm_joints_rad": self.data.qpos[self.arm_qpos].tolist(),
                            "gripper": "close" if closed else "open",
                            "pinch_xyz": self.data.site_xpos[self.pinch].tolist(),
                            "sprout_forearm_xyz": self.data.site_xpos[
                                self.forearm
                            ].tolist(),
                        }
                    )
                print(f"SIM {phase:9s} map target={xyz.round(4).tolist()}", flush=True)
                if failure:
                    print(f"SIM BLOCKED: {failure}", flush=True)
                    break
        finally:
            renderer.close()
        try:
            import imageio.v2 as imageio
        except ImportError:
            write_mp4(frames, output / "preview.mp4", fps)
        else:
            imageio.mimwrite(
                output / "preview.mp4",
                frames,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        cv2.imwrite(
            str(output / "preview.png"),
            cv2.cvtColor(
                frames[-1] if failure else frames[len(frames) // 2], cv2.COLOR_RGB2BGR
            ),
        )
        (output / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
        return {
            "simulation": "kinematic_command_preview",
            "completed": failure is None,
            "failure": failure,
            "frames": len(frames),
            "plan": plan.to_dict(),
            "inter_robot_contact_pairs": sorted(contacts),
            "force_validation": False,
            "representation": "forearm_box" if self.box_mode else "sprout_model",
            "sprout_joint_feasibility_checked": not self.box_mode,
            "white_tape_xyz": self.scene.get("white_tape_xyz"),
            "white_tape_source": self.scene.get("white_tape_source"),
            "simulation_gaps": [
                "Wrist visual servo is represented by a hover, "
                "not simulated camera feedback",
                (
                    "Forearm box follows tool kinematically; "
                    "Sprout joints and other body parts are not simulated"
                    if self.box_mode
                    else "Forearm follows through articulated IK, "
                    "not simulated grasp contact"
                ),
                "Gripper linkage is visualized; "
                "closure force and compliant dynamics are not validated",
            ],
            "body_pose_source": self.scene["body_pose_source"],
            "video": "preview.mp4",
        }
