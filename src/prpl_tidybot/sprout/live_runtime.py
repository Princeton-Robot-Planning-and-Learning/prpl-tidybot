"""Concrete supervised Sprout runtime: ceiling servo + local compliant arm.

All configuration comes from the hashed preview bundle. Construction/observe
are read-only. Hardware holding starts only inside execute(), after approval.
Grasp/support confirmations are live operator observations, never fabricated
sensor evidence. Confirmations hold the last commanded pose with bounded waits.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from prpl_tidybot.sprout.approach import TopDownGeometry, WristCalibration
from prpl_tidybot.sprout.compliant_arm import HardwareLimits, LocalCompliantArm
from prpl_tidybot.sprout.compliant_backend import (
    CompliantBackend,
    JointSample,
    MotionLimits,
)
from prpl_tidybot.sprout.live_config import validate_config
from prpl_tidybot.sprout.live_kinematics import LiveKinematics, rotation_error, top_down
from prpl_tidybot.sprout.live_perception import LivePerception
from prpl_tidybot.sprout.read_robot_state import read_arm
from prpl_tidybot.sprout.sequence import ForearmSkill, GraspPose, Recipe, Stopped


class Confirmations:
    """Nonce-bound, phase-specific laptop observations; never reuse a prior yes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pending: tuple[str, str] | None = None
        self.accepted = threading.Event()

    def request(self, phase: str) -> None:
        with self._lock:
            self.accepted.clear()
            self.pending = (phase, uuid.uuid4().hex)
            print(
                json.dumps(
                    {
                        "event": "operator_confirmation",
                        "phase": phase,
                        "token": self.pending[1],
                    }
                ),
                flush=True,
            )

    def confirm(self, phase: str, token: str) -> None:
        with self._lock:
            if self.pending != (phase, token):
                raise RuntimeError("Unexpected/stale operator confirmation")
            self.pending = None
            self.accepted.set()


class LiveScene:
    """Calibrated map FK/IK, fresh target/base state and verified phase evidence."""

    def __init__(
        self,
        config: dict,
        review: dict,
        kinematics: LiveKinematics,
        perception: LivePerception,
        confirmations: Confirmations,
    ) -> None:
        self.config, self.review, self.kinematics, self.perception = (
            config,
            review,
            kinematics,
            perception,
        )
        self.confirmations = confirmations
        self.phase = "preflight"
        self.held = self.supported = False
        self.grasp_reading: float | None = None
        self.original_grasp = GraspPose(**review["plan"]["grasp"])

    def localize(self) -> GraspPose:
        grasp, _ = self.perception.target()
        old = self.review["observed"]["grasp_xyz"]
        tolerance = self.review["revalidation_tolerances"]["grasp_xyz"]
        if np.max(np.abs(np.asarray(old) - [grasp.x, grasp.y, grasp.z])) > tolerance:
            raise RuntimeError("Forearm moved beyond reviewed target; preview again")
        return grasp

    def pose(self, sample: JointSample) -> GraspPose:
        return self.kinematics.pose(sample.q)

    def solve(self, pose: GraspPose, seed: JointSample) -> tuple[float, ...]:
        return self.kinematics.solve(
            np.array([pose.x, pose.y, pose.z]), top_down(pose.yaw), seed.q
        )

    def guard(self, sample: JointSample) -> None:
        base, _ = self.perception.base()
        self.perception.image()  # fail on stale camera, even during target occlusion
        old = self.review["observed"]["base_xyyaw"]
        p = self.config["servo"]
        if (
            np.linalg.norm(base[:2] - old[:2]) > p["base_tolerance_m"]
            or abs(math.remainder(base[2] - old[2], 2 * math.pi))
            > p["base_tolerance_rad"]
        ):
            raise RuntimeError("Base moved; stopping fixed-base forearm cycle")
        if (
            self.held
            and self.phase not in ("open", "retreat")
            and self.grasp_reading is not None
            and abs(sample.gripper - self.grasp_reading)
            > self.config["motion"]["gripper_tolerance"]
        ):
            raise RuntimeError("Gripper changed after grasp confirmation")

    def validate_path(self, sample: JointSample, q: tuple[float, ...]) -> None:
        self.kinematics.check_path(
            sample.q,
            q,
            held=self.held,
            on_support=self.phase in ("close", "lower", "open"),
        )

    def preflight(self, plan, sample: JointSample) -> None:
        self.guard(sample)
        if self.phase == "preflight":
            delta = (
                np.asarray(sample.q) - self.review["observed"]["arm_joints_rad"] + np.pi
            ) % (2 * np.pi) - np.pi
            if (
                max(abs(delta))
                > self.config["revalidation_tolerances"]["arm_joints_rad"]
            ):
                raise RuntimeError("Arm moved during controller startup; preview again")
            if (
                max(abs(v) for v in sample.dq)
                > self.config["motion"]["settled_speed_rad_s"]
            ):
                raise RuntimeError("Arm must be stationary at startup")
        self.kinematics.check_state(sample.q)
        if abs(plan.grasp.z - self.original_grasp.z) > 1e-6:
            raise RuntimeError("Support height changed")

    def grasp_confirmed(self, sample: JointSample) -> bool:
        p = self.config["servo"]
        return (
            self.held
            and p["grasp_reading_min"] <= sample.gripper <= p["grasp_reading_max"]
        )

    def support_confirmed(self, sample: JointSample) -> bool:
        if not self.supported:
            return False
        current = self.pose(sample)
        goal = self.original_grasp
        return (
            np.linalg.norm(
                np.array([current.x - goal.x, current.y - goal.y, current.z - goal.z])
            )
            <= self.config["motion"]["position_tolerance_m"]
            and max(abs(v) for v in sample.dq)
            <= self.config["motion"]["settled_speed_rad_s"]
        )

    def released(self, sample: JointSample) -> bool:
        return (
            abs(sample.gripper - self.config["motion"]["open_position"])
            <= self.config["motion"]["gripper_tolerance"]
        )


class LiveBackend(CompliantBackend):
    """Measured compliant motion with full initial orientation and ceiling servo."""

    scene: LiveScene

    def _joint_pregrasp(self, pose: GraspPose) -> None:
        """Execute the reviewed joint-space approach with compliant tracking."""
        p, settings = self.limits, self.scene.config["servo"]
        sample = self._sample()
        origin = np.asarray(sample.q)
        seed = self.scene.review.get("pregrasp_joints_rad")
        if seed is None:
            raise RuntimeError("Reviewed joint-space pregrasp endpoint missing")
        goal = np.asarray(
            self.scene.kinematics.solve(
                np.array([pose.x, pose.y, pose.z]), top_down(pose.yaw), seed
            )
        )
        delta = (goal - origin + np.pi) % (2 * np.pi) - np.pi
        for i, joint in enumerate(self.scene.kinematics.joints):
            if self.scene.kinematics.model.jnt_limited[joint]:
                delta[i] = goal[i] - origin[i]
        velocity = np.asarray(self.scene.config["hardware"]["velocity_rad_s"]) * 0.5
        duration = max(float(np.max(np.abs(delta) / velocity)), p.period_s)
        # Check entire joint path before sending its first command. Partition it
        # to preserve the per-command branch-jump guard.
        previous = origin
        for f in np.linspace(
            0,
            1,
            max(2, int(np.ceil(max(abs(delta)) / settings["max_joint_step_rad"])) + 1),
        ):
            q = (origin + f * delta + np.pi) % (2 * np.pi) - np.pi
            self.scene.kinematics.check_path(previous, q)
            previous = q
        fraction, stable = 0.0, 0
        deadline = time.monotonic() + p.timeout_s
        print(
            "REAL pregrasp: reviewed joint-space approach with compliant controller",
            flush=True,
        )
        while time.monotonic() < deadline:
            sample = self._sample()
            fraction = min(1.0, fraction + p.period_s / duration)
            q = (origin + fraction * delta + np.pi) % (2 * np.pi) - np.pi
            self._send(sample, tuple(q), self._gripper)
            xyz, rot = self.scene.kinematics.full_pose(sample.q)
            arrived = (
                fraction == 1
                and np.linalg.norm(xyz - [pose.x, pose.y, pose.z])
                <= p.position_tolerance_m
                and np.linalg.norm(rotation_error(top_down(pose.yaw), rot))
                <= settings["orientation_tolerance_rad"]
                and max(abs(v) for v in sample.dq) <= p.settled_speed_rad_s
            )
            stable = stable + 1 if arrived else 0
            if stable >= p.stable_samples:
                return
            self._wait()
        raise TimeoutError("Joint-space pregrasp timed out")

    def move(self, phase: str, pose: GraspPose) -> None:
        self.scene.phase = phase
        self.geometry.validate_pose(pose)
        if (
            phase == "pregrasp"
            and self.scene.review.get("pregrasp_joints_rad") is not None
        ):
            self._joint_pregrasp(pose)
            return
        sample = self._sample()
        origin, initial_rotation = self.scene.kinematics.full_pose(sample.q)
        target = np.array([pose.x, pose.y, pose.z])
        final_rotation = top_down(pose.yaw)
        angle = np.linalg.norm(rotation_error(final_rotation, initial_rotation))
        p, s = self.limits, self.scene.config["servo"]
        duration = max(
            np.linalg.norm(target - origin) / p.translation_speed_m_s,
            angle / s["orientation_speed_rad_s"],
            p.period_s,
        )
        qa, qb = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(qa, initial_rotation.ravel())
        mujoco.mju_mat2Quat(qb, final_rotation.ravel())
        if qa @ qb < 0:
            qb *= -1
        fraction, stable, last = 0.0, 0, -math.inf
        deadline = time.monotonic() + p.timeout_s
        print(f"REAL {phase}: target {target.tolist()}", flush=True)
        while time.monotonic() < deadline:
            sample = self._sample()
            xyz, actual_rotation = self.scene.kinematics.full_pose(sample.q)
            fraction = min(1.0, fraction + p.period_s / duration)
            aim = origin + fraction * (target - origin)
            if np.linalg.norm(aim - xyz) > p.max_command_lead_m:
                raise RuntimeError("Compliant arm lagged behind commanded path")
            quat = qa * (1 - fraction) + qb * fraction
            quat /= np.linalg.norm(quat)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            q = self.scene.kinematics.solve(aim, rotation.reshape(3, 3), sample.q)
            self._send(sample, q, self._gripper)
            if sample.captured_at > last:
                arrived = (
                    fraction == 1
                    and np.linalg.norm(xyz - target) <= p.position_tolerance_m
                    and np.linalg.norm(rotation_error(final_rotation, actual_rotation))
                    <= s["orientation_tolerance_rad"]
                    and max(abs(v) for v in sample.dq) <= p.settled_speed_rad_s
                )
                stable = stable + 1 if arrived else 0
                last = sample.captured_at
            if stable >= p.stable_samples:
                return
            self._wait()
        raise TimeoutError(f"{phase} arrival timed out")

    def align(self, pregrasp: GraspPose) -> GraspPose:
        p = self.scene.config["servo"]
        deadline = time.monotonic() + p["timeout_s"]
        stable, travel, updates = 0, 0.0, 0
        initial = np.array([pregrasp.x, pregrasp.y])
        last_capture = time.time()
        while time.monotonic() < deadline and updates < p["max_updates"]:
            sample = self._sample()
            # Hold the last commanded pose while waiting for a post-motion image.
            self._send(sample, self._hold_q, self._gripper)
            target, captured = self.scene.perception.target()
            if captured <= last_capture:
                self._wait()
                continue
            last_capture = captured
            updates += 1
            measured = self.scene.pose(sample)
            error = np.array([target.x - measured.x, target.y - measured.y])
            if (
                np.linalg.norm([target.x - initial[0], target.y - initial[1]])
                > p["max_travel_m"]
            ):
                raise RuntimeError(
                    "Visual target moved outside reviewed correction envelope"
                )
            if np.linalg.norm(error) <= p["tolerance_m"]:
                stable += 1
                if stable >= p["stable_frames"]:
                    self._grasp = GraspPose(
                        measured.x, measured.y, self._grasp.z, pregrasp.yaw
                    )
                    self.scene.original_grasp = self._grasp
                    return self._grasp
            else:
                stable = 0
                step = error * min(1.0, p["max_step_m"] / np.linalg.norm(error))
                travel += np.linalg.norm(step)
                if travel > p["max_travel_m"]:
                    raise RuntimeError("Visual correction travel budget exceeded")
                self.move(
                    "ceiling_align",
                    GraspPose(
                        measured.x + step[0],
                        measured.y + step[1],
                        pregrasp.z,
                        pregrasp.yaw,
                    ),
                )
                last_capture = time.time()
            self._wait()
        raise TimeoutError("Visual alignment timed out")

    def _confirmation(self, phase: str) -> None:
        self.scene.confirmations.request(phase)
        deadline = (
            time.monotonic() + self.scene.config["servo"]["confirmation_timeout_s"]
        )
        while time.monotonic() < deadline:
            sample = self._sample()
            self._send(sample, self._hold_q, self._gripper)
            if self.scene.confirmations.accepted.is_set():
                return
            self._wait()
        raise TimeoutError(f"Operator {phase} confirmation timed out")

    def close_gripper(self) -> None:
        self.scene.phase = "close"
        p, s = self.limits, self.scene.config["servo"]
        deadline, stable, last, last_time = (
            time.monotonic() + p.timeout_s,
            0,
            None,
            -math.inf,
        )
        while time.monotonic() < deadline:
            sample = self._sample()
            self._gripper = min(
                p.close_position, self._gripper + p.gripper_position_rate_s * p.period_s
            )
            self._send(sample, self._hold_q, self._gripper)
            if sample.captured_at > last_time:
                stalled = (
                    last is not None
                    and abs(sample.gripper - last) <= s["gripper_stall_epsilon"]
                )
                contact_candidate = (
                    s["grasp_reading_min"] <= sample.gripper <= s["grasp_reading_max"]
                    and self._gripper - sample.gripper > p.gripper_tolerance
                )
                stable = stable + 1 if stalled and contact_candidate else 0
                last, last_time = sample.gripper, sample.captured_at
            if sample.gripper > s["grasp_reading_max"]:
                raise RuntimeError("Gripper closed beyond expected forearm width")
            if stable >= s["gripper_stall_samples"]:
                self._gripper = sample.gripper
                self._send(sample, self._hold_q, self._gripper)
                self._confirmation("grasp")
                confirmed = self._sample()
                if (
                    not s["grasp_reading_min"]
                    <= confirmed.gripper
                    <= s["grasp_reading_max"]
                ):
                    raise RuntimeError(
                        "Gripper left contact interval during confirmation"
                    )
                self.scene.held, self.scene.grasp_reading = True, confirmed.gripper
                return
            self._wait()
        raise TimeoutError("No stable gripper contact candidate; lift refused")

    def support_confirmed(self) -> bool:
        if not self.scene.supported:
            if self.scene.phase != "lower":
                return False
            self._confirmation("support")
            self.scene.supported = True
        return self.scene.support_confirmed(self._sample())

    def open_gripper(self) -> None:
        self.scene.phase = "open"
        super().open_gripper()
        self.scene.held = False


class LiveRuntime:
    def __init__(
        self,
        review_path: Path,
        *,
        perception_factory=LivePerception,
        arm_reader=read_arm,
        arm_connect=LocalCompliantArm.connect,
        kinematics_factory=LiveKinematics,
    ) -> None:
        self.review_path = review_path
        self.review = json.loads(review_path.read_text())
        if not {"live.json", "reference.jpg"}.issubset(
            self.review.get("artifact_sha256", {})
        ):
            raise ValueError("Review must bind live configuration and reference image")
        self.config = json.loads((review_path.parent / "live.json").read_text())
        validate_config(self.config)
        self.geometry = TopDownGeometry(**self.config["geometry"])
        self.motion = MotionLimits(**self.config["motion"])
        self.hardware = HardwareLimits(**self.config["hardware"])
        model_path = os.environ.get("SPROUT_TIDYBOT_MODEL")
        if model_path is None:
            import kinder

            model_path = str(
                Path(kinder.__file__).parent
                / "envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
            )
        path = Path(model_path)
        self._model_path = path
        self.base_plan = self.config.get("base_plan")
        if self.base_plan != self.review.get("base_plan"):
            raise ValueError("Review/config base paths differ")
        if self.base_plan:
            from prpl_tidybot.sprout.base_planning import (
                transit_radius,
                validate_base_plan,
            )

            validate_base_plan(self.base_plan)
            if not np.allclose(
                self.base_plan["path_xyyaw"][0],
                self.review["observed"]["base_xyyaw"],
                atol=1e-8,
            ):
                raise ValueError("Base path must start at reviewed measured pose")
            minimum_radius = transit_radius(
                path, self.review["observed"]["arm_joints_rad"]
            )
            if self.base_plan["radius_m"] < minimum_radius + 0.04:
                raise ValueError(
                    "Base plan does not bound measured robot transit posture"
                )
        base_pose = self.review["observed"]["base_xyyaw"]
        if self.base_plan and self.base_plan["required"]:
            from prpl_tidybot.sprout.base_motion import validate_navigation
            from prpl_tidybot.sprout.base_planning import validate_base_plan

            validate_base_plan(self.base_plan)
            validate_navigation(self.config.get("navigation", {}))
            base_pose = self.base_plan["path_xyyaw"][-1]
        self.kinematics = kinematics_factory(path, base_pose, self.config)
        self.perception = perception_factory(
            self.config, review_path.parent / "reference.jpg"
        )
        self.confirmations = Confirmations()
        self.scene = LiveScene(
            self.config,
            self.review,
            self.kinematics,
            self.perception,
            self.confirmations,
        )
        self._reader, self._connect = arm_reader, arm_connect
        self._stop = threading.Event()
        self._arm_lock = threading.Lock()
        self._arm: Any = None
        self._base_motion: Any = None

    def observe(self) -> dict:
        deadline = time.monotonic() + self.config["servo"]["startup_timeout_s"]
        while True:
            if self._stop.is_set():
                raise Stopped("Stopped during observation")
            try:
                grasp, captured = self.perception.target()
                base, base_time = self.perception.base()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        arm = self._reader(self.config.get("arm_host", "192.168.1.10"))
        # Validate every saved trajectory segment before activating hardware.
        trajectory = json.loads(
            (self.review_path.parent / "trajectory.json").read_text()
        )
        self.kinematics.validate_trajectory(trajectory, arm["joints_rad"])
        if self.review.get("pregrasp_joints_rad") is not None:
            endpoints = [
                f["arm_joints_rad"] for f in trajectory if f["phase"] == "pregrasp"
            ]
            if not endpoints or not np.allclose(
                endpoints[-1], self.review["pregrasp_joints_rad"], atol=1e-8
            ):
                raise ValueError("Reviewed pregrasp endpoint differs from trajectory")
        # Planning may take seconds. Reacquire, rather than returning old data.
        grasp, captured = self.perception.target()
        base, base_time = self.perception.base()
        arm = self._reader(self.config.get("arm_host", "192.168.1.10"))
        if self.config.get("detection", {}).get("body_obstacle", {}).get("enabled"):
            self.perception.body()
        self._observed = True
        return {
            "ready": not self._stop.is_set(),
            "captured_at_unix": min(captured, base_time, arm["received_at_unix"]),
            "grasp_xyz": [grasp.x, grasp.y, grasp.z],
            "base_xyyaw": list(base),
            "arm_joints_rad": arm["joints_rad"],
        }

    def confirm(self, phase: str, token: str) -> None:
        self.confirmations.confirm(phase, token)

    def stop(self) -> None:
        self._stop.set()
        if self._base_motion is not None:
            self._base_motion.stop()
        with self._arm_lock:
            if self._arm is not None:
                self._arm.stop()

    def execute(self, review: dict) -> dict:
        if (
            self._stop.is_set()
            or review != self.review
            or not review.get("execution_ready")
        ):
            raise RuntimeError("Stopped or mismatched execution review")
        # Revalidation must have completed in this process before activation.
        if not getattr(self, "_observed", False):
            raise RuntimeError("Read-only observation required before execution")
        if getattr(self, "_used", False):
            raise RuntimeError("One cycle per approved runtime session")
        # Refuse a second controller owner. Merely launching legacy servers may
        # leave an initialized hardware controller from a prior client.
        from prpl_tidybot.sprout.process_guard import require_exclusive_motion

        require_exclusive_motion()
        self._used = True
        if self.base_plan and self.base_plan["required"]:
            from prpl_tidybot.sprout.base_motion import BaseMotion

            self._base_motion = BaseMotion(
                self.config["navigation"],
                self.perception,
                lambda: self._reader(self.config.get("arm_host", "192.168.1.10")),
                self._stop,
            )
            self._base_motion.execute(
                self.base_plan, np.asarray(self.review["observed"]["arm_joints_rad"])
            )
            base, _ = self.perception.base()
            target = np.asarray(self.base_plan["path_xyyaw"][-1])
            if (
                np.linalg.norm(base[:2] - target[:2])
                > self.config["servo"]["base_tolerance_m"]
                or abs(math.remainder(base[2] - target[2], 2 * math.pi))
                > self.config["servo"]["base_tolerance_rad"]
            ):
                raise RuntimeError(
                    "Base arrival outside arm-plan tolerance; preview again"
                )
            # Use measured arrival pose, not the ideal planned base pose.
            self.kinematics = LiveKinematics(self._model_path, list(base), self.config)
            arm_state = self._reader(self.config.get("arm_host", "192.168.1.10"))
            trajectory = json.loads(
                (self.review_path.parent / "trajectory.json").read_text()
            )
            previous = arm_state["joints_rad"]
            for frame in trajectory:
                self.kinematics.check_path(
                    previous,
                    frame["arm_joints_rad"],
                    held=frame["phase"] in ("lift", "lower"),
                )
                previous = frame["arm_joints_rad"]
            import copy

            self.scene.review = copy.deepcopy(self.review)
            self.scene.review["observed"]["base_xyyaw"] = list(base)
            self.scene.review["observed"]["arm_joints_rad"] = arm_state["joints_rad"]
            self.scene.kinematics = self.kinematics
            self.scene.localize()
        # Existing backend takes wrist limits; this implementation overrides the
        # alignment with calibrated ceiling XY servo and uses these only as an
        # explicit correction envelope.
        p = self.config["servo"]
        wrist = WristCalibration(
            (0, 0),
            ((1, 0), (0, 1)),
            p["tolerance_m"],
            p["max_step_m"],
            p["max_travel_m"],
            p["camera_max_age_s"],
            p["timeout_s"],
            p["stable_frames"],
            p["max_updates"],
            self.motion.position_tolerance_m,
            p["orientation_tolerance_rad"],
        )
        arm = self._connect(self.hardware, commissioning_verified=True)
        try:
            with self._arm_lock:
                self._arm = arm
                if self._stop.is_set():
                    raise Stopped("Stop requested during startup")
            backend = LiveBackend(arm, self.scene, self.geometry, self.motion, wrist)
            return ForearmSkill(
                backend, Recipe(**review["recipe"]), self.geometry
            ).cycle()
        finally:
            with self._arm_lock:
                self._arm = None
                arm.close()

    def close(self) -> None:
        self.stop()
        self.perception.close()


def create_runtime(review_path: Path) -> LiveRuntime:
    return LiveRuntime(review_path)
