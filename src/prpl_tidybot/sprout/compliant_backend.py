"""Predefined forearm cycle over a NUC-local compliant joint session.

Calibration/perception/contact checks are required collaborators, never fake
defaults. This module does not start hardware or provide a force-contact stop.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from prpl_tidybot.sprout.approach import (
    TopDownGeometry,
    WristAligner,
    WristCalibration,
    WristObservation,
)
from prpl_tidybot.sprout.sequence import GraspPose, Plan, Stopped

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointSample:
    """Measured feedback with NUC monotonic acquisition time, never RPC time."""

    q: tuple[float, ...]
    dq: tuple[float, ...]
    gripper: float
    captured_at: float


class CompliantArm(Protocol):
    """Measured arm feedback and bounded commands used by the skill backend."""

    controller_name: str

    def read(self) -> JointSample:
        """Read measured joints, gripper position, and acquisition time."""

    def command(self, q: tuple[float, ...], gripper: float) -> None:
        """Send a joint target and normalized gripper position."""

    def stop(self) -> None:
        """Latch stop; no open, retract, or subsequent accepted command."""


class CalibratedScene(Protocol):
    """Required lab integration. All reads/checks must be bounded in time.

    solve/pose must use the SAME calibrated pinch point and map transform;
    pose must reject orientations outside the calibrated top-down envelope.
    validate_path must check swept links, joint limits, attached forearm, fixed
    base, IK residual and discontinuities, not just the endpoint/floor.
    guard must check fresh scene and commissioned joint/contact limits.
    No method here may command motion. Contact evidence is independent of the
    commanded gripper position. No supplied implementation invents evidence.
    """

    def localize(self) -> GraspPose:
        """Return the validated forearm grasp pose in the calibrated map frame."""

    def preflight(self, plan: Plan, sample: JointSample) -> None:
        """Validate the planned cycle against measured state before motion."""

    def pose(self, sample: JointSample) -> GraspPose:
        """Return the measured pinch pose in the calibrated map frame."""

    def solve(self, pose: GraspPose, seed: JointSample) -> tuple[float, ...]:
        """Solve the target pinch pose from a measured joint seed."""

    def validate_path(self, sample: JointSample, q: tuple[float, ...]) -> None:
        """Reject a commanded joint segment that violates geometry guards."""

    def guard(self, sample: JointSample) -> None:
        """Validate fresh observations and commissioned operating limits."""

    def wrist(self) -> WristObservation | None:
        """Return a wrist observation when a calibrated wrist camera is available."""

    def grasp_confirmed(self, sample: JointSample) -> bool:
        """Return whether independent evidence confirms the forearm grasp."""

    def support_confirmed(self, sample: JointSample) -> bool:
        """Return whether evidence confirms the forearm is supported."""

    def released(self, sample: JointSample) -> bool:
        """Return whether measured gripper state confirms release."""


@dataclass(frozen=True)
class MotionLimits:
    """Required commissioning inputs; no hardware speed/force defaults."""

    period_s: float
    timeout_s: float
    feedback_max_age_s: float
    translation_speed_m_s: float
    max_command_lead_m: float
    position_tolerance_m: float
    yaw_tolerance_rad: float
    settled_speed_rad_s: float
    stable_samples: int
    open_position: float
    close_position: float
    gripper_position_rate_s: float
    gripper_tolerance: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in ("open_position", "close_position"):
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Gripper positions must be in [0, 1]")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid motion limit: {name}")
        if not isinstance(self.stable_samples, int) or self.stable_samples < 2:
            raise ValueError("Arrival requires at least two distinct samples")
        if self.open_position >= self.close_position:
            raise ValueError("Expected open_position < close_position")
        if self.period_s >= self.timeout_s:
            raise ValueError("Period must be shorter than timeout")


class CompliantBackend:
    """Backend for ForearmSkill: slow streamed targets and measured arrivals.

    Stop shares a short command lock, not the cycle lock. Long camera/IK calls
    happen outside it; a stopped backend cannot enqueue a late result. The local
    arm session separately watches command loss. Exceptions are stopped by
    ForearmSkill; standalone callers must also stop on any failed operation.
    """

    def __init__(
        self,
        arm: CompliantArm,
        scene: CalibratedScene,
        geometry: TopDownGeometry,
        limits: MotionLimits,
        wrist: WristCalibration,
    ) -> None:
        if arm.controller_name != "joint_compliant":
            raise ValueError("Sprout requires the joint compliant controller")
        self.arm, self.scene, self.geometry = arm, scene, geometry
        self.limits, self.wrist_calibration = limits, wrist
        self._stopped = threading.Event()
        self._command_lock = threading.Lock()
        self._grasp: GraspPose | None = None
        self._gripper: float | None = None
        self._hold_q: tuple[float, ...] | None = None

    def _check(self) -> None:
        if self._stopped.is_set():
            raise Stopped("Compliant Sprout backend stopped")

    def _sample(self) -> JointSample:
        self._check()
        sample = self.arm.read()
        values = (*sample.q, *sample.dq, sample.gripper, sample.captured_at)
        if (
            len(sample.q) != 7
            or len(sample.dq) != 7
            or not all(math.isfinite(v) for v in values)
            or not 0 <= sample.gripper <= 1
            or not 0
            <= time.monotonic() - sample.captured_at
            <= self.limits.feedback_max_age_s
        ):
            raise RuntimeError("Invalid or stale arm feedback")
        self.scene.guard(sample)
        self._check()
        return sample

    def _send(self, sample: JointSample, q: tuple[float, ...], grip: float) -> None:
        if len(q) != 7 or not all(math.isfinite(v) for v in q):
            raise ValueError("Invalid joint target")
        self.scene.validate_path(sample, q)
        with self._command_lock:
            self._check()
            # An expensive IK/path check must not authorize a stale command.
            if time.monotonic() - sample.captured_at > self.limits.feedback_max_age_s:
                raise RuntimeError("Feedback expired during planning")
            self.arm.command(q, grip)
            self._hold_q = q

    def _wait(self) -> None:
        if self._stopped.wait(self.limits.period_s):
            self._check()

    def localize(self) -> GraspPose:
        """Return the validated forearm grasp pose in the calibrated map frame."""
        self._check()
        self._grasp = self.scene.localize()
        self.geometry.validate_pose(self._grasp)
        return self._grasp

    def preflight(self, plan: Plan) -> None:
        """Validate the planned cycle against measured state before motion."""
        sample = self._sample()
        for pose in (
            plan.pregrasp,
            plan.grasp,
            plan.lifted,
            plan.put_down,
            plan.retreat,
        ):
            self.geometry.validate_pose(pose)
        if (
            abs(sample.gripper - self.limits.open_position)
            > self.limits.gripper_tolerance
        ):
            raise RuntimeError("Start with the empty gripper already open")
        self.scene.preflight(plan, sample)
        if self._gripper is None:
            self._gripper = sample.gripper

    def move(self, phase: str, pose: GraspPose) -> None:
        """Stream a bounded motion and wait for measured arrival."""
        self.geometry.validate_pose(pose)
        p = self.limits
        sample = self._sample()
        start = self.scene.pose(sample)
        self.geometry.validate_pose(start)
        if abs(math.remainder(pose.yaw - start.yaw, 2 * math.pi)) > p.yaw_tolerance_rad:
            raise ValueError(
                "Orient at the calibrated hover before starting this recipe"
            )
        if self._gripper is None:
            raise RuntimeError("Preflight required")
        origin = np.array([start.x, start.y, start.z])
        goal = np.array([pose.x, pose.y, pose.z])
        delta = goal - origin
        distance = float(np.linalg.norm(delta))
        direction = delta / distance if distance else delta
        progress = 0.0
        stable = 0
        last_capture = -math.inf
        deadline = time.monotonic() + p.timeout_s
        LOG.info("%s: target map xyz=%s", phase, goal.tolist())
        while time.monotonic() < deadline:
            sample = self._sample()
            measured = self.scene.pose(sample)
            self.geometry.validate_pose(measured)
            xyz = np.array([measured.x, measured.y, measured.z])
            if (
                abs(math.remainder(measured.yaw - pose.yaw, 2 * math.pi))
                > p.yaw_tolerance_rad
            ):
                raise RuntimeError("Tool yaw left the calibrated envelope")
            # A lagging/blocked compliant arm cannot accumulate a far-away target.
            candidate = min(distance, progress + p.translation_speed_m_s * p.period_s)
            aim_xyz = origin + candidate * direction
            if np.linalg.norm(aim_xyz - xyz) > p.max_command_lead_m:
                raise RuntimeError("Compliant tracking lag exceeds command lead limit")
            aim = replace(
                pose, x=float(aim_xyz[0]), y=float(aim_xyz[1]), z=float(aim_xyz[2])
            )
            self.geometry.validate_pose(aim)
            self._send(sample, self.scene.solve(aim, sample), self._gripper)
            progress = candidate
            if sample.captured_at > last_capture:
                arrived = (
                    progress >= distance
                    and np.linalg.norm(xyz - goal) <= p.position_tolerance_m
                    and max(abs(v) for v in sample.dq) <= p.settled_speed_rad_s
                )
                stable = stable + 1 if arrived else 0
                last_capture = sample.captured_at
            if stable >= p.stable_samples:
                LOG.info(
                    "%s: measured arrival, error=%.4f m",
                    phase,
                    np.linalg.norm(xyz - goal),
                )
                return
            self._wait()
        raise TimeoutError(f"{phase}: measured arrival timed out")

    def align(self, pregrasp: GraspPose) -> GraspPose:
        """Refine the grasp alignment using fresh visual observations."""
        if self._grasp is None:
            raise RuntimeError("Localize before alignment")
        aligner = WristAligner(self.wrist_calibration, pregrasp, time.monotonic())
        while True:
            sample = self._sample()
            target, done = aligner.update(
                self.scene.wrist(), self.scene.pose(sample), now=time.monotonic()
            )
            if done:
                self._grasp = replace(
                    self._grasp, x=target.x, y=target.y, yaw=target.yaw
                )
                return self._grasp
            self.move("wrist_align", target)
            # Next image must postdate the completed correction, not its start.
            aligner.last_update = time.monotonic()

    def _grip(self, target: float, closing: bool) -> None:
        if self._gripper is None or self._hold_q is None:
            raise RuntimeError("Move to grasp before gripper operation")
        deadline = time.monotonic() + self.limits.timeout_s
        LOG.info(
            "%s gripper with configured driver limits",
            "Closing" if closing else "Opening",
        )
        while time.monotonic() < deadline:
            sample = self._sample()
            if not closing and not self.scene.support_confirmed(sample):
                raise RuntimeError("Support lost during release")
            confirmed = (
                self.scene.grasp_confirmed(sample)
                if closing
                else self.scene.released(sample)
            )
            if confirmed:
                # Stop increasing closure when independently confirmed contact
                # is obtained; freeze at measured position, not full closure.
                self._gripper = sample.gripper
                self._send(sample, self._hold_q, self._gripper)
                return
            step = self.limits.gripper_position_rate_s * self.limits.period_s
            self._gripper += max(-step, min(step, target - self._gripper))
            self._send(sample, self._hold_q, self._gripper)
            self._wait()
        raise TimeoutError("Gripper confirmation timed out")

    def close_gripper(self) -> None:
        """Close gradually and require grasp evidence before lifting."""
        self._grip(self.limits.close_position, closing=True)

    def grasp_confirmed(self) -> bool:
        """Return whether independent evidence confirms the forearm grasp."""
        return self.scene.grasp_confirmed(self._sample())

    def support_confirmed(self) -> bool:
        """Return whether evidence confirms the forearm is supported."""
        return self.scene.support_confirmed(self._sample())

    def open_gripper(self) -> None:
        """Open the gripper only after confirming forearm support."""
        if not self.support_confirmed():
            raise RuntimeError("Support lost; refusing release")
        self._grip(self.limits.open_position, closing=False)

    def stop(self) -> None:
        """Latch the stop request and prevent subsequent motion commands."""
        with self._command_lock:
            self._stopped.set()
            self.arm.stop()
