"""Wire an approved recipe to the NUC-local compliant backend.

The lab must supply CalibratedScene and a read-only snapshot source. No default
scene fabricates camera freshness, collision clearance, contact or support.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

from prpl_tidybot.sprout.approach import TopDownGeometry, WristCalibration
from prpl_tidybot.sprout.compliant_arm import HardwareLimits, LocalCompliantArm
from prpl_tidybot.sprout.compliant_backend import (
    CalibratedScene,
    CompliantBackend,
    MotionLimits,
)
from prpl_tidybot.sprout.sequence import ForearmSkill, GraspPose, Plan, Recipe


class _ReviewedScene:
    """Keep each localization/servo update within the approved grasp envelope."""

    def __init__(
        self, scene: CalibratedScene, review: dict[str, Any], wrist: WristCalibration
    ) -> None:
        self.scene, self.review, self.wrist_limits = scene, review, wrist

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scene, name)

    def localize(self) -> GraspPose:
        """Return the validated forearm grasp pose in the calibrated map frame."""
        actual = self.scene.localize()
        expected = self.review["observed"]["grasp_xyz"]
        tolerance = self.review["revalidation_tolerances"]["grasp_xyz"]
        if any(
            abs(a - b) > tolerance
            for a, b in zip(expected, (actual.x, actual.y, actual.z))
        ):
            raise RuntimeError("Target changed after approval; preview again")
        return actual

    def preflight(self, plan: Plan, sample: Any) -> None:
        """Validate the planned cycle against measured state before motion."""
        approved = self.review["plan"]["grasp"]
        actual = plan.grasp
        if (
            math.hypot(actual.x - approved["x"], actual.y - approved["y"])
            > self.wrist_limits.max_travel_m
            or abs(actual.z - approved["z"])
            > self.review["revalidation_tolerances"]["grasp_xyz"]
            or abs(math.remainder(actual.yaw - approved["yaw"], 2 * math.pi))
            > self.wrist_limits.hover_yaw_tolerance_rad
        ):
            raise RuntimeError("Refined grasp outside reviewed envelope; preview again")
        self.scene.preflight(plan, sample)


class ReviewedCompliantRuntime:
    """Constructing this runtime/reading its snapshot does not activate an arm.

    Pass observe/execute to approve_and_execute. The snapshot and scene adapter
    must share calibration/target identity. A failed scene check or cycle closes
    cyclic control without issuing an open/home/retract command.
    """

    def __init__(
        self,
        scene: CalibratedScene,
        snapshot: Callable[[], dict[str, Any]],
        geometry: TopDownGeometry,
        motion: MotionLimits,
        wrist: WristCalibration,
        hardware: HardwareLimits,
        *,
        commissioning_verified: bool = False,
    ) -> None:
        self.scene, self.snapshot = scene, snapshot
        self.geometry, self.motion, self.wrist, self.hardware = (
            geometry,
            motion,
            wrist,
            hardware,
        )
        self.commissioning_verified = commissioning_verified
        self._stop_requested = threading.Event()
        self._arm_lock = threading.Lock()
        self._active_arm: LocalCompliantArm | None = None

    def stop(self) -> None:
        """Latch cancellation, including during startup; never open/retract."""
        self._stop_requested.set()
        with self._arm_lock:
            if self._active_arm is not None:
                self._active_arm.stop()

    def observe(self) -> dict[str, Any]:
        """Read the target and robot state for approval revalidation."""
        sample = self.snapshot()
        return {
            **sample,
            "ready": sample.get("ready") is True and self.commissioning_verified,
        }

    def execute(self, review: dict[str, Any]) -> dict[str, Any]:
        """Run the approved cycle through the compliant backend."""
        if (
            self._stop_requested.is_set()
            or not self.commissioning_verified
            or review.get("execution_ready") is not True
            or review.get("completed") is not True
        ):
            raise RuntimeError("Hardware commissioning and successful review required")
        recipe = Recipe(**review["recipe"])
        recipe.validate()
        scene = _ReviewedScene(self.scene, review, self.wrist)
        scene.localize()
        arm = LocalCompliantArm.connect(self.hardware, commissioning_verified=True)
        try:
            with self._arm_lock:
                self._active_arm = arm
                if self._stop_requested.is_set():
                    raise RuntimeError("Stop requested during hardware startup")
            backend = CompliantBackend(
                arm, scene, self.geometry, self.motion, self.wrist
            )
            skill = ForearmSkill(backend, recipe, self.geometry)
            return skill.cycle()
        finally:
            with self._arm_lock:
                self._active_arm = None
                arm.close()
