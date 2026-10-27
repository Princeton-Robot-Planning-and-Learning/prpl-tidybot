"""Predefined left-forearm cycle, independent of ABLE and hardware imports.

The offline fake is executable today. The compliant backend requires calibrated
scene integration and a commissioned NUC-local session before hardware use.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from prpl_tidybot.sprout.approach import TopDownGeometry


@dataclass(frozen=True)
class GraspPose:
    """Calibrated pinch pose in the gravity-aligned map frame, metres/radians.

    Orientation is supplied by the backend's calibrated top-down grasp. This
    recipe only translates it: it does not infer tool axes or solve IK.
    """

    x: float
    y: float
    z: float
    yaw: float
    frame: str = "map"
    target: str = "sprout_left_forearm"

    def validate(self) -> None:
        """Reject non-finite coordinates or an unsupported pose frame."""
        if self.frame != "map" or self.target != "sprout_left_forearm":
            raise ValueError("Expected Sprout's left forearm in the map frame")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z, self.yaw)):
            raise ValueError("Non-finite grasp pose")
        if self.z <= 0:
            raise ValueError("Pinch height must be above the floor")

    def above(self, height: float) -> GraspPose:
        """Return a pose translated upward by the requested distance."""
        return replace(self, z=self.z + height)


@dataclass(frozen=True)
class Recipe:
    """Distances are recipe requests, never validated hardware safety limits."""

    pregrasp_height_m: float
    retreat_height_m: float
    lift_height_m: float = 0.15
    sprout_mode: str = "passive"

    def validate(self) -> None:
        """Validate the predefined lift, approach, and passive-arm assumptions."""
        for value in (
            self.pregrasp_height_m,
            self.retreat_height_m,
            self.lift_height_m,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Recipe heights must be finite and positive")
        if self.lift_height_m > 0.15:
            raise ValueError("This first-trial recipe is limited to a 15 cm request")
        if self.sprout_mode != "passive":
            raise ValueError("Active assistance requires a separately validated recipe")


@dataclass(frozen=True)
class Plan:
    """Calibrated target poses for each stage of the forearm cycle."""

    pregrasp: GraspPose
    grasp: GraspPose
    lifted: GraspPose
    put_down: GraspPose
    retreat: GraspPose

    def to_dict(self) -> dict[str, Any]:
        """Serialize the phase targets for preview and approval."""
        return asdict(self)


def make_plan(grasp: GraspPose, recipe: Recipe) -> Plan:
    """Preserve the initial support pose; 'up' is map +Z, never tool +Z."""
    grasp.validate()
    recipe.validate()
    return Plan(
        grasp.above(recipe.pregrasp_height_m),
        grasp,
        grasp.above(recipe.lift_height_m),
        grasp,
        grasp.above(recipe.retreat_height_m),
    )


class Backend(Protocol):
    """NUC-local implementation contract for fake and compliant backends.

    All operations must check cancellation and fresh force/joint/perception
    feedback during motion, enforce calibrated limits, and have deadlines.
    Motion methods return only on measured arrival, never on command issuance.
    stop must be callable concurrently, halt motion, and retain gripper state.
    A stopped skill must not enqueue subsequent commands.
    """

    def localize(self) -> GraspPose:
        """Fresh, unambiguous left-forearm grasp, with calibrated height and yaw."""

    def preflight(self, plan: Plan) -> None:
        """Check whole paths, body/floor clearance, joint/force limits and mode.

        Require empty/open gripper, fixed base, operator readiness, approved
        force settings, calibrated top-down tool orientation and feasible
        attached-limb motion. No robot motion is allowed here.
        """

    def move(self, phase: str, pose: GraspPose) -> None:
        """Guarded compliant translation; lowering ends at verified arrival."""

    def align(self, pregrasp: GraspPose) -> GraspPose:
        """Visual servo at hover; return refined grasp, preserving support Z.

        Reject lost/stale/ambiguous detections, timeout and excessive correction.
        Check each correction's swept path before commanding it.
        """

    def close_gripper(self) -> None:
        """Close using verified force and speed limits, not full-force position."""

    def grasp_confirmed(self) -> bool:
        """Fresh evidence of a stable grasp; commanded closure is insufficient."""

    def support_confirmed(self) -> bool:
        """Fresh evidence the forearm is supported before release."""

    def open_gripper(self) -> None:
        """Open and verify release, retaining the tool pose."""

    def stop(self) -> None:
        """Cancel and halt arm/base without opening or retracting the gripper."""


class Stopped(RuntimeError):
    """A latched stop requires a new server session before any new cycle."""


class ForearmSkill:
    """One cycle per call; stop is independent of the motion lock."""

    def __init__(
        self, backend: Backend, recipe: Recipe, geometry: TopDownGeometry
    ) -> None:
        recipe.validate()
        self.backend = backend
        self.recipe = recipe
        self.geometry = geometry
        self._stopped = threading.Event()
        self._motion = threading.Lock()

    def stop(self) -> None:
        """Latch the stop request and prevent subsequent motion commands."""
        self._stopped.set()
        self.backend.stop()

    def _check(self) -> None:
        if self._stopped.is_set():
            raise Stopped("Stop latched; inspect the scene before restarting server")

    def cycle(self) -> dict[str, Any]:
        """Run the guarded grasp, lift, lower, release, and retreat sequence."""
        if not self._motion.acquire(blocking=False):
            raise RuntimeError("Another forearm cycle is running")
        completed: list[str] = []
        try:
            self._check()
            plan = make_plan(self.backend.localize(), self.recipe)
            self.geometry.validate_pose(plan.grasp)
            self.backend.preflight(plan)
            self._check()
            self.backend.move("pregrasp", plan.pregrasp)
            completed.append("pregrasp")
            self._check()
            refined = self.backend.align(plan.pregrasp)
            refined.validate()
            # Visual alignment may refine XY/yaw, but must not silently change
            # the measured support height or identity.
            if not math.isclose(refined.z, plan.grasp.z, abs_tol=1e-6):
                raise ValueError("Visual alignment changed the support height")
            plan = make_plan(refined, self.recipe)
            self.geometry.validate_pose(plan.grasp)
            self.backend.preflight(plan)
            completed.append("align")
            self._check()
            self.backend.move("descend", plan.grasp)
            completed.append("descend")
            self._check()
            self.backend.close_gripper()
            self._check()
            if not self.backend.grasp_confirmed():
                raise RuntimeError("Grasp unconfirmed; lift refused")
            completed.append("close")
            self._check()
            self.backend.move("lift", plan.lifted)
            completed.append("lift")
            self._check()
            self.backend.move("lower", plan.put_down)
            completed.append("lower")
            self._check()
            if not self.backend.support_confirmed():
                raise RuntimeError("Support unconfirmed; gripper remains closed")
            self._check()
            self.backend.open_gripper()
            completed.append("open")
            self._check()
            self.backend.move("retreat", plan.retreat)
            completed.append("retreat")
            self._check()
            return {"completed": completed, "plan": plan.to_dict()}
        except BaseException:
            self.stop()
            raise
        finally:
            self._motion.release()
