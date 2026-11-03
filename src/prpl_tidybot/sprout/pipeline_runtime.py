# Hardware imports remain lazy for workstation simulation.
# pylint: disable=import-outside-toplevel
"""Sprout adapter for main's RPC → RealTidyBotEnv → Runner → PlanExecutor path."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from prpl_tidybot.sprout.live_config import model_digest
from prpl_tidybot.sprout.live_kinematics import LiveKinematics, top_down
from prpl_tidybot.sprout.live_perception import LivePerception
from prpl_tidybot.sprout.pipeline import (
    SproutPipeline,
    arm_pairs,
    base_pairs,
    recipe_phases,
)
from prpl_tidybot.sprout.read_robot_state import read_arm
from prpl_tidybot.sprout.sequence import Stopped
from prpl_tidybot.sprout.urdf_collision import urdf_digest


def real_interface(config: dict):
    """Use the same RPC interfaces as main, with the requested slow limits."""
    if model_digest(Path(os.environ["SPROUT_TIDYBOT_MODEL"])) != config["model_sha256"]:
        raise ValueError("NUC TidyBot model/assets differ from the reviewed model")
    if urdf_digest() != config["collision_model_sha256"]:
        raise ValueError("NUC PRPL collision model differs from the reviewed model")

    from prpl_tidybot.interfaces.base_interface import RealBaseInterface
    from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
    from prpl_tidybot.interfaces.interface import RealInterface
    from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface

    # As in main's cylinder-shelf config, unused on-robot camera slots are
    # placeholders. Forearm perception uses the actual ceiling-camera publisher.
    arm = RealArmInterface(reset_arm=False, control_limits=config["hardware"])
    try:
        base = RealBaseInterface(
            marker_detector_host=config["perception_host"],
            control_limits=config["navigation"],
        )
    except BaseException:
        arm.close()
        raise
    return RealInterface(arm, base, FakeCameraInterface())


class PipelineRuntime:
    """Approval/stop transport adapter, without direct motor ownership."""

    def __init__(
        self,
        review_path: Path,
        *,
        interface_factory=real_interface,
        perception_factory=LivePerception,
        arm_reader=read_arm,
        kinematics_factory=LiveKinematics,
    ) -> None:
        self.review_path = review_path
        self.review = json.loads(review_path.read_text())
        self.config = json.loads((review_path.parent / "live.json").read_text())
        self.trajectory = json.loads(
            (review_path.parent / "trajectory.json").read_text()
        )
        # Explicit user request: no camera acquisition-age cutoff, including
        # during motion. Connection, image identity/calibration and detection
        # failures still propagate. Hardware command watchdogs are unchanged.
        self.perception = perception_factory(
            self.config,
            review_path.parent / "reference.jpg",
            check_acquisition_age=False,
        )
        self.factory, self.reader = interface_factory, arm_reader
        self.kinematics_factory = kinematics_factory
        self.stopped = threading.Event()
        self.interface: Any = None
        self.pipeline: SproutPipeline | None = None
        self._used = False
        self.kinematics: Any = None
        self.target_offset = np.zeros(3)
        self._close_lock = threading.Lock()

    def observe(self) -> dict:
        """Read current target/joints/map pose without activating RPC controllers."""
        deadline = time.monotonic() + self.config["servo"]["startup_timeout_s"]
        while True:
            if self.stopped.is_set():
                raise Stopped("Stopped before execution")
            try:
                target, _ = self.perception.target()
                base, _ = self.perception.base()
                arm = self.reader(self.config.get("arm_host", "192.168.1.10"))
                return {
                    "ready": True,
                    "grasp_xyz": [target.x, target.y, target.z],
                    "base_xyyaw": list(base),
                    "arm_joints_rad": arm["joints_rad"],
                }
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def stop(self) -> None:
        """Use main's interface cleanup to stop RPC control loops."""
        self.stopped.set()
        self._close_interface()

    def _close_interface(self) -> None:
        with self._close_lock:
            interface, self.interface = self.interface, None
            if interface is not None:
                interface.close()

    def confirm(self, _phase: str, _token: str) -> None:
        """There are no additional grasp/support confirmation gates."""
        raise RuntimeError("No phase confirmation is pending")

    def execute(self, _review: dict) -> dict:
        """Run the approved predefined recipe using main's unmodified executors."""
        if self.stopped.is_set() or self._used:
            raise Stopped("Stopped or already executed")
        self._used = True
        self.interface = self.factory(self.config)
        if self.stopped.is_set():
            self._close_interface()
            raise Stopped("Stopped during interface startup")
        try:
            self.pipeline = SproutPipeline(self.interface, self.config, self.stopped)
            plan = self.review.get("base_plan")
            if plan and plan["required"]:
                self.pipeline.run(
                    "base transit", base_pairs(self.pipeline.state, plan["path_xyyaw"])
                )
            completed = []
            for phase, frames in recipe_phases(self.trajectory):
                if phase in ("base", "base_transit"):
                    continue
                if self.kinematics is not None and phase != "pregrasp":
                    frames = self._corrected_frames(frames)
                self.pipeline.run(phase, arm_pairs(self.pipeline.state, frames))
                completed.append(phase)
                if phase == "pregrasp":
                    self._align()
            return {"completed": completed, "execution_backend": "prpl_main_runner"}
        finally:
            self._close_interface()

    def _align(self) -> None:
        """Feed ceiling-XY correction plans through the same main arm executor."""
        assert self.pipeline is not None
        state = self.pipeline.state
        robot = state.get_object_from_name("robot")
        base = [
            state.get(robot, k) for k in ("pos_base_x", "pos_base_y", "pos_base_rot")
        ]
        self.kinematics = self.kinematics_factory(
            Path(os.environ["SPROUT_TIDYBOT_MODEL"]), base, self.config
        )
        p = self.config["servo"]
        for _ in range(p["max_updates"]):
            if self.stopped.is_set():
                raise Stopped("Stopped during ceiling alignment")
            state = self.pipeline.state
            q = [state.get(robot, f"joint_{j+1}") for j in range(7)]
            xyz, _ = self.kinematics.full_pose(q)
            target, _ = self.perception.target()
            self.target_offset = (
                np.array([target.x, target.y, target.z])
                - self.review["observed"]["grasp_xyz"]
            )
            delta = np.array([target.x, target.y]) - xyz[:2]
            if np.linalg.norm(delta) <= p["tolerance_m"]:
                return
            if np.linalg.norm(delta) > p["max_travel_m"]:
                raise RuntimeError(
                    "Target outside the preview's servo correction range"
                )
            delta *= min(1.0, p["max_step_m"] / np.linalg.norm(delta))
            goal = xyz.copy()
            goal[:2] += delta
            solved = self.kinematics.solve(goal, top_down(target.yaw), q)
            self.kinematics.check_path(q, solved)
            self.pipeline.run(
                "ceiling alignment",
                arm_pairs(state, [{"arm_joints_rad": solved, "gripper": "open"}]),
            )
        raise RuntimeError("Ceiling alignment did not converge")

    def _corrected_frames(self, frames: list[dict]) -> list[dict]:
        """Preserve the servo correction and measured arrival base in later phases."""
        assert self.pipeline is not None
        state = self.pipeline.state
        robot = state.get_object_from_name("robot")
        seed = [state.get(robot, f"joint_{j+1}") for j in range(7)]
        yaw = self.review["plan"]["grasp"]["yaw"]
        corrected = []
        for frame in frames:
            goal = np.asarray(frame["pinch_xyz"]) + self.target_offset
            q = self.kinematics.solve(goal, top_down(yaw), seed)
            self.kinematics.check_path(
                seed, q, held=frame["phase"] in ("lift", "lower")
            )
            corrected.append(dict(frame, arm_joints_rad=list(q)))
            seed = q
        return corrected

    def close(self) -> None:
        """Release RPC and perception resources on success, failure or Ctrl-C."""
        self.stop()
        self.perception.close()


def create_runtime(review_path: Path) -> PipelineRuntime:
    """Factory used by the laptop approval/stop transport."""
    return PipelineRuntime(review_path)
