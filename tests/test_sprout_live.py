# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Concrete adapter integration and failure paths, without hardware activation."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import kinder
import numpy as np
import pytest

from prpl_tidybot.sprout import compliant_backend, live_runtime
from prpl_tidybot.sprout.compliant_arm import HardwareLimits
from prpl_tidybot.sprout.compliant_backend import JointSample, MotionLimits
from prpl_tidybot.sprout.live_config import (
    CHECKS,
    calibration_digest,
    model_digest,
    validate_config,
)
from prpl_tidybot.sprout.live_kinematics import LiveKinematics, top_down
from prpl_tidybot.sprout.live_perception import LivePerception, fresh_time
from prpl_tidybot.sprout.live_runtime import Confirmations, LiveRuntime
from prpl_tidybot.sprout.sequence import GraspPose, Stopped


def config():
    """Synthetic test limits only; never installed as lab configuration."""
    return {
        "schema": "sprout_live_v1",
        "execution_enabled": True,
        "confirmation_mode": "operator",
        "verified": dict.fromkeys(CHECKS, True),
        "geometry": {
            "floor_z_arm_m": -0.395,
            "pad_drop_m": 0.0294,
            "minimum_clearance_m": 0.01,
            "downward_error_budget_m": 0.005,
        },
        "hardware": asdict(
            HardwareLimits(
                (-math.pi,) * 7,
                (math.pi,) * 7,
                (0.1,) * 7,
                (0.2,) * 7,
                (20.0,) * 7,
                10,
                5,
                0.5,
                0.1,
                2,
            )
        ),
        "motion": asdict(
            MotionLimits(
                0.01, 10, 0.1, 0.2, 0.03, 0.002, 0.01, 0.02, 2, 0, 0.7, 1, 0.01
            )
        ),
        "servo": {
            "tolerance_m": 0.002,
            "max_step_m": 0.01,
            "max_travel_m": 0.03,
            "timeout_s": 3,
            "stable_frames": 2,
            "max_updates": 30,
            "camera_max_age_s": 1,
            "base_tolerance_m": 0.01,
            "base_tolerance_rad": 0.01,
            "path_resolution_rad": 0.01,
            "max_joint_step_rad": 0.3,
            "orientation_speed_rad_s": 0.1,
            "orientation_tolerance_rad": 0.01,
            "confirmation_timeout_s": 1,
            "gripper_stall_samples": 3,
            "gripper_stall_epsilon": 0.001,
            "grasp_reading_min": 0.3,
            "grasp_reading_max": 0.4,
            "collision_margin_m": 0.005,
            "startup_timeout_s": 1,
        },
        "obstacles": [
            {"name": "far test box", "center": [10, 10, 10], "half_size": [0.1] * 3}
        ],
        "held_forearm": {"center": [0, 0, 0], "half_size": [0.02] * 3},
        "revalidation_tolerances": {
            "grasp_xyz": 0.01,
            "base_xyyaw": 0.01,
            "arm_joints_rad": 0.01,
        },
    }


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Provide the rig test fixture or callback."""
    clock = SimpleNamespace(now=100.0)
    timer = SimpleNamespace(
        time=lambda: clock.now,
        monotonic=lambda: clock.now,
        sleep=lambda amount: setattr(clock, "now", clock.now + amount),
    )
    monkeypatch.setattr(live_runtime, "time", timer)
    monkeypatch.setattr(compliant_backend, "time", timer)
    monkeypatch.setattr(live_runtime.LiveBackend, "_wait", lambda _: timer.sleep(0.01))
    monkeypatch.setenv("SPROUT_TIDYBOT_MODEL", "/unused-in-fake.xml")

    class Arm:
        """Test double for Arm."""

        controller_name = "joint_compliant"
        q = (0.5, 0, 0.3, 0, 0, 0, 0)
        gripper = 0.0
        stopped = False
        frozen = False
        commands = None

        def read(self):
            """Provide the read test fixture or callback."""
            return JointSample(self.q, (0.0,) * 7, self.gripper, clock.now)

        def command(self, q, gripper):
            """Provide the command test fixture or callback."""
            if self.stopped:
                raise Stopped("test stop")
            self.commands.append((q, gripper))
            if not self.frozen:
                self.q, self.gripper = q, min(gripper, 0.35)

        def stop(self):
            """Provide the stop test fixture or callback."""
            self.stopped = True

        close = stop

    class Perception:
        """Test double for Perception."""

        moved = False
        stale = False
        lost = False

        def __init__(self, *_):
            pass

        def target(self):
            """Provide the target test fixture or callback."""
            if self.lost:
                raise RuntimeError("tape lost")
            return GraspPose(0.5, 0, 0.1, 0), clock.now

        def base(self):
            """Provide the base test fixture or callback."""
            return np.array([1 if self.moved else 0, 0, 0]), clock.now

        def image(self):
            """Provide the image test fixture or callback."""
            if self.stale:
                raise RuntimeError("stale camera")
            return None, clock.now

        def close(self):
            """Provide the close test fixture or callback."""

    class Kinematics:
        """Test double for Kinematics."""

        def __init__(self, *_):
            pass

        def pose(self, q):
            """Provide the pose test fixture or callback."""
            return GraspPose(*q[:4])

        def full_pose(self, q):
            """Provide the full pose test fixture or callback."""
            return np.array(q[:3]), top_down(q[3])

        def solve(self, xyz, rotation, seed):
            """Provide the solve test fixture or callback."""
            return (*map(float, xyz), 0.0, 0.0, 0.0, 0.0)

        def check_state(self, *_, **__):
            """Provide the check state test fixture or callback."""

        def check_path(self, *_, **__):
            """Provide the check path test fixture or callback."""

        def validate_trajectory(self, *_, **__):
            """Provide the validate trajectory test fixture or callback."""

    arm = Arm()
    arm.commands = []
    connects = []

    def connect(*_, **__):
        """Provide the connect test fixture or callback."""
        connects.append(True)
        return arm

    report = {
        "execution_ready": True,
        "observed": {
            "grasp_xyz": [0.5, 0, 0.1],
            "base_xyyaw": [0, 0, 0],
            "arm_joints_rad": list(arm.q),
        },
        "plan": {"grasp": {"x": 0.5, "y": 0, "z": 0.1, "yaw": 0}},
        "recipe": {
            "pregrasp_height_m": 0.1,
            "retreat_height_m": 0.1,
            "lift_height_m": 0.03,
        },
        "revalidation_tolerances": {"grasp_xyz": 0.01},
        "artifact_sha256": {"live.json": "test", "reference.jpg": "test"},
    }
    (tmp_path / "live.json").write_text(json.dumps(config()))
    (tmp_path / "review.json").write_text(json.dumps(report))
    (tmp_path / "trajectory.json").write_text("[]")
    runtime = LiveRuntime(
        tmp_path / "review.json",
        perception_factory=Perception,
        arm_reader=lambda _: {"joints_rad": list(arm.q), "received_at_unix": clock.now},
        arm_connect=connect,
        kinematics_factory=Kinematics,
    )
    confirmations = []
    original = runtime.confirmations.request

    def operator(phase):
        """Provide the operator test fixture or callback."""
        confirmations.append((phase, arm.q[2], arm.gripper))
        original(phase)
        runtime.confirm(*runtime.confirmations.pending)

    runtime.confirmations.request = operator
    return SimpleNamespace(
        runtime=runtime,
        arm=arm,
        connects=connects,
        report=report,
        confirmations=confirmations,
        clock=clock,
    )


def test_concrete_runtime_full_supervised_cycle(rig):
    """Verify concrete runtime full supervised cycle."""
    assert rig.runtime.observe()["ready"]
    assert not rig.connects and not rig.arm.commands
    result = rig.runtime.execute(rig.report)
    assert result["completed"] == [
        "pregrasp",
        "align",
        "descend",
        "close",
        "lift",
        "lower",
        "open",
        "retreat",
    ]
    assert [item[0] for item in rig.confirmations] == ["grasp", "support"]
    assert all(abs(item[1] - 0.1) < 0.002 for item in rig.confirmations)
    assert rig.arm.stopped and rig.arm.gripper <= 0.01
    assert rig.arm.q[2] == pytest.approx(0.2)


def test_cannot_activate_without_observation_or_after_stop(rig):
    """Verify cannot activate without observation or after stop."""
    with pytest.raises(RuntimeError, match="observation"):
        rig.runtime.execute(rig.report)
    rig.runtime.observe()
    rig.runtime.stop()
    with pytest.raises(RuntimeError, match="Stopped"):
        rig.runtime.execute(rig.report)
    assert not rig.connects


@pytest.mark.parametrize("fault", ["stale", "moved", "lost"])
def test_scene_faults_stop_before_any_command(rig, fault):
    """Verify scene faults stop before any command."""
    rig.runtime.observe()
    setattr(rig.runtime.perception, fault, True)
    with pytest.raises(RuntimeError):
        rig.runtime.execute(rig.report)
    assert not rig.arm.commands and rig.arm.stopped


def test_no_support_confirmation_never_releases(rig):
    """Verify no support confirmation never releases."""
    rig.runtime.observe()
    original = rig.runtime.confirmations.request

    def operator(phase):
        """Provide the operator test fixture or callback."""
        if phase == "support":
            Confirmations.request(rig.runtime.confirmations, phase)
        else:
            original(phase)

    rig.runtime.confirmations.request = operator
    with pytest.raises(TimeoutError, match="support"):
        rig.runtime.execute(rig.report)
    assert rig.arm.stopped and rig.arm.gripper == pytest.approx(0.35)
    assert rig.arm.q[2] == pytest.approx(0.1)


def test_stop_during_grasp_confirmation_never_lifts(rig):
    """Verify stop during grasp confirmation never lifts."""
    rig.runtime.observe()
    rig.runtime.confirmations.request = lambda _: rig.runtime.stop()
    with pytest.raises(Stopped):
        rig.runtime.execute(rig.report)
    assert rig.arm.q[2] == pytest.approx(0.1)
    assert rig.arm.gripper == pytest.approx(0.35)


def test_confirmation_nonce_cannot_be_replayed():
    """Verify confirmation nonce cannot be replayed."""
    confirmation = Confirmations()
    confirmation.request("grasp")
    old = confirmation.pending
    confirmation.confirm(*old)
    confirmation.request("support")
    with pytest.raises(RuntimeError, match="stale"):
        confirmation.confirm(*old)
    assert not confirmation.accepted.is_set()


@pytest.mark.parametrize(
    "stamp", [None, float("nan"), time.time() - 5, time.time() + 5]
)
def test_camera_requires_finite_recent_acquisition(stamp):
    """Verify camera requires finite recent acquisition."""
    with pytest.raises(RuntimeError):
        fresh_time(stamp, 1)


def test_config_rejects_uncommissioned_force_limits():
    """Verify config rejects uncommissioned force limits."""
    cfg = config()
    validate_config(cfg)
    cfg["hardware"]["gripper_force_percent"] = None
    with pytest.raises(ValueError, match="gripper_force"):
        validate_config(cfg)


def test_actual_model_ik_and_obstacle_rejection():
    """Verify actual model ik and obstacle rejection."""

    model = (
        Path(kinder.__file__).parent
        / "envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
    )
    cfg = config()
    cfg["model_sha256"] = model_digest(model)
    kin = LiveKinematics(model, [0, 0, 0], cfg)
    q = np.array([0, -0.35, math.pi - 0.001, -2.55, 0, -0.87, math.pi / 2])
    xyz, rotation = kin.full_pose(q)
    solution = kin.solve(xyz + [0, 0, 0.002], rotation, q)
    assert np.linalg.norm(kin.full_pose(solution)[0] - xyz - [0, 0, 0.002]) < 0.001
    kin.check_path(q, solution)
    cfg["obstacles"][0]["center"] = xyz.tolist()
    with pytest.raises(RuntimeError, match="collision envelope"):
        kin.check_state(q)


def test_live_camera_projects_selected_patch_and_rejects_stale_or_wrong_calibration(
    tmp_path,
):
    """Verify live camera projects selected patch and rejects stale or wrong
    calibration."""

    frame = np.zeros((256, 128, 3), np.uint8)
    frame[48:64, 48:64] = np.random.default_rng(7).integers(
        0, 255, (16, 16, 3), dtype=np.uint8
    )
    reference = tmp_path / "reference.jpg"
    cv2.imwrite(str(reference), frame)
    projection = {
        "camera_matrix": [[100, 0, 64], [0, 100, 64], [0, 0, 1]],
        "rvec": [math.pi, 0, 0],
        "tvec": [0, 0, 1],
        "patch_height_m": 0.1,
        "patch_to_grasp_xy_m": [0.01, 0],
        "grasp_height_m": 0.08,
        "grasp_yaw_rad": 0,
    }
    cfg = config()
    cfg["perception_host"] = "fake"
    cfg["detection"] = {
        "camera_index": 0,
        "patch_xywh": [48, 48, 16, 16],
        "search_xywh": [32, 32, 48, 48],
        "minimum_quality": 0.8,
        "minimum_margin": 0.1,
        "projection": projection,
        "camera_calibration": {
            "serial": "test",
            "image_height": 128,
            "image_width": 128,
        },
    }
    meta = {
        "captured_at_unix": time.time(),
        "serial": "test",
        "calibration_sha256": calibration_digest(projection),
    }
    images = {"jpeg": reference.read_bytes(), "cameras": [meta, {}]}
    markers = {"poses": {0: [1, 2, 0.3]}, "pose_captured_at_unix": {0: time.time()}}

    def source(host, port):
        """Provide the source test fixture or callback."""
        return SimpleNamespace(
            get=lambda: images if port == 6003 else markers, close=lambda: None
        )

    perception = LivePerception(cfg, reference, source_factory=source)
    target, stamp = perception.target()
    assert [target.x, target.y, target.z] == pytest.approx([-0.062, 0.072, 0.08])
    assert stamp == meta["captured_at_unix"]
    assert perception.base()[0] == pytest.approx([1, 2, 0.3])
    meta["calibration_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="calibration differs"):
        perception.image()
    meta["captured_at_unix"] -= 10
    with pytest.raises(RuntimeError, match="stale"):
        perception.image()


def test_reviewed_joint_pregrasp_uses_compliant_commands(rig):
    """Verify reviewed joint pregrasp uses compliant commands."""
    goal = [0.5, 0, 0.2, 0, 0, 0, 0]
    rig.report["pregrasp_joints_rad"] = goal
    rig.runtime.review["pregrasp_joints_rad"] = goal
    rig.runtime.kinematics.joints = []  # fixture has Cartesian-as-joint kinematics
    (rig.runtime.review_path.parent / "trajectory.json").write_text(
        json.dumps([{"phase": "pregrasp", "arm_joints_rad": goal}])
    )
    assert rig.runtime.observe()["ready"]
    result = rig.runtime.execute(rig.report)
    assert "pregrasp" in result["completed"]
    assert rig.arm.stopped and len(rig.arm.commands) > 10


def test_joint_pregrasp_endpoint_must_match_reviewed_trajectory(rig):
    """Verify joint pregrasp endpoint must match reviewed trajectory."""
    rig.report["pregrasp_joints_rad"] = [0.5, 0, 0.2, 0, 0, 0, 0]
    rig.runtime.review["pregrasp_joints_rad"] = rig.report["pregrasp_joints_rad"]
    with pytest.raises(ValueError, match="pregrasp endpoint"):
        rig.runtime.observe()
    assert not rig.connects


@pytest.mark.parametrize("operation", ["move", "align", "close", "confirm"])
def test_live_backend_rejects_uninitialized_motion(rig, operation):
    """An out-of-order command must fail before sending any arm target."""
    backend = live_runtime.LiveBackend(
        rig.arm,
        rig.runtime.scene,
        rig.runtime.geometry,
        rig.runtime.motion,
        live_runtime.WristCalibration(
            (0, 0), ((1, 0), (0, 1)), 0.01, 0.01, 0.1, 1, 1, 2, 10, 0.01, 0.1
        ),
    )
    target = GraspPose(0.5, 0.0, 0.1, 0.0)
    operations = {
        "move": lambda: backend.move("descend", target),
        "align": lambda: backend.align(target),
        "close": backend.close_gripper,
        "confirm": lambda: backend._confirmation("grasp"),
    }
    with pytest.raises(RuntimeError, match="required"):
        operations[operation]()
    assert not rig.arm.commands
    assert not rig.confirmations
