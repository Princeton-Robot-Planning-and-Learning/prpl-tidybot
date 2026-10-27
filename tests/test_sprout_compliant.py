"""Compliant integration tests with synthetic feedback; never connect hardware."""

import sys
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from prpl_tidybot.sprout import compliant_arm, compliant_backend
from prpl_tidybot.sprout.approach import WristCalibration, WristObservation
from prpl_tidybot.sprout.compliant_arm import HardwareLimits, LocalCompliantArm
from prpl_tidybot.sprout.compliant_backend import (
    CompliantBackend,
    JointSample,
    MotionLimits,
)
from prpl_tidybot.sprout.offline_server import SYNTHETIC_GEOMETRY
from prpl_tidybot.sprout.reviewed_runtime import ReviewedCompliantRuntime
from prpl_tidybot.sprout.sequence import ForearmSkill, GraspPose, Recipe, Stopped


@pytest.fixture
def rig(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        compliant_backend, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )

    class Arm:
        controller_name = "joint_compliant"

        def __init__(self):
            self.q = (0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0)
            self.grip = 0.0
            self.commands = []
            self.stopped = False
            self.frozen = False

        def read(self):
            return JointSample(self.q, (0.0,) * 7, self.grip, clock.now)

        def command(self, q, gripper):
            assert not self.stopped
            self.commands.append((q, gripper))
            if not self.frozen:
                self.q, self.grip = q, gripper

        def stop(self):
            self.stopped = True

    class Scene:
        def __init__(self):
            self.support = True

        def localize(self):
            return GraspPose(0.5, 0, 0.1, 0)

        def preflight(self, plan, sample):
            pass

        def pose(self, sample):
            return GraspPose(*sample.q[:4])

        def solve(self, pose, seed):
            return (pose.x, pose.y, pose.z, pose.yaw, 0.0, 0.0, 0.0)

        def validate_path(self, sample, q):
            pass

        def guard(self, sample):
            pass

        def wrist(self):
            return WristObservation((0, 0), clock.now)

        def grasp_confirmed(self, sample):
            return sample.gripper >= 0.3

        def support_confirmed(self, sample):
            return self.support

        def released(self, sample):
            return sample.gripper <= 0.01

    arm, scene = Arm(), Scene()
    limits = MotionLimits(
        0.01, 10, 0.1, 0.2, 0.03, 0.002, 0.01, 0.02, 2, 0.0, 0.7, 1.0, 0.01
    )
    wrist = WristCalibration(
        (0, 0), ((0.001, 0), (0, 0.001)), 1, 0.01, 0.05, 0.1, 3, 2, 20, 0.01, 0.01
    )
    backend = CompliantBackend(arm, scene, SYNTHETIC_GEOMETRY, limits, wrist)

    def wait():
        clock.now += limits.period_s

    backend._wait = wait
    skill = ForearmSkill(backend, Recipe(0.1, 0.1), SYNTHETIC_GEOMETRY)
    return SimpleNamespace(
        clock=clock, arm=arm, scene=scene, backend=backend, skill=skill
    )


def test_full_cycle_streams_bounded_steps_and_preserves_contact_height(rig):
    result = rig.skill.cycle()
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
    commands = rig.arm.commands
    assert min(q[2] for q, _ in commands) >= 0.1 - 1e-9
    assert max(grip for _, grip in commands) < 0.7  # stops closing at confirmed contact
    assert commands[-1][0][2] == pytest.approx(0.2)
    assert commands[-1][1] <= 0.01
    for (a, _), (b, _) in pairwise(commands):
        assert np.linalg.norm(np.array(a[:3]) - b[:3]) <= 0.002 + 1e-9


def test_blocked_arm_does_not_descend_or_close(rig):
    rig.arm.frozen = True
    with pytest.raises(RuntimeError, match="tracking lag"):
        rig.skill.cycle()
    assert rig.arm.stopped
    assert all(grip == 0 for _, grip in rig.arm.commands)


def test_missing_support_never_releases(rig):
    rig.scene.support = False
    with pytest.raises(RuntimeError, match="Support unconfirmed"):
        rig.skill.cycle()
    assert rig.arm.stopped and rig.arm.grip >= 0.3
    assert rig.arm.q[2] == pytest.approx(0.1)


def test_stale_feedback_refuses_first_command(rig):
    original = rig.arm.read
    rig.arm.read = lambda: replace(original(), captured_at=rig.clock.now - 1)
    with pytest.raises(RuntimeError, match="stale"):
        rig.skill.cycle()
    assert rig.arm.commands == [] and rig.arm.stopped


def test_stop_during_ik_drops_late_solution(rig):
    original = rig.scene.solve

    def solve(pose, sample):
        rig.skill.stop()
        return original(pose, sample)

    rig.scene.solve = solve
    with pytest.raises(Stopped):
        rig.skill.cycle()
    assert rig.arm.commands == []


def test_repeated_feedback_cannot_establish_arrival(rig):
    original = rig.arm.read
    rig.arm.read = lambda: replace(original(), captured_at=100.0)
    with pytest.raises(RuntimeError, match="stale"):
        rig.skill.cycle()
    assert rig.arm.grip == 0


def test_angular_backend_rejected(rig):
    rig.arm.controller_name = "angular"
    with pytest.raises(ValueError, match="compliant"):
        CompliantBackend(
            rig.arm,
            rig.scene,
            SYNTHETIC_GEOMETRY,
            rig.backend.limits,
            rig.backend.wrist_calibration,
        )


def hardware_limits():
    return HardwareLimits(
        (-7.0,) * 7,
        (7.0,) * 7,
        (0.1,) * 7,
        (0.2,) * 7,
        (20.0,) * 7,
        10.0,
        5.0,
        0.5,
        0.1,
        2.0,
    )


def test_hardware_connection_requires_explicit_commissioning():
    with pytest.raises(RuntimeError, match="verified first"):
        LocalCompliantArm.connect(hardware_limits())


@pytest.mark.parametrize("moved", [False, True])
def test_reviewed_runtime_binds_target_and_runs_compliant_cycle(
    rig, monkeypatch, moved
):
    connected = []

    def connect(*args, **kwargs):
        connected.append(True)
        return rig.arm

    monkeypatch.setattr(LocalCompliantArm, "connect", connect)
    monkeypatch.setattr(
        CompliantBackend,
        "_wait",
        lambda _: setattr(rig.clock, "now", rig.clock.now + 0.01),
    )
    rig.arm.close = rig.arm.stop
    runtime = ReviewedCompliantRuntime(
        rig.scene,
        lambda: {"ready": True},
        SYNTHETIC_GEOMETRY,
        rig.backend.limits,
        rig.backend.wrist_calibration,
        hardware_limits(),
        commissioning_verified=True,
    )
    report = {
        "execution_ready": True,
        "completed": True,
        "recipe": {
            "pregrasp_height_m": 0.1,
            "retreat_height_m": 0.1,
            "lift_height_m": 0.03,
        },
        "observed": {"grasp_xyz": [0.5 + (0.1 if moved else 0), 0, 0.1]},
        "plan": {"grasp": {"x": 0.5, "y": 0, "z": 0.1, "yaw": 0}},
        "revalidation_tolerances": {"grasp_xyz": 0.005},
    }
    if moved:
        with pytest.raises(RuntimeError, match="Target changed"):
            runtime.execute(report)
        assert connected == []
    else:
        assert runtime.execute(report)["completed"][-1] == "retreat"
        assert connected == [True] and rig.arm.stopped


def test_local_watchdog_latches_hold_without_open_or_retract(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        compliant_arm, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    made = []

    class Controller:
        def __init__(self, commands, **kwargs):
            self.commands = commands
            made.append(kwargs)

        def control_callback(self, arm):
            if self.commands.empty():
                return np.zeros(7), arm.gripper_pos
            _, grip = self.commands.get_nowait()
            return np.zeros(7), grip

    arm = SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        tau=np.zeros(7),
        gripper_pos=0.35,
        feedback_time=100.0,
    )
    session = LocalCompliantArm(arm, hardware_limits(), Controller)
    session._callback(arm)
    session.command((0.01,) * 7, 0.6)
    clock.now += 0.6
    arm.feedback_time = clock.now
    _, grip = session._callback(arm)
    assert grip == 0.35
    assert session.stop_reason == "command watchdog"
    assert made[-1]["max_velocity"] == (0.1,) * 7
    with pytest.raises(Stopped):
        session.command((0.0,) * 7, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gripper_force_percent", 0),
        ("gripper_speed_percent", 101),
        ("velocity_rad_s", (float("nan"),) * 7),
    ],
)
def test_invalid_driver_limits(field, value):
    with pytest.raises(ValueError):
        replace(hardware_limits(), **{field: value})


def test_connect_uses_real_compliant_controller_without_retract_or_open(monkeypatch):
    from prpl_tidybot.third_party.arm_controller import JointCompliantController

    calls = []

    class FakeHardware:
        def __init__(self):
            self.q = np.zeros(7)
            self.dq = np.zeros(7)
            self.tau = np.zeros(7)
            self.gripper_pos = 0.35
            self.actuator_count = 7

        def gravity(self):
            return np.zeros(7)

        def set_joint_limits(self, **kwargs):
            calls.append(("joint_limits", kwargs))

        def init_cyclic(self, callback, **kwargs):
            calls.append(("start", kwargs))
            self.feedback_time = compliant_arm.time.monotonic()
            _, grip = callback(self)
            assert grip == 0.35

        def stop_cyclic(self):
            calls.append(("stop", {}))

        def disconnect(self):
            calls.append(("disconnect", {}))

    monkeypatch.setitem(
        sys.modules,
        "prpl_tidybot.third_party.kinova",
        SimpleNamespace(TorqueControlledArm=FakeHardware),
    )
    session = LocalCompliantArm.connect(hardware_limits(), commissioning_verified=True)
    assert isinstance(session._controller, JointCompliantController)
    assert session._controller.otg_inp.max_velocity == [0.1] * 7
    assert session._controller.otg_inp.max_acceleration == [0.2] * 7
    assert calls[1] == (
        "start",
        {"gripper_force_limit": 10.0, "gripper_speed_limit": 5.0},
    )
    session.stop()
    with pytest.raises(Stopped):
        session.command((0.0,) * 7, 0.0)
    session.close()
    assert [name for name, _ in calls] == [
        "joint_limits",
        "start",
        "stop",
        "stop",
        "disconnect",
    ]


def test_gripper_limit_applied_to_first_cyclic_frame(monkeypatch):
    """Load actual Kortex wrapper with only its unused model import stubbed."""
    import importlib.util
    from pathlib import Path
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "pinocchio", ModuleType("pinocchio"))
    path = Path(__file__).parents[1] / "src/prpl_tidybot/third_party/kinova.py"
    spec = importlib.util.spec_from_file_location("kinova_limits_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = module.TorqueControlledArm.__new__(module.TorqueControlledArm)
    arm.cyclic_running = False
    arm.actuator_count = 0
    arm.actuator_device_ids = []
    arm.motor_cmd = SimpleNamespace()
    arm.base_command = SimpleNamespace()
    feedback = SimpleNamespace(
        interconnect=SimpleNamespace(
            gripper_feedback=SimpleNamespace(motor=[SimpleNamespace(position=35.0)])
        )
    )
    frames = []

    def refresh(*args):
        frames.append(vars(arm.motor_cmd).copy())
        return feedback

    arm.base_cyclic = SimpleNamespace(RefreshFeedback=lambda: feedback, Refresh=refresh)
    arm.send_options = None
    arm.base = SimpleNamespace(SetServoingMode=lambda _: None)
    enum = SimpleNamespace(Value=lambda _: 1)
    module.Base_pb2 = SimpleNamespace(
        ServoingModeInformation=SimpleNamespace, LOW_LEVEL_SERVOING=1
    )
    module.ActuatorConfig_pb2 = SimpleNamespace(
        ControlModeInformation=SimpleNamespace, ControlMode=enum
    )
    monkeypatch.setattr(module.os, "sched_setscheduler", lambda *args: None)
    module.threading = SimpleNamespace(
        Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    arm.init_cyclic(lambda _: None, gripper_force_limit=10, gripper_speed_limit=5)
    assert frames == [{"position": 35.0, "velocity": 0, "force": 10}]
    assert arm.gripper_speed_limit == 5
    with pytest.raises(ValueError):
        arm.init_cyclic(lambda _: None, gripper_force_limit=float("nan"))
