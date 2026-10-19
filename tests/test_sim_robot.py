"""Tests for `prpl_tidybot.sim_robot`: a kinematic TidyBot that executes the arm, base
and gripper commands the real robot would, with the real arm server's own toss."""

import inspect
import math
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
from spatialmath import SE2

from prpl_tidybot.sim_robot import (
    ARM_SOFT_ACCEL_LIMIT_DEG_S2,
    ARM_SOFT_SPEED_LIMIT_DEG_S,
    BASE_MAX_ACCEL,
    BASE_MAX_VEL,
    COMPLIANT_MAX_ACCEL,
    COMPLIANT_MAX_VEL,
    SimArmInterface,
    SimBaseInterface,
    SimWorld,
)
from prpl_tidybot.third_party import arm_server
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    CARRY_ARM_CONF,
    TopDownPickGeometry,
    top_down_joint_targets,
)
from prpl_tidybot.toss import plan_toss_swing, toss_release_pose, toss_windup_pose


@pytest.fixture(autouse=True)
def _no_release_wait(monkeypatch):
    """The arm server waits 0.75 s of wall time before stopping its torque loop; the
    sim clock still advances by that much."""
    monkeypatch.setattr(
        arm_server, "time", SimpleNamespace(sleep=lambda _s: None, time=time.time)
    )


def _world(**kwargs):
    defaults = {"base_pose": (0.0, 0.0, 0.0), "cube": (1.5, 0.0, 0.0)}
    defaults.update(kwargs)
    return SimWorld(**defaults)


def _wrapped_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def test_the_toss_swing_is_the_trajectory_the_arm_server_streams():
    """The swing executed in sim is, cycle for cycle at 1 kHz, the trajectory the real
    arm server plans from the measured wind-up pose; the gripper starts opening at
    `gripper_release_ms` and the wind-up ends exactly at the wind-up pose."""
    world = _world()
    arm = SimArmInterface(world)
    for _ in range(40):
        arm.execute_action(list(CARRY_ARM_CONF), 1.0)
    arm.toss(heading=0.0, max_vel=120.0, gripper_release_ms=675)
    swing = world.swings[-1]
    planned = plan_toss_swing(
        toss_windup_pose(0.0), 0.0, max_vel=120.0, gripper_release_ms=675
    )
    assert swing["executed_deg"].shape == planned.shape
    assert np.abs(_wrapped_deg(swing["executed_deg"] - planned)).max() < 1e-3
    assert swing["release_index"] == 675
    assert np.abs(_wrapped_deg(swing["windup_end_deg"] - toss_windup_pose(0))).max() < (
        1e-3
    )
    assert np.abs(
        _wrapped_deg(np.degrees(arm.get_arm_state()) - toss_release_pose(0.0))
    ).max() == pytest.approx(0.0, abs=1e-3)
    assert world.gripper_target == 0.0


def test_the_sim_toss_is_the_arm_servers_own_toss(monkeypatch):
    """`toss` runs `arm_server.Arm.toss` itself, so its checks hold in sim too: an arm
    that stops more than 1 deg short of the wind-up pose is not swung."""
    world = _world()
    arm = SimArmInterface(world)
    world.angular_stop_short_deg = 2.0
    with pytest.raises(RuntimeError, match="not swinging"):
        arm.toss(heading=0.0, max_vel=100.0, gripper_release_ms=500)
    assert not world.swings
    calls = []
    original = arm_server.Arm.toss

    def spy(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(arm_server.Arm, "toss", spy)
    world.angular_stop_short_deg = 0.0
    arm.toss(heading=0.0, max_vel=100.0, gripper_release_ms=500)
    assert calls == [{"heading": 0.0, "max_vel": 100.0, "gripper_release_ms": 500}]


def test_angular_moves_are_synchronized_at_the_arm_servers_soft_limits():
    """A Kortex REACH_JOINT_ANGLES move under the 30 deg/s, 80 deg/s^2 soft limits the
    arm server sets: every joint starts and stops together, the furthest-moving one
    at the speed limit."""
    world = _world()
    arm = SimArmInterface(world)
    arm.release()
    start = np.degrees(arm.get_arm_state())
    goal = list(np.radians(start + np.array([60.0, 10.0, 0.0, 30.0, 0.0, -20.0, 0.0])))
    t0 = world.t
    arm.move_angular(goal)
    path = np.asarray(world.last_angular_path_deg)
    speed = np.abs(np.diff(path, axis=0)).max(axis=0) / world.angular_dt
    assert speed[0] == pytest.approx(ARM_SOFT_SPEED_LIMIT_DEG_S, rel=0.02)
    assert speed[1] == pytest.approx(ARM_SOFT_SPEED_LIMIT_DEG_S / 6, rel=0.02)
    assert world.t - t0 == pytest.approx(60 / 30 + 30 / 80, rel=0.02)
    assert np.abs(_wrapped_deg(path[-1] - np.degrees(goal))).max() < 1e-6


def test_angular_moves_take_the_short_way_only_on_the_unlimited_joints():
    """Joints 1, 3, 5 and 7 turn without limit and go the short way to a [0, 360)
    target; joints 2, 4 and 6 have stops and cannot cross 180 deg."""
    world = _world()
    arm = SimArmInterface(world)
    arm.release()
    start = np.radians([170.0, 0.0, 180.0, 140.0, 0.0, 0.0, 90.0])
    world.q[:] = start
    goal = list(np.radians([-170.0, 0.0, 180.0, -140.0, 0.0, 0.0, 90.0]))
    arm.move_angular(goal)
    path = np.asarray(world.last_angular_path_deg)
    assert path[:, 0].max() == pytest.approx(190.0, abs=1e-3)  # through 180
    assert path[:, 3].min() == pytest.approx(-140.0, abs=1e-3)  # through 0
    assert np.abs(path[:, 3]).max() <= 140.0 + 1e-3


def test_the_compliant_loop_follows_the_controllers_trajectory_generator():
    """Streamed joint targets (`execute_action`) are tracked through the compliant
    controller's own Ruckig limits; a target across the wrap goes the short way."""
    world = _world()
    arm = SimArmInterface(world)
    target = list(RETRACT_ARM_CONF)
    target[0] = math.radians(179.0)
    world.q[0] = math.radians(-179.0)
    for _ in range(30):
        arm.execute_action(target, 0.0)
    assert math.degrees(world.q[0]) == pytest.approx(-181.0, abs=1e-3)
    world.q[:] = RETRACT_ARM_CONF
    arm = SimArmInterface(world)
    goal = list(np.array(RETRACT_ARM_CONF) + np.radians([90, 0, 0, 0, 0, 0, 0]))
    trace = []
    for _ in range(40):
        arm.execute_action(goal, 0.0)
        trace.append(world.q[0])
    speed = np.abs(np.diff(trace)).max() / 0.1
    assert speed == pytest.approx(COMPLIANT_MAX_VEL[0], rel=0.02)


def test_the_gripper_holds_the_cube_only_when_closed_around_it():
    """Closing at the grasp pose reads ~0.49 (the 5 cm cube) and carries the cube;
    closing anywhere else reads ~1.0 and leaves it on the floor."""
    geom = TopDownPickGeometry()
    world = _world(cube=(0.67, 0.0, 0.0))
    arm = SimArmInterface(world)
    hover, grasp = top_down_joint_targets(0.55, 0.0, 0.0, geom, seed=RETRACT_ARM_CONF)
    arm.release()
    arm.move_angular(hover)
    assert arm.set_gripper_angular(1.0) == pytest.approx(1.0)
    assert not world.held
    arm.set_gripper_angular(0.0)
    arm.move_angular(grasp)
    assert 0.2 < arm.set_gripper_angular(1.0) < 0.9
    assert world.held
    arm.move_angular(hover)
    assert world.cube_position()[2] > 0.1


def test_the_base_drives_under_the_base_controllers_limits():
    """Position targets go through the base controller's Ruckig limits, in its
    continuous heading (a wrapped target does not spin the base a full turn)."""
    world = _world()
    base = SimBaseInterface(world)
    arm = SimArmInterface(world)
    xs = []
    for _ in range(80):
        base.execute_action(SE2(2.0, 0.0, 0.0))
        arm.execute_action(arm.get_arm_state(), 0.0)
        xs.append(base.get_base_state().x)
    assert xs[-1] == pytest.approx(2.0, abs=1e-3)
    assert np.diff(xs).max() / 0.1 == pytest.approx(BASE_MAX_VEL[0], rel=0.02)
    world.base[2] = math.radians(179.0)
    for _ in range(30):
        base.get_base_state()
        base.execute_action(SE2(2.0, 0.0, math.radians(-179.0)))
        arm.execute_action(arm.get_arm_state(), 0.0)
    assert math.degrees(world.base[2]) == pytest.approx(181.0, abs=1e-3)


def test_a_base_left_without_commands_stops():
    """As on the robot, the base holds when its command stream stops (an arm-only
    move), instead of finishing the last target."""
    world = _world()
    base = SimBaseInterface(world)
    arm = SimArmInterface(world)
    base.execute_action(SE2(2.0, 0.0, 0.0))
    arm.execute_action(arm.get_arm_state(), 0.0)
    for _ in range(30):
        arm.execute_action(arm.get_arm_state(), 0.0)
    assert world.base[0] < 0.5


def test_the_sim_limits_are_the_real_controllers():
    """The limits copied into the sim are the ones the arm server, the compliant
    controller and the base controller use."""
    limits = {}

    class _Kortex:
        def set_joint_limits(self, speed_limits, acceleration_limits):
            """Record what Arm.__init__ sets."""
            limits["speed"], limits["accel"] = speed_limits, acceleration_limits

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(arm_server, "TorqueControlledArm", _Kortex)
        mp.setattr(arm_server, "IKSolver", lambda **_kw: None)
        arm_server.Arm()
    assert limits["speed"] == (ARM_SOFT_SPEED_LIMIT_DEG_S,) * 7
    assert limits["accel"] == (ARM_SOFT_ACCEL_LIMIT_DEG_S2,) * 7

    fake = SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        tau=np.zeros(7),
        gripper_pos=0.0,
        actuator_count=7,
    )
    fake.gravity = lambda: np.zeros(7)
    controller = arm_server.JointCompliantController(queue.Queue(1))
    controller.control_callback(fake)
    assert list(controller.otg_inp.max_velocity) == pytest.approx(COMPLIANT_MAX_VEL)
    assert list(controller.otg_inp.max_acceleration) == pytest.approx(
        COMPLIANT_MAX_ACCEL
    )

    base_controller = pytest.importorskip("prpl_tidybot.third_party.base_controller")
    params = inspect.signature(base_controller.Vehicle.__init__).parameters
    assert params["max_vel"].default == pytest.approx(BASE_MAX_VEL)
    assert params["max_accel"].default == pytest.approx(BASE_MAX_ACCEL)
