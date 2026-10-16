"""A kinematic TidyBot for ``scripts/toss_cycles.py mode=sim``.

The point is to watch the robot's commanded paths before a real run, so the arm, base
and gripper commands ``toss_cycles`` sends are executed as the robot's own controllers
would turn them into motion, and nothing else is modelled:

* onboard-controller moves (``move_angular``, the grasp sequence and the toss
  wind-up) run through the real arm server (``third_party/arm_server.Arm``) on a
  simulated Kortex arm, whose REACH_JOINT_ANGLES is a synchronized trapezoid at the
  soft limits the arm server sets;
* the toss is ``arm_server.Arm.toss`` itself: the same wind-up check, and the same
  1 kHz swing from ``prpl_tidybot.toss``, executed cycle by cycle;
* streamed joint targets (``execute_action``) and base targets follow the compliant
  arm controller's and the base controller's Ruckig trajectory generators with their
  limits, tracked perfectly.

The cube is only there so the loop can continue: it rides in a closed gripper, and a
thrown cube lands ``landing_offset`` ahead of the base. Time is the sim's own clock
(``SimWorld.t``), which is what the video plays.
"""

from __future__ import annotations

import logging
import math
import queue
import re
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from prpl_utils.structs import Image
from ruckig import (  # type: ignore[import-not-found]  # pylint: disable=no-name-in-module
    InputParameter,
    OutputParameter,
    Ruckig,
)
from spatialmath import SE2

from prpl_tidybot.interfaces.arm_interface import ArmInterface
from prpl_tidybot.interfaces.base_interface import BaseInterface, unwrap_heading
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
from prpl_tidybot.interfaces.interface import Interface
from prpl_tidybot.real_sim.perceivers.target_source import TargetSource
from prpl_tidybot.third_party import arm_server
from prpl_tidybot.third_party.constants import (
    MOUNTING_OFFSET,
    POLICY_CONTROL_PERIOD,
    RETRACT_ARM_CONF,
)
from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics
from prpl_tidybot.toss import (
    TOSS_GRIPPER_RELEASE_MS,
    TOSS_MAX_VEL,
    trapezoidal_motion_profile,
)

_logger = logging.getLogger(__name__)

# The ANGULAR_TRAJECTORY soft limits arm_server.Arm.__init__ sets (deg/s, deg/s^2).
ARM_SOFT_SPEED_LIMIT_DEG_S = 30
ARM_SOFT_ACCEL_LIMIT_DEG_S2 = 80
# JointCompliantController's Ruckig limits (rad/s, rad/s^2).
COMPLIANT_MAX_VEL = 4 * [math.radians(80)] + 3 * [math.radians(140)]
COMPLIANT_MAX_ACCEL = 4 * [math.radians(240)] + 3 * [math.radians(450)]
# The base controller's (Vehicle) Ruckig limits on (x, y, heading).
BASE_MAX_VEL = (0.5, 0.5, 1.57)
BASE_MAX_ACCEL = (0.25, 0.25, 0.79)
# Both controllers hold where they are once commands stop for this long.
_COMMAND_TIMEOUT_S = 2.5 * POLICY_CONTROL_PERIOD
# Kortex joints 2, 4 and 6 have hard stops; the others turn without limit.
_LIMITED_JOINTS = (1, 3, 5)
# Kortex's move_angular is waited on for at most this long (third_party/kinova.py).
_ANGULAR_TIMEOUT_S = 20.0

_TICK = 0.001
_CUBE_HALF = 0.025
# The 2F-85 crosses its full stroke in about 0.6 s at full speed.
_GRIPPER_RATE = 1.0 / 0.6
# Closed on the 5 cm cube the gripper read 0.49 (conf/toss_cycles.yaml).
GRASP_READING = 0.49
_ARM_MOUNT_Z = 0.3948
_FLIGHT_S = 0.8


class SimWorld:
    """The simulated robot and cube, and the record of everything they did.

    ``q`` is the arm's continuous joint angles (rad) and ``base`` the base's
    (x, y, continuous heading); both are tracked perfectly. ``samples`` is the
    trajectory the video and the safety summary are made from.
    """

    def __init__(
        self,
        base_pose: Sequence[float],
        cube: Sequence[float],
        landing_offset: Sequence[float] = (1.2, 0.0),
        record_period: float = 0.005,
    ) -> None:
        self.t = 0.0
        self.base = np.array(base_pose, dtype=float)
        self.q = np.array(RETRACT_ARM_CONF, dtype=float)
        self.gripper = 0.0
        self.gripper_target = 0.0
        self.held = False
        self.cyclic = False
        self.label = "start"
        self.sublabel = ""
        self.toss_setting = ""
        self.landing_offset = (float(landing_offset[0]), float(landing_offset[1]))
        self.samples: list[tuple[Any, ...]] = []
        self.swings: list[dict[str, Any]] = []
        # Stretches of the clock the video plays in slow motion (the swings).
        self.slow_spans: list[tuple[float, float]] = []
        self.warnings: list[str] = []
        self.angular_dt = _TICK
        # Lets a test make the onboard controller stop short of its goal.
        self.angular_stop_short_deg = 0.0
        self.last_angular_path_deg: NDArray[np.float64] | None = None
        self._geom = TopDownPickGeometry()
        self._cube = np.array([cube[0], cube[1], _CUBE_HALF, cube[2]], dtype=float)
        self._flight: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = (
            None
        )
        self._record_period = record_period
        self._last_record = -math.inf

        self._arm_otg = Ruckig(7, _TICK)
        self._arm_in = InputParameter(7)
        self._arm_out = OutputParameter(7)
        self._arm_in.max_velocity = COMPLIANT_MAX_VEL
        self._arm_in.max_acceleration = COMPLIANT_MAX_ACCEL
        self._arm_target: NDArray[np.float64] | None = None
        self._arm_quiet = 0.0
        self._arm_vel = np.zeros(7)

        self._base_otg = Ruckig(3, _TICK)
        self._base_in = InputParameter(3)
        self._base_out = OutputParameter(3)
        self._base_in.max_velocity = list(BASE_MAX_VEL)
        self._base_in.max_acceleration = list(BASE_MAX_ACCEL)
        self._base_target: NDArray[np.float64] | None = None
        self._base_quiet = 0.0
        self._base_vel = np.zeros(3)
        self._record(force=True)

    # Commands.

    def start_compliant(self) -> None:
        """The torque loop starts holding wherever the arm is."""
        self.cyclic = True
        self._arm_vel = np.zeros(7)
        self._arm_target = self.q.copy()
        self._arm_quiet = 0.0

    def stop_compliant(self) -> None:
        """The torque loop stops; the onboard controller holds the arm."""
        self.cyclic = False
        self._arm_target = None
        self._arm_vel = np.zeros(7)

    def command_arm(self, goal: Sequence[float], gripper: float) -> None:
        """A compliant-loop joint target, unwrapped onto the arm's current angles as
        JointCompliantController does."""
        goal_arr = np.asarray(goal, dtype=float)
        self._arm_target = self.q + np.mod(goal_arr - self.q + np.pi, 2 * np.pi) - np.pi
        self._arm_quiet = 0.0
        self.gripper_target = float(gripper)

    def command_base(self, target: Sequence[float]) -> None:
        """A base position target (x, y, continuous heading)."""
        self._base_target = np.asarray(target, dtype=float)
        self._base_quiet = 0.0

    def idle(self, seconds: float) -> None:
        """Let time pass with no new command."""
        self.advance(seconds)

    def play_arm_path(self, path_rad: NDArray[np.float64]) -> None:
        """Move the arm along `path_rad` (one row per 1 ms tick)."""
        for row in path_rad:
            self.q = np.array(row, dtype=float)
            self.advance(_TICK)

    def place_cube(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Put the cube down on the floor at (x, y)."""
        self.held = False
        self._flight = None
        self._cube = np.array([x, y, _CUBE_HALF, yaw], dtype=float)

    def throw_cube(self) -> None:
        """The gripper opened mid-swing: the cube leaves the pinch point and lands
        ``landing_offset`` (forward, left) from the base."""
        if not self.held:
            return
        self.held = False
        x, y, theta = self.base
        fwd, left = self.landing_offset
        land = np.array(
            [
                x + fwd * math.cos(theta) - left * math.sin(theta),
                y + fwd * math.sin(theta) + left * math.cos(theta),
                _CUBE_HALF,
            ]
        )
        self._flight = (self.t, self.pinch_world(), land)
        self._cube = np.array([land[0], land[1], _CUBE_HALF, theta])

    # Clock.

    def advance(self, seconds: float) -> None:
        """Run the controllers forward by `seconds`, in 1 ms ticks."""
        for _ in range(max(1, int(round(seconds / _TICK)))):
            self._tick()

    def _tick(self) -> None:
        self.t += _TICK
        if self.cyclic and self._arm_target is not None:
            self._arm_quiet += _TICK
            if self._arm_quiet > _COMMAND_TIMEOUT_S:
                self._arm_target = self.q.copy()
            self.q, self._arm_vel = _otg_step(
                self._arm_otg,
                self._arm_in,
                self._arm_out,
                self.q,
                self._arm_vel,
                self._arm_target,
            )
        if self._base_target is not None:
            self._base_quiet += _TICK
            if self._base_quiet > _COMMAND_TIMEOUT_S:
                self._base_target = self.base.copy()
            self.base, self._base_vel = _otg_step(
                self._base_otg,
                self._base_in,
                self._base_out,
                self.base,
                self._base_vel,
                self._base_target,
            )
        self._step_gripper()
        if self._flight is not None and self.t - self._flight[0] >= _FLIGHT_S:
            self._flight = None
        self._record()

    def _step_gripper(self) -> None:
        step = _GRIPPER_RATE * _TICK
        if self.gripper_target > self.gripper:
            self.gripper = min(self.gripper_target, self.gripper + step)
            if self.held:
                self.gripper = min(self.gripper, GRASP_READING)
            elif self.gripper >= GRASP_READING and self._cube_between_fingers():
                self.gripper = GRASP_READING
                self.held = True
        elif self.gripper_target < self.gripper:
            self.gripper = max(self.gripper_target, self.gripper - step)
            if self.held and self.gripper < GRASP_READING - 0.05:
                pinch = self.pinch_world()
                self.place_cube(pinch[0], pinch[1], self._cube[3])

    def _record(self, force: bool = False) -> None:
        if not force and self.t - self._last_record < self._record_period:
            return
        self._last_record = self.t
        cube = None if self.held else self.cube_position()
        self.samples.append(
            (
                self.t,
                self.base.copy(),
                self.q.copy(),
                self.gripper,
                cube,
                self.label,
                self.sublabel,
            )
        )

    # Geometry.

    def pinch_world(self) -> NDArray[np.float64]:
        """The gripper's pinch point in the map frame."""
        pos = forward_kinematics(list(self.q), self._geom)[0]
        x, y, theta = self.base
        c, s = math.cos(theta), math.sin(theta)
        return np.array(
            [
                x + MOUNTING_OFFSET * c + c * pos[0] - s * pos[1],
                y + MOUNTING_OFFSET * s + s * pos[0] + c * pos[1],
                _ARM_MOUNT_Z + pos[2],
            ]
        )

    def cube_position(self) -> NDArray[np.float64]:
        """The cube's (x, y, z, yaw) now: in the gripper, in flight or at rest."""
        if self.held:
            pinch = self.pinch_world()
            return np.array([*pinch, self._cube[3]])
        if self._flight is not None:
            t0, start, land = self._flight
            u = min(1.0, (self.t - t0) / _FLIGHT_S)
            pos = start + u * (land - start)
            pos[2] += 4 * 0.5 * u * (1 - u)
            return np.array([*pos, self._cube[3]])
        return self._cube.copy()

    def cube_at_rest(self) -> NDArray[np.float64]:
        """Where the cube is, or will be once a throw lands: what the ceiling cameras
        report after the settle wait."""
        return self._cube.copy() if not self.held else self.cube_position()

    def _cube_between_fingers(self) -> bool:
        pinch = self.pinch_world()
        cube = self.cube_position()
        return bool(
            math.hypot(pinch[0] - cube[0], pinch[1] - cube[1]) < 0.02
            and abs(pinch[2] - cube[2]) < _CUBE_HALF
        )


def _otg_step(
    otg: Ruckig,
    inp: InputParameter,
    out: OutputParameter,
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    inp.current_position = list(position)
    inp.current_velocity = list(velocity)
    inp.current_acceleration = [0.0] * len(position)
    inp.target_position = list(target)
    inp.target_velocity = [0.0] * len(position)
    otg.update(inp, out)
    return np.array(out.new_position), np.array(out.new_velocity)


class SimKortexArm:
    """The part of ``third_party.kinova.TorqueControlledArm`` that
    ``arm_server.Arm`` uses, on a :class:`SimWorld`. Angles are Kortex degrees."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self.actuator_count = 7
        self.speed_limits: tuple[float, ...] = (ARM_SOFT_SPEED_LIMIT_DEG_S,) * 7
        self.acceleration_limits: tuple[float, ...] = (ARM_SOFT_ACCEL_LIMIT_DEG_S2,) * 7

    @property
    def cyclic_running(self) -> bool:
        """Whether the torque loop is running."""
        return self.world.cyclic

    @property
    def q(self) -> NDArray[np.float64]:
        """Joint angles (rad)."""
        return self.world.q.copy()

    @property
    def gripper_pos(self) -> float:
        """The gripper (0 open, 1 closed)."""
        return self.world.gripper

    def set_joint_limits(
        self, speed_limits: Sequence[float], acceleration_limits: Sequence[float]
    ) -> None:
        """The ANGULAR_TRAJECTORY soft limits for onboard moves."""
        self.speed_limits = tuple(speed_limits)
        self.acceleration_limits = tuple(acceleration_limits)

    def clear_faults(self) -> None:
        """Nothing faults in sim."""

    def init_cyclic(self, control_callback: Any) -> None:
        """Start the torque loop (simulated by the compliant controller's OTG)."""
        del control_callback
        self.world.start_compliant()

    def stop_cyclic(self) -> None:
        """Stop the torque loop. ``Arm.release`` first waits 0.75 s for the arm to
        stop, which the sim clock shows as a pause."""
        self.world.idle(0.75)
        self.world.stop_compliant()

    def move_angular(self, joint_angles: Sequence[float], timeout: float = 20) -> None:
        """REACH_JOINT_ANGLES: a synchronized trapezoidal joint move at the soft
        limits, the short way on the unlimited joints."""
        assert not self.world.cyclic, "Arm must be in high-level servoing mode"
        start = np.degrees(self.world.q)
        goal = np.asarray(joint_angles, dtype=float)
        diffs = (goal - start + 180.0) % 360.0 - 180.0
        for j in _LIMITED_JOINTS:
            diffs[j] = _signed_deg(goal[j]) - _signed_deg(start[j])
        short = np.minimum(np.abs(diffs), self.world.angular_stop_short_deg)
        diffs = diffs - np.sign(diffs) * short
        max_diff = float(np.abs(diffs).max())
        if max_diff < 1e-9:
            self.world.last_angular_path_deg = start[None, :]
            return
        _, pos = trapezoidal_motion_profile(
            max_diff,
            max_vel=float(min(self.speed_limits)),
            max_accel=float(min(self.acceleration_limits)),
            max_decel=float(min(self.acceleration_limits)),
            step_size=self.world.angular_dt,
        )
        path = start + np.outer(pos.astype(float) / max_diff, diffs)
        path[-1] = start + diffs
        duration = len(path) * self.world.angular_dt
        if duration > min(timeout, _ANGULAR_TIMEOUT_S):
            self.world.warnings.append(
                f"t={self.world.t:.1f}s: an onboard move takes {duration:.1f} s, past "
                f"the {timeout:.0f} s the driver waits; the next command would be sent "
                "while the arm is still moving."
            )
        self.world.last_angular_path_deg = path
        self.world.play_arm_path(np.radians(path))

    def get_measured_joint_angles(self) -> list[float]:
        """Joint angles in [0, 360) degrees, as Kortex reports them."""
        return [float(v) for v in np.degrees(self.world.q) % 360.0]

    def get_feedback_joint_positions(self) -> list[float]:
        """Actuator positions (deg) from one cyclic feedback frame."""
        return self.get_measured_joint_angles()

    def run_position_trajectory(
        self, trajectory: Sequence[Sequence[float]], gripper_release_step: int
    ) -> None:
        """Stream `trajectory` (cycles x joints, degrees) at 1 kHz, opening the gripper
        from cycle `gripper_release_step` on, as ``TorqueControlledArm`` does."""
        assert not self.world.cyclic, "Torque loop must be stopped"
        world = self.world
        traj = np.asarray(trajectory, dtype=float)
        start = world.q.copy()
        t0 = world.t
        executed = np.empty_like(traj)
        world.sublabel = f"swing, 1 kHz position stream ({world.toss_setting})"
        for i, row in enumerate(traj):
            if i == gripper_release_step:
                world.gripper_target = 0.0
                world.throw_cube()
                world.sublabel += ": gripper opening"
            world.q = start + np.radians(row - traj[0])
            executed[i] = np.degrees(world.q)
            world.advance(_TICK)
        world.swings.append(
            {
                "executed_deg": executed,
                "release_index": int(gripper_release_step),
                "windup_end_deg": np.degrees(start),
                "t_start": t0,
                "t_release": t0 + gripper_release_step * _TICK,
                "t_end": world.t,
            }
        )
        world.slow_spans.append((t0, world.t))

    def gripper_to(self, value: float, timeout: float = 4.0) -> float:
        """Move the gripper; returns where the fingers stopped (on the cube, ~0.49)."""
        world = self.world
        world.gripper_target = float(value)
        deadline = world.t + timeout
        while world.t < deadline:
            before = world.gripper
            world.advance(0.01)
            if abs(world.gripper - before) < 1e-9:
                break
        return world.gripper


def _signed_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


class SimArmInterface(ArmInterface):
    """The arm, through the real arm server's code on a :class:`SimKortexArm`."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        kortex = SimKortexArm(world)
        # Built as Arm.__init__ does, without connecting to hardware.
        server = arm_server.Arm.__new__(arm_server.Arm)
        server.arm = kortex
        kortex.set_joint_limits(
            speed_limits=(7 * (ARM_SOFT_SPEED_LIMIT_DEG_S,)),
            acceleration_limits=(7 * (ARM_SOFT_ACCEL_LIMIT_DEG_S2,)),
        )
        server.command_queue = queue.Queue(1)
        server.controller = None
        server.reset(reset_arm=False)
        self.server: Any = server

    def get_arm_state(self) -> list[float]:
        return [float(v) for v in self.server.get_joint_angles()]

    def get_gripper_state(self) -> float:
        return float(self.server.get_gripper_position())

    def execute_action(self, arm_goal: list[float], gripper_goal: float) -> None:
        if not self.world.cyclic:
            raise RuntimeError(
                "execute_action with the torque loop released; on the robot the arm "
                "server's command queue would fill and block."
            )
        self.world.command_arm(arm_goal, gripper_goal)
        self.world.advance(POLICY_CONTROL_PERIOD)

    def release(self) -> None:
        self.server.release()

    def resume(self) -> None:
        self.server.resume()

    def note_step(self, name: str) -> None:
        self.world.sublabel = name

    def move_angular(self, goal: list[float]) -> list[float]:
        return [float(v) for v in self.server.move_joints_angular(list(goal))]

    def set_gripper_angular(self, position: float, timeout: float = 4.0) -> float:
        return float(self.server.gripper_angular(position, timeout))

    def toss(
        self,
        heading: float = 0.0,
        max_vel: float = TOSS_MAX_VEL,
        gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
    ) -> None:
        cycle = re.match(r"Cycle \d+", self.world.label)
        self.world.label = f"{cycle.group(0)}: toss" if cycle else "toss"
        self.world.toss_setting = (
            f"{max_vel:.0f} deg/s, release at {gripper_release_ms} ms"
        )
        self.world.sublabel = f"wind-up, onboard controller ({self.world.toss_setting})"
        try:
            self.server.toss(
                heading=heading, max_vel=max_vel, gripper_release_ms=gripper_release_ms
            )
        finally:
            self.world.sublabel = ""


class SimBaseInterface(BaseInterface):
    """The base, through the base controller's trajectory generator."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self._last_odom_heading = float(world.base[2])

    def get_base_state(self) -> SE2:
        self._last_odom_heading = float(self.world.base[2])
        x, y, theta = self.world.base
        return SE2(float(x), float(y), float(theta))

    def get_map_base_state(self) -> SE2:
        x, y, theta = self.world.base
        return SE2(float(x), float(y), float(theta))

    def execute_action(self, action: SE2) -> None:
        heading = unwrap_heading(action.theta(), self._last_odom_heading)
        self.world.command_base([action.x, action.y, heading])


class SimInterface(Interface):
    """Arm, base and (fake) cameras of a :class:`SimWorld`; map and odom coincide."""

    def __init__(self, world: SimWorld) -> None:
        self.arm_interface = SimArmInterface(world)
        self.base_interface = SimBaseInterface(world)
        self.camera_interface = FakeCameraInterface()

    def get_base_state(self) -> SE2:
        return self.base_interface.get_base_state()

    def get_map_base_state(self) -> SE2:
        return self.base_interface.get_map_base_state()

    def get_arm_state(self) -> list[float]:
        return self.arm_interface.get_arm_state()

    def get_gripper_state(self) -> float:
        return self.arm_interface.get_gripper_state()

    def get_wrist_image(self) -> Image:
        return self.camera_interface.get_wrist_image()

    def get_base_image(self) -> Image:
        return self.camera_interface.get_base_image()


class SimCubeTargetSource(TargetSource):
    """The ceiling localizer's view of the sim cube: where it rests."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def get_target(self) -> tuple[float, float, float]:
        x, y, z, _ = self.world.cube_at_rest()
        return (float(x), float(y), float(z))

    def get_yaw(self) -> float:
        return float(self.world.cube_at_rest()[3])

    def alternates(self) -> list[tuple[float, float, float]]:
        """One estimate only: there is one (perfect) camera."""
        return []


class SimOperator:
    """Answers toss_cycles' prompts in sim: puts a cube found in the bin back on the
    floor at `put_back`, and then asks the person (`ask`) or answers yes."""

    def __init__(
        self, world: SimWorld, put_back: Sequence[float], ask: Any = None
    ) -> None:
        self.world = world
        self.put_back = (float(put_back[0]), float(put_back[1]))
        self.ask = ask
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        _logger.info("Sim operator: %s", prompt.strip())
        if "in the bin" in prompt:
            self.world.place_cube(*self.put_back)
        if self.ask is not None:
            return str(self.ask(prompt))
        return "y"
