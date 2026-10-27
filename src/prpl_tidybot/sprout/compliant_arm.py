"""NUC-local owner of the existing JointCompliantController.

No automatic retract, home, gripper open, or fault clearing. Do not run beside
the arm server or another arm client. This is an integration API, not a turnkey
hardware launcher: scene calibration and physical stop testing are still needed.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from prpl_tidybot.sprout.compliant_backend import JointSample
from prpl_tidybot.sprout.sequence import Stopped


@dataclass(frozen=True)
class HardwareLimits:
    """Explicit lab values; gripper percentages are NOT newton limits.

    Joint bounds use wrapped [-pi, pi) radians, matching read_robot_state. Joint torque
    bounds include gravity, and cannot be interpreted as Cartesian force bounds.
    """

    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    acceleration_rad_s2: tuple[float, ...]
    measured_torque_limit_nm: tuple[float, ...]
    gripper_force_percent: float
    gripper_speed_percent: float
    command_timeout_s: float
    feedback_timeout_s: float
    startup_timeout_s: float

    def __post_init__(self) -> None:
        for values in (
            self.joint_min_rad,
            self.joint_max_rad,
            self.velocity_rad_s,
            self.acceleration_rad_s2,
            self.measured_torque_limit_nm,
        ):
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                raise ValueError("Expected seven finite values per joint limit")
        if any(a >= b for a, b in zip(self.joint_min_rad, self.joint_max_rad)):
            raise ValueError("Invalid joint range")
        if any(
            v <= 0
            for v in (
                *self.velocity_rad_s,
                *self.acceleration_rad_s2,
                *self.measured_torque_limit_nm,
            )
        ):
            raise ValueError("Joint dynamic limits must be positive")
        for v in (self.gripper_force_percent, self.gripper_speed_percent):
            if not math.isfinite(v) or not 0 < v <= 100:
                raise ValueError("Gripper limits must be in (0, 100]")
        for v in (
            self.command_timeout_s,
            self.feedback_timeout_s,
            self.startup_timeout_s,
        ):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Timeouts must be finite and positive")


class LocalCompliantArm:
    """Single local session, suitable for CompliantBackend.

    Command loss latches a measured-pose hold in the cyclic callback, retaining
    the measured gripper position. Explicit stop exits cyclic control to the
    Kinova position controller. Neither is a certified emergency stop; stale
    feedback/communication loss requires the physical stop procedure.
    """

    controller_name = "joint_compliant"

    def __init__(
        self, arm: Any, limits: HardwareLimits, controller_factory: Callable[..., Any]
    ) -> None:
        self.arm, self.limits = arm, limits
        self._factory = controller_factory
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._ready = threading.Event()
        self._sample: JointSample | None = None
        self._controller = self._new_controller()
        self._holding = False
        self._last_command = time.monotonic()
        self.stop_reason: str | None = None

    @classmethod
    def connect(
        cls, limits: HardwareLimits, *, commissioning_verified: bool = False
    ) -> LocalCompliantArm:
        """Explicit hardware activation; call only on the NUC after commissioning.

        This starts compliant holding at the current pose. Keep physical stop
        available. No RPC listener or remote execution is enabled by this call.
        """
        if not commissioning_verified:
            raise RuntimeError(
                "Calibration, scene guards, gripper limits and physical stop must be verified first"
            )
        from prpl_tidybot.third_party.arm_controller import JointCompliantController
        from prpl_tidybot.third_party.kinova import TorqueControlledArm

        arm: Any = TorqueControlledArm()  # type: ignore[no-untyped-call]
        session = cls(arm, limits, JointCompliantController)
        try:
            arm.set_joint_limits(
                speed_limits=tuple(math.degrees(v) for v in limits.velocity_rad_s),
                acceleration_limits=tuple(
                    math.degrees(v) for v in limits.acceleration_rad_s2
                ),
            )
            arm.init_cyclic(
                session._callback,
                gripper_force_limit=limits.gripper_force_percent,
                gripper_speed_limit=limits.gripper_speed_percent,
            )
            if not session._ready.wait(limits.startup_timeout_s):
                raise TimeoutError("No compliant feedback during startup")
            session.read()
        except BaseException:
            try:
                session.stop()
            finally:
                arm.disconnect()
            raise
        return session

    def _new_controller(self) -> Any:
        return self._factory(
            self._queue,
            max_velocity=self.limits.velocity_rad_s,
            max_acceleration=self.limits.acceleration_rad_s2,
        )

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _callback(self, arm: Any) -> Any:
        with self._lock:
            now = time.monotonic()
            sample = JointSample(
                tuple((np.asarray(arm.q) + np.pi) % (2 * np.pi) - np.pi),
                tuple(arm.dq),
                float(arm.gripper_pos),
                float(arm.feedback_time),
            )
            p = self.limits
            if not np.all(
                np.isfinite(
                    [
                        *sample.q,
                        *sample.dq,
                        *arm.tau,
                        sample.gripper,
                        sample.captured_at,
                    ]
                )
            ):
                self.stop_reason = "non-finite feedback"
                self._stopped.set()
                # Never initialize a hold controller from invalid measurements.
                # The caller observes the latch and exits cyclic control.
                raise RuntimeError(self.stop_reason)
            if not 0 <= now - sample.captured_at <= p.feedback_timeout_s:
                self.stop_reason = "stale feedback"
            elif now - self._last_command > p.command_timeout_s:
                self.stop_reason = "command watchdog"
            elif (
                not 0 <= sample.gripper <= 1
                or np.any(np.array(sample.q) < p.joint_min_rad)
                or np.any(np.array(sample.q) > p.joint_max_rad)
                or np.any(np.abs(sample.dq) > p.velocity_rad_s)
                or np.any(np.abs(arm.tau) > p.measured_torque_limit_nm)
            ):
                self.stop_reason = "joint/speed/torque feedback guard"
            if self.stop_reason is not None:
                self._stopped.set()
            if self._stopped.is_set() and not self._holding:
                self._drain()
                self._controller = self._new_controller()
                self._holding = True
            self._sample = sample
            result = self._controller.control_callback(arm)
            self._ready.set()
            return result

    def read(self) -> JointSample:
        with self._lock:
            if self._stopped.is_set():
                raise Stopped(self.stop_reason or "Local compliant session stopped")
            if self._sample is None:
                raise RuntimeError("No cyclic feedback yet")
            return self._sample

    def command(self, q: tuple[float, ...], gripper: float) -> None:
        p = self.limits
        if (
            len(q) != 7
            or not np.all(np.isfinite(q))
            or np.any(np.array(q) < p.joint_min_rad)
            or np.any(np.array(q) > p.joint_max_rad)
            or not math.isfinite(gripper)
            or not 0 <= gripper <= 1
        ):
            raise ValueError("Command outside commissioned joint/gripper bounds")
        with self._lock:
            if self._stopped.is_set():
                raise Stopped(self.stop_reason or "Local compliant session stopped")
            self._drain()
            self._queue.put_nowait((np.array(q), gripper))
            self._last_command = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._stopped.set()
            self.stop_reason = self.stop_reason or "explicit stop"
            self._drain()
        # Do not hold the callback lock while joining the cyclic thread.
        self.arm.stop_cyclic()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.arm.disconnect()
