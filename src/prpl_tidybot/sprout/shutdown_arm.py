# Hardware and optional model imports stay lazy for offline use.
# pylint: disable=import-outside-toplevel
"""Supervised high-level Retract action after all motion clients stop.

No fault clearing or gripper opening. Require stable joint feedback through the
server, or ACTION_END for direct Kortex control. Failures prevent server teardown
and poweroff.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from pathlib import Path

import numpy as np


def retract(arm, messages, timeout: float = 45.0) -> None:
    """Request supervised retraction and wait for completion or stop."""
    mode = messages.ServoingModeInformation()
    mode.servoing_mode = messages.SINGLE_LEVEL_SERVOING
    arm.base.SetServoingMode(mode)
    query = messages.RequestedActionType()
    query.action_type = messages.REACH_JOINT_ANGLES
    actions = [
        a for a in arm.base.ReadAllActions(query).action_list if a.name == "Retract"
    ]
    if len(actions) != 1:
        raise RuntimeError(
            "Expected exactly one stored Kinova Retract action; not powering off"
        )
    done = threading.Event()
    result = []

    def notification(event):
        if event.action_event in (messages.ACTION_END, messages.ACTION_ABORT):
            result.append(event.action_event)
            done.set()

    handle = arm.base.OnNotificationActionTopic(
        notification, messages.NotificationOptions()
    )
    try:
        print("REAL SHUTDOWN: executing stored Retract action", flush=True)
        arm.base.ExecuteActionFromReference(actions[0].handle)
        if not done.wait(timeout):
            raise TimeoutError("Retract timed out; not powering off")
        if result[-1] != messages.ACTION_END:
            raise RuntimeError("Retract aborted; not powering off")
        print("Retract reported ACTION_END", flush=True)
    except BaseException:
        arm.base.Stop()
        raise
    finally:
        arm.base.Unsubscribe(handle)


def retract_via_server(
    arm, target, *, timeout: float = 45.0, period: float = 0.1
) -> None:
    """Hold the existing gripper state and verify stable retract joint feedback."""
    target = np.asarray(target, float)
    stable = 0
    previous = None
    deadline = time.monotonic() + timeout
    try:
        # Do not reset(), which opens the gripper and invokes an implicit retract.
        arm.resume()
        gripper = arm.get_gripper_position()
        if not np.isfinite(gripper):
            raise RuntimeError("Invalid gripper feedback; retract refused")
        while time.monotonic() < deadline:
            joints = np.asarray(arm.get_joint_angles(), float)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                raise RuntimeError("Invalid arm feedback; retract refused")
            error = (joints - target + np.pi) % (2 * np.pi) - np.pi
            movement = (
                math_delta(joints, previous) if previous is not None else float("inf")
            )
            settled = np.max(np.abs(error)) < 0.02 and movement < 0.002
            stable = stable + 1 if settled else 0
            print(
                f"RETRACT max joint error={np.max(np.abs(error)):.4f} rad, "
                f"stable={stable}/5",
                flush=True,
            )
            if stable >= 5:
                print("Retract reached and stable; servers may now stop", flush=True)
                return
            arm.execute_action_angular(target.tolist(), gripper)
            previous = joints.copy()
            time.sleep(period)
        raise TimeoutError(
            "Retract did not settle; not stopping servers or powering off"
        )
    finally:
        arm.close()


def math_delta(current: np.ndarray, previous: np.ndarray) -> float:
    """Maximum wrapped joint change between consecutive feedback samples."""
    return float(np.max(np.abs((current - previous + np.pi) % (2 * np.pi) - np.pi)))


def server_available() -> bool:
    """Only a refused local connection permits fallback to direct Kortex control."""
    from prpl_tidybot.third_party.constants import ARM_RPC_HOST, ARM_RPC_PORT

    try:
        with socket.create_connection((ARM_RPC_HOST, ARM_RPC_PORT), timeout=2):
            return True
    except ConnectionRefusedError:
        return False


def main(*, server: bool = False, auto: bool = False) -> None:
    """Retract using the running arm server, or direct Kortex when none is running."""
    if auto:
        server = server_available()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
            comm = (process / "comm").read_text()
        except OSError:
            continue
        conflicts = [b"sprout.remote_execute", b"test_arm_", b"test_gamepad_teleop"]
        if not server:
            conflicts.append(b"third_party.arm_server")
        if comm.startswith("python") and any(name in command for name in conflicts):
            raise RuntimeError("Competing motion process; retract refused")

    def interrupted(*_):
        raise KeyboardInterrupt("Shutdown connection interrupted")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    if server:
        from prpl_tidybot.third_party.arm_server import ArmManager
        from prpl_tidybot.third_party.constants import (
            ARM_RPC_HOST,
            ARM_RPC_PORT,
            POLICY_CONTROL_PERIOD,
            RETRACT_ARM_CONF,
            RPC_AUTHKEY,
        )

        manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        manager.connect()
        print("Retracting via running arm server", flush=True)
        arm = manager.Arm()  # type: ignore[attr-defined] # pylint: disable=no-member
        retract_via_server(arm, RETRACT_ARM_CONF, period=POLICY_CONTROL_PERIOD)
        return

    from kortex_api.autogen.messages import Base_pb2

    from prpl_tidybot.third_party.kinova import TorqueControlledArm

    print("No arm RPC server: using stored Kinova Retract action", flush=True)
    arm = TorqueControlledArm()  # type: ignore[no-untyped-call]
    try:
        retract(arm, Base_pb2)
    finally:
        arm.disconnect()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
