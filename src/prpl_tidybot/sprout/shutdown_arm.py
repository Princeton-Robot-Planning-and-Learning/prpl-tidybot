# Hardware and optional model imports stay lazy for offline use.
# pylint: disable=import-outside-toplevel
"""Supervised high-level Retract action after all motion clients stop.

No fault clearing, gripper opening or cyclic controller activation. Unlike the
legacy retract test, distinguish ACTION_ABORT and timeout from successful end.
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path


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


def main() -> None:
    """Run the shutdown arm command-line entry point."""
    # Imports here keep all tests/read-only import paths free of hardware access.

    from kortex_api.autogen.messages import Base_pb2

    from prpl_tidybot.third_party.kinova import TorqueControlledArm

    # This module is copied standalone; do not depend on deployed Sprout modules.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
            comm = (process / "comm").read_text()
        except OSError:
            continue
        if comm.startswith("python") and any(
            name in command
            for name in (
                b"third_party.arm_server",
                b"sprout.remote_execute",
                b"test_arm_",
                b"test_gamepad_teleop",
            )
        ):
            raise RuntimeError("Competing motion process; retract refused")

    def interrupted(*_):
        raise KeyboardInterrupt("Shutdown connection interrupted")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    # The vendor driver has no type annotations.
    arm = TorqueControlledArm()  # type: ignore[no-untyped-call]
    try:
        retract(arm, Base_pb2)
    finally:
        arm.disconnect()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
