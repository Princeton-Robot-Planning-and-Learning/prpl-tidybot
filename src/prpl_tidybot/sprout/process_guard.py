"""Reject known competing hardware owners without terminating them."""

import os
from pathlib import Path


def require_exclusive_motion():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            comm = (process / "comm").read_text()
        except (OSError, ProcessLookupError):
            continue
        if not comm.startswith("python"):
            continue
        if any(
            name in command
            for name in (
                "third_party.arm_server",
                "third_party.base_server",
                "sprout.remote_execute",
                "test_gamepad_teleop",
                "test_arm_",
            )
        ):
            raise RuntimeError(
                "Competing robot controller process "
                + process.name
                + "; stop it in its own terminal first"
            )
