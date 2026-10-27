"""Offline RPC rehearsal. This module cannot connect to robot hardware."""

# Synthetic callbacks retain the hardware-facing interface signatures.
# pylint: disable=unused-argument

from __future__ import annotations

import argparse
import json
import math
import threading
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

from prpl_tidybot.sprout.approach import TopDownGeometry
from prpl_tidybot.sprout.sequence import ForearmSkill, GraspPose, Plan, Recipe, Stopped

# Synthetic test geometry ONLY, not a commissioned Sprout contact height.
SYNTHETIC_GEOMETRY = TopDownGeometry(-0.395, 0.0295, 0.012, 0.010)


class FakeBackend:
    """Synthetic geometry and confirmations, explicitly not a physical simulation."""

    def __init__(self, delay: float = 0.25) -> None:
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("Delay must be finite and nonnegative")
        self.delay = delay
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.closed = False
        self.events: list[str] = []
        self.grasp = GraspPose(0.5, 0.0, 0.10, 0.0)

    def _operation(self, name: str) -> None:
        if self.cancelled.wait(self.delay):
            raise Stopped("Offline backend cancelled")
        with self.lock:
            if self.cancelled.is_set():
                raise Stopped("Offline backend cancelled")
            self.events.append(name)

    def localize(self) -> GraspPose:
        """Return the synthetic target without accessing hardware."""
        self._operation("synthetic_localization")
        return self.grasp

    def preflight(self, plan: Plan) -> None:
        """Record the synthetic preflight event."""
        self._operation("synthetic_preflight")

    def move(self, phase: str, pose: GraspPose) -> None:
        """Record a synthetic motion phase without commanding hardware."""
        self._operation(phase)

    def align(self, pregrasp: GraspPose) -> GraspPose:
        """Record synthetic alignment and return the configured target."""
        self._operation("synthetic_alignment")
        return self.grasp

    def close_gripper(self) -> None:
        """Record a synthetic close and mark the simulated grasp."""
        self._operation("close")
        with self.lock:
            if self.cancelled.is_set():
                raise Stopped("Offline backend cancelled")
            self.closed = True

    def grasp_confirmed(self) -> bool:
        """Return the synthetic grasp evidence."""
        self._operation("synthetic_grasp_confirmation")
        return True

    def support_confirmed(self) -> bool:
        """Return the synthetic support evidence."""
        self._operation("synthetic_support_confirmation")
        return True

    def open_gripper(self) -> None:
        """Record a synthetic open and clear the simulated grasp."""
        self._operation("open")
        with self.lock:
            if self.cancelled.is_set():
                raise Stopped("Offline backend cancelled")
            self.closed = False

    def stop(self) -> None:
        """Latch the stop request and prevent subsequent motion commands."""
        with self.lock:
            self.cancelled.set()
            self.events.append("stop")

    def status(self) -> dict[str, Any]:
        """Return synthetic backend state for the rehearsal client."""
        with self.lock:
            return {
                "mode": "offline_fake",
                "stopped": self.cancelled.is_set(),
                "gripper_closed": self.closed,
                "events": list(self.events),
            }


class Service:
    """Expose only the offline cycle, latched stop, and status."""

    def __init__(self, delay: float) -> None:
        self.backend = FakeBackend(delay)
        self.skill = ForearmSkill(self.backend, Recipe(0.10, 0.10), SYNTHETIC_GEOMETRY)

    def cycle(self) -> dict[str, Any]:
        """Run one synthetic forearm cycle."""
        return {"mode": "offline_fake", **self.skill.cycle()}

    def stop(self) -> dict[str, Any]:
        """Latch the stop request and prevent subsequent motion commands."""
        self.skill.stop()
        return self.status()

    def status(self) -> dict[str, Any]:
        """Return the synthetic service state."""
        return self.backend.status()


def main() -> None:
    """Run the offline server command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=50012)
    parser.add_argument("--authkey-file", type=Path, required=True)
    parser.add_argument("--phase-delay", type=float, default=0.25)
    args = parser.parse_args()
    authkey = args.authkey_file.read_bytes()
    if len(authkey) < 32:
        parser.error("Use a private authkey file containing at least 32 random bytes")
    service = Service(args.phase_delay)

    class Manager(BaseManager):
        """RPC manager exposing only the offline rehearsal service."""

    Manager.register(
        "Sprout", callable=lambda: service, exposed=("cycle", "stop", "status")
    )
    server = Manager(address=("127.0.0.1", args.port), authkey=authkey).get_server()
    address = server.address
    if not isinstance(address, tuple):
        raise TypeError("Offline server requires a TCP loopback address")
    print(json.dumps({"mode": "offline_fake", "port": address[1]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()
