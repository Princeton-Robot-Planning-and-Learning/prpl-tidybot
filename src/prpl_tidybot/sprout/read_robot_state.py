"""Read-only NUC telemetry snapshot; also runnable as a standalone copied file.

Uses Kortex RefreshFeedback and the existing ceiling-marker publisher protocol.
Never constructs Arm/RealArmInterface/Base/RealBaseInterface: those initialize
controllers. A received marker reply is not proof of camera acquisition freshness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any


def vector(value: Any, length: int) -> list[float]:
    """Require a finite vector of the expected dimension."""
    result = [float(v) for v in value]
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"Expected {length} finite values")
    return result


def read_arm(host: str) -> dict[str, Any]:
    """Open a telemetry session only; do not set servo modes or send commands."""
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Session_pb2
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.UDPTransport import UDPTransport

    transport = UDPTransport()
    router = RouterClient(transport, RouterClient.basicErrorCallback)
    manager = SessionManager(router)
    opened = False
    try:
        transport.connect(host, 10001)
        session = Session_pb2.CreateSessionInfo()
        session.username = os.environ.get("KINOVA_USERNAME", "admin")
        session.password = os.environ.get("KINOVA_PASSWORD", "admin")
        session.session_inactivity_timeout = 10000
        session.connection_inactivity_timeout = 2000
        manager.CreateSession(session)
        opened = True
        options = RouterClientSendOptions()
        options.timeout_ms = 2000
        feedback = BaseCyclicClient(router).RefreshFeedback(options=options)
        degrees = vector([a.position for a in feedback.actuators], 7)
        joints = [
            (math.radians(v) + math.pi) % (2 * math.pi) - math.pi for v in degrees
        ]
        return {
            "joints_rad": joints,
            "joints_deg_raw": degrees,
            "received_at_unix": time.time(),
            "source": "Kortex BaseCyclic.RefreshFeedback",
        }
    finally:
        try:
            if opened:
                options = RouterClientSendOptions()
                options.timeout_ms = 1000
                manager.CloseSession(options)
        finally:
            transport.disconnect()


def read_base(host: str, port: int, robot_id: int) -> dict[str, Any]:
    """Request one publisher reply; never substitute cached/zero poses."""
    with Client((host, port), authkey=b"secret password") as connection:
        connection.send(None)
        if not connection.poll(3.0):
            raise TimeoutError("No ceiling-marker reply within 3 seconds")
        payload = connection.recv()
    if robot_id not in payload.get("poses", {}):
        raise ValueError(f"Robot {robot_id} not visible to ceiling-marker detector")
    return {
        "xyyaw": vector(payload["poses"][robot_id], 3),
        "frame": "map",
        "robot_id": robot_id,
        "received_at_unix": time.time(),
        "captured_at_unix": payload.get("pose_captured_at_unix", {}).get(robot_id),
        "acquisition_timestamp_available": payload.get("pose_captured_at_unix", {}).get(
            robot_id
        )
        is not None,
        "source": "PRPL ceiling-marker publisher",
    }


def validate_snapshot(state: dict[str, Any], max_age_s: float = 120) -> None:
    """Validate snapshot for visualization, not hardware authorization."""
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be finite and positive")
    if state.get("schema") != "sprout_robot_snapshot_v1":
        raise ValueError("Unknown robot snapshot schema")
    if state["base"].get("frame") != "map":
        raise ValueError("Base snapshot must use the calibrated map frame")
    vector(state["arm"]["joints_rad"], 7)
    vector(state["base"]["xyyaw"], 3)
    stamps = [float(state[key]["received_at_unix"]) for key in ("arm", "base")]
    now = time.time()
    if any(not math.isfinite(t) or not 0 <= now - t <= max_age_s for t in stamps):
        raise ValueError(
            "Snapshot is old or clock is ahead; capture again/check clocks"
        )
    if abs(stamps[0] - stamps[1]) > 5:
        raise ValueError("Arm/base reads are more than 5 seconds apart; capture again")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-host", default="192.168.1.10")
    parser.add_argument("--marker-host", default="192.168.0.131")
    parser.add_argument("--marker-port", type=int, default=6002)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        "Reading arm feedback and ceiling-marker map pose (no motion commands)...",
        flush=True,
    )
    arm = read_arm(args.arm_host)
    base = read_base(args.marker_host, args.marker_port, args.robot_id)
    state = {"schema": "sprout_robot_snapshot_v1", "arm": arm, "base": base}
    validate_snapshot(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    print(json.dumps(state, indent=2), flush=True)
    print(
        f"Saved {args.output}. Read-only feedback; base acquisition time is included when published. Not an execution clearance."
    )


if __name__ == "__main__":
    main()
