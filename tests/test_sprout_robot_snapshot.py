# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Telemetry reads must never initialize controllers or accept missing poses."""

import math
import sys
import time
from types import SimpleNamespace

import pytest

from prpl_tidybot.sprout import read_robot_state as snapshot


def state():
    """Provide the state test fixture or callback."""
    return {
        "schema": "sprout_robot_snapshot_v1",
        "arm": {"joints_rad": [0.0] * 7, "received_at_unix": time.time()},
        "base": {"frame": "map", "xyyaw": [1, 2, 0.3], "received_at_unix": time.time()},
    }


@pytest.mark.parametrize("fault", ["old", "future", "skew", "nan", "odom", "short"])
def test_reject_invalid_snapshot(fault):
    """Verify reject invalid snapshot."""
    value = state()
    if fault == "old":
        value["arm"]["received_at_unix"] -= 121
    elif fault == "future":
        value["arm"]["received_at_unix"] += 20
    elif fault == "skew":
        value["arm"]["received_at_unix"] -= 6
    elif fault == "nan":
        value["base"]["xyyaw"][0] = math.nan
    elif fault == "odom":
        value["base"]["frame"] = "odom"
    else:
        value["arm"]["joints_rad"] = [0] * 6
    with pytest.raises(ValueError):
        snapshot.validate_snapshot(value)


def test_valid_snapshot():
    """Verify valid snapshot."""
    snapshot.validate_snapshot(state())


@pytest.mark.parametrize(
    "payload,ready,error",
    [
        ({}, True, ValueError),
        ({}, False, TimeoutError),
        ({"poses": {0: [1, 2, 0.3]}}, True, None),
    ],
)
def test_marker_reply(monkeypatch, payload, ready, error):
    """Verify marker reply."""

    class Connection:
        """Test double for Connection."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def send(self, request):
            """Provide the send test fixture or callback."""
            assert request is None

        def poll(self, timeout):
            """Provide the poll test fixture or callback."""
            return ready

        def recv(self):
            """Provide the recv test fixture or callback."""
            return payload

    monkeypatch.setattr(snapshot, "Client", lambda *args, **kwargs: Connection())
    if error:
        with pytest.raises(error):
            snapshot.read_base("fake", 6002, 0)
    else:
        assert snapshot.read_base("fake", 6002, 0)["xyyaw"] == [1, 2, 0.3]


def test_arm_only_uses_telemetry_and_closes_session(monkeypatch):
    """Verify arm only uses telemetry and closes session."""
    calls = []

    class Transport:
        """Test double for Transport."""

        def connect(self, *args):
            """Provide the connect test fixture or callback."""
            calls.append("connect")

        def disconnect(self):
            """Provide the disconnect test fixture or callback."""
            calls.append("disconnect")

    class Manager:
        """Test double for Manager."""

        def __init__(self, router):
            pass

        def CreateSession(self, info):
            """Provide the CreateSession test fixture or callback."""
            calls.append("session")

        def CloseSession(self, options):
            """Provide the CloseSession test fixture or callback."""
            calls.append("close")

    class Router:
        """Test double for Router."""

        basicErrorCallback = None

        def __init__(self, *args):
            pass

    class Feedback:
        """Test double for Feedback."""

        def __init__(self, router):
            pass

        def RefreshFeedback(self, *, options):
            """Provide the RefreshFeedback test fixture or callback."""
            calls.append("feedback")
            return SimpleNamespace(
                actuators=[
                    SimpleNamespace(position=v) for v in [0, 90, 180, 270, 360, 10, 20]
                ]
            )

    modules = {
        "autogen.client_stubs.BaseCyclicClientRpc": {"BaseCyclicClient": Feedback},
        "autogen.messages": {
            "Session_pb2": SimpleNamespace(CreateSessionInfo=SimpleNamespace)
        },
        "RouterClient": {
            "RouterClient": Router,
            "RouterClientSendOptions": SimpleNamespace,
        },
        "SessionManager": {"SessionManager": Manager},
        "UDPTransport": {"UDPTransport": Transport},
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, "kortex_api." + name, SimpleNamespace(**attrs))
    result = snapshot.read_arm("fake")
    assert result["joints_rad"][:5] == pytest.approx(
        [0, math.pi / 2, -math.pi, -math.pi / 2, 0]
    )
    assert calls == ["connect", "session", "feedback", "close", "disconnect"]
