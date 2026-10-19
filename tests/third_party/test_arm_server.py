"""Tests for third_party/arm_server.py: however many clients connect, at most one 1 kHz
torque loop commands the arm.

The real `ArmManager` server runs in-process against a fake Kortex session, so the tests
count the torque loops a sequence of clients leaves running.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from prpl_tidybot.third_party import arm_server


class _FakeKortexArm:
    """Stands in for `TorqueControlledArm` (one Kortex session); tracks its torque
    loop."""

    sessions: list["_FakeKortexArm"] = []

    def __init__(self) -> None:
        self.cyclic_running = False
        self.q = np.zeros(7)
        _FakeKortexArm.sessions.append(self)

    def set_joint_limits(self, speed_limits, acceleration_limits):
        """No-op."""
        del speed_limits, acceleration_limits

    def clear_faults(self):
        """No-op."""

    def init_cyclic(self, control_callback):
        """Start the torque loop; asserts as the real one does."""
        del control_callback
        assert not self.cyclic_running, "Cyclic thread is already running"
        self.cyclic_running = True

    def stop_cyclic(self):
        """Stop the torque loop."""
        self.cyclic_running = False

    def disconnect(self):
        """No-op."""


def _live_loops() -> int:
    return sum(session.cyclic_running for session in _FakeKortexArm.sessions)


_SERVERS: list = []  # keeps each test's in-process RPC server alive


@pytest.fixture(name="arm_rpc")
def _arm_rpc(monkeypatch):
    """An in-process arm RPC server, as on the NUC; each call of the returned factory is
    one client process's `manager.Arm()`."""
    monkeypatch.setattr(_FakeKortexArm, "sessions", [])
    monkeypatch.setattr(arm_server, "TorqueControlledArm", _FakeKortexArm)
    monkeypatch.setattr(arm_server, "IKSolver", lambda ee_offset: None)
    monkeypatch.setattr(arm_server, "time", SimpleNamespace(sleep=lambda _: None))
    manager = arm_server.ArmManager(address=("127.0.0.1", 0), authkey=b"test")
    server = manager.get_server()
    _SERVERS.append(server)
    # Only the accepter: serve_forever resets sys.stdout and sys.exit()s, which breaks
    # pytest's capture.
    server.stop_event = threading.Event()
    threading.Thread(target=server.accepter, daemon=True).start()

    def client():
        connection = arm_server.ArmManager(address=server.address, authkey=b"test")
        connection.connect()
        return connection.Arm()  # pylint: disable=no-member

    return client


def test_a_client_that_never_closed_leaves_no_second_torque_loop(arm_rpc):
    """The first client dies without close() (Ctrl-C, 2026-10-16 runs 21c/21d); the next
    client's reset() must not start a second loop beside the first's."""
    arm_rpc().reset(reset_arm=False)
    arm_rpc().reset(reset_arm=False)
    assert _live_loops() == 1


def test_a_client_after_one_that_closed_gets_one_torque_loop(arm_rpc):
    """The normal sequence still works: close(), then a new client's reset()."""
    first = arm_rpc()
    first.reset(reset_arm=False)
    first.close()
    assert _live_loops() == 0
    arm_rpc().reset(reset_arm=False)
    assert _live_loops() == 1
