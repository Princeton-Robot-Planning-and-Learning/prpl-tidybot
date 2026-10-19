"""Tests for third_party/base_server.py: however many clients connect, at most one
`Vehicle` control loop drives the base."""

import threading

import pytest

from prpl_tidybot.third_party import base_server


class _FakeVehicle:
    """Stands in for `Vehicle`; tracks whether its control loop runs."""

    vehicles: list["_FakeVehicle"] = []

    def __init__(self, max_vel, max_accel) -> None:
        del max_vel, max_accel
        self.control_loop_running = False
        _FakeVehicle.vehicles.append(self)

    def start_control(self):
        """Start the control loop."""
        self.control_loop_running = True

    def stop_control(self):
        """Stop the control loop."""
        self.control_loop_running = False


def _live_loops() -> int:
    return sum(vehicle.control_loop_running for vehicle in _FakeVehicle.vehicles)


_SERVERS: list = []  # keeps each test's in-process RPC server alive


@pytest.fixture(name="base_rpc")
def _base_rpc(monkeypatch):
    """An in-process base RPC server; each call is one client's `manager.Base()`."""
    monkeypatch.setattr(_FakeVehicle, "vehicles", [])
    monkeypatch.setattr(base_server, "Vehicle", _FakeVehicle)
    manager = base_server.BaseManager(address=("127.0.0.1", 0), authkey=b"test")
    server = manager.get_server()
    _SERVERS.append(server)
    # Only the accepter: serve_forever resets sys.stdout and sys.exit()s, which breaks
    # pytest's capture.
    server.stop_event = threading.Event()
    threading.Thread(target=server.accepter, daemon=True).start()

    def client():
        connection = base_server.BaseManager(address=server.address, authkey=b"test")
        connection.connect()
        return connection.Base()  # pylint: disable=no-member

    return client


def test_a_client_that_never_closed_leaves_no_second_control_loop(base_rpc):
    """The first client dies without close(); the next client's reset() must not start a
    second loop beside the first's."""
    base_rpc().reset()
    base_rpc().reset()
    assert _live_loops() == 1


def test_a_client_after_one_that_closed_gets_one_control_loop(base_rpc):
    """The normal sequence still works: close(), then a new client's reset()."""
    first = base_rpc()
    first.reset()
    first.close()
    assert _live_loops() == 0
    base_rpc().reset()
    assert _live_loops() == 1
