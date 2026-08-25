"""Tests for marker_detector/client.py."""

from typing import Any

import pytest

from prpl_tidybot.marker_detector import client as client_module
from prpl_tidybot.marker_detector.client import MarkerDetectorClient


class _FakeConnection:
    """In-memory stand-in for `multiprocessing.connection.Connection`.

    Payloads queued via `queue_payload` become available to `poll`/`recv` one at a time.
    Every `poll` call records the timeout it was invoked with so tests can assert on
    blocking vs non-blocking reads.
    """

    def __init__(self) -> None:
        self.pending: list[dict[str, Any]] = []
        self.sent: list[Any] = []
        self.poll_timeouts: list[float] = []
        self.closed = False

    def queue_payload(self, payload: dict[str, Any]) -> None:
        """Make `payload` available to the next poll/recv pair."""
        self.pending.append(payload)

    def poll(self, timeout: float = 0.0) -> bool:
        """Report whether a payload is pending; record the timeout used."""
        self.poll_timeouts.append(timeout)
        return bool(self.pending)

    def recv(self) -> dict[str, Any]:
        """Pop and return the oldest pending payload."""
        return self.pending.pop(0)

    def send(self, obj: Any) -> None:
        """Record an outgoing request."""
        self.sent.append(obj)

    def close(self) -> None:
        """Mark the connection closed."""
        self.closed = True


@pytest.fixture(name="fake_conn")
def _fake_conn(monkeypatch: pytest.MonkeyPatch) -> _FakeConnection:
    conn = _FakeConnection()
    monkeypatch.setattr(client_module, "Client", lambda *args, **kwargs: conn)
    return conn


def test_constructor_pipelines_first_request(fake_conn: _FakeConnection):
    """Connecting sends the initial request so the publisher can respond."""
    MarkerDetectorClient()
    assert fake_conn.sent == [None]


def test_get_latest_receives_payload_and_pipelines_next_request(
    fake_conn: _FakeConnection,
):
    """A delivered payload is returned, cached, and followed by a new request."""
    client = MarkerDetectorClient()
    payload = {"poses": {0: (1.0, 2.0, 0.5)}}
    fake_conn.queue_payload(payload)
    assert client.get_latest() == payload
    # Initial request plus the re-request after the successful receive.
    assert fake_conn.sent == [None, None]
    # Nothing new pending: the cached payload is returned.
    assert client.get_latest() == payload


def test_get_latest_uses_constructor_timeout_by_default(fake_conn: _FakeConnection):
    """Without a per-call override, poll waits up to `poll_timeout_s`."""
    client = MarkerDetectorClient(poll_timeout_s=0.7)
    client.get_latest()
    assert fake_conn.poll_timeouts == [0.7]


def test_get_latest_timeout_override_is_passed_to_poll(fake_conn: _FakeConnection):
    """A per-call `timeout_s` overrides the constructor default."""
    client = MarkerDetectorClient(poll_timeout_s=0.7)
    client.get_latest(timeout_s=0.0)
    assert fake_conn.poll_timeouts == [0.0]


def test_nonblocking_read_returns_empty_dict_before_first_payload(
    fake_conn: _FakeConnection,
):
    """With nothing ever delivered, the cache is an empty dict."""
    del fake_conn  # Patched in by the fixture; only its emptiness matters.
    client = MarkerDetectorClient()
    assert client.get_latest(timeout_s=0.0) == {}


def test_stale_warning_is_rate_limited(
    fake_conn: _FakeConnection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """One warning per `stale_warning_s` window, none while the cache is fresh."""
    del fake_conn  # Patched in by the fixture; only its emptiness matters.
    clock = {"now": 100.0}
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock["now"])
    client = MarkerDetectorClient(stale_warning_s=2.0)

    # Within the freshness window: no warning.
    clock["now"] = 101.0
    client.get_latest(timeout_s=0.0)
    assert "stale" not in capsys.readouterr().out

    # Past the window: exactly one warning...
    clock["now"] = 103.0
    client.get_latest(timeout_s=0.0)
    assert capsys.readouterr().out.count("stale") == 1

    # ...and none again until another `stale_warning_s` elapses.
    clock["now"] = 104.0
    client.get_latest(timeout_s=0.0)
    assert "stale" not in capsys.readouterr().out
    clock["now"] = 106.0
    client.get_latest(timeout_s=0.0)
    assert capsys.readouterr().out.count("stale") == 1


def test_fresh_payload_resets_staleness(
    fake_conn: _FakeConnection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Receiving a payload restarts the staleness clock."""
    clock = {"now": 100.0}
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock["now"])
    client = MarkerDetectorClient(stale_warning_s=2.0)

    clock["now"] = 110.0
    fake_conn.queue_payload({"poses": {}})
    client.get_latest(timeout_s=0.0)

    # Only 1s stale relative to the fresh receive: no warning.
    clock["now"] = 111.0
    client.get_latest(timeout_s=0.0)
    assert "stale" not in capsys.readouterr().out


def test_close_closes_connection(fake_conn: _FakeConnection):
    """Close() tears down the underlying connection."""
    client = MarkerDetectorClient()
    client.close()
    assert fake_conn.closed
