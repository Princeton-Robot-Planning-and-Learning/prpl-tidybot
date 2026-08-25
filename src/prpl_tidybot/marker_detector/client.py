"""Subscriber client for the marker-detector publisher socket.

Wraps the request/reply handshake so callers (the real base interface, the real-mode
target perceivers) can ask for the latest published payload without each one re-
implementing the socket dance.
"""

import time
from multiprocessing.connection import Client, Connection
from typing import Any

from prpl_tidybot.marker_detector.constants import MARKER_DETECTOR_PORT
from prpl_tidybot.third_party.constants import CONN_AUTHKEY, SERVER_HOSTNAME


class MarkerDetectorClient:
    """Single-connection subscriber for `MarkerDetectorServer`.

    `get_latest()` blocks up to `poll_timeout_s` waiting for a fresh payload
    from the publisher; if none arrives in that window it returns the last
    cached payload (or an empty dict if nothing has ever arrived). Each
    successful receive immediately pipelines the next request, so back-to-back
    calls only ever wait on the publisher's own refresh cadence.

    A per-call `timeout_s` overrides the constructor default. `timeout_s=0.0`
    never blocks: it drains a payload that has already arrived and otherwise
    returns the cache immediately. Fixed-cadence control loops should use
    that form — a blocking read stalls the loop for up to one publisher
    period per call, while the non-blocking form returns a payload at most
    one publisher period staler. When the cache hasn't been refreshed for
    `stale_warning_s` seconds (detector down, marker occluded), a
    rate-limited warning is printed so silent-stale data doesn't go
    unnoticed.
    """

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = MARKER_DETECTOR_PORT,
        poll_timeout_s: float = 1.0,
        stale_warning_s: float = 2.0,
    ) -> None:
        self._conn: Connection = Client((host, port), authkey=CONN_AUTHKEY)
        self._poll_timeout_s = poll_timeout_s
        self._stale_warning_s = stale_warning_s
        self._conn.send(None)
        self._last: dict[str, Any] = {}
        self._last_fresh_time = time.monotonic()
        self._last_stale_warning_time = 0.0

    def get_latest(self, timeout_s: float | None = None) -> dict[str, Any]:
        """Return the most recent published payload (cached when none is fresh).

        Waits up to `timeout_s` (default: the constructor's `poll_timeout_s`)
        for a payload newer than the cached one; pass `0.0` for a
        non-blocking read.
        """
        effective_timeout = self._poll_timeout_s if timeout_s is None else timeout_s
        if self._conn.poll(timeout=effective_timeout):
            self._last = self._conn.recv()
            self._conn.send(None)
            self._last_fresh_time = time.monotonic()
        else:
            self._warn_if_stale()
        return self._last

    def _warn_if_stale(self) -> None:
        now = time.monotonic()
        stale_for = now - self._last_fresh_time
        if stale_for <= self._stale_warning_s:
            return
        if now - self._last_stale_warning_time <= self._stale_warning_s:
            return
        self._last_stale_warning_time = now
        print(
            f"warning: marker-detector payload is {stale_for:.1f}s stale; "
            "returning cached payload"
        )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
