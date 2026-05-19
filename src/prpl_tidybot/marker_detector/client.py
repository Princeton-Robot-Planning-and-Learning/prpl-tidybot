"""Subscriber client for the marker-detector publisher socket.

Wraps the request/reply handshake so callers (the real base interface, the
real-mode target perceivers) can ask for the latest published payload without
each one re-implementing the socket dance.
"""

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
    """

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = MARKER_DETECTOR_PORT,
        poll_timeout_s: float = 1.0,
    ) -> None:
        self._conn: Connection = Client((host, port), authkey=CONN_AUTHKEY)
        self._poll_timeout_s = poll_timeout_s
        self._conn.send(None)
        self._last: dict[str, Any] = {}

    def get_latest(self) -> dict[str, Any]:
        """Return the most recent published payload (cached on timeout)."""
        if self._conn.poll(timeout=self._poll_timeout_s):
            self._last = self._conn.recv()
            self._conn.send(None)
        else:
            print(
                "warning: marker detector did not respond within "
                f"{self._poll_timeout_s:.1f}s; returning cached payload"
            )
        return self._last

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
