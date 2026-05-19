"""Generic multiprocessing-connection publisher.

A `Publisher` runs a worker thread that repeatedly calls `get_data()` and
stamps a timestamp, plus a per-connection thread that blocks until the client
sends a sentinel and replies with the most recent payload. Subclasses override
`get_data` (e.g. `MarkerDetectorServer`, `CameraServer`).

Adapted from `yixuanhuang98/tidybot_server/server/publisher.py`.
"""

import time
from multiprocessing.connection import Connection, Listener
from threading import Thread
from typing import Any, Optional

from prpl_tidybot.third_party.constants import CONN_AUTHKEY


class Publisher:
    """Threaded publisher that ships the latest `get_data()` result over a socket."""

    def __init__(self, hostname: str = "localhost", port: int = 6000) -> None:
        self.listener = Listener((hostname, port), authkey=CONN_AUTHKEY)
        self.data: Optional[Any] = None
        self.timestamp: Optional[float] = None

    def get_data(self) -> Any:
        """Produce the next payload. Subclasses override; base impl is a heartbeat."""
        time.sleep(0.033)
        return {"timestamp": time.time()}

    def worker(self) -> None:
        """Continuously refresh `self.data` from `get_data()`."""
        while True:
            data = self.get_data()
            self.data = data
            self.timestamp = time.time()

    def clean_up(self) -> None:
        """Subclass hook for tearing down resources on shutdown."""
        print("Cleaning up")

    def handle_conn(self, conn: Connection) -> None:
        """Per-client loop: wait for fresh data, then reply on each recv()."""
        last_timestamp = self.timestamp
        try:
            while True:
                while self.timestamp == last_timestamp:
                    time.sleep(0.0001)
                last_timestamp = self.timestamp

                conn.recv()
                conn.send(self.data)
        except (ConnectionResetError, EOFError, BrokenPipeError):
            pass

    def run(self) -> None:
        """Bind the listener and serve forever, one thread per connection."""
        try:
            Thread(target=self.worker, daemon=True).start()
            assert isinstance(self.listener.address, tuple)
            address, port = self.listener.address
            print(f"Waiting for connections ({address}:{port})")
            while True:
                conn = self.listener.accept()
                print(f"Connected! ({address}:{port})")
                Thread(target=self.handle_conn, args=(conn,), daemon=True).start()
        finally:
            self.clean_up()
