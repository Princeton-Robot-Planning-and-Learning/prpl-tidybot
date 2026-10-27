"""NUC execution gate over SSH stdin, with laptop heartbeat and latched stop.

Uses the concrete live_runtime adapter and its reviewed commissioning inputs.
No private model assets are needed by the transport. EOF/stop/heartbeat loss and
SIGINT/SIGTERM/SIGHUP request a software stop, not a certified emergency stop.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from prpl_tidybot.sprout.review import approve_and_execute, digest


class Session:
    """Only an exact review approval allows motion; all stop conditions latch."""

    def __init__(self, review_hash: str, timeout_s: float = 3.0) -> None:
        self.review_hash, self.timeout_s = review_hash, timeout_s
        self.approved = threading.Event()
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.last_heartbeat = time.monotonic()
        self._runtime: Any = None
        self._lock = threading.Lock()

    def bind(self, runtime: Any) -> None:
        with self._lock:
            self._runtime = runtime
            if self.stopped.is_set():
                runtime.stop()

    def stop(self) -> None:
        self.stopped.set()
        with self._lock:
            if self._runtime is not None:
                self._runtime.stop()

    def receive(self, message: dict[str, Any]) -> None:
        if self.stopped.is_set():
            return
        kind = message.get("kind")
        if kind == "approve" and not self.approved.is_set():
            if message.get("review_sha256") != self.review_hash:
                self.stop()
                return
            self.approved.set()
        elif kind == "confirm" and self.approved.is_set():
            try:
                if self._runtime is None:
                    raise RuntimeError("Runtime not ready")
                self._runtime.confirm(message.get("phase"), message.get("token"))
            except (RuntimeError, ValueError, AttributeError):
                self.stop()
                return
        elif kind != "heartbeat":
            self.stop()
            return
        self.last_heartbeat = time.monotonic()

    def read(self, stream: TextIO) -> None:
        try:
            while not self.finished.is_set():
                line = stream.readline(4096)
                if not line or len(line) >= 4096:
                    break
                message = json.loads(line)
                if not isinstance(message, dict):
                    break
                self.receive(message)
        except (ValueError, OSError):
            pass
        finally:
            if not self.finished.is_set():
                self.stop()

    def watch(self) -> None:
        while not self.finished.wait(0.1):
            if (
                self.stopped.is_set()
                or time.monotonic() - self.last_heartbeat > self.timeout_s
            ):
                self.stop()
                return

    def prompt(self, message: str) -> str:
        while not self.approved.wait(0.05):
            if self.stopped.is_set():
                raise RuntimeError("Laptop approval connection stopped")
        if self.stopped.is_set():
            raise RuntimeError("Laptop stop latched")
        return "yes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        default="prpl_tidybot.sprout.live_runtime:create_runtime",
        help="Read-only module:factory(review_path) on the NUC",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    session = Session(digest(args.review))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        # Only set the latch in a signal handler; the watchdog performs stop.
        signal.signal(sig, lambda *_: session.stopped.set())
    threading.Thread(target=session.read, args=(sys.stdin,), daemon=True).start()
    threading.Thread(target=session.watch, daemon=True).start()
    try:
        # Existing review validation runs BEFORE importing the runtime factory.
        runtime: Any = None

        def observe() -> dict[str, Any]:
            nonlocal runtime
            if session.stopped.is_set():
                raise RuntimeError("Laptop stop latched before hardware observation")
            module, factory = args.runtime.split(":", 1)
            runtime = getattr(importlib.import_module(module), factory)(args.review)
            session.bind(runtime)
            if session.stopped.is_set():
                raise RuntimeError("Laptop stop latched")
            return runtime.observe()

        def execute(review: dict[str, Any]) -> Any:
            if session.stopped.is_set():
                raise RuntimeError("Laptop stop latched before execution")
            return runtime.execute(review)

        result = approve_and_execute(
            args.review, observe, execute, prompt=session.prompt, interactive=True
        )
        print(json.dumps(result), flush=True)
    finally:
        session.finished.set()
        session.stop()
        if runtime is not None and hasattr(runtime, "close"):
            runtime.close()


if __name__ == "__main__":
    main()
