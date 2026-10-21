"""A continuous, read-only log of the robot's motion while the skill server runs, so a
real session can be replayed in sim: ``<log dir>/motion.jsonl``.

A background thread samples at ``hz`` and appends one compact JSON line per sample:

* ``t``: Unix time (s);
* ``i``: the indices of the skill calls in progress (``skill_calls.jsonl``'s ``i``);
* ``live``: whether the arm's readings below are live measurements. On the real arm
  they are only while the arm server's cyclic (torque) loop runs: its joint angles
  and gripper position are updated by that loop alone, so while it is released (an
  angular move, a gripper move under the onboard controller, the toss) they hold
  their last value. Then ``live`` is false and ``q`` and ``g`` are null;
* ``q``: the arm's joint angles (rad) as the arm interface reports them;
* ``g``, ``gc``: the gripper's reading and the skills' last gripper command (the
  command is always known);
* ``odom``, ``map``: the base's odometry and map-frame poses (x, y, theta);
* ``step``: the step label, only on the samples where it changed.

A sample identical to the last one written (to 0.1 mrad, 0.1 mm, the same calls and
step) is not written, except once per ``heartbeat_s``: an idle robot costs almost
nothing, and a replay holds the last value.

It never commands anything and takes no lock a skill holds: each reader is a plain
attribute read or an RPC on this thread's own connection (multiprocessing proxies
open one per thread). A reader that fails gives ``null`` for its field; a sample
that runs past its slot makes the sampler skip the missed slots rather than catch up.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

Reader = Callable[[], Any]


def _rounded(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return [round(float(v), digits) for v in value]


def se2_xyt(pose: Any) -> list[float]:
    """(x, y, theta) of a spatialmath SE2 (or of a sequence)."""
    if hasattr(pose, "theta"):
        return [float(pose.x), float(pose.y), float(pose.theta())]
    return [float(v) for v in pose]


class MarkerPoseReader:
    """The robot's map pose from the marker detector's publisher, on a connection of
    its own (:class:`~prpl_tidybot.marker_detector.client.MarkerDetectorClient` is one
    request/reply connection, not safe to share with the skills' thread), and never
    waiting: each call takes whatever payload has arrived and returns the latest
    pose (None before the first)."""

    def __init__(self, host: str, port: int | None = None) -> None:
        # pylint: disable=import-outside-toplevel
        from multiprocessing.connection import Client

        from prpl_tidybot.marker_detector.constants import MARKER_DETECTOR_PORT
        from prpl_tidybot.third_party.constants import CONN_AUTHKEY

        self._conn = Client(
            (host, MARKER_DETECTOR_PORT if port is None else port),
            authkey=CONN_AUTHKEY,
        )
        self._conn.send(None)
        self._pose: list[float] | None = None

    def __call__(self) -> list[float] | None:
        while self._conn.poll(0):
            payload = self._conn.recv()
            self._conn.send(None)
            pose = (payload.get("poses") or {}).get(0)
            if pose is not None:
                self._pose = [float(v) for v in pose[:3]]
        return self._pose

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()


class MotionLog:  # pylint: disable=too-many-instance-attributes
    """The sampler.

    ``readers`` maps each field (``live``, ``q``, ``g``, ``gc``, ``odom``, ``map``,
    ``i``, ``step``) to a no-argument callable; ``clock`` and ``wait`` (wait up to the
    given seconds, returning early on stop) can be faked in tests.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        path: str | Path,
        readers: dict[str, Reader],
        *,
        hz: float = 20.0,
        flush_s: float = 1.0,
        heartbeat_s: float = 1.0,
        clock: Callable[[], float] = time.time,
        wait: Callable[[float], Any] | None = None,
        on_close: Sequence[Callable[[], Any]] = (),
    ) -> None:
        if hz <= 0:
            raise ValueError(f"MotionLog: hz must be positive (got {hz})")
        self.path = Path(path)
        self._readers = dict(readers)
        self._period = 1.0 / float(hz)
        self._flush_s = float(flush_s)
        self._heartbeat_s = float(heartbeat_s)
        self._clock = clock
        self._stop = threading.Event()
        self._wait = wait if wait is not None else self._stop.wait
        self._on_close = list(on_close)
        # Shared only between the sampler and close(), never with a skill.
        self._io = threading.Lock()
        self._buffer: list[str] = []
        self._closed = False
        self._thread: threading.Thread | None = None
        self._last: dict[str, Any] = {}
        self._last_written_t = -float("inf")
        self._failing: set[str] = set()
        self.samples = 0
        self.written = 0
        self.skipped = 0

    def start(self) -> "MotionLog":
        """Start sampling in a daemon thread."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="motion-log", daemon=True
        )
        self._thread.start()
        return self

    def _read(self, name: str) -> Any:
        reader = self._readers.get(name)
        if reader is None:
            return None
        try:
            value = reader()
        except Exception as e:  # pylint: disable=broad-exception-caught
            if name not in self._failing:
                self._failing.add(name)
                _logger.warning("motion log: reading %s failed: %s", name, e)
            return None
        self._failing.discard(name)
        return value

    def sample(self, now: float) -> dict[str, Any] | None:
        """Take one sample at ``now``; returns the line written (None if suppressed as
        unchanged)."""
        self.samples += 1
        live = self._read("live") if "live" in self._readers else None
        q, g = self._read("q"), self._read("g")
        if live is not None:
            # Live only if it was so on both sides of the reads (a release or resume
            # in between makes them suspect).
            live = bool(live) and bool(self._read("live"))
            if not live:
                q = g = None
        state = {
            "i": list(self._read("i") or ()),
            "live": live,
            "q": _rounded(q),
            "g": _rounded(g),
            "gc": _rounded(self._read("gc")),
            "odom": _rounded(self._read("odom")),
            "map": _rounded(self._read("map")),
            "step": self._read("step"),
        }
        changed = state != self._last
        if not changed and now - self._last_written_t < self._heartbeat_s - 1e-6:
            return None
        line: dict[str, Any] = {"t": round(now, 3)}
        line.update(state)
        if state["step"] == self._last.get("step"):
            del line["step"]
        self._last = state
        self._last_written_t = now
        text = json.dumps(line, separators=(",", ":"))
        with self._io:
            if self._closed:
                return None
            self._buffer.append(text)
        self.written += 1
        return line

    def flush(self) -> None:
        """Append the buffered lines to the file."""
        with self._io:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer or self._closed:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    def _run(self) -> None:
        next_t = self._clock()
        last_flush = next_t
        while not self._stop.is_set():
            now = self._clock()
            if now < next_t:
                self._wait(next_t - now)
                continue
            missed = int((now - next_t) / self._period + 1e-9)
            if missed:
                self.skipped += missed
                next_t += missed * self._period
            self.sample(now)
            next_t += self._period
            if now - last_flush >= self._flush_s:
                self.flush()
                last_flush = now

    def close(self, timeout: float = 2.0) -> None:
        """Stop sampling, write what is buffered and close.

        Returns within about
        ``timeout`` even if a reader hangs (the thread is a daemon).
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                _logger.warning("motion log: sampler still in a read at shutdown")
        with self._io:
            self._flush_locked()
            self._closed = True
        for release in self._on_close:
            try:
                release()
            except Exception as e:  # pylint: disable=broad-exception-caught
                _logger.warning("motion log: closing a reader failed: %s", e)
