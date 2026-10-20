"""Record a toss from the ceiling cameras and read where the cube landed.

`TossCapture` films the toss from the top ceiling camera (through any `Renderer`,
normally `CeilingCameraRenderer` on the ceiling image publisher) and, once the cube has
settled, reads the marker detector's published targets. Any visible cube-face marker
gives the landing position in the map frame, which `measure_landing` turns into a
distance and bearing from the robot's map pose at the throw. Both land in the run's log
dir: ``toss.mp4`` and ``toss_outcome.json``.

The detector projects every target marker at `TARGET_MARKER_HEIGHT` (the floor), while a
marker on the top face of a resting cube sits a cube's height above it, so the reported
position slides toward the camera's nadir by (cube height / camera height) times the
cube's distance from the nadir.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from prpl_tidybot.marker_detector.constants import TARGET_MARKER_IDS
from prpl_tidybot.rendering import Renderer
from prpl_tidybot.video import write_mp4

_logger = logging.getLogger(__name__)

# Target markers taped under the cylinders of the cylinder-shelf env.
CYLINDER_MARKER_IDS = (35, 36, 37)

# Cube-face markers looked for after a toss: every published target marker
# except the cylinders'. 23 is the cube marker the shelf3d env picks.
DEFAULT_CUBE_FACE_MARKER_IDS = tuple(
    marker_id for marker_id in TARGET_MARKER_IDS if marker_id not in CYLINDER_MARKER_IDS
)

# Robot index of the TidyBot in the detector's "poses" (as RealBaseInterface).
_ROBOT_INDEX = 0


class MarkerPayloadSource(Protocol):
    """Anything with `MarkerDetectorClient`'s `get_latest` / `close`."""

    def get_latest(self) -> dict[str, Any]:
        """The latest published payload."""

    def close(self) -> None:
        """Release the connection."""


def validate_cube_marker_ids(ids: Sequence[int]) -> tuple[int, ...]:
    """The ids as a tuple; raise unless they are a non-empty subset of
    `TARGET_MARKER_IDS`, the only markers the detector publishes as targets."""
    result = tuple(int(i) for i in ids)
    if not result:
        raise ValueError("At least one cube-face marker id is required.")
    unknown = sorted(set(result) - set(TARGET_MARKER_IDS))
    if unknown:
        raise ValueError(
            f"Cube-face marker ids {unknown} are not in TARGET_MARKER_IDS "
            f"{TARGET_MARKER_IDS}; the detector never publishes them."
        )
    return result


def robot_pose_from_payload(
    payload: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    """The robot's map-frame ``(x, y, theta)`` in a detector payload, or None."""
    poses = payload.get("poses") or {}
    if _ROBOT_INDEX not in poses:
        return None
    x, y, theta = poses[_ROBOT_INDEX][:3]
    return float(x), float(y), float(theta)


def measure_landing(
    robot_pose_at_throw: Sequence[float] | None,
    targets: Mapping[int, Sequence[float]],
    cube_marker_ids: Sequence[int],
) -> dict[str, Any]:
    """Where the cube landed, from the detector's ``targets`` after it settled.

    Every visible cube-face marker is logged; the first visible one in
    ``cube_marker_ids`` order is the landing. Distance is from the robot's base centre
    at the throw and bearing is relative to its heading, wrapped to (-180, 180] deg,
    positive to the robot's left. ``status`` is ``"landed"`` or
    ``"no_cube_marker_visible"``.
    """
    visible = {
        int(marker_id): (float(targets[marker_id][0]), float(targets[marker_id][1]))
        for marker_id in cube_marker_ids
        if marker_id in targets
    }
    robot = (
        None if robot_pose_at_throw is None else [float(v) for v in robot_pose_at_throw]
    )
    landing: dict[str, Any] | None = None
    if visible:
        marker_id = next(i for i in cube_marker_ids if i in visible)
        x, y = visible[marker_id]
        distance = bearing = None
        if robot is not None:
            dx, dy = x - robot[0], y - robot[1]
            distance = math.hypot(dx, dy)
            bearing = math.atan2(
                math.sin(math.atan2(dy, dx) - robot[2]),
                math.cos(math.atan2(dy, dx) - robot[2]),
            )
        landing = {
            "marker_id": marker_id,
            "x": x,
            "y": y,
            "distance_m": distance,
            "bearing_rad": bearing,
            "bearing_deg": None if bearing is None else math.degrees(bearing),
        }
    return {
        "status": "landed" if landing is not None else "no_cube_marker_visible",
        "robot_map_pose_at_throw": robot,
        "visible_cube_markers": {str(i): list(xy) for i, xy in visible.items()},
        "landing": landing,
    }


class TossCapture:
    """Film a toss from the ceiling and log where the cube landed.

    Call :meth:`start` right before the toss (it reads the robot's map pose and starts
    sampling ``renderer`` at ``fps`` on a thread) and :meth:`finish` right after it
    returns. ``finish`` waits ``settle_s`` for the cube to come to rest, reads the
    detector, keeps filming until ``post_toss_record_s`` has passed since the toss
    returned, then writes ``toss.mp4`` and ``toss_outcome.json`` to ``log_dir`` and
    returns the outcome.
    """

    def __init__(
        self,
        renderer: Renderer,
        marker_client: MarkerPayloadSource,
        log_dir: Path | str,
        cube_marker_ids: Sequence[int] = DEFAULT_CUBE_FACE_MARKER_IDS,
        fps: int = 15,
        post_toss_record_s: float = 3.0,
        settle_s: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._renderer = renderer
        self._marker_client = marker_client
        self._log_dir = Path(log_dir)
        self._cube_marker_ids = validate_cube_marker_ids(cube_marker_ids)
        self._fps = int(fps)
        self._post_toss_record_s = float(post_toss_record_s)
        self._settle_s = float(settle_s)
        self._sleep_fn = sleep_fn
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._robot_pose: tuple[float, float, float] | None = None

    def start(self) -> None:
        """Read the robot pose at the throw and start filming."""
        self._robot_pose = robot_pose_from_payload(self._marker_client.get_latest())
        if self._robot_pose is None:
            _logger.warning("Robot pose not published at the throw.")
        self._frames = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def finish(self, toss: Mapping[str, Any]) -> dict[str, Any]:
        """Wait for the cube to settle, read its landing, stop filming, write both
        files; ``toss`` (the toss parameters) is copied into the JSON."""
        self._sleep_fn(self._settle_s)
        payload = self._marker_client.get_latest()
        outcome = measure_landing(
            self._robot_pose, payload.get("targets") or {}, self._cube_marker_ids
        )
        self._sleep_fn(max(0.0, self._post_toss_record_s - self._settle_s))
        video = self._stop_and_write_video()
        outcome["toss"] = dict(toss)
        outcome["cube_marker_ids"] = list(self._cube_marker_ids)
        outcome["video"] = video
        out_path = self._log_dir / "toss_outcome.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(outcome, indent=2))
        if outcome["landing"] is None:
            _logger.warning(
                "No cube-face marker (%s) visible after the toss; logged to %s.",
                list(self._cube_marker_ids),
                out_path,
            )
        else:
            landing = outcome["landing"]
            _logger.info(
                "Cube landed at (%.3f, %.3f) (marker %d), %s from the robot; "
                "logged to %s.",
                landing["x"],
                landing["y"],
                landing["marker_id"],
                (
                    "unknown distance"
                    if landing["distance_m"] is None
                    else f"{landing['distance_m']:.3f} m at "
                    f"{landing['bearing_deg']:+.1f} deg"
                ),
                out_path,
            )
        return outcome

    def close(self) -> None:
        """Stop filming if still running and release the renderer and client."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._renderer.close()
        self._marker_client.close()

    def _record_loop(self) -> None:
        period = 1.0 / self._fps
        next_time = time.monotonic()
        while not self._stop.is_set():
            frame = self._renderer.render()
            if frame is not None:
                with self._frames_lock:
                    self._frames.append(np.asarray(frame, dtype=np.uint8))
            next_time += period
            self._stop.wait(max(0.0, next_time - time.monotonic()))

    def _stop_and_write_video(self) -> str | None:
        # A renderer that has not delivered its first frame yet would leave the
        # clip empty on a very short capture; give it a moment.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._frames_lock:
                if self._frames:
                    break
            time.sleep(0.01)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._frames_lock:
            frames = list(self._frames)
        if not frames:
            _logger.warning("No ceiling-camera frames arrived; no toss video.")
            return None
        write_mp4(frames, self._log_dir / "toss.mp4", fps=self._fps)
        return "toss.mp4"
