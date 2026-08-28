"""Per-tick trajectory recording for the planner pipeline.

Each call to :meth:`RecordingPerceiver.step` (or :meth:`reset`) snapshots
the new sim state and asks the recorder to persist a per-tick directory
under the Hydra runtime log dir:

    ${hydra:runtime.output_dir}/trajectory/000000/
        state.pkl      # the ObjectCentricState
        real.png       # real_env.render() at this tick (BGR-encoded on disk)
        meta.json      # {idx, timestamp}
        shadow.png     # the shadow sim's render of `state` (rendered at finish)

During the rollout, the only background work is a single serializer thread
that pops a snapshot off a queue and writes ``state.pkl``, ``real.png``,
and ``meta.json``. Both operations release the GIL during their heavy
phases (pickle for state-sized objects is microseconds; ``cv.imwrite``
releases the GIL during PNG compression), so the rollout thread is not
starved.

The shadow sim's ``set_state`` + ``render`` is **deferred to finish()**.
PyBullet's Python bindings don't reliably release the GIL during those
calls, so running them on a background thread during the rollout still
starves the control loop — visible on real hardware as a per-step pause
identical to the issue this recorder was meant to fix (#45). Rendering
in one sequential sweep after the rollout is the only configuration
that's been verified to not slow the control loop.

That sweep is the slow part of ending a rollout, so it is kept short:
at most ``max_frames`` ticks are rendered (a stride over the rollout,
first and last tick always kept, same as the plan preview), ticks whose
state is identical to the previously rendered one reuse its frame, and
progress is logged as it goes. The resulting ``video.mp4`` is a gist of
the rollout at ``fps``, not a real-time replay; the per-tick
``state.pkl`` / ``real.png`` are still written for every tick.

Trajectory capture is always on whenever a log dir is supplied; the
boolean `record.video` config flag controls only whether `finish()`
also composes the `(real, shadow)` panels into `video.mp4` alongside
the trajectory dir.
"""

from __future__ import annotations

import json
import logging
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import cv2 as cv
import gymnasium
import numpy as np
from prpl_utils.real_sim import Perceiver
from relational_structs import ObjectCentricState

from prpl_tidybot.video import subsample_indices, write_mp4

_RealObsType = TypeVar("_RealObsType")
_SENTINEL: Any = object()
_logger = logging.getLogger(__name__)
_PROGRESS_EVERY = 25


class _ShadowSim(Protocol):
    """Minimal protocol the recorder needs from a shadow sim."""

    def reset(self, *, seed: int | None = ...) -> Any:
        """Reset the underlying sim; gymnasium's order-enforcing wrapper requires this
        before any render call."""

    def set_state(self, state: ObjectCentricState) -> None:
        """Teleport the sim to `state` so the subsequent render shows it."""

    def render(self) -> Any:
        """Return an RGB frame for the sim's current state."""


@dataclass(frozen=True)
class _TickPayload:
    """Single per-tick record handed to the serializer worker."""

    idx: int
    timestamp: float
    state: ObjectCentricState
    real_frame: np.ndarray | None


class TrajectoryRecorder:
    """Capture (state, real frame, timestamp) per tick to disk; render shadow at end.

    During the rollout, one background **serializer** worker thread pops
    captured snapshots off a queue and writes ``state.pkl``, ``real.png``,
    and ``meta.json`` per tick dir. Pickling small dataclasses and PNG
    encoding both release the GIL during their heavy phases, so the
    rollout thread is not starved.

    At :meth:`finish`, after the serializer is drained, this class sweeps
    the per-tick dirs sequentially: for each saved ``state.pkl``, set the
    shadow sim's state, render it, and write ``shadow.png`` next to the
    other files. This is where the pybullet work lives — moved OFF the
    rollout because PyBullet's bindings don't reliably release the GIL,
    so even a background-thread shadow worker starves the control loop
    (#45 follow-up).

    With ``compose_video=True`` :meth:`finish` additionally renders shadow
    frames for at most ``max_frames`` ticks (``None`` renders every tick)
    and composes the ``(real, shadow)`` panels of those ticks into
    ``video.mp4`` alongside the trajectory dir.
    """

    _TICK_DIR_FORMAT = "{:06d}"

    def __init__(
        self,
        log_dir: Path | str,
        shadow_sim: _ShadowSim,
        real_env: gymnasium.Env,
        seed: int = 0,
        fps: int = 10,
        compose_video: bool = False,
        max_frames: int | None = 100,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._trajectory_dir = self._log_dir / "trajectory"
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._shadow_sim = shadow_sim
        self._real_env = real_env
        self._fps = fps
        self._compose_video = compose_video
        self._max_frames = max_frames
        # Prime the shadow sim: gymnasium's order-enforcing wrapper raises
        # if render is called before reset. Same seed as the rollout so any
        # static / non-state-set env content (camera positions, world
        # layout) matches between real and sim in modes where both are the
        # same env. Done at construction so the finish-time render sweep
        # can call render straight away.
        self._shadow_sim.reset(seed=seed)

        self._serialize_queue: queue.Queue[Any] = queue.Queue()
        self._serialize_thread = threading.Thread(
            target=self._serialize_loop, daemon=True, name="trajectory-serializer"
        )
        self._serialize_thread.start()

    @property
    def trajectory_dir(self) -> Path:
        """Directory holding the per-tick subdirs."""
        return self._trajectory_dir

    def capture(self, idx: int, state: ObjectCentricState) -> None:
        """Enqueue tick `idx` for async serialization.

        Non-blocking: the only synchronous work is ``real_env.render()``
        (a non-blocking cache read for :class:`CeilingCameraRenderer`) and
        a single ``queue.put`` call.
        """
        real_frame = self._real_env.render()
        real_frame_arr: np.ndarray | None = (
            np.asarray(real_frame) if real_frame is not None else None
        )
        payload = _TickPayload(
            idx=idx,
            timestamp=time.time(),
            state=state,
            real_frame=real_frame_arr,
        )
        self._serialize_queue.put(payload)

    def finish(self) -> Path | None:
        """Drain the serializer, render shadows, optionally compose `video.mp4`.

        Returns the path to the video if ``compose_video=True`` and at
        least one pair of (real, shadow) frames was produced; otherwise
        ``None``.
        """
        self._serialize_queue.put(_SENTINEL)
        self._serialize_thread.join()
        if not self._compose_video:
            # Shadow rendering only feeds the mp4. Skipping it when no video
            # was requested keeps `finish()` fast — the pybullet sweep was
            # the dominant cost otherwise.
            return None
        tick_dirs = [
            d
            for d in sorted(self._trajectory_dir.iterdir())
            if (d / "state.pkl").exists()
        ]
        selected = [
            tick_dirs[i] for i in subsample_indices(len(tick_dirs), self._max_frames)
        ]
        self._render_shadows_from_disk(selected, len(tick_dirs))
        return self._compose_video_from_disk(selected)

    def _serialize_loop(self) -> None:
        while True:
            item = self._serialize_queue.get()
            if item is _SENTINEL:
                return
            self._serialize_tick(item)

    def _serialize_tick(self, payload: _TickPayload) -> None:
        tick_dir = self._trajectory_dir / self._TICK_DIR_FORMAT.format(payload.idx)
        tick_dir.mkdir(exist_ok=True)
        with open(tick_dir / "state.pkl", "wb") as f:
            pickle.dump(payload.state, f)
        if payload.real_frame is not None:
            cv.imwrite(
                str(tick_dir / "real.png"),
                cv.cvtColor(payload.real_frame, cv.COLOR_RGB2BGR),
            )
        with open(tick_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"idx": payload.idx, "timestamp": payload.timestamp}, f)

    def _render_shadows_from_disk(self, tick_dirs: list[Path], num_ticks: int) -> None:
        """Render shadow.png for the saved state in each of `tick_dirs`.

        Sequential, after the rollout. The shadow sim is touched only here, so there is
        no thread-safety surface area on it and no concurrent pybullet load on the
        rollout thread. A tick whose state equals the previously rendered state reuses
        that frame instead of rendering again (long holds render identically).
        """
        _logger.info(
            "Rendering %d of %d recorded ticks through the shadow sim...",
            len(tick_dirs),
            num_ticks,
        )
        start = time.monotonic()
        last_state: ObjectCentricState | None = None
        last_frame: np.ndarray | None = None
        rendered = 0
        for count, tick_dir in enumerate(tick_dirs, start=1):
            with open(tick_dir / "state.pkl", "rb") as f:
                state = pickle.load(f)
            if last_state is not None and state == last_state:
                frame = last_frame
            else:
                self._shadow_sim.set_state(state)
                raw = self._shadow_sim.render()
                frame = None if raw is None else np.asarray(raw, dtype=np.uint8)
                rendered += 1
            last_state, last_frame = state, frame
            if frame is None:
                continue
            cv.imwrite(
                str(tick_dir / "shadow.png"), cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            )
            if count % _PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start
                _logger.info(
                    "  shadow frames %d/%d (%.0fs elapsed, ~%.0fs left)",
                    count,
                    len(tick_dirs),
                    elapsed,
                    elapsed / count * (len(tick_dirs) - count),
                )
        _logger.info(
            "Shadow sweep done: %d rendered, %d reused, %.1fs.",
            rendered,
            len(tick_dirs) - rendered,
            time.monotonic() - start,
        )

    def _compose_video_from_disk(self, tick_dirs: list[Path]) -> Path | None:
        start = time.monotonic()
        frames: list[np.ndarray] = []
        for tick_dir in tick_dirs:
            real_path = tick_dir / "real.png"
            shadow_path = tick_dir / "shadow.png"
            if not real_path.exists() or not shadow_path.exists():
                continue
            real_bgr = cv.imread(str(real_path))
            shadow_bgr = cv.imread(str(shadow_path))
            real_rgb = cv.cvtColor(real_bgr, cv.COLOR_BGR2RGB)
            shadow_rgb = cv.cvtColor(shadow_bgr, cv.COLOR_BGR2RGB)
            frames.append(_hstack_frames(real_rgb, shadow_rgb))
        if not frames:
            _logger.info("No (real, shadow) frame pairs; skipping video.mp4.")
            return None
        video_path = self._log_dir / "video.mp4"
        write_mp4(frames, video_path, fps=self._fps)
        _logger.info(
            "Wrote %s (%d frames at %d fps, %.1fs).",
            video_path,
            len(frames),
            self._fps,
            time.monotonic() - start,
        )
        return video_path


class RecordingPerceiver(
    Generic[_RealObsType], Perceiver[_RealObsType, ObjectCentricState]
):
    """Perceiver wrapper that hands every produced state to a recorder.

    Wiring recording at the perceiver layer guarantees one captured tick
    per real-env tick: every state the rollout ever sees flows through
    this method. The state type is pinned to ``ObjectCentricState``
    because that is what :class:`TrajectoryRecorder` needs for its shadow
    sim.
    """

    def __init__(
        self,
        inner: Perceiver[_RealObsType, ObjectCentricState],
        recorder: TrajectoryRecorder,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._idx = 0

    def reset(self, obs: _RealObsType, info: dict[str, Any]) -> ObjectCentricState:
        state = self._inner.reset(obs, info)
        self._idx = 0
        self._recorder.capture(self._idx, state)
        return state

    def step(self, obs: _RealObsType, info: dict[str, Any]) -> ObjectCentricState:
        state = self._inner.step(obs, info)
        self._idx += 1
        self._recorder.capture(self._idx, state)
        return state


def _hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize the shorter frame to match heights (preserving aspect ratio), then concat
    horizontally.

    Avoids the black padding bands the previous pad-based impl produced when the real
    and shadow renderers had different resolutions.
    """
    if left.shape[0] != right.shape[0]:
        target_h = max(left.shape[0], right.shape[0])
        left = _resize_to_height(left, target_h)
        right = _resize_to_height(right, target_h)
    return np.concatenate([left, right], axis=1)


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    if frame.shape[0] == target_h:
        return frame
    new_w = max(1, int(round(frame.shape[1] * target_h / frame.shape[0])))
    return cv.resize(frame, (new_w, target_h))
