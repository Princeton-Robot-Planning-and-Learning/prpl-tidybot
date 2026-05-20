"""Side-by-side real+sim video recording for the planner pipeline.

Each `capture(state)` call:

  1. Calls `real_env.render()` for the "real" panel — whatever the
     current run's `real_env` exposes (kinder render in sim mode, the
     FakeInterface's base camera in fake mode, the real base camera in
     real mode).
  2. `set_state(state)` on a shadow kinder env and renders it for the
     "sim" panel — i.e. "what the agent thinks the world looks like".
  3. Horizontally concatenates the two frames into the buffer.

`finish()` writes the buffer to an mp4 via moviepy. Frame compositing
is generic — the `shadow_sim` argument is anything with a `set_state`
plus a `render() -> ndarray | None`; `prpl_tidybot.sim_env.KinderSimEnv`
satisfies that protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import cv2 as cv
import gymnasium
import numpy as np
from moviepy import ImageSequenceClip
from prpl_utils.real_sim import Perceiver
from relational_structs import ObjectCentricState

_RealObsType = TypeVar("_RealObsType")


class _ShadowSim(Protocol):
    """Minimal protocol the Recorder needs from a shadow sim."""

    def reset(self, *, seed: int | None = ...) -> Any:
        """Reset the underlying sim; required by gymnasium order
        enforcement before any render call."""

    def set_state(self, state: ObjectCentricState) -> None:
        """Teleport the sim to `state` so subsequent renders show it."""

    def render(self) -> Any:
        """Return an RGB frame for the sim's current state."""


class Recorder:
    """Capture (real | sim) side-by-side frames per step, write to mp4."""

    def __init__(
        self,
        real_env: gymnasium.Env,
        shadow_sim: _ShadowSim,
        video_path: str | Path,
        fps: int = 10,
        seed: int = 0,
    ) -> None:
        self._real_env = real_env
        self._shadow_sim = shadow_sim
        self._video_path = Path(video_path)
        self._fps = fps
        self._frames: list[np.ndarray] = []
        # gymnasium's order-enforcing wrapper raises if you call render
        # before reset, so prime the shadow sim. Pass the same seed used
        # for the rollout so that any static / non-state-set env content
        # (camera positions, world layout) matches between real and sim
        # in modes where both are the same kinder env.
        self._shadow_sim.reset(seed=seed)

    @property
    def frames(self) -> list[np.ndarray]:
        """Frames captured so far (snapshot of the internal list)."""
        return list(self._frames)

    def capture(self, state: ObjectCentricState) -> None:
        """Append one composed frame for `state` to the buffer.

        Skips silently if either side fails to produce a frame (e.g.
        `real_env.render()` returns None).
        """
        real_frame = self._real_env.render()
        self._shadow_sim.set_state(state)
        sim_frame = self._shadow_sim.render()
        if real_frame is None or sim_frame is None:
            return
        composed = _hstack_frames(np.asarray(real_frame), np.asarray(sim_frame))
        self._frames.append(composed.astype(np.uint8))

    def finish(self) -> Path | None:
        """Write the captured frames to `video_path` via moviepy.

        Returns the output path, or `None` if no frames have been
        captured (e.g. no `capture` calls or every render returned None).
        """
        if not self._frames:
            return None
        clip = ImageSequenceClip(self._frames, fps=self._fps)
        clip.write_videofile(str(self._video_path), logger=None)
        return self._video_path


class RecordingPerceiver(
    Generic[_RealObsType], Perceiver[_RealObsType, ObjectCentricState]
):
    """Perceiver wrapper that captures a Recorder frame for every produced state.

    Wiring recording at the perceiver layer (rather than via a Runner hook or
    subclass) is what guarantees one frame per real-env tick: the perceiver is
    the single point through which every state — initial and per-tick — passes,
    so wrapping it makes the "one frame per outer Runner.step" failure mode
    structurally impossible. The state type is pinned to ``ObjectCentricState``
    because that is what :class:`Recorder` needs to drive its shadow sim.
    """

    def __init__(
        self,
        inner: Perceiver[_RealObsType, ObjectCentricState],
        recorder: "Recorder",
    ) -> None:
        self._inner = inner
        self._recorder = recorder

    def reset(self, obs: _RealObsType, info: dict[str, Any]) -> ObjectCentricState:
        state = self._inner.reset(obs, info)
        self._recorder.capture(state)
        return state

    def step(self, obs: _RealObsType, info: dict[str, Any]) -> ObjectCentricState:
        state = self._inner.step(obs, info)
        self._recorder.capture(state)
        return state


def _hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize the shorter frame to match heights (preserving aspect ratio), then
    concat horizontally."""
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
