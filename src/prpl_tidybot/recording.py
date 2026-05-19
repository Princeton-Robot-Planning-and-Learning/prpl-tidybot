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
from typing import Any, Protocol

import gymnasium
import numpy as np
from moviepy import ImageSequenceClip
from relational_structs import ObjectCentricState


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


def _hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pad shorter frame with black to match heights, then concat horizontally."""
    if left.shape[0] != right.shape[0]:
        target_h = max(left.shape[0], right.shape[0])
        left = _pad_to_height(left, target_h)
        right = _pad_to_height(right, target_h)
    return np.concatenate([left, right], axis=1)


def _pad_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    pad_h = target_h - frame.shape[0]
    if pad_h == 0:
        return frame
    pad = np.zeros((pad_h, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    return np.concatenate([frame, pad], axis=0)
