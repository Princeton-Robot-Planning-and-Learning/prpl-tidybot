"""Tests for prpl_tidybot.recording.Recorder.

Recorder is generic over the shadow-sim protocol; we test it with tiny stubs to keep the
tests fast (no pybullet). End-to-end coverage that the full planner pipeline writes a
video lives in `test_pipeline.py`.
"""

from pathlib import Path

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from relational_structs import ObjectCentricState

from prpl_tidybot.recording import Recorder


class _StubRealEnv(gymnasium.Env):
    """Toy real_env whose render returns a uniform-color frame."""

    observation_space = spaces.Discrete(1)
    action_space = spaces.Discrete(1)

    def __init__(self, color: int, shape: tuple[int, int] = (4, 4)) -> None:
        self._color = color
        self._h, self._w = shape

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return 0, {}

    def step(self, action):
        del action
        return 0, 0.0, False, False, {}

    def render(self):
        return np.full((self._h, self._w, 3), self._color, dtype=np.uint8)


class _StubShadowSim:
    """Toy shadow_sim mirroring the protocol Recorder needs."""

    def __init__(self, color: int) -> None:
        self._color = color
        self.last_state: ObjectCentricState | None = None
        self.reset_called: bool = False

    def reset(self, *, seed: int | None = None) -> tuple:
        """Mark that reset was called."""
        del seed
        self.reset_called = True
        return None, {}

    def set_state(self, state: ObjectCentricState) -> None:
        """Record the state the recorder set."""
        self.last_state = state

    def render(self) -> np.ndarray:
        """Return a uniform-color sim frame."""
        return np.full((4, 4, 3), self._color, dtype=np.uint8)


def test_recorder_resets_shadow_sim_at_construction(tmp_path: Path) -> None:
    """Gymnasium order-enforcing wrappers raise on render-before-reset; Recorder must
    reset the shadow once at construction."""
    real = _StubRealEnv(color=10)
    shadow = _StubShadowSim(color=20)
    Recorder(real, shadow, tmp_path / "out.mp4")
    assert shadow.reset_called


def test_recorder_captures_hstacked_frames(tmp_path: Path) -> None:
    """Capture() produces a horizontal stack of real and sim frames."""
    real = _StubRealEnv(color=10)
    shadow = _StubShadowSim(color=20)
    recorder = Recorder(real, shadow, tmp_path / "out.mp4")
    recorder.capture(state=None)  # type: ignore[arg-type]
    recorder.capture(state=None)  # type: ignore[arg-type]
    assert len(recorder.frames) == 2
    frame = recorder.frames[0]
    # Real on the left (color 10), sim on the right (color 20).
    assert frame.shape == (4, 8, 3)
    assert (frame[:, :4] == 10).all()
    assert (frame[:, 4:] == 20).all()


def test_recorder_finish_writes_video(tmp_path: Path) -> None:
    """Finish() writes an mp4 at the configured path with non-zero size."""
    real = _StubRealEnv(color=30)
    shadow = _StubShadowSim(color=60)
    video_path = tmp_path / "out.mp4"
    recorder = Recorder(real, shadow, video_path, fps=5)
    for _ in range(3):
        recorder.capture(state=None)  # type: ignore[arg-type]
    assert recorder.finish() == video_path
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_recorder_finish_no_frames_returns_none(tmp_path: Path) -> None:
    """No captures → no video, returns None."""
    real = _StubRealEnv(color=0)
    shadow = _StubShadowSim(color=0)
    recorder = Recorder(real, shadow, tmp_path / "out.mp4")
    assert recorder.finish() is None
    assert not (tmp_path / "out.mp4").exists()


def test_recorder_skips_when_real_renders_none(tmp_path: Path) -> None:
    """If real_env.render() returns None, the capture is skipped silently."""

    class _NoneRender(_StubRealEnv):
        def render(self):  # type: ignore[override]
            return None

    real = _NoneRender(color=0)
    shadow = _StubShadowSim(color=20)
    recorder = Recorder(real, shadow, tmp_path / "out.mp4")
    recorder.capture(state=None)  # type: ignore[arg-type]
    assert not recorder.frames


def test_recorder_pads_heights_for_hstack(tmp_path: Path) -> None:
    """Frames of different heights get zero-padded so hstack succeeds."""
    real = _StubRealEnv(color=10, shape=(4, 4))
    shadow = _StubShadowSim(color=20)

    class _TallerShadow(_StubShadowSim):
        def render(self) -> np.ndarray:  # type: ignore[override]
            return np.full((6, 4, 3), self._color, dtype=np.uint8)

    shadow = _TallerShadow(color=20)
    recorder = Recorder(real, shadow, tmp_path / "out.mp4")
    recorder.capture(state=None)  # type: ignore[arg-type]
    frame = recorder.frames[0]
    assert frame.shape == (6, 8, 3)
    # Real panel rows 0-3 are color 10; rows 4-5 are pad (zero).
    assert (frame[:4, :4] == 10).all()
    assert (frame[4:, :4] == 0).all()
    # Sim panel is full color 20.
    assert (frame[:, 4:] == 20).all()


@pytest.mark.usefixtures()
def test_recorder_skips_when_sim_renders_none(tmp_path: Path) -> None:
    """If shadow_sim.render() returns None, the capture is skipped."""
    real = _StubRealEnv(color=0)

    class _NoneSim(_StubShadowSim):
        def render(self):  # type: ignore[override]
            return None

    shadow = _NoneSim(color=0)
    recorder = Recorder(real, shadow, tmp_path / "out.mp4")
    recorder.capture(state=None)  # type: ignore[arg-type]
    assert not recorder.frames
