"""Tests for prpl_tidybot.recording.TrajectoryRecorder.

The recorder is generic over the shadow-sim protocol; we test it with tiny stubs to keep
the tests fast (no pybullet). End-to-end coverage that the full planner pipeline writes
a trajectory dir + optional video lives in `test_pipeline.py`.
"""

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import gymnasium
import numpy as np
from gymnasium import spaces

from prpl_tidybot.recording import TrajectoryRecorder, _hstack_frames


@dataclass(frozen=True)
class _StubState:
    """Stand-in for an ObjectCentricState.

    Hashable + pickleable.
    """

    label: str


class _StubRealEnv(gymnasium.Env):
    """Toy real_env whose render returns a uniform-color frame."""

    observation_space = spaces.Discrete(1)
    action_space = spaces.Discrete(1)

    def __init__(self, color: int, shape: tuple[int, int] = (4, 4)) -> None:
        self._color = color
        self._h, self._w = shape

    def reset(self, *, seed=None, options=None):
        """Reset to a fixed dummy observation."""
        super().reset(seed=seed)
        del options
        return 0, {}

    def step(self, action):
        """No-op step; returns the same dummy obs every call."""
        del action
        return 0, 0.0, False, False, {}

    def render(self):
        """Return a uniform-color frame."""
        return np.full((self._h, self._w, 3), self._color, dtype=np.uint8)


class _StubShadowSim:
    """Toy shadow_sim mirroring the protocol TrajectoryRecorder needs."""

    def __init__(self, color: int, shape: tuple[int, int] = (4, 4)) -> None:
        self._color = color
        self._h, self._w = shape
        self.last_state: object | None = None
        self.reset_called: bool = False

    def reset(self, *, seed: int | None = None) -> tuple:
        """Mark that reset was called."""
        del seed
        self.reset_called = True
        return None, {}

    def set_state(self, state) -> None:
        """Record the state the recorder set."""
        self.last_state = state

    def render(self) -> np.ndarray:
        """Return a uniform-color sim frame."""
        return np.full((self._h, self._w, 3), self._color, dtype=np.uint8)


def _make_recorder(
    tmp_path: Path,
    *,
    real_color: int = 10,
    shadow_color: int = 20,
    compose_video: bool = False,
    fps: int = 5,
) -> tuple[TrajectoryRecorder, _StubRealEnv, _StubShadowSim]:
    real = _StubRealEnv(color=real_color)
    shadow = _StubShadowSim(color=shadow_color)
    recorder = TrajectoryRecorder(
        log_dir=tmp_path,
        shadow_sim=shadow,
        real_env=real,
        fps=fps,
        compose_video=compose_video,
    )
    return recorder, real, shadow


def test_recorder_resets_shadow_sim_at_construction(tmp_path: Path) -> None:
    """Gymnasium order-enforcing wrappers raise on render-before-reset; the recorder
    must reset the shadow once at construction."""
    recorder, _, shadow = _make_recorder(tmp_path)
    try:
        assert shadow.reset_called
    finally:
        recorder.finish()


def test_recorder_writes_per_tick_dir_with_state_real_shadow_meta(
    tmp_path: Path,
) -> None:
    """Each capture creates a zero-padded subdir under trajectory/ with state.pkl,
    real.png, shadow.png and meta.json.

    `compose_video=True` is required for the shadow render sweep at finish(); without it
    shadow.png isn't written.
    """
    recorder, _, _ = _make_recorder(
        tmp_path, real_color=30, shadow_color=60, compose_video=True
    )
    state = _StubState(label="t0")
    recorder.capture(idx=0, state=state)  # type: ignore[arg-type]
    recorder.capture(idx=1, state=_StubState(label="t1"))  # type: ignore[arg-type]
    recorder.finish()  # drains both worker threads

    tick0 = tmp_path / "trajectory" / "000000"
    tick1 = tmp_path / "trajectory" / "000001"
    assert tick0.is_dir() and tick1.is_dir()
    for tick in (tick0, tick1):
        assert (tick / "state.pkl").exists()
        assert (tick / "real.png").exists()
        assert (tick / "shadow.png").exists()
        assert (tick / "meta.json").exists()

    # state.pkl round-trips the supplied state.
    with open(tick0 / "state.pkl", "rb") as f:
        assert pickle.load(f) == state

    # meta.json carries idx + timestamp.
    with open(tick0 / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["idx"] == 0
    assert isinstance(meta["timestamp"], float)

    # Frames round-trip through cv's BGR encoding back to RGB.
    real_bgr = cv.imread(str(tick0 / "real.png"))
    real_rgb = cv.cvtColor(real_bgr, cv.COLOR_BGR2RGB)
    assert (real_rgb == 30).all()
    shadow_bgr = cv.imread(str(tick0 / "shadow.png"))
    shadow_rgb = cv.cvtColor(shadow_bgr, cv.COLOR_BGR2RGB)
    assert (shadow_rgb == 60).all()


def test_recorder_finish_with_compose_video_writes_mp4(tmp_path: Path) -> None:
    """`compose_video=True` produces video.mp4 alongside trajectory/."""
    recorder, _, _ = _make_recorder(tmp_path, compose_video=True)
    for i in range(3):
        state = _StubState(label=f"t{i}")
        recorder.capture(idx=i, state=state)  # type: ignore[arg-type]
    video_path = recorder.finish()
    assert video_path == tmp_path / "video.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_recorder_finish_without_compose_video_skips_mp4(tmp_path: Path) -> None:
    """Default `compose_video=False` writes per-tick dirs only — no video.mp4."""
    recorder, _, _ = _make_recorder(tmp_path, compose_video=False)
    recorder.capture(idx=0, state=_StubState(label="t0"))  # type: ignore[arg-type]
    assert recorder.finish() is None
    assert not (tmp_path / "video.mp4").exists()
    # But the trajectory dir is still populated.
    assert (tmp_path / "trajectory" / "000000" / "state.pkl").exists()


def test_recorder_compose_video_no_captures_returns_none(tmp_path: Path) -> None:
    """No captures + compose_video=True → no video, returns None."""
    recorder, _, _ = _make_recorder(tmp_path, compose_video=True)
    assert recorder.finish() is None
    assert not (tmp_path / "video.mp4").exists()


def test_recorder_skips_real_png_when_real_renders_none(tmp_path: Path) -> None:
    """If real_env.render() returns None, the per-tick dir omits real.png; state and
    shadow still land (shadow requires compose_video=True to be written)."""

    class _NoneReal(_StubRealEnv):
        def render(self):  # type: ignore[override]
            return None

    shadow = _StubShadowSim(color=20)
    recorder = TrajectoryRecorder(
        log_dir=tmp_path,
        shadow_sim=shadow,
        real_env=_NoneReal(color=0),
        compose_video=True,
    )
    recorder.capture(idx=0, state=_StubState(label="t0"))  # type: ignore[arg-type]
    recorder.finish()

    tick = tmp_path / "trajectory" / "000000"
    assert (tick / "state.pkl").exists()
    assert (tick / "shadow.png").exists()
    assert not (tick / "real.png").exists()


def test_recorder_skips_shadow_png_when_shadow_renders_none(tmp_path: Path) -> None:
    """With compose_video=True the shadow render sweep runs at finish(), but if
    shadow_sim.render() returns None for a tick the corresponding shadow.png is skipped
    (state.pkl + real.png + meta.json still land)."""

    class _NoneShadow(_StubShadowSim):
        def render(self):  # type: ignore[override]
            return None

    real = _StubRealEnv(color=10)
    recorder = TrajectoryRecorder(
        log_dir=tmp_path,
        shadow_sim=_NoneShadow(color=0),
        real_env=real,
        compose_video=True,
    )
    recorder.capture(idx=0, state=_StubState(label="t0"))  # type: ignore[arg-type]
    recorder.finish()

    tick = tmp_path / "trajectory" / "000000"
    assert (tick / "state.pkl").exists()
    assert (tick / "real.png").exists()
    assert not (tick / "shadow.png").exists()


def test_recorder_capture_is_non_blocking(tmp_path: Path) -> None:
    """The rollout's hot-path call must not block on heavy shadow rendering.

    We install a deliberately-slow shadow render and verify capture() returns before the
    render completes.
    """

    class _SlowShadow(_StubShadowSim):
        def render(self) -> np.ndarray:  # type: ignore[override]
            time.sleep(0.5)
            return super().render()

    recorder = TrajectoryRecorder(
        log_dir=tmp_path,
        shadow_sim=_SlowShadow(color=20),
        real_env=_StubRealEnv(color=10),
    )
    try:
        t0 = time.time()
        recorder.capture(idx=0, state=_StubState(label="t0"))  # type: ignore[arg-type]
        elapsed = time.time() - t0
        # Should be well under the 0.5s shadow render budget — capture is just
        # a render of the (instant) stub real_env plus two queue puts.
        assert elapsed < 0.1, f"capture took {elapsed:.3f}s; expected non-blocking"
    finally:
        # finish() waits for the slow shadow render to actually complete.
        recorder.finish()


def test_hstack_resizes_shorter_frame_to_match_heights():
    """Different-height inputs get resized (preserving aspect ratio) before concat — no
    black padding bands on the shorter panel."""
    # Real 4x4 (uniform 10); shadow 6x4 (uniform 20).
    real = np.full((4, 4, 3), 10, dtype=np.uint8)
    shadow = np.full((6, 4, 3), 20, dtype=np.uint8)
    composed = _hstack_frames(real, shadow)
    # Real (4x4) is resized to height 6 -> width round(4 * 6 / 4) = 6, so the
    # composite is height 6, width 6 (real) + 4 (shadow) = 10.
    assert composed.shape == (6, 10, 3)
    # Real panel stays uniform color 10 after resize.
    assert (composed[:, :6] == 10).all()
    # Shadow panel keeps its original uniform color 20.
    assert (composed[:, 6:] == 20).all()


def test_hstack_no_op_when_heights_already_match():
    """Same-height inputs are concatenated as-is."""
    real = np.full((4, 4, 3), 10, dtype=np.uint8)
    shadow = np.full((4, 4, 3), 20, dtype=np.uint8)
    composed = _hstack_frames(real, shadow)
    assert composed.shape == (4, 8, 3)
    assert (composed[:, :4] == 10).all()
    assert (composed[:, 4:] == 20).all()


def test_recorder_trajectory_dir_property(tmp_path: Path) -> None:
    """The recorder exposes the trajectory subdirectory for downstream callers."""
    recorder, _, _ = _make_recorder(tmp_path)
    try:
        assert recorder.trajectory_dir == tmp_path / "trajectory"
        assert recorder.trajectory_dir.is_dir()
    finally:
        recorder.finish()
