"""Tests for `prpl_tidybot.preview`.

Covers the small synchronous behavior of `preview_or_abort` and the helper that reaches
into a BilevelPlanningAgent for its planned-state sequence. The integration with
`pipeline.py` (Hydra-config-driven enable / disable, end-to-end abort path) lives in
`test_pipeline.py` follow-up tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest
from kinder_bilevel_planning.agent import AgentFailure

from prpl_tidybot.preview import planned_states_from_agent, preview_or_abort


@dataclass
class _StubShadowSim:
    """Minimal shadow sim: returns a uniform-color frame per `render()`."""

    color: int = 50
    shape: tuple[int, int] = (8, 8)
    reset_called: bool = False
    set_states: list[Any] = field(default_factory=list)

    def reset(self, *, seed: int | None = None) -> tuple:
        """Mark reset called; ignore seed."""
        del seed
        self.reset_called = True
        return None, {}

    def set_state(self, state: Any) -> None:
        """Record the state the recorder set."""
        self.set_states.append(state)

    def render(self) -> np.ndarray:
        """Return a uniform-color frame."""
        return np.full((self.shape[0], self.shape[1], 3), self.color, dtype=np.uint8)


@dataclass
class _NoneRenderShadowSim(_StubShadowSim):
    """Shadow sim whose render() always yields None."""

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """Return None for every state — emulates an unrenderable sim."""
        return None


def test_returns_none_and_does_not_prompt_when_no_planned_states(tmp_path: Path):
    """Empty trajectory → no preview written, no prompt fired, returns None."""
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    result = preview_or_abort(
        planned_states=[],
        shadow_sim=_StubShadowSim(),
        log_dir=tmp_path,
        prompt_fn=_record,
    )

    assert result is None
    assert not prompted
    assert not (tmp_path / "preview.mp4").exists()


def test_writes_mp4_and_returns_path_on_approval(tmp_path: Path):
    """Approval path: each planned state is rendered through the shadow sim and a
    preview.mp4 lands under log_dir."""
    states = [object(), object(), object()]
    shadow = _StubShadowSim(color=80)

    out = preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        shadow_sim=shadow,
        log_dir=tmp_path,
        fps=5,
        prompt_fn=lambda _msg: "y",
    )

    assert out == tmp_path / "preview.mp4"
    assert out.exists()
    assert out.stat().st_size > 0
    assert shadow.reset_called
    assert shadow.set_states == states


@pytest.mark.parametrize("answer", ["n", "no", "", "   ", "anything-else", "0", "Y "])
def test_raises_agent_failure_on_any_non_yes_answer(tmp_path: Path, answer: str):
    """Anything other than `y` / `yes` (case- and whitespace-insensitive) rejects."""
    if answer.strip().lower() in ("y", "yes"):
        pytest.skip("'y' / 'yes' are the approval path; covered separately")

    with pytest.raises(AgentFailure, match="rejected"):
        preview_or_abort(
            planned_states=[object()],  # type: ignore[list-item]
            shadow_sim=_StubShadowSim(),
            log_dir=tmp_path,
            prompt_fn=lambda _msg: answer,
        )


def test_writes_no_preview_when_shadow_renders_none(tmp_path: Path):
    """If the shadow sim returns None for every state, there's nothing to encode;
    preview_or_abort returns None and does not prompt."""
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    result = preview_or_abort(
        planned_states=[object(), object()],  # type: ignore[list-item]
        shadow_sim=_NoneRenderShadowSim(),
        log_dir=tmp_path,
        prompt_fn=_record,
    )

    assert result is None
    assert not prompted
    assert not (tmp_path / "preview.mp4").exists()


def test_creates_log_dir_if_missing(tmp_path: Path):
    """preview.mp4 lands even when log_dir hasn't been created by another component
    yet."""
    nested = tmp_path / "deep" / "not-yet-made"
    out = preview_or_abort(
        planned_states=[object()],  # type: ignore[list-item]
        shadow_sim=_StubShadowSim(),
        log_dir=nested,
        prompt_fn=lambda _msg: "y",
    )

    assert out == nested / "preview.mp4"
    assert out.exists()


def test_rendered_frames_have_consistent_dimensions(tmp_path: Path):
    """Round-trip the preview through cv to confirm it decodes back at the shadow sim's
    reported frame shape — guards against accidental shape mismatches if the helper
    grows BGR/RGB conversions in a follow-up."""
    shape = (12, 16)
    states = [object() for _ in range(4)]
    out = preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        shadow_sim=_StubShadowSim(color=120, shape=shape),
        log_dir=tmp_path,
        fps=8,
        prompt_fn=lambda _msg: "y",
    )
    assert out is not None
    cap = cv.VideoCapture(str(out))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok
    # mp4 codecs sometimes round up to multiples of 2/16; loose check.
    assert frame.shape[0] >= shape[0] and frame.shape[1] >= shape[1]


def test_planned_states_from_agent_reads_private_attribute():
    """Helper reads ``_planned_states`` from the agent so the private-attribute reach is
    in one place (easy to swap when upstream grows an accessor)."""

    class _StubAgent:
        def __init__(self):
            self._planned_states = ["a", "b", "c"]

    assert planned_states_from_agent(_StubAgent()) == ["a", "b", "c"]


def test_planned_states_from_agent_returns_empty_when_missing():
    """An agent without ``_planned_states`` (e.g. before reset()) yields an empty list
    rather than raising."""

    class _BareAgent:
        pass

    assert not planned_states_from_agent(_BareAgent())
