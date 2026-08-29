"""Tests for `prpl_tidybot.preview`.

Covers the small synchronous behavior of `preview_or_abort` (including how
magic-skill gaps are shown) and the helper that reaches into a
BilevelPlanningAgent for its plan. The integration with
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
from kinder_models.structs import SkillCall
from relational_structs import Object, Type
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.preview import (
    find_floor_violation,
    planned_trajectory_from_agent,
    preview_or_abort,
)
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.structs import TidyBotObservation


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


def test_subsamples_long_trajectories_keeping_first_and_last(tmp_path: Path):
    """Plans longer than `max_frames` are strided down before rendering: the shadow sim
    sees at most max_frames + 1 states, always including the first and last, in
    order."""
    states = list(range(203))
    shadow = _StubShadowSim()

    out = preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        shadow_sim=shadow,
        log_dir=tmp_path,
        max_frames=50,
        prompt_fn=lambda _msg: "y",
    )

    assert out is not None
    assert len(shadow.set_states) <= 51
    assert shadow.set_states[0] == states[0]
    assert shadow.set_states[-1] == states[-1]
    assert shadow.set_states == sorted(shadow.set_states)


def test_renders_every_state_when_max_frames_is_none(tmp_path: Path):
    """`max_frames=None` opts out of subsampling entirely."""
    states = [object() for _ in range(120)]
    shadow = _StubShadowSim()

    preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        shadow_sim=shadow,
        log_dir=tmp_path,
        max_frames=None,
        prompt_fn=lambda _msg: "y",
    )

    assert shadow.set_states == states


def test_short_trajectories_are_not_subsampled(tmp_path: Path):
    """Plans at or under `max_frames` render every state unchanged."""
    states = [object() for _ in range(10)]
    shadow = _StubShadowSim()

    preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        shadow_sim=shadow,
        log_dir=tmp_path,
        max_frames=50,
        prompt_fn=lambda _msg: "y",
    )

    assert shadow.set_states == states


def test_planned_trajectory_from_agent_reads_private_attributes():
    """Helper reads ``_planned_states`` / ``_planned_actions`` from the agent so the
    private-attribute reach is in one place (easy to swap when upstream grows an
    accessor)."""

    class _StubAgent:
        def __init__(self):
            self._planned_states = ["a", "b", "c"]
            self._planned_actions = [1, 2]

    assert planned_trajectory_from_agent(_StubAgent()) == (["a", "b", "c"], [1, 2])


def test_planned_trajectory_from_agent_returns_empty_when_missing():
    """An agent without a plan (e.g. before reset()) yields empty lists rather than
    raising."""

    class _BareAgent:
        pass

    assert planned_trajectory_from_agent(_BareAgent()) == ([], [])


# ---------------------------------------------------------------------------
# Magic gaps
# ---------------------------------------------------------------------------


def _skill_call(predicted: Any) -> SkillCall:
    robot_type = Type("robot")
    return SkillCall(
        "Pick",
        (Object("robot", robot_type), Object("cylinder0", robot_type)),
        np.array([0.8, 0.0]),
        predicted,
    )


def test_gap_adds_banner_frames_and_lists_gap_in_prompt(tmp_path: Path):
    """A SkillCall action holds the pre-gap frame under a banner for gap_hold_seconds
    and the prompt names the gap; the post-gap (predicted) state is rendered too."""
    states = [object(), object(), object(), object()]
    actions = [np.zeros(11), _skill_call(states[2]), np.zeros(11)]
    shadow = _StubShadowSim(shape=(64, 96))
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    out = preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        planned_actions=actions,
        shadow_sim=shadow,
        log_dir=tmp_path,
        fps=4,
        gap_hold_seconds=1.0,
        prompt_fn=_record,
    )

    assert out is not None
    assert shadow.set_states == states
    cap = cv.VideoCapture(str(out))
    try:
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    # 4 state frames + 4 held banner frames (1 s at 4 fps).
    assert num_frames == 8
    assert len(prompted) == 1
    assert "1 magic gap" in prompted[0]
    assert "step 1: Pick(robot, cylinder0)" in prompted[0]


def test_gap_states_survive_subsampling(tmp_path: Path):
    """Striding never drops the states on either side of a gap."""
    states = list(range(200))
    actions: list[Any] = [np.zeros(11)] * 199
    actions[77] = _skill_call(states[78])
    shadow = _StubShadowSim(shape=(32, 32))

    preview_or_abort(
        planned_states=states,  # type: ignore[arg-type]
        planned_actions=actions,
        shadow_sim=shadow,
        log_dir=tmp_path,
        max_frames=20,
        prompt_fn=lambda _msg: "y",
    )

    assert 77 in shadow.set_states and 78 in shadow.set_states
    assert shadow.set_states[0] == 0 and shadow.set_states[-1] == 199
    assert shadow.set_states == sorted(shadow.set_states)
    assert len(shadow.set_states) <= 24


def test_no_gap_prompt_without_skill_calls(tmp_path: Path):
    """Plain action arrays leave the prompt gap-free."""
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    preview_or_abort(
        planned_states=[object(), object()],  # type: ignore[list-item]
        planned_actions=[np.zeros(11)],
        shadow_sim=_StubShadowSim(),
        log_dir=tmp_path,
        prompt_fn=_record,
    )
    assert "magic gap" not in prompted[0]


def test_rejects_mismatched_action_count(tmp_path: Path):
    """The action list must be one shorter than the state list."""
    with pytest.raises(ValueError, match="planned actions"):
        preview_or_abort(
            planned_states=[object(), object()],  # type: ignore[list-item]
            planned_actions=[np.zeros(11), np.zeros(11)],
            shadow_sim=_StubShadowSim(),
            log_dir=tmp_path,
            prompt_fn=lambda _msg: "y",
        )


def _state_at(x: float, y: float):
    """A planner state with the base at map-frame (x, y)."""
    obs = TidyBotObservation(
        arm_conf=[0.0] * 7,
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=x, y=y, theta=0.0),
        gripper=0.0,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    return PrplLab3DPerceiver().step(obs, {})


_FLOOR = (-1.83, 1.83, -1.83, 1.83)


def test_find_floor_violation_reports_the_first_state_outside_the_margin():
    """The first state whose base is not base_margin inside the floor is reported
    with its index and position; a plan that stays inside gives None."""
    inside = [_state_at(0.0, 0.0), _state_at(1.4, -1.2), _state_at(-1.0, 1.4)]
    assert find_floor_violation(inside, _FLOOR, base_margin=0.37) is None
    # 1.6 is on the floor but less than 0.37 m from its edge.
    plan = inside + [_state_at(1.6, 0.0), _state_at(1.9, 0.0)]
    assert find_floor_violation(plan, _FLOOR, base_margin=0.0) == (4, 1.9, 0.0)
    assert find_floor_violation(plan, _FLOOR, base_margin=0.37) == (3, 1.6, 0.0)


def test_plan_off_the_floor_is_refused_without_prompting(tmp_path: Path):
    """A plan with a base position outside the floor bounds is refused before the
    operator is asked, the preview is still written, and the message names the
    offending state."""
    states = [_state_at(0.0, 0.0), _state_at(1.0, 0.5), _state_at(1.9, -1.4)]
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    with pytest.raises(AgentFailure, match=r"state 2 puts the base at \(1.90, -1.40\)"):
        preview_or_abort(
            planned_states=states,
            shadow_sim=_StubShadowSim(),
            log_dir=tmp_path,
            prompt_fn=_record,
            floor_bounds=_FLOOR,
            base_margin=0.37,
        )
    assert not prompted
    assert (tmp_path / "preview.mp4").exists()


def test_plan_inside_the_floor_is_prompted_as_before(tmp_path: Path):
    """With the bounds set, a plan that stays inside them still goes to the prompt."""
    states = [_state_at(0.0, 0.0), _state_at(1.3, 0.6)]
    prompted: list[str] = []

    def _record(msg: str) -> str:
        prompted.append(msg)
        return "y"

    out = preview_or_abort(
        planned_states=states,
        shadow_sim=_StubShadowSim(),
        log_dir=tmp_path,
        prompt_fn=_record,
        floor_bounds=_FLOOR,
        base_margin=0.37,
    )
    assert out is not None and len(prompted) == 1


def test_predicted_state_after_a_gap_is_checked_too(tmp_path: Path):
    """The predicted post-gap state is part of the plan and is held to the floor
    bounds like any other."""
    outside = _state_at(-1.9, 0.0)
    states = [_state_at(0.0, 0.0), outside, _state_at(0.0, 0.0)]
    actions = [_skill_call(outside), np.zeros(11)]
    with pytest.raises(AgentFailure, match="state 1"):
        preview_or_abort(
            planned_states=states,
            planned_actions=actions,
            shadow_sim=_StubShadowSim(),
            log_dir=tmp_path,
            prompt_fn=lambda _msg: "y",
            floor_bounds=_FLOOR,
            base_margin=0.37,
        )
