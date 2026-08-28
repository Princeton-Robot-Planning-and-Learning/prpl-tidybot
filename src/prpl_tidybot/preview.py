"""Pause-and-confirm plan preview for the planner pipeline.

`preview_or_abort` is intended to run AFTER the agent has produced a
plan and BEFORE the executor starts driving the real env. It walks the
planned trajectory through a shadow sim, composes an mp4 of the per-
state renders, and blocks on stdin for operator approval. If the
operator rejects the plan it raises ``kinder_bilevel_planning.agent.
AgentFailure`` so the Runner exits cleanly without any motion being
commanded.

Magic skills leave gaps in the plan: a ``SkillCall`` action whose
policy the planner did not simulate, followed by the state its option
model predicted. The preview makes each gap explicit. The state before
the gap is held for ``gap_hold_seconds`` under a banner naming the
skill, the predicted post-gap state follows, and the approval prompt
lists every gap, so the operator knows which parts of the plan will be
carried out by something other than the planned motion.

The preview is a gist-check, not a replay: by default the planned
trajectory is subsampled to at most `max_frames` states before
rendering, since the per-state shadow-sim render is the dominant cost
of the preview (#92). The first and last states and the states on
either side of a gap are always kept.

Frames come from the shadow sim only (no real-env panel), and approval
is plain stdin. Side-by-side comparison with real frames, alternative
approval channels, and inline replan-without-restart can land in
follow-ups.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from kinder_bilevel_planning.agent import AgentFailure
from kinder_models.structs import SkillCall
from prpl_utils.utils import render_textbox_on_image
from relational_structs import ObjectCentricState

from prpl_tidybot.video import subsample_indices, write_mp4

# Default prompt indirection so tests can swap a fake stdin without
# monkeypatching the `input` builtin.
PromptFn = Callable[[str], str]

_logger = logging.getLogger(__name__)

_GAP_BANNER_COLOR = (180, 30, 30, 255)


def preview_or_abort(
    planned_states: Sequence[ObjectCentricState],
    shadow_sim,
    log_dir: Path | str,
    seed: int = 0,
    fps: int = 10,
    max_frames: int | None = 50,
    prompt_fn: PromptFn = input,
    planned_actions: Sequence[Any] | None = None,
    gap_hold_seconds: float = 1.0,
) -> Path | None:
    """Render the plan through `shadow_sim`, save an mp4, prompt for approval.

    ``planned_actions`` (one fewer than ``planned_states``; action ``i``
    leads from state ``i`` to state ``i + 1``) is scanned for
    ``SkillCall`` entries. Each one is a magic gap: the preview holds the
    pre-gap state under a banner for ``gap_hold_seconds``, then shows the
    predicted post-gap state, and the prompt lists the gap. With
    ``planned_actions=None`` the plan is previewed as gap-free.

    Returns the path to the written preview (``log_dir/preview.mp4``) on
    approval, or ``None`` if there was nothing to preview. Raises
    :class:`AgentFailure` if the operator rejects the plan. Any other
    response is treated as rejection.

    When the trajectory is longer than `max_frames`, it is strided down
    to about that many states; the first and last states and the states
    on either side of every gap are always kept. Pass ``max_frames=None``
    to render every state.

    `shadow_sim` is reset once here (with `seed`) so the first
    `set_state` lands on a clean env — same protocol the recorder uses.
    The caller is responsible for instantiating the sim; reusing the
    recorder's shadow sim is fine since `set_state` is idempotent.
    """
    if not planned_states:
        return None
    gaps = _find_gaps(planned_states, planned_actions)
    keep: set[int] = set()
    for index in gaps:
        keep.update({index, index + 1})
    selected = subsample_indices(len(planned_states), max_frames, keep)
    _logger.info(
        "Rendering plan preview (%d of %d planned states, %d magic gap(s))...",
        len(selected),
        len(planned_states),
        len(gaps),
    )
    render_start = time.monotonic()
    shadow_sim.reset(seed=seed)
    hold_frames = max(1, int(round(fps * gap_hold_seconds)))
    frames: list[np.ndarray] = []
    for index in selected:
        shadow_sim.set_state(planned_states[index])
        frame = shadow_sim.render()
        if frame is None:
            continue
        frame = np.asarray(frame, dtype=np.uint8)
        frames.append(frame)
        if index in gaps:
            banner = _gap_banner(frame, gaps[index], index)
            frames.extend([banner] * hold_frames)
    render_time = time.monotonic() - render_start
    if not frames:
        return None
    out_path = Path(log_dir) / "preview.mp4"
    encode_start = time.monotonic()
    write_mp4(frames, out_path, fps=fps)
    encode_time = time.monotonic() - encode_start
    _logger.info(
        "Preview: rendered %d/%d planned states in %.2fs, encoded in %.2fs.",
        len(selected),
        len(planned_states),
        render_time,
        encode_time,
    )
    answer = prompt_fn(_prompt_text(out_path, gaps)).strip().lower()
    if answer not in ("y", "yes"):
        raise AgentFailure(f"Plan preview rejected by operator (answer={answer!r})")
    return out_path


def _find_gaps(
    planned_states: Sequence[ObjectCentricState],
    planned_actions: Sequence[Any] | None,
) -> dict[int, SkillCall]:
    """Map the index of each pre-gap state to the SkillCall leaving it."""
    if planned_actions is None:
        return {}
    if len(planned_actions) != len(planned_states) - 1:
        raise ValueError(
            f"Expected {len(planned_states) - 1} planned actions for "
            f"{len(planned_states)} planned states, got {len(planned_actions)}."
        )
    return {
        index: action
        for index, action in enumerate(planned_actions)
        if isinstance(action, SkillCall)
    }


def _gap_banner(frame: np.ndarray, call: SkillCall, index: int) -> np.ndarray:
    """Overlay a banner naming the magic skill on a copy of `frame`."""
    text = f"MAGIC GAP at step {index}: {call}\n(not simulated; executed externally)"
    return np.asarray(
        render_textbox_on_image(
            frame.copy(),
            text,
            top_offset_frac=0.35,
            bottom_offset_frac=0.35,
            left_offset_frac=0.05,
            right_offset_frac=0.05,
            textbox_color=_GAP_BANNER_COLOR,
        ),
        dtype=np.uint8,
    )


def _prompt_text(out_path: Path, gaps: dict[int, SkillCall]) -> str:
    lines = [f"\nPlan preview written to {out_path}."]
    if gaps:
        lines.append(
            f"The plan has {len(gaps)} magic gap(s) that will be carried out "
            "outside the planned motion:"
        )
        for index, call in sorted(gaps.items()):
            lines.append(f"  - step {index}: {call}")
    lines.append("Approve and execute? [y/N]: ")
    return "\n".join(lines)


def planned_trajectory_from_agent(
    agent,
) -> tuple[list[ObjectCentricState], list[Any]]:
    """Pull the planned states and actions out of a `BilevelPlanningAgent`.

    Isolated in one place so the private-attribute reach is easy to find and replace if
    the upstream agent grows a public accessor. Returns empty lists before the agent has
    planned.
    """
    states = getattr(agent, "_planned_states", None)
    actions = getattr(agent, "_planned_actions", None)
    if states is None or actions is None:
        return [], []
    return list(states), list(actions)
