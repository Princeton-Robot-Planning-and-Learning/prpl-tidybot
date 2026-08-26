"""Pause-and-confirm plan preview for the planner pipeline.

`preview_or_abort` is intended to run AFTER the agent has produced a
plan and BEFORE the executor starts driving the real env. It walks the
planned trajectory through a shadow sim, composes an mp4 of the per-
state renders, and blocks on stdin for operator approval. If the
operator rejects the plan it raises ``kinder_bilevel_planning.agent.
AgentFailure`` so the Runner exits cleanly without any motion being
commanded.

The preview is a gist-check, not a replay: by default the planned
trajectory is subsampled to at most `max_frames` states (first and last
always kept) before rendering, since the per-state shadow-sim render is
the dominant cost of the preview (#92).

This is a deliberately small first cut — frames come from the shadow
sim only (no real-env panel), and approval is plain stdin. Side-by-
side comparison with real frames, alternative approval channels, and
inline replan-without-restart can land in follow-ups.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from kinder_bilevel_planning.agent import AgentFailure
from relational_structs import ObjectCentricState

from prpl_tidybot.video import write_mp4

# Default prompt indirection so tests can swap a fake stdin without
# monkeypatching the `input` builtin.
PromptFn = Callable[[str], str]

_logger = logging.getLogger(__name__)


def preview_or_abort(
    planned_states: list[ObjectCentricState],
    shadow_sim,
    log_dir: Path | str,
    seed: int = 0,
    fps: int = 10,
    max_frames: int | None = 50,
    prompt_fn: PromptFn = input,
) -> Path | None:
    """Render `planned_states` through `shadow_sim`, save an mp4, prompt for approval.

    Returns the path to the written preview (``log_dir/preview.mp4``) on
    approval, or ``None`` if there was nothing to preview. Raises
    :class:`AgentFailure` if the operator rejects the plan. Any other
    response is treated as rejection.

    When the trajectory is longer than `max_frames`, it is strided down
    to at most that many states (plus the final state, which is always
    kept). Pass ``max_frames=None`` to render every state.

    `shadow_sim` is reset once here (with `seed`) so the first
    `set_state` lands on a clean env — same protocol the recorder uses.
    The caller is responsible for instantiating the sim; reusing the
    recorder's shadow sim is fine since `set_state` is idempotent.
    """
    if not planned_states:
        return None
    preview_states = _subsample(planned_states, max_frames)
    _logger.info(
        "Rendering plan preview (%d of %d planned states)...",
        len(preview_states),
        len(planned_states),
    )
    render_start = time.monotonic()
    shadow_sim.reset(seed=seed)
    frames: list[np.ndarray] = []
    for state in preview_states:
        shadow_sim.set_state(state)
        frame = shadow_sim.render()
        if frame is None:
            continue
        frames.append(np.asarray(frame, dtype=np.uint8))
    render_time = time.monotonic() - render_start
    if not frames:
        return None
    out_path = Path(log_dir) / "preview.mp4"
    encode_start = time.monotonic()
    write_mp4(frames, out_path, fps=fps)
    encode_time = time.monotonic() - encode_start
    _logger.info(
        "Preview: rendered %d/%d planned states in %.2fs, encoded in %.2fs.",
        len(preview_states),
        len(planned_states),
        render_time,
        encode_time,
    )
    answer = (
        prompt_fn(
            f"\nPlan preview written to {out_path}.\n" "Approve and execute? [y/N]: "
        )
        .strip()
        .lower()
    )
    if answer not in ("y", "yes"):
        raise AgentFailure(f"Plan preview rejected by operator (answer={answer!r})")
    return out_path


def _subsample(
    states: list[ObjectCentricState], max_frames: int | None
) -> list[ObjectCentricState]:
    """Stride `states` down to at most `max_frames` entries, keeping the first and last
    state; the result may hold one extra frame when the stride does not land on the
    final state."""
    if max_frames is None or len(states) <= max_frames:
        return states
    if max_frames < 1:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    stride = math.ceil(len(states) / max_frames)
    subsampled = states[::stride]
    if (len(states) - 1) % stride != 0:
        subsampled.append(states[-1])
    return subsampled


def planned_states_from_agent(agent) -> list[ObjectCentricState]:
    """Pull the planned-state sequence out of a `BilevelPlanningAgent`.

    Isolated in one place so the private-attribute reach is easy to find and replace if
    the upstream agent grows a public accessor.
    """
    states = getattr(agent, "_planned_states", None)
    if states is None:
        return []
    return list(states)
