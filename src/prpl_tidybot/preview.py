"""Pause-and-confirm plan preview for the planner pipeline.

`preview_or_abort` is intended to run AFTER the agent has produced a
plan and BEFORE the executor starts driving the real env. It walks the
planned trajectory through a shadow sim, composes an mp4 of the per-
state renders, and blocks on stdin for operator approval. If the
operator rejects the plan it raises ``kinder_bilevel_planning.agent.
AgentFailure`` so the Runner exits cleanly without any motion being
commanded.

This is a deliberately small first cut — frames come from the shadow
sim only (no real-env panel), and approval is plain stdin. Side-by-
side comparison with real frames, alternative approval channels, and
inline replan-without-restart can land in follow-ups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from kinder_bilevel_planning.agent import AgentFailure
from moviepy import ImageSequenceClip
from relational_structs import ObjectCentricState

# Default prompt indirection so tests can swap a fake stdin without
# monkeypatching the `input` builtin.
PromptFn = Callable[[str], str]


def preview_or_abort(
    planned_states: list[ObjectCentricState],
    shadow_sim,
    log_dir: Path | str,
    seed: int = 0,
    fps: int = 10,
    prompt_fn: PromptFn = input,
) -> Path | None:
    """Render `planned_states` through `shadow_sim`, save an mp4, prompt for approval.

    Returns the path to the written preview (``log_dir/preview.mp4``) on
    approval, or ``None`` if there was nothing to preview. Raises
    :class:`AgentFailure` if the operator rejects the plan. Any other
    response is treated as rejection.

    `shadow_sim` is reset once here (with `seed`) so the first
    `set_state` lands on a clean env — same protocol the recorder uses.
    The caller is responsible for instantiating the sim; reusing the
    recorder's shadow sim is fine since `set_state` is idempotent.
    """
    if not planned_states:
        return None
    shadow_sim.reset(seed=seed)
    frames: list[np.ndarray] = []
    for state in planned_states:
        shadow_sim.set_state(state)
        frame = shadow_sim.render()
        if frame is None:
            continue
        frames.append(np.asarray(frame, dtype=np.uint8))
    if not frames:
        return None
    out_path = Path(log_dir) / "preview.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `moviepy` wants BGR-style frames written via its own encoder; we
    # already have RGB from the kinder env, so feed straight in.
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(out_path), logger=None)
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


def planned_states_from_agent(agent) -> list[ObjectCentricState]:
    """Pull the planned-state sequence out of a `BilevelPlanningAgent`.

    Isolated in one place so the private-attribute reach is easy to find
    and replace if the upstream agent grows a public accessor.
    """
    states = getattr(agent, "_planned_states", None)
    if states is None:
        return []
    return list(states)
