"""Replace the model's shelf placements with a teleoperated demonstration.

The refined kinder plan's placement — a level insertion at a model-derived
height — has been awkward on the real robot (cans dropped at an angle, rolled
out). A single teleoperated demonstration of one good top-shelf placement
(recorded by ``scripts/record_place_waypoints.py``) is more reliable, and it
generalises to every can on that board: the demonstrated arm motion is planar
(only joints 2/4/6 move), so the release lands at a fixed offset relative to
the base, and the cans on one board differ only in lateral position. Shifting
the holonomic base laterally by each can's placement x therefore reproduces
the demonstrated placement for each of them.

``rewrite_places_with_demo`` splices the demonstration into a refined
trajectory: each targeted place's base-staging and arm insertion are replaced
by a base drive to the laterally-shifted demonstration pose plus the
demonstrated arm waypoints (densified, gripper held closed, released at the
demonstrated release waypoint, then following any waypoints recorded after
the release as the retract — or reversing the forward path when none were
recorded). Places not in the target set (e.g. a different board) are left
untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from relational_structs import ObjectCentricState

# Densify the demonstrated joint path to this maximum per-step change in any
# single joint (rad), so the executor's carrot has closely spaced targets
# rather than four sparse waypoints it would cut the corners between.
_JOINT_STEP = 0.03
# Base-drive densification: max change per pair in x/y (m) and rotation (rad).
_BASE_STEP = 0.05


def load_demo(path: str | Path) -> dict[str, Any]:
    """Load and validate a place demonstration JSON."""
    demo = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("base_map", "waypoints", "release_index"):
        if key not in demo:
            raise ValueError(f"demo missing {key!r}")
    if not 0 <= demo["release_index"] < len(demo["waypoints"]):
        raise ValueError("demo release_index out of range")
    return demo


def _interpolate_joints(a: Sequence[float], b: Sequence[float]) -> list[list[float]]:
    """Joint configs from ``a`` (exclusive) to ``b`` (inclusive), <= _JOINT_STEP."""
    steps = max(1, int(math.ceil(max(abs(y - x) for x, y in zip(a, b)) / _JOINT_STEP)))
    return [
        [x + (y - x) * (k / steps) for x, y in zip(a, b)] for k in range(1, steps + 1)
    ]


def _base_drive_pairs(
    carry_state: ObjectCentricState,
    target: tuple[float, float, float],
    robot_name: str,
) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
    """Base-only pairs driving from the carry state's base to ``target``."""
    robot = carry_state.get_object_from_name(robot_name)
    start = (
        float(carry_state.get(robot, "pos_base_x")),
        float(carry_state.get(robot, "pos_base_y")),
        float(carry_state.get(robot, "pos_base_rot")),
    )
    d = (target[0] - start[0], target[1] - start[1], target[2] - start[2])
    steps = max(
        1,
        int(math.ceil(max(abs(d[0]), abs(d[1]), abs(d[2]) / 1.0) / _BASE_STEP)),
    )
    pairs = []
    prev = start
    for k in range(1, steps + 1):
        point = (
            start[0] + d[0] * k / steps,
            start[1] + d[1] * k / steps,
            start[2] + d[2] * k / steps,
        )
        state = carry_state.copy()
        state.set(robot, "pos_base_x", prev[0])
        state.set(robot, "pos_base_y", prev[1])
        state.set(robot, "pos_base_rot", prev[2])
        action = np.zeros(11)
        action[0] = point[0] - prev[0]
        action[1] = point[1] - prev[1]
        action[2] = point[2] - prev[2]
        pairs.append((state, action))
        prev = point
    return pairs


def _arm_pairs(
    base: tuple[float, float, float],
    configs: list[list[float]],
    carry_state: ObjectCentricState,
    robot_name: str,
    release_config_index: int,
) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
    """Arm-only pairs walking through ``configs``, base held at ``base``.

    The gripper is held (closed, carried from the grasp) until the pair that
    reaches ``configs[release_config_index]``, which commands an open.
    """
    robot = carry_state.get_object_from_name(robot_name)

    def base_state(joints: Sequence[float], finger: float) -> ObjectCentricState:
        state = carry_state.copy()
        state.set(robot, "pos_base_x", base[0])
        state.set(robot, "pos_base_y", base[1])
        state.set(robot, "pos_base_rot", base[2])
        for j in range(7):
            state.set(robot, f"joint_{j + 1}", float(joints[j]))
        state.set(robot, "finger_state", finger)
        return state

    pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []
    prev = configs[0]
    for i in range(1, len(configs)):
        action = np.zeros(11)
        action[3:10] = np.asarray(configs[i]) - np.asarray(prev)
        if i == release_config_index:
            action[10] = 1.0  # release
        pairs.append((base_state(prev, 0.9), action))
        prev = configs[i]
    return pairs


def rewrite_places_with_demo(
    states: list[ObjectCentricState],
    actions: list[NDArray[np.floating]],
    place_targets_x: list[float | None],
    demo: dict[str, Any],
    fk: Callable[[Sequence[float], tuple[float, float, float]], Sequence[float]],
    robot_name: str = "robot",
    place_delta_z: list[float] | None = None,
    shift_config: Callable[[Sequence[float], float], Sequence[float]] | None = None,
) -> tuple[list[ObjectCentricState], list[NDArray[np.floating]]]:
    """Splice the demonstration into the targeted placements.

    ``place_targets_x`` is one entry per place in trajectory (release) order:
    the desired end-effector map x for that placement, or None to leave it
    unchanged (a place on a different board). ``fk`` maps (7 joints, base SE2
    tuple) to an end-effector map position; it grounds the demonstration's own
    release x so the per-can base shift is exact.

    ``place_delta_z`` (aligned with ``place_targets_x``) raises or lowers a
    place's demonstrated arm configurations by that height, via
    ``shift_config`` (config, delta_z) -> config: the demo defines where the
    gripper goes, so a can that hangs differently below the gripper (a taller
    or shorter object) is placed at the same board height by shifting the
    whole insertion vertically.
    """
    base_map = demo["base_map"]
    demo_base: tuple[float, float, float] = (
        float(base_map[0]),
        float(base_map[1]),
        float(base_map[2]),
    )
    waypoints = [list(map(float, w)) for w in demo["waypoints"]]
    release_index = int(demo["release_index"])
    demo_release_x = float(fk(waypoints[release_index], demo_base)[0])

    # Densified absolute joint configs: the forward path (carry -> release),
    # then the retract. If the demonstration recorded waypoints after the
    # release, those are the demonstrated retract; otherwise the forward path
    # is reversed back to the carry pose.
    forward = [waypoints[0]]
    for i in range(1, release_index + 1):
        forward.extend(_interpolate_joints(forward[-1], waypoints[i]))
    release_config_index = len(forward) - 1
    configs = list(forward)
    if release_index + 1 < len(waypoints):
        for i in range(release_index + 1, len(waypoints)):
            configs.extend(_interpolate_joints(configs[-1], waypoints[i]))
    else:
        for i in range(release_index - 1, -1, -1):
            configs.extend(_interpolate_joints(configs[-1], waypoints[i]))

    release_positions = [i for i, a in enumerate(actions) if float(a[10]) > 0.5]
    if len(release_positions) != len(place_targets_x):
        raise ValueError(
            f"{len(release_positions)} releases but {len(place_targets_x)} "
            "place targets"
        )

    # Each place spans from the base run before its release's arm run through
    # the end of that arm run (the retract), replaced in place. Work back to
    # front so indices stay valid.
    deltas_z = (
        place_delta_z if place_delta_z is not None else [0.0] * len(place_targets_x)
    )
    out_states = list(states)
    out_actions = list(actions)
    for release_pos, target_x, delta_z in reversed(
        list(zip(release_positions, place_targets_x, deltas_z))
    ):
        arm_start = release_pos
        while arm_start > 0 and not _is_base(out_actions[arm_start - 1]):
            arm_start -= 1
        base_start = arm_start
        while base_start > 0 and _is_base(out_actions[base_start - 1]):
            base_start -= 1
        arm_end = release_pos
        while arm_end + 1 < len(out_actions) and not _is_base(out_actions[arm_end + 1]):
            arm_end += 1
        if target_x is None:
            continue
        carry_state = out_states[base_start]
        base_target = (
            demo_base[0] + (target_x - demo_release_x),
            demo_base[1],
            demo_base[2],
        )
        place_configs = configs
        if abs(delta_z) > 1e-4 and shift_config is not None:
            place_configs = [list(shift_config(c, delta_z)) for c in configs]
        # Ramp the arm from the carry pose to the demo's first waypoint rather
        # than commanding it in a single step. The compliant controller cannot
        # track a large jump (joint 2, the shoulder, in particular) and trips a
        # FOLLOWING_ERROR fault; densify this lead-in to _JOINT_STEP like the
        # rest of the demo path. Unwrap the demo path onto the carry pose's
        # branch first, per joint, so no joint whose demo value differs from the
        # carry value by ~2*pi sweeps the long way around.
        robot = carry_state.get_object_from_name(robot_name)
        carry_joints = [
            float(carry_state.get(robot, f"joint_{j + 1}")) for j in range(7)
        ]
        offsets = [
            round((carry_joints[j] - place_configs[0][j]) / (2 * math.pi)) * 2 * math.pi
            for j in range(7)
        ]
        place_configs = [[c[j] + offsets[j] for j in range(7)] for c in place_configs]
        lead_in = _interpolate_joints(carry_joints, place_configs[0])
        full_configs = [carry_joints] + lead_in + place_configs[1:]
        splice_release_index = len(lead_in) + release_config_index
        base_pairs = _base_drive_pairs(carry_state, base_target, robot_name)
        arm_pairs = _arm_pairs(
            base_target, full_configs, carry_state, robot_name, splice_release_index
        )
        new_pairs = base_pairs + arm_pairs
        # The trailing state (arm_end + 1) is preserved as the segment's exit
        # state; splice the new pairs' states/actions over [base_start, arm_end].
        new_states = [s for s, _ in new_pairs] + [out_states[arm_end + 1]]
        new_actions = [a for _, a in new_pairs]
        out_states[base_start : arm_end + 2] = new_states
        out_actions[base_start : arm_end + 1] = new_actions
    return out_states, out_actions


def _is_base(action: NDArray[np.floating]) -> bool:
    return bool(np.any(np.abs(np.asarray(action)[0:3]) > 1e-4))
