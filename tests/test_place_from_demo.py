"""Tests for place_from_demo.py."""

import math

import numpy as np
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DEnvTypeFeatures,
    Kinematic3DRobotType,
)
from relational_structs import Object
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.place_from_demo import rewrite_places_with_demo

_ROBOT = Object("robot", Kinematic3DRobotType)
_FEATS = Kinematic3DEnvTypeFeatures[Kinematic3DRobotType]


def _robot_state(base, joints, finger=0.9):
    values = {f: 0.0 for f in _FEATS}
    values["pos_base_x"], values["pos_base_y"], values["pos_base_rot"] = base
    for j in range(7):
        values[f"joint_{j + 1}"] = joints[j]
    values["finger_state"] = finger
    return create_state_from_dict(
        {_ROBOT: values}, {Kinematic3DRobotType: list(_FEATS)}
    )


def _fk(joints, base):
    # A stand-in forward kinematics: the map release x is the base x plus a
    # fixed reach, so the per-can base shift is exact and simple to assert.
    return (base[0] + 0.5, base[1], 0.9)


def _arm_action(delta7, release=False):
    a = np.zeros(11)
    a[3:10] = delta7
    if release:
        a[10] = 1.0
    return a


def test_rewrite_splices_demo_and_reaches_release():
    """The splice replaces the place with the demonstrated arm motion: exactly
    one release, and summing the arm deltas from the demo's first waypoint
    reaches the demonstrated release waypoint."""
    demo_w0 = [0.0, 1.7, 0.0, 1.0, 0.0, 0.3, 0.0]
    demo_release = [0.0, 1.9, 0.0, 1.2, 0.0, 0.3, 0.0]
    demo = {
        "base_map": [1.0, 0.0, 0.0],
        "waypoints": [demo_w0, demo_release],
        "release_index": 1,
    }

    carry_state = _robot_state((0.9, 0.0, 0.0), [0.0, 0.5, 0.0, 1.0, 0.0, 0.3, 0.0])
    states = [carry_state, carry_state, carry_state]
    actions = [_arm_action([0.0] * 7), _arm_action([0.0] * 7, release=True)]

    _, out_actions = rewrite_places_with_demo(
        states, actions, place_targets_x=[1.5], demo=demo, fk=_fk
    )

    releases = sum(int(float(a[10]) > 0.5) for a in out_actions)
    assert releases == 1

    # The arm walk starts at the demo's first waypoint; accumulating the deltas
    # up to the release action reaches the demonstrated release waypoint (the
    # demo defines the placement, no lead-in added). After the release the walk
    # retraces to the start, so only check up to the release.
    joints = np.array(demo_w0)
    reached_release_at_open = False
    for a in out_actions:
        if np.any(np.abs(np.asarray(a)[0:3]) > 1e-4):
            continue
        joints = joints + np.asarray(a)[3:10]
        if float(a[10]) > 0.5:
            reached_release_at_open = np.allclose(joints, demo_release, atol=1e-6)
    assert reached_release_at_open, joints


def test_rewrite_unwraps_demo_onto_carry_branch():
    """When a joint's demo value differs from the carry value by ~2*pi, the demo
    path is unwrapped onto the carry pose's branch, so the executor's transition
    from the carry pose to the first waypoint does not sweep the long way
    around: the first arm target stays within pi of the carry pose on that
    joint."""
    carry_joints = [3.0, 0.5, 0.0, 1.0, 0.0, 0.3, 0.0]
    # Same physical joint-1 angle as carry, but expressed ~2*pi lower.
    demo_w0 = [3.0 - 2 * math.pi, 0.6, 0.0, 1.0, 0.0, 0.3, 0.0]
    demo_release = [3.0 - 2 * math.pi, 0.7, 0.0, 1.0, 0.0, 0.3, 0.0]
    demo = {
        "base_map": [1.0, 0.0, 0.0],
        "waypoints": [demo_w0, demo_release],
        "release_index": 1,
    }
    carry_state = _robot_state((0.9, 0.0, 0.0), carry_joints)
    states = [carry_state, carry_state, carry_state]
    actions = [_arm_action([0.0] * 7), _arm_action([0.0] * 7, release=True)]

    out_states, out_actions = rewrite_places_with_demo(
        states, actions, place_targets_x=[1.5], demo=demo, fk=_fk
    )

    # The first arm pair's state holds the (unwrapped) first target; joint 1 must
    # be on the carry branch (~3.0), not ~2*pi away at the raw demo value.
    robot = out_states[0].get_object_from_name("robot")
    first_arm_state = next(
        s
        for s, a in zip(out_states, out_actions)
        if not np.any(np.abs(np.asarray(a)[0:3]) > 1e-4)
    )
    j1 = float(first_arm_state.get(robot, "joint_1"))
    assert abs(j1 - 3.0) <= math.pi, j1
