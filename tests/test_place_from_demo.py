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


def test_rewrite_densifies_carry_to_first_waypoint():
    """The carry pose to the demo's first waypoint is walked in steps no larger
    than the densification limit, so the compliant controller never gets a single
    large jump (the FOLLOWING_ERROR cause)."""
    carry_joints = [0.0, 0.5, 0.0, 1.0, 0.0, 0.3, 0.0]
    # The demo's first waypoint is far from the carry pose on joint 2 (index 1).
    demo_w0 = [0.0, 1.7, 0.0, 1.0, 0.0, 0.3, 0.0]
    demo_release = [0.0, 1.9, 0.0, 1.2, 0.0, 0.3, 0.0]
    demo = {
        "base_map": [1.0, 0.0, 0.0],
        "waypoints": [demo_w0, demo_release],
        "release_index": 1,
    }

    # A trajectory whose single place is a carry state, one arm step, a release.
    carry_state = _robot_state((0.9, 0.0, 0.0), carry_joints)
    states = [carry_state, carry_state, carry_state]
    actions = [
        _arm_action([0.0] * 7),
        _arm_action([0.0] * 7, release=True),
    ]

    out_states, out_actions = rewrite_places_with_demo(
        states,
        actions,
        place_targets_x=[1.5],
        demo=demo,
        fk=_fk,
    )

    # Every arm step in the spliced result stays within the densification limit.
    max_step = 0.0
    releases = 0
    for a in out_actions:
        arm = np.asarray(a)[3:10]
        base = np.asarray(a)[0:3]
        if np.any(np.abs(base) > 1e-4):
            continue  # base drive pair
        max_step = max(max_step, float(np.max(np.abs(arm))))
        releases += int(float(a[10]) > 0.5)
    assert max_step <= 0.03 + 1e-9, max_step
    assert releases == 1

    # The arm actually starts from the carry pose: summing arm deltas from the
    # carry joints reaches the demo's first waypoint, then the release waypoint.
    joints = np.array(carry_joints)
    reached_w0 = False
    for a in out_actions:
        if np.any(np.abs(np.asarray(a)[0:3]) > 1e-4):
            continue
        joints = joints + np.asarray(a)[3:10]
        if np.allclose(joints, demo_w0, atol=1e-6):
            reached_w0 = True
    assert reached_w0, "lead-in never lands on the demo's first waypoint"


def test_rewrite_unwraps_lead_in_to_carry_branch():
    """When a joint's demo value differs from the carry value by ~2*pi, the
    lead-in is unwrapped onto the carry branch instead of sweeping the long way
    around: no intermediate command is more than pi from the carry pose."""
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

    joints = np.array(carry_joints)
    for a in out_actions:
        if np.any(np.abs(np.asarray(a)[0:3]) > 1e-4):
            continue
        joints = joints + np.asarray(a)[3:10]
        # Joint 1 never departs from the carry branch by more than a step.
        assert abs(joints[0] - 3.0) <= 0.03 + 1e-9, joints[0]
