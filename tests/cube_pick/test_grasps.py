"""Tests for the analytic cube grasp sampler."""

import numpy as np
import pytest

from prpl_tidybot.cube_pick.grasps import (
    CubeGraspSampler,
    nearest_equivalent_yaw,
    upright_cube_frame,
)


def test_straight_down_candidates_come_first_and_point_down():
    sampler = CubeGraspSampler(cube_size=0.05)
    first = sampler.candidates()[0]
    assert first.tilt == 0.0
    tool = first.tool_pose(upright_cube_frame(np.array([1.0, 2.0, 0.045]), 0.0))
    assert np.allclose(tool[:3, 2], [0, 0, -1])
    # Pads' centre 1 cm above the cube centre; the tool frame 5.2 mm behind it.
    assert tool[2, 3] == pytest.approx(0.045 + 0.010 + 0.0052)
    assert np.allclose(tool[:2, 3], [1.0, 2.0])


def test_tilts_rotate_about_the_closing_axis():
    sampler = CubeGraspSampler(cube_size=0.05, tilts_deg=(0.0, 30.0))
    for cand in sampler.candidates():
        down = sampler._candidate(cand.yaw_offset, 0.0, cand.opening, cand.depth)
        # The closing axis (tool x) is the same as for the straight-down grasp, so the
        # pads stay parallel to the faces they close on.
        assert np.allclose(cand.cube_from_tool[:3, 0], down.cube_from_tool[:3, 0])
        angle = np.arccos(np.clip(-cand.cube_from_tool[2, 2], -1, 1))
        assert angle == pytest.approx(abs(cand.tilt), abs=1e-9)


def test_tilted_candidates_keep_the_pads_at_the_same_point():
    sampler = CubeGraspSampler(cube_size=0.05)
    pinches = set()
    for cand in sampler.candidates():
        pinch = cand.cube_from_tool[:3, 3] + cand.cube_from_tool[:3, 2] * 0.0052
        pinches.add(tuple(np.round(pinch, 9)))
    assert pinches == {(0.0, 0.0, 0.01), (0.0, 0.0, 0.016)}


def test_openings_are_partial_and_ordered():
    cands = CubeGraspSampler(cube_size=0.05).candidates()
    openings = [c.opening for c in cands if c.tilt == 0.0]
    assert openings[0] == pytest.approx(0.062)
    assert max(openings) < 0.085


def test_every_wall_direction_has_a_lean_away():
    sampler = CubeGraspSampler(cube_size=0.05, tilts_deg=(20.0,))
    leans = set()
    for cand in sampler.candidates():
        # Where the wrist goes: back along the approach axis, projected horizontally.
        back = -cand.cube_from_tool[:3, 2]
        leans.add(tuple(np.round(np.sign(np.round(back[:2], 6)), 0)))
    assert {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)} <= leans


def test_standoff_is_back_along_the_approach():
    cand = CubeGraspSampler(cube_size=0.05, tilts_deg=(20.0,)).candidates()[0]
    frame = upright_cube_frame(np.zeros(3), 0.3)
    tool = cand.tool_pose(frame)
    standoff = cand.standoff_pose(frame, 0.1)
    assert np.linalg.norm(standoff[:3, 3] - tool[:3, 3]) == pytest.approx(0.1)
    assert (tool[:3, 3] - standoff[:3, 3]) @ tool[:3, 2] == pytest.approx(0.1)


def test_nearest_equivalent_yaw():
    assert nearest_equivalent_yaw(0.1, 1.6) == pytest.approx(0.1 + np.pi / 2)
    assert nearest_equivalent_yaw(0.1, -3.0) == pytest.approx(0.1 - np.pi)


def test_panda_grasps_map_to_the_same_pads_centre_and_axes():
    from prpl_tidybot.cube_pick.learned_grasps import panda_to_tool

    panda = np.eye(4)
    panda[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    panda[:3, 3] = [0.3, 0.1, 0.2]
    tool = panda_to_tool(panda)
    assert np.allclose(tool[:3, :3], panda[:3, :3])
    pads_panda = panda[:3, 3] + panda[:3, 2] * 0.1034
    pads_tool = tool[:3, 3] + tool[:3, 2] * 0.0052
    assert np.allclose(pads_panda, pads_tool)
