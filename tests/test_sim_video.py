"""Tests for `prpl_tidybot.sim_video`: the path check and the video of a sim run."""

import numpy as np
import pytest

from prpl_tidybot.sim_robot import SimWorld
from prpl_tidybot.sim_video import (
    SimMarkers,
    gl_available,
    render_video,
    safety_summary,
)

MARKERS = SimMarkers(
    toss_pose=(0.13, -1.13, 1.5708),
    floor_bounds=(-1.5, 1.7, -1.5, 1.5),
    bin_box=(-0.13, 0.39, 1.45, 1.91),
)


def _hold(world, label, joints_deg, base=None):
    world.label = label
    if base is not None:
        world.base[:] = base
    world.q[:] = np.radians(joints_deg)
    world.advance(0.1)


def test_the_path_check_flags_the_floor_the_base_and_the_joint_stops():
    """Each step reports contact of the arm or gripper with the floor or the base,
    its least clearance above the floor, its least margin to a joint stop, and how
    far the base centre went outside the floor bounds."""
    world = SimWorld(base_pose=(0.0, 0.0, 0.0), cube=(1.0, 0.0, 0.0))
    _hold(world, "retract", [0, -20, 180, -146, 0, -50, 90])
    _hold(world, "into the floor", [125, 113, -10, 65, -1, 7, 72])
    _hold(world, "into the base", [33, 108, -15, 115, 0, -19, 30])
    _hold(world, "off the floor", [0, -20, 180, -146, 0, -50, 90], base=(1.9, 0, 0))
    steps = {s["step"]: s for s in safety_summary(world, MARKERS)["steps"]}
    retract = steps["retract"]
    assert not retract["contacts"]
    assert retract["min_floor_clearance_m"]["gripper"] > 0.3
    assert retract["min_joint_limit_margin_deg"]["joint"] == 4
    assert retract["min_joint_limit_margin_deg"]["margin"] == pytest.approx(
        1.3, abs=0.1
    )
    floor = steps["into the floor"]
    assert any("world" in pair for pair in floor["contacts"])
    assert floor["min_floor_clearance_m"]["gripper"] <= 0
    assert any("base_link" in pair for pair in steps["into the base"]["contacts"])
    assert steps["off the floor"]["base_outside_floor_bounds_m"] == pytest.approx(0.2)


@pytest.mark.skipif(not gl_available(), reason="no OpenGL context for MuJoCo")
def test_the_swing_plays_in_slow_motion(tmp_path):
    """The video plays the clock in real time, and the swings `slowdown` times
    slower."""
    import imageio.v3 as iio  # pylint: disable=import-outside-toplevel

    world = SimWorld(base_pose=(0.0, 0.0, 0.0), cube=(1.0, 0.0, 0.0))
    world.label = "drive"
    world.advance(1.0)
    world.slow_spans.append((world.t, world.t + 1.0))
    world.advance(1.0)
    path = render_video(world, MARKERS, tmp_path / "v.mp4", fps=10, slowdown=4)
    frames = sum(1 for _ in iio.imiter(path))
    assert frames == pytest.approx(10 + 40, abs=2)
