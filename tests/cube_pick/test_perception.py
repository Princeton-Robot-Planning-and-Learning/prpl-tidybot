"""Tests for the perception helpers that need no learned model."""

import numpy as np

from prpl_tidybot.cube_pick.camera import PinholeIntrinsics, RGBDFrame
from prpl_tidybot.cube_pick.perception import CubePrior, depth_prompt


def _looking_down(height: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])  # camera z down, x along world x
    pose[:3, 3] = [0.0, 0.0, height]
    return pose


def test_depth_prompt_finds_the_cube_when_the_prior_is_off():
    intr = PinholeIntrinsics.from_diagonal_fov(
        width=320, height=240, diagonal_fov_deg=67
    )
    depth = np.full((240, 320), 0.40, np.float32)
    # A 5 cm cube centred at world (0.03, 0.0): its top face 5 cm nearer the camera.
    frame = RGBDFrame(
        np.zeros((240, 320, 3), np.uint8), depth, intr, _looking_down(0.40)
    )
    uv, _ = frame.project_world(np.array([[0.005, -0.025, 0.05], [0.055, 0.025, 0.05]]))
    u0, u1 = sorted(uv[:, 0].astype(int))
    v0, v1 = sorted(uv[:, 1].astype(int))
    depth[v0:v1, u0:u1] = 0.35
    prior = CubePrior(position=np.array([0.0, 0.0, 0.025]), size=0.05, sigma=0.05)
    centre, box = depth_prompt(frame, prior)
    assert u0 <= centre[0] <= u1 and v0 <= centre[1] <= v1
    assert box[0] <= u0 + 2 and box[2] >= u1 - 2


def test_depth_prompt_gives_up_on_a_flat_view():
    intr = PinholeIntrinsics.from_diagonal_fov(
        width=320, height=240, diagonal_fov_deg=67
    )
    depth = np.full((240, 320), 0.40, np.float32)
    frame = RGBDFrame(
        np.zeros((240, 320, 3), np.uint8), depth, intr, _looking_down(0.40)
    )
    prior = CubePrior(position=np.array([0.0, 0.0, 0.025]), size=0.05)
    assert depth_prompt(frame, prior) is None
