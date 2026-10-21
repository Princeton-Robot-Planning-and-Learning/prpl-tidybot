"""Tests for the occlusion-aware cube detector (``prpl_tidybot.cube_detector``) and the
shared look (``CeilingCubeDetectorSource.look``), on real ceiling frames from the
2026-10-20/21 runs and on a synthetic bin.

Each file in ``tests/data/cube_detector`` is one saved frame set reduced to the median
of its last ten stitched frames (top camera over bottom, 1280 x 1440), a quality-95
JPEG:

* ``corner_8_10-29-48_53`` and ``corner_1_10-29-48_29``: the 2026-10-21 run's two
  evaluation tosses that scored but were reported as "no cube found": the cube rests
  in the far bin's -x corner against its wall, in the wall's shadow. The positions are
  eye-labelled (the top face's centre pixel in the top camera, at the face's height).
* ``held_16-24-41_00``: the cube held in the closed gripper, the arm stretched out
  past the robot's marker exclusion.
* ``floor_17-24-13_00``: a cube on the floor, which the c49c24c localizer fitted at
  (-0.824, -0.559).
"""

import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bin_localizer import BinFit, CeilingBinSource
from prpl_tidybot.cube_detector import (
    CeilingCubeDetectorSource,
    camera_centre,
    detect_cube,
    occluder_mask,
    occluding_walls,
)
from prpl_tidybot.cube_localizer import CeilingCamera, load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")
DATA = Path(__file__).resolve().parent / "data" / "cube_detector"
FLOOR = (-1.846, 2.046, -1.846, 1.846)
SIDE = 0.05
# The eye-labelled cube centres (top camera pixel (662, 224) and (659.5, 229)).
CORNER_8 = (0.0725, 1.3946)
CORNER_1 = (0.0648, 1.3798)
# The robot's marker pose in each set (the localizer's exclusion square).
ROBOT = {
    "corner_8_10-29-48_53": (0.533, -0.652),
    "corner_1_10-29-48_29": (0.536, -0.465),
    "held_16-24-41_00": (0.044, -0.130),
    "floor_17-24-13_00": (0.753, -0.718),
}


def _frame(name):
    frame = cv.imread(str(DATA / f"{name}.jpg"))
    assert frame is not None and frame.shape == (1440, 1280, 3)
    return frame


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    return load_ceiling_cameras(SERIALS)


class _FakeImages:
    """Serves a fresh copy of one stitched frame per call, counting the calls."""

    def __init__(self, frame):
        self._frame = frame
        self.calls = 0

    def get_image(self):
        """A copy of the frame."""
        self.calls += 1
        return self._frame.copy()

    def close(self):
        """Nothing to release."""


class _FakeMarkers:
    """A marker detector that only reports the robot's pose."""

    def __init__(self, robot_xy):
        self._robot = robot_xy

    def get_latest(self):
        """A payload with the robot pose and no targets."""
        return {"poses": {0: (self._robot[0], self._robot[1], 0.0)}, "targets": {}}

    def close(self):
        """Nothing to release."""


class _NoImages:
    """An image client the bin source must never read in a shared look."""

    def get_image(self):
        """Fails: the look grabs once, through the cube source."""
        raise AssertionError("the bin source grabbed its own frames")

    def close(self):
        """Nothing to release."""


def _sources(name, look_frames=3):
    frame = _frame(name)
    images = _FakeImages(frame)
    cube = CeilingCubeDetectorSource(
        SERIALS,
        image_client=images,
        marker_client=_FakeMarkers(ROBOT[name]),
        floor_bounds=FLOOR,
        frame_period_s=0.0,
        look_frames=look_frames,
        threads=2,
    )
    bin_source = CeilingBinSource(
        SERIALS,
        image_client=_NoImages(),
        marker_client=_FakeMarkers(ROBOT[name]),
        length=0.39,
        width=0.29,
        height=0.21,
        wall=0.005,
        method="rim_edges",
    )
    return cube, bin_source, images


@pytest.mark.parametrize(
    "name, spot",
    [("corner_8_10-29-48_53", CORNER_8), ("corner_1_10-29-48_29", CORNER_1)],
)
def test_a_cube_in_the_bins_corner_is_found_in_the_bin(name, spot):
    """The two scored tosses the c49c24c look reported as "no cube found": found, inside
    the bin, within 3 cm of where the eye puts them."""
    cube, bin_source, _ = _sources(name)
    look = cube.look(bin_source)
    assert look.bin_fit is not None
    assert look.cube is not None
    assert look.cube.in_bin
    assert math.hypot(look.cube.x - spot[0], look.cube.y - spot[1]) < 0.03


def test_a_held_cube_is_no_cube():
    """The cube in the closed gripper, past the robot's marker exclusion, is not
    reported as a cube on the floor."""
    cube, bin_source, _ = _sources("held_16-24-41_00")
    assert cube.look(bin_source).cube is None


def test_a_floor_cube_is_where_the_floor_localizer_put_it():
    """A cube on the floor fits within 1 cm of the c49c24c floor localizer's pose, and
    is not in the bin."""
    cube, bin_source, _ = _sources("floor_17-24-13_00")
    look = cube.look(bin_source)
    assert look.cube is not None and not look.cube.in_bin
    assert math.hypot(look.cube.x + 0.824, look.cube.y + 0.559) < 0.01


def test_the_standalone_localizer_runs_the_same_detector():
    """get_target (no bin known) finds the floor cube too."""
    cube, _, _ = _sources("floor_17-24-13_00")
    x, y, _ = cube.get_target()
    assert math.hypot(x + 0.824, y + 0.559) < 0.01


def test_one_look_grabs_its_frames_once_for_the_bin_and_the_cube():
    """The bin and the cube come from one grab of look_frames frames (the bin source
    never reads its own client)."""
    cube, bin_source, images = _sources("corner_8_10-29-48_53", look_frames=4)
    look = cube.look(bin_source)
    assert images.calls == 4
    assert look.num_frames == 4
    assert look.bin_fit is not None and look.cube is not None


def test_a_missed_look_saves_its_frames_with_the_bin(tmp_path):
    """A look with no cube saves its frames and the bin it was scored against."""
    frame = _frame("held_16-24-41_00")
    cube = CeilingCubeDetectorSource(
        SERIALS,
        image_client=_FakeImages(frame),
        marker_client=_FakeMarkers(ROBOT["held_16-24-41_00"]),
        floor_bounds=FLOOR,
        frame_period_s=0.0,
        look_frames=2,
        failure_dir=tmp_path,
        retries=0,
    )
    _, bin_source, _ = _sources("held_16-24-41_00")
    assert cube.look(bin_source).cube is None
    (saved,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(list(saved.glob("frame_*.png"))) == 2
    info = __import__("json").loads((saved / "localization.json").read_text())
    assert info["bin"] is not None and "candidates" in info


# --------------------------------------------------------------- synthetic bin


def _camera_at(x, y, z):
    """A pinhole camera at (x, y, z) looking straight down (image u along +x, v along
    -y)."""
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    rvec = cv.Rodrigues(rot)[0].reshape(3)
    tvec = -rot @ np.array([x, y, z])
    k = np.array([[770.0, 0.0, 640.0], [0.0, 770.0, 360.0], [0.0, 0.0, 1.0]])
    return CeilingCamera(k, rvec, tvec)


SYNTH_BIN = BinFit(0.0, 0.0, 0.0, 0.39, 0.29, 0.21, 0.005)


def _hidden_by_slab(cam_pos, point, lo, hi):
    """Whether the segment camera -> point passes through the box [lo, hi]."""
    d = point - cam_pos
    t0, t1 = 0.0, 0.999
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if not lo[k] <= cam_pos[k] <= hi[k]:
                return False
            continue
        a, b = (lo[k] - cam_pos[k]) / d[k], (hi[k] - cam_pos[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return t0 <= t1


def test_the_walls_between_camera_and_cube_are_the_occluders():
    """With the camera off the bin's -x side, a cube inside is hidden only by the -x
    wall; one beyond the +x wall by both x walls; one on the camera's side by none."""
    cam = _camera_at(-1.0, 0.0, 2.3)
    assert camera_centre(cam) == pytest.approx([-1.0, 0.0, 2.3])
    flags = occluding_walls(
        cam, SYNTH_BIN, np.array([[0.0, 0.0], [0.4, 0.0], [-0.4, 0.0]])
    )
    # Walls: +x, -x, +y, -y.
    assert flags[0].tolist() == [False, True, False, False]
    assert flags[1].tolist() == [True, True, False, False]
    assert flags[2].tolist() == [False, False, False, False]


def test_the_occluder_mask_is_the_walls_projection():
    """The projected -x wall covers exactly the floor points inside the bin whose ray to
    the camera passes through that wall (ray-slab test), away from its edge."""
    cam = _camera_at(-1.0, 0.0, 2.3)
    mask = occluder_mask(cam, SYNTH_BIN, [False, True, False, False], (720, 1280))
    centre = camera_centre(cam)
    lo = np.array([-0.195, -0.145, 0.0])
    hi = np.array([-0.19, 0.145, 0.21])
    rng = np.random.default_rng(0)
    checked = hidden = 0
    for x, y in rng.uniform([-0.19, -0.14], [0.19, 0.14], (400, 2)):
        point = np.array([x, y, 0.0])
        u, v = cam.project(point[None])[0]
        window = mask[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2]
        if window.min() != window.max():
            continue  # within a pixel of the wall's edge
        checked += 1
        truth = _hidden_by_slab(centre, point, lo, hi)
        hidden += truth
        assert (window[1, 1] > 0.5) == truth
    assert checked > 350 and 0 < hidden < checked


def test_a_mostly_hidden_cube_is_a_miss_not_a_guess():
    """A synthetic cube in the bin behind its -x wall, less than min_visible of it in
    sight, is not reported; the same cube in the open is."""
    cam = _camera_at(-1.0, 0.0, 2.3)
    hidden_xy, open_xy = (-0.13, 0.0), (0.1, 0.0)  # ~1/4 and all of it in sight
    for xy, found in ((hidden_xy, False), (open_xy, True)):
        img = np.full((720, 1280, 3), (60, 90, 130), np.uint8)  # warm cardboard
        local = np.array(
            [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0, 1)]
        )
        world = local * SIDE + [xy[0], xy[1], 0.0]
        poly = cv.convexHull(cam.project(world).astype(np.float32)).astype(np.int32)
        cv.fillConvexPoly(img, poly, (200, 180, 160))  # pale blue-grey
        occ = occluder_mask(cam, SYNTH_BIN, [False, True, False, False], (720, 1280))
        img[occ > 0.5] = (40, 60, 90)  # the wall over it
        fit = detect_cube([img], [cam], (-1.0, 1.0, -1.0, 1.0), SIDE, SYNTH_BIN)
        assert (fit is not None) == found
        if found:
            assert fit.in_bin and math.hypot(fit.x - xy[0], fit.y - xy[1]) < 0.01


def test_the_real_pipeline_runs_this_detector():
    """The real cube source (conf/env/floor-cube-o1.yaml: the skill server's and
    toss_cycles') is the occlusion-aware detector's."""
    text = (Path(__file__).resolve().parent.parent / "conf" / "env").joinpath(
        "floor-cube-o1.yaml"
    )
    assert "prpl_tidybot.cube_detector.CeilingCubeDetectorSource" in text.read_text()
