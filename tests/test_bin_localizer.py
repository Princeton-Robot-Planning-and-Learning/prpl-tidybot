"""Tests for the ceiling-camera bin localizer: an open cardboard box found by its
darkness against the local floor and fitted as a known 3D box silhouette."""

import json
import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bin_localizer import (
    BinFit,
    BinModel,
    CeilingBinSource,
    ConstantBinSource,
    bin_silhouette,
    cube_in_bin,
    find_bin_candidates,
    localize_bin,
)
from prpl_tidybot.cube_localizer import load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")  # the PRPL lab's top and bottom ceiling cameras
H, W = 720, 1280
MODEL = BinModel(length=0.40, width=0.32, height=0.19, wall=0.005)
SEARCH = (-1.8, 2.0, -1.8, 1.8)


def _lab_to_bgr(lab: tuple[int, int, int]) -> tuple[int, int, int]:
    px = np.array([[lab]], np.uint8)
    b, g, r = (int(v) for v in cv.cvtColor(px, cv.COLOR_LAB2BGR)[0, 0])
    return b, g, r


# Colours sampled from a real ceiling frame of the bin (2026-10-15): the floor is
# the same warm hue as the cardboard, so only brightness separates them; the bin's
# outer walls read L ~60-90 and its shadowed inside L ~35, against a floor that
# ranges from L ~205 in the middle of the room to ~115 in the dim corner by the wall.
FLOOR_LAB = (206, 138, 148)
WALL_BGR = _lab_to_bgr((85, 140, 156))
INSIDE_BGR = _lab_to_bgr((38, 140, 142))
FURNITURE_BGR = _lab_to_bgr((30, 128, 128))
CUBE_BGR = _lab_to_bgr((185, 135, 126))


def _floor(cam_index):
    """The floor, dimming towards the top-right corner of the room as in the lab."""
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    if cam_index == 0:
        dim = np.clip(1.0 - 0.45 * (cols / W) * (1.0 - rows / H) ** 2, 0.5, 1.0)
    else:
        dim = np.ones((H, W), np.float32)
    lab = np.empty((H, W, 3), np.uint8)
    lab[:, :, 0] = np.clip(FLOOR_LAB[0] * dim, 0, 255).astype(np.uint8)
    lab[:, :, 1] = FLOOR_LAB[1]
    lab[:, :, 2] = FLOOR_LAB[2]
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR).astype(np.float32)


def _box_polygon(cam, x, y, yaw, length, width, z0, z1):
    """Image convex hull of a box of footprint length x width between heights z0, z1."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for z in (z0, z1):
                lx, ly = sx * length, sy * width
                pts.append([x + c * lx - s * ly, y + s * lx + c * ly, z])
    hull = cv.convexHull(cam.project(np.array(pts)).astype(np.float32))
    return hull.reshape(-1, 2)


def _render(cameras, bins=(), cubes=(), patches=(), seed=0, noise=3.0):
    """A stitched (top over bottom) frame: the floor, open bins at `bins` (x, y, yaw),
    cubes at `cubes` (x, y) and dark floor patches at `patches` (x0, x1, y0, y1)."""
    rng = np.random.default_rng(seed)
    halves = []
    for i, cam in enumerate(cameras):
        img = _floor(i)
        for x0, x1, y0, y1 in patches:
            corners = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
            poly = cam.project(corners.astype(float)).astype(np.int32)
            cv.fillConvexPoly(img, poly, FURNITURE_BGR, lineType=cv.LINE_AA)
        for x, y, yaw in bins:
            outer = bin_silhouette(cam, x, y, yaw, MODEL).astype(np.int32)
            cv.fillConvexPoly(img, outer, WALL_BGR, lineType=cv.LINE_AA)
            # Through the open top: the inner walls and the bottom, in shadow.
            t = MODEL.wall
            opening = _box_polygon(
                cam,
                x,
                y,
                yaw,
                MODEL.length - 2 * t,
                MODEL.width - 2 * t,
                MODEL.height,
                MODEL.height,
            ).astype(np.int32)
            cv.fillConvexPoly(img, opening, INSIDE_BGR, lineType=cv.LINE_AA)
        for x, y in cubes:
            poly = _box_polygon(cam, x, y, 0.0, 0.05, 0.05, 0.0, 0.05)
            cv.fillConvexPoly(img, poly.astype(np.int32), CUBE_BGR, lineType=cv.LINE_AA)
        img += rng.normal(0, noise, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    return load_ceiling_cameras(SERIALS)


def _yaw_err(a, b):
    """Smallest difference between two bin yaws (a box repeats every 180 degrees)."""
    d = (a - b + math.pi / 2) % math.pi - math.pi / 2
    return abs(d)


@pytest.mark.parametrize(
    "pose",
    [
        (-0.6, -0.9, math.radians(20)),  # open floor, bottom camera
        (1.10, 1.40, 0.0),  # where it stood on 2026-10-15: dim corner, by the wall
        (-1.3, 1.2, math.radians(-35)),  # far corner of the top camera
        (0.3, 0.0, math.radians(60)),  # across the stitch line
        (-0.4, 0.15, math.radians(-10)),  # in the camera-overlap strip
        (0.5, -0.15, math.radians(80)),  # in the overlap strip, nearly sideways
    ],
)
def test_localize_recovers_a_known_pose(cameras, pose):
    """A bin rendered at a known pose through the real calibration is recovered."""
    frames = [_render(cameras, bins=[pose], seed=s) for s in range(3)]
    fit = localize_bin(frames, cameras, MODEL, search_bounds=SEARCH)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015
    assert _yaw_err(fit.yaw, pose[2]) < math.radians(3.0)
    assert -math.pi / 2 <= fit.yaw < math.pi / 2
    assert fit.iou > 0.8
    assert (fit.length, fit.width, fit.height) == (0.40, 0.32, 0.19)


def test_a_cube_inside_the_bin_does_not_break_the_fit(cameras):
    """The pale cube lying in the bin is a hole in the dark mask; holes are filled."""
    pose = (-0.6, 0.9, math.radians(15))
    frame = _render(cameras, bins=[pose], cubes=[(-0.62, 0.88)])
    fit = localize_bin([frame], cameras, MODEL, search_bounds=SEARCH)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015


def test_a_bin_beside_dark_furniture_is_still_found(cameras):
    """Furniture next to the bin (as the shelf beside it on 2026-10-15) neither hides it
    nor drags the fit, as long as the two stay apart in the image (the rim leans ~7 cm
    towards x here, off the camera's nadir)."""
    pose = (0.9, 1.25, 0.0)
    frame = _render(cameras, bins=[pose], patches=[(1.27, 1.7, 1.0, 1.6)])
    fit = localize_bin([frame], cameras, MODEL, search_bounds=SEARCH)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.02


def test_nothing_on_the_floor_gives_none(cameras):
    """An empty floor (dim corner included) yields no bin."""
    assert (
        localize_bin([_render(cameras)], cameras, MODEL, search_bounds=SEARCH) is None
    )


def test_a_large_dark_patch_is_not_a_bin(cameras):
    """A dark region much larger than the bin (a rug, a table's shadow) is rejected."""
    frame = _render(cameras, patches=[(-1.2, -0.2, -1.3, -0.5)])
    assert localize_bin([frame], cameras, MODEL, search_bounds=SEARCH) is None


def test_a_small_dark_patch_is_not_a_bin(cameras):
    """Something dark but cube-sized (the robot's markers, a shoe) is not a bin."""
    frame = _render(cameras, patches=[(0.4, 0.5, -0.8, -0.7)])
    assert localize_bin([frame], cameras, MODEL, search_bounds=SEARCH) is None


def test_the_robot_footprint_is_excluded(cameras):
    """Anything inside an excluded square (the robot) is ignored."""
    frame = _render(cameras, bins=[(0.5, -0.6, 0.0)])
    assert (
        localize_bin(
            [frame],
            cameras,
            MODEL,
            search_bounds=SEARCH,
            exclude=[(0.5, -0.6, 0.45)],
        )
        is None
    )


def test_a_bin_outside_the_search_box_is_ignored(cameras):
    """Only the configured search box is searched."""
    frame = _render(cameras, bins=[(-0.6, -0.9, 0.0)])
    assert (
        localize_bin([frame], cameras, MODEL, search_bounds=(0.0, 2.0, -1.8, 1.8))
        is None
    )


def test_candidates_are_found_near_the_bin(cameras):
    """Candidate search puts a seed within a few cm of the bin."""
    frame = _render(cameras, bins=[(-0.8, 0.6, 0.4)])
    cands = find_bin_candidates(frame, cameras, SEARCH, MODEL)
    assert any(math.hypot(x + 0.8, y - 0.6) < 0.06 for x, y in cands)


def _fit(x=0.0, y=0.0, yaw=0.0):
    return BinFit(x=x, y=y, yaw=yaw, length=0.40, width=0.32, height=0.19, wall=0.005)


@pytest.mark.parametrize(
    "cube_xy, inside",
    [
        ((0.0, 0.0), True),
        ((0.17, 0.0), True),  # 2.5 cm from the inner wall along the length
        ((0.0, 0.13), True),
        ((0.19, 0.0), False),  # inside the outer footprint but on the wall/margin
        ((0.0, 0.155), False),
        ((0.25, 0.0), False),
        ((0.0, -0.3), False),
    ],
)
def test_cube_in_bin_uses_the_inner_footprint(cube_xy, inside):
    """The cube counts only when it is inside the walls, by the margin."""
    assert cube_in_bin(cube_xy, _fit(), margin=0.01) is inside


def test_cube_in_bin_scores_a_region_strictly_inside_the_bin_by_default():
    """The default margin (in_bin_margin, 5 cm) makes the scoring region strictly
    smaller than the bin's interior: a cube by the inner wall is in the bin but does not
    score."""
    assert cube_in_bin((0.14, 0.0), _fit())
    assert not cube_in_bin((0.17, 0.0), _fit())
    assert cube_in_bin((0.17, 0.0), _fit(), margin=0.01)
    assert not cube_in_bin((0.0, 0.11), _fit())


def test_a_bin_touching_dark_furniture_is_rejected(cameras):
    """Merged with furniture, the dark region spills far past any bin outline; that is
    reported as no bin rather than guessed."""
    frame = _render(cameras, bins=[(0.9, 1.25, 0.0)], patches=[(1.1, 1.7, 1.0, 1.6)])
    assert localize_bin([frame], cameras, MODEL, search_bounds=SEARCH) is None


def test_cube_in_bin_follows_the_bins_pose():
    """The footprint is the bin's, translated and rotated."""
    rotated = _fit(1.0, -0.5, math.pi / 2)  # length now along y
    assert cube_in_bin((1.0, -0.5 + 0.17), rotated, margin=0.01)
    assert not cube_in_bin((1.0 + 0.17, -0.5), rotated, margin=0.01)


def test_bin_fit_serialises_to_the_contract():
    """``as_dict`` is exactly the skill-server contract's bin entry, plain floats."""
    d = _fit(0.1, 0.2, 0.3).as_dict()
    assert set(d) == {"x", "y", "yaw", "length", "width", "height"}
    assert all(isinstance(v, float) for v in d.values())
    json.dumps(d)


class _FakeImages:
    """Serves a fixed list of stitched frames, cycling."""

    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def get_image(self):
        """The next frame."""
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return frame

    def close(self):
        """Nothing to release."""


class _FakeMarkers:
    """A marker detector that only reports the robot's pose."""

    def __init__(self, robot_pose):
        self._robot_pose = robot_pose

    def get_latest(self):
        """A payload with the robot pose and no targets."""
        return {"poses": {0: self._robot_pose}, "targets": {}}

    def close(self):
        """Nothing to release."""


def _source(frames, robot=(0.49, -0.49, math.pi)):
    return CeilingBinSource(
        serials=SERIALS,
        image_client=_FakeImages(frames),
        marker_client=_FakeMarkers(robot),
        length=MODEL.length,
        width=MODEL.width,
        height=MODEL.height,
        wall=MODEL.wall,
        search_bounds=SEARCH,
        num_frames=2,
        frame_period_s=0.0,
    )


def test_ceiling_bin_source_localizes(cameras):
    """The source returns the fitted bin, and its centre at rim height as a target."""
    pose = (-0.5, 0.8, math.radians(-12))
    source = _source([_render(cameras, bins=[pose], seed=s) for s in range(2)])
    fit = source.localize()
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015
    x, y, z = source.get_target()
    assert (x, y) == pytest.approx((fit.x, fit.y))
    assert z == pytest.approx(MODEL.height)
    assert source.last_fit == fit


def test_ceiling_bin_source_ignores_the_robot(cameras):
    """A bin under the robot's reported pose is the robot's footprint, not a bin."""
    source = _source([_render(cameras, bins=[(0.49, -0.49, 0.0)])])
    with pytest.raises(RuntimeError, match="no bin"):
        source.localize()


def test_ceiling_bin_source_raises_when_no_bin_is_visible(cameras):
    """With no bin in view the source raises rather than guessing."""
    with pytest.raises(RuntimeError, match="no bin"):
        _source([_render(cameras)]).get_target()


def test_a_missed_bin_saves_its_frames_and_inputs(cameras, tmp_path):
    """No bin found: the exact frames it fitted and the inputs it used are saved under
    failure_dir (as the cube's misses are), so the miss can be replayed."""
    frames = [_render(cameras, seed=s) for s in range(2)]
    source = CeilingBinSource(
        serials=SERIALS,
        image_client=_FakeImages(frames),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        length=MODEL.length,
        width=MODEL.width,
        height=MODEL.height,
        wall=MODEL.wall,
        search_bounds=SEARCH,
        num_frames=2,
        frame_period_s=0.0,
        min_saturation=130.0,
        failure_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="no bin"):
        source.localize()
    (saved,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert saved.name.endswith("_bin")
    pngs = sorted(saved.glob("frame_*.png"))
    assert len(pngs) == 2
    assert np.array_equal(cv.imread(str(pngs[0])), frames[0])
    info = json.loads((saved / "localization.json").read_text())
    assert info["object"] == "bin"
    assert info["search_bounds"] == list(SEARCH)
    assert info["exclusions"] == [[0.49, -0.49, 0.45]]
    assert info["min_iou"] == pytest.approx(0.7)
    assert info["min_saturation"] == pytest.approx(130.0)
    assert info["model"] == {
        "length": MODEL.length,
        "width": MODEL.width,
        "height": MODEL.height,
        "wall": MODEL.wall,
    }
    assert info["serials"] == list(SERIALS)


def test_a_found_bin_saves_nothing(cameras, tmp_path):
    """Only misses are saved."""
    pose = (-0.5, 0.8, math.radians(-12))
    source = _source([_render(cameras, bins=[pose], seed=s) for s in range(2)])
    source._failure_dir = tmp_path  # pylint: disable=protected-access
    source.localize()
    assert not list(tmp_path.iterdir())


def test_constant_bin_source_reports_its_pose():
    """Fake mode's source answers with the configured bin."""
    source = ConstantBinSource(x=-1.0, y=0.5, yaw=0.2, length=0.4, width=0.32)
    fit = source.localize()
    assert (fit.x, fit.y, fit.yaw, fit.length, fit.width) == (-1.0, 0.5, 0.2, 0.4, 0.32)
    assert source.get_target() == pytest.approx((-1.0, 0.5, fit.height))


def test_the_toss_cycles_config_builds_both_bin_sources():
    """conf/toss_cycles.yaml's bin block instantiates for fake and resolves for real."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    import hydra  # pylint: disable=import-outside-toplevel
    from hydra import (  # pylint: disable=import-outside-toplevel
        compose,
        initialize_config_dir,
    )

    conf = Path(__file__).resolve().parent.parent / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf)):
        cfg = compose(config_name="toss_cycles", overrides=["mode=fake"])
    source = hydra.utils.instantiate(cfg.bin.pipelines.fake)
    assert source.localize().length == cfg.bin.length
    real = cfg.bin.pipelines.real
    assert real.serials == list(SERIALS)
    assert real.darkness == cfg.bin.darkness


# Sampled from the 2026-10-20 ceiling frame of the same-side bin (camera 1): floor
# HSV saturation ~80, the bin's shadow on the floor ~80-100 (darker floor), the
# cardboard ~160-200. Darkness alone counts the shadow as bin; saturation does not.
TODAY_FLOOR_BGR = (163, 188, 232)
TODAY_SHADOW_BGR = (110, 137, 179)
TODAY_WALL_BGR = (33, 60, 116)
TODAY_INSIDE_BGR = (55, 83, 141)
TODAY_MODEL = BinModel(length=0.39, width=0.29, height=0.21, wall=0.005)


def _render_with_shadow(cameras, pose, shadow_offset, seed=0):
    """Today's colours: the bin at `pose` and its shadow on the floor, the bin's
    footprint shifted by `shadow_offset` (m) and grown 4 cm."""
    rng = np.random.default_rng(seed)
    x, y, yaw = pose
    halves = []
    for cam in cameras:
        img = np.full((H, W, 3), TODAY_FLOOR_BGR, np.float32)
        shadow = _box_polygon(
            cam,
            x + shadow_offset[0],
            y + shadow_offset[1],
            yaw,
            TODAY_MODEL.length + 0.04,
            TODAY_MODEL.width + 0.04,
            0.0,
            0.0,
        ).astype(np.int32)
        cv.fillConvexPoly(img, shadow, TODAY_SHADOW_BGR, lineType=cv.LINE_AA)
        outer = bin_silhouette(cam, x, y, yaw, TODAY_MODEL).astype(np.int32)
        cv.fillConvexPoly(img, outer, TODAY_WALL_BGR, lineType=cv.LINE_AA)
        t = TODAY_MODEL.wall
        opening = _box_polygon(
            cam,
            x,
            y,
            yaw,
            TODAY_MODEL.length - 2 * t,
            TODAY_MODEL.width - 2 * t,
            TODAY_MODEL.height,
            TODAY_MODEL.height,
        ).astype(np.int32)
        cv.fillConvexPoly(img, opening, TODAY_INSIDE_BGR, lineType=cv.LINE_AA)
        img += rng.normal(0, 3.0, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.mark.parametrize(
    "pose, shadow",
    [
        ((-0.297, -0.776, math.radians(89.0)), (0.04, 0.06)),  # 2026-10-20 spot
        ((0.6, -0.9, math.radians(20)), (-0.05, 0.03)),
    ],
)
def test_a_saturation_mask_fits_the_bin_and_not_its_shadow(cameras, pose, shadow):
    """2026-10-20: the darkness mask took the bin's shadow for bin, and the fit sat
    2 cm off the ArUco marker in the bin's centre. With ``min_saturation`` the mask is
    the cardboard only and the fit lands within 1 cm."""
    frames = [_render_with_shadow(cameras, pose, shadow, seed=s) for s in range(2)]
    fit = localize_bin(
        frames, cameras, TODAY_MODEL, search_bounds=SEARCH, min_saturation=130.0
    )
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.01
    assert _yaw_err(fit.yaw, pose[2]) < math.radians(3.0)


def test_the_ceiling_bin_source_passes_min_saturation_on(cameras):
    """CeilingBinSource(min_saturation=...) fits with the saturation mask."""
    pose, shadow = (-0.297, -0.776, math.radians(89.0)), (0.04, 0.06)
    source = CeilingBinSource(
        SERIALS,
        image_client=_FakeImages(
            [_render_with_shadow(cameras, pose, shadow, seed=s) for s in range(2)]
        ),
        marker_client=_FakeMarkers((5.0, 5.0)),
        length=TODAY_MODEL.length,
        width=TODAY_MODEL.width,
        height=TODAY_MODEL.height,
        wall=TODAY_MODEL.wall,
        search_bounds=SEARCH,
        num_frames=2,
        frame_period_s=0.0,
        min_saturation=130.0,
    )
    fit = source.localize()
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.01
