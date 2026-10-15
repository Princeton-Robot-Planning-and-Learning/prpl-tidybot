"""Tests for the ceiling-camera cube localizer (no markers: colour + 3D silhouette
fit)."""

import json
import math

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.cube_localizer import (
    CeilingCubeTargetSource,
    cube_silhouette,
    find_cube_candidates,
    fit_cube_pose,
    in_calibrated_region,
    load_ceiling_cameras,
    localize_cube,
)

SERIALS = ("AD8A29AE", "5A5C29AE")  # the PRPL lab's top and bottom ceiling cameras
H, W = 720, 1280
FLOOR_BOUNDS = (-1.5, 1.7, -1.5, 1.5)
SIDE = 0.05


def _lab_to_bgr(lab: tuple[int, int, int]) -> tuple[int, int, int]:
    px = np.array([[lab]], np.uint8)
    b, g, r = (int(v) for v in cv.cvtColor(px, cv.COLOR_LAB2BGR)[0, 0])
    return b, g, r


FLOOR_BGR = _lab_to_bgr((206, 138, 148))  # the lab floor: warm, light
CUBE_BGR = _lab_to_bgr((185, 135, 126))  # the cube: pale blue-grey


def _render(cameras, cubes, seed=0, noise=3.0):
    """A stitched (top over bottom) frame of the floor with cubes at `cubes`."""
    rng = np.random.default_rng(seed)
    halves = []
    for cam in cameras:
        img = np.full((H, W, 3), FLOOR_BGR, np.float32)
        for x, y, yaw in cubes:
            poly = cube_silhouette(cam, x, y, yaw, SIDE).astype(np.int32)
            cv.fillConvexPoly(img, poly, CUBE_BGR, lineType=cv.LINE_AA)
        img += rng.normal(0, noise, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    return load_ceiling_cameras(SERIALS)


def _yaw_err(a, b):
    """Smallest difference between two cube yaws (a cube repeats every 90 degrees)."""
    d = (a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return abs(d)


@pytest.mark.parametrize(
    "pose",
    [
        (-1.09, 1.07, math.radians(-19.5)),
        (0.78, 1.15, 0.0),
        (-0.8, -1.0, math.radians(30)),
    ],
)
def test_localize_recovers_a_known_pose(cameras, pose):
    """A cube rendered at a known pose through the real calibration is recovered."""
    frames = [_render(cameras, [pose], seed=s) for s in range(5)]
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.006
    assert _yaw_err(fit.yaw, pose[2]) < math.radians(2.5)
    assert fit.iou > 0.8
    assert -math.pi / 4 <= fit.yaw < math.pi / 4


def test_localize_on_the_camera_seam(cameras):
    """A cube straddling the two cameras' stitch line is still localized."""
    pose = (-0.8, -0.01, math.radians(10))
    fit = localize_cube([_render(cameras, [pose])], cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.01


@pytest.mark.parametrize("pose", [(-0.184, 0.13, 0.1), (-0.5, 0.11, -0.3)])
def test_a_cube_beside_the_stitch_line_fits_well(cameras, pose):
    """Rows masked at the stitch line are unknown, not 'not cube', so a cube whose
    silhouette crosses them still scores a high IoU (seen on the robot 2026-10-14:
    IoU 0.57 and rejected)."""
    fit = localize_cube([_render(cameras, [pose])], cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert fit.iou > 0.8
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.01


def _render_inconsistent(cameras, pose_top, pose_bottom, seed=0):
    """A frame in which the two cameras disagree about where the cube is, as when
    their extrinsics are inconsistent (seen on the robot: 2.6 cm at the seam)."""
    top = _render(cameras, [pose_top], seed=seed)
    bottom = _render(cameras, [pose_bottom], seed=seed)
    return np.vstack([top[:H], bottom[H:]])


def test_inconsistent_cameras_pick_one_view_instead_of_blending(cameras):
    """With the cameras 2.6 cm apart on the same cube, the localizer keeps the better
    single-camera fit (high IoU, on one of the two answers), reports both, and flags the
    disagreement, instead of a blended pose that matches neither image."""
    top_pose, bottom_pose = (-0.184, 0.10, 0.1), (-0.177, 0.074, 0.1)
    frames = [
        _render_inconsistent(cameras, top_pose, bottom_pose, seed=s) for s in range(3)
    ]
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert fit.iou > 0.8
    assert (
        min(math.hypot(fit.x - p[0], fit.y - p[1]) for p in (top_pose, bottom_pose))
        < 0.006
    )
    assert len(fit.per_camera) == 2
    assert fit.camera_disagreement_m == pytest.approx(0.0269, abs=0.006)


def test_a_cube_one_camera_sees_well_passes_the_screen(cameras):
    """The quick screen keeps the better camera's fit rather than averaging both: on
    the robot (run 17) the top camera fit the cube at 0.84 and the bottom one at 0.52,
    the average 0.54 fell under the threshold, and the cube was never found."""
    top_pose, bottom_pose = (-0.184, 0.10, 0.1), (-0.15, 0.065, 0.5)
    frames = [
        _render_inconsistent(cameras, top_pose, bottom_pose, seed=s) for s in range(3)
    ]
    assert fit_cube_pose(frames, cameras, top_pose[:2], SIDE).iou < 0.6
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert fit.iou > 0.8


def test_prefers_the_camera_whose_calibrated_region_holds_the_cube(cameras):
    """Outside a camera's calibrated region its extrinsics are least trustworthy, so
    when only one camera's calibrated region contains the cube that camera wins, even
    if the other fits slightly better."""
    pose = (-0.18, 0.105, 0.0)
    inside = [
        in_calibrated_region(c, i, pose[0], pose[1], SERIALS)
        for i, c in enumerate(cameras)
    ]
    assert inside.count(True) == 1  # the seam strip: only one calibrated region
    frames = [_render(cameras, [pose], seed=s) for s in range(3)]
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE, serials=SERIALS)
    assert fit is not None
    assert fit.camera == inside.index(True)


def test_a_calibrated_camera_that_fits_badly_does_not_veto_a_good_one(cameras):
    """The calibrated-region preference applies among fits that pass the threshold:
    in run 17 the calibrated camera fit the cube at 0.53 and the other at 0.84, the
    calibrated one was chosen, failed the threshold, and no cube was reported."""
    pose = (-0.18, 0.105, 0.0)
    inside = [
        in_calibrated_region(c, i, pose[0], pose[1], SERIALS)
        for i, c in enumerate(cameras)
    ]
    good = inside.index(False)
    frames = []
    for s in range(3):
        frame = _render(cameras, [], seed=s)
        for i, cam in enumerate(cameras):
            side = SIDE if i == good else SIDE * 0.6  # a poor view of the cube
            poly = cube_silhouette(cam, *pose, side).astype(np.int32)
            cv.fillConvexPoly(frame[i * H : (i + 1) * H], poly, CUBE_BGR)
        frames.append(frame)
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE, serials=SERIALS)
    assert fit is not None
    assert fit.camera == good


def test_consistent_cameras_agree(cameras):
    """When the calibration is consistent, both cameras' fits land together."""
    pose = (-0.3, 0.05, 0.2)
    fit = localize_cube([_render(cameras, [pose])], cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert len(fit.per_camera) == 2
    assert fit.camera_disagreement_m < 0.006


def test_nothing_on_the_floor_gives_none(cameras):
    """An empty floor yields no cube."""
    assert (
        localize_cube([_render(cameras, [])], cameras, FLOOR_BOUNDS, side=SIDE) is None
    )


def test_a_cube_outside_the_drivable_box_is_ignored(cameras):
    """A cube the base could not reach is not reported."""
    frame = _render(cameras, [(1.85, -0.2, 0.0)])
    assert localize_cube([frame], cameras, FLOOR_BOUNDS, side=SIDE) is None


def test_a_large_blob_is_not_a_cube(cameras):
    """A cube-coloured patch far larger than the cube is rejected."""
    frame = _render(cameras, [])
    # A chair-sized dark-blue patch in the top camera's view.
    cv.rectangle(frame, (300, 300), (420, 420), CUBE_BGR, thickness=-1)
    assert localize_cube([frame], cameras, FLOOR_BOUNDS, side=SIDE) is None


def test_the_robot_footprint_is_excluded(cameras):
    """Anything inside an excluded square (the robot) is ignored."""
    frame = _render(cameras, [(0.5, -0.5, 0.0)])
    assert (
        localize_cube(
            [frame], cameras, FLOOR_BOUNDS, side=SIDE, exclude=[(0.5, -0.5, 0.45)]
        )
        is None
    )


def test_candidates_are_found_near_the_cube(cameras):
    """Candidate search puts a seed within a few cm of the cube."""
    frame = _render(cameras, [(0.3, 0.9, 0.2)])
    cands = find_cube_candidates(frame, cameras, FLOOR_BOUNDS, side=SIDE)
    assert any(math.hypot(x - 0.3, y - 0.9) < 0.04 for x, y in cands)


def _with_clutter(frame):
    """Cube-coloured clutter seen on the robot: a slab of shelf, a thin strip of tape,
    and specks; none is the size and shape a cube would make at its spot."""
    out = frame.copy()
    cv.rectangle(out, (900, 150), (1000, 165), CUBE_BGR, -1)  # strip, 100x15
    cv.rectangle(out, (200, 900), (260, 960), CUBE_BGR, -1)  # slab, 4x a cube
    for u, v in [(400, 300), (420, 310), (1100, 1200)]:
        cv.circle(out, (u, v), 3, CUBE_BGR, -1)  # specks
    return out


def test_only_cube_sized_cube_shaped_blobs_are_candidates(cameras):
    """Each candidate costs a full pose fit, so clutter that cannot be the cube (wrong
    size for where it lies on the floor, or a thin strip) is not one."""
    pose = (0.3, 0.9, 0.2)
    frame = _with_clutter(_render(cameras, [pose]))
    cands = find_cube_candidates(frame, cameras, (-1.85, 2.05, -1.85, 1.85), side=SIDE)
    assert cands
    assert all(math.hypot(x - pose[0], y - pose[1]) < 0.04 for x, y in cands)


def test_blobs_cut_by_the_outer_image_border_are_not_candidates(cameras):
    """The vignetted image border passes the colour test in cube-sized pieces (seen on
    the robot 2026-10-15: a 'cube' at (-1.86, +1.43) with IoU 0.93). A cube there is
    only partly in view of every camera, so it is not a candidate."""
    frame = _render(cameras, [])
    cv.rectangle(frame, (0, 200), (12, 230), CUBE_BGR, -1)
    cands = find_cube_candidates(frame, cameras, (-3.0, 3.0, -3.0, 3.0), side=SIDE)
    assert not cands


def test_a_decoy_that_wins_the_quick_screen_does_not_hide_the_cube(cameras):
    """A blob that looks perfect in the last few frames but is gone from the rest (a
    flicker) wins the quick screen and then fails the full refit; the localizer moves
    on to the next candidate instead of giving up (run 16, 2026-10-15)."""
    cube = (0.3, 0.9, 0.2)
    decoy = (-0.9, -0.6, 0.0)
    frames = [_render(cameras, [], seed=s) for s in range(20)]
    for k, frame in enumerate(frames):
        for i, cam in enumerate(cameras):
            half = frame[i * H : (i + 1) * H]
            # the real cube reads smaller than a perfect silhouette, as on the robot
            poly = cube_silhouette(cam, *cube, SIDE * 0.8).astype(np.int32)
            cv.fillConvexPoly(half, poly, CUBE_BGR, lineType=cv.LINE_AA)
            if k >= 17:  # the decoy is only in the last three frames
                poly = cube_silhouette(cam, *decoy, SIDE).astype(np.int32)
                cv.fillConvexPoly(half, poly, CUBE_BGR)
    fit = localize_cube(frames, cameras, FLOOR_BOUNDS, side=SIDE)
    assert fit is not None
    assert math.hypot(fit.x - cube[0], fit.y - cube[1]) < 0.02


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


def test_target_source_reports_position_and_yaw(cameras):
    """The target source returns the cube centre and its yaw."""
    pose = (-0.5, 0.8, math.radians(-12))
    frames = [_render(cameras, [pose], seed=s) for s in range(4)]
    source = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages(frames),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=4,
        frame_period_s=0.0,
    )
    x, y, z = source.get_target()
    assert math.hypot(x - pose[0], y - pose[1]) < 0.006
    assert z == pytest.approx(SIDE / 2)
    assert _yaw_err(source.get_yaw(), pose[2]) < math.radians(2.5)


def test_target_source_raises_when_no_cube_is_visible(cameras):
    """With no cube in view the source raises rather than guessing."""
    source = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages([_render(cameras, [])]),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=2,
        frame_period_s=0.0,
    )
    with pytest.raises(RuntimeError, match="no cube"):
        source.get_target()


def test_a_failed_localization_saves_its_frames_and_inputs(cameras, tmp_path):
    """When no cube is found, every attempt (the first and its one retry) leaves the
    exact frames it fitted and the inputs it used, so the miss can be replayed."""
    frames = [_render(cameras, [], seed=s) for s in range(2)]
    source = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages(frames),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=2,
        frame_period_s=0.0,
        failure_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="no cube"):
        source.get_target()
    attempts = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(attempts) == 2
    for attempt in attempts:
        saved = sorted(attempt.glob("frame_*.png"))
        assert len(saved) == 2
        assert np.array_equal(cv.imread(str(saved[0])), frames[0])
        info = json.loads((attempt / "localization.json").read_text())
        assert info["exclusions"] == [[0.49, -0.49, 0.45]]
        assert info["floor_bounds"] == list(FLOOR_BOUNDS)
        assert info["min_iou"] == pytest.approx(0.6)
        assert "candidates" in info


def test_a_failed_localization_is_retried_once(cameras, tmp_path):
    """A miss is retried once with fresh frames before the source gives up."""
    pose = (-0.5, 0.8, 0.1)
    empty = [_render(cameras, [], seed=s) for s in range(2)]
    cube = [_render(cameras, [pose], seed=s) for s in range(2)]
    source = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages(empty + cube),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=2,
        frame_period_s=0.0,
        failure_dir=tmp_path,
    )
    x, y, _ = source.get_target()
    assert math.hypot(x - pose[0], y - pose[1]) < 0.006
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_the_other_cameras_estimate_is_offered_as_a_fallback(cameras):
    """When the cameras disagree on the cube, the less confident camera's estimate is
    kept as a fallback target for a missed grasp; when they agree there is none."""
    top_pose, bottom_pose = (-0.184, 0.10, 0.1), (-0.15, 0.065, 0.1)
    source = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages(
            [
                _render_inconsistent(cameras, top_pose, bottom_pose, seed=s)
                for s in range(2)
            ]
        ),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=2,
        frame_period_s=0.0,
    )
    x, y, _ = source.get_target()
    ((ax, ay, _),) = source.alternates()
    _, other = sorted(
        (top_pose, bottom_pose), key=lambda p: math.hypot(x - p[0], y - p[1])
    )
    assert math.hypot(ax - other[0], ay - other[1]) < 0.01

    agree = CeilingCubeTargetSource(
        serials=SERIALS,
        image_client=_FakeImages(
            [_render(cameras, [top_pose], seed=s) for s in range(2)]
        ),
        marker_client=_FakeMarkers((0.49, -0.49, math.pi)),
        num_frames=2,
        frame_period_s=0.0,
    )
    agree.get_target()
    assert not agree.alternates()
