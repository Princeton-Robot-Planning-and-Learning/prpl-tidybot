"""Tests for toss_outcome.py: ceiling-camera video and landing readout of a toss."""

import json
import math

import pytest

from prpl_tidybot.marker_detector.ceiling_image_client import FakeCeilingImageClient
from prpl_tidybot.marker_detector.client import FakeMarkerDetectorClient
from prpl_tidybot.marker_detector.constants import TARGET_MARKER_IDS
from prpl_tidybot.rendering import CeilingCameraRenderer
from prpl_tidybot.toss_outcome import (
    DEFAULT_CUBE_FACE_MARKER_IDS,
    TossCapture,
    measure_landing,
    validate_cube_marker_ids,
)


def test_default_cube_face_ids_are_targets_that_avoid_the_cylinder_markers():
    """The defaults are the target markers other than 35-37 (under the cylinders)."""
    assert set(DEFAULT_CUBE_FACE_MARKER_IDS) <= set(TARGET_MARKER_IDS)
    assert not {35, 36, 37} & set(DEFAULT_CUBE_FACE_MARKER_IDS)
    assert 23 in DEFAULT_CUBE_FACE_MARKER_IDS


def test_validate_cube_marker_ids_rejects_ids_the_detector_does_not_publish():
    """Only TARGET_MARKER_IDS are published as targets, so anything else is an error;
    a robot sticker id (10) or a ChArUco id (0) would never be seen."""
    assert validate_cube_marker_ids([23, 40]) == (23, 40)
    with pytest.raises(ValueError):
        validate_cube_marker_ids([10])
    with pytest.raises(ValueError):
        validate_cube_marker_ids([])


def test_measure_landing_relative_to_the_robot():
    """Distance from the base centre and bearing relative to the base heading."""
    outcome = measure_landing(
        robot_pose_at_throw=(1.0, 1.0, math.pi / 2),
        targets={23: (1.0, 3.0), 35: (0.0, 0.0)},
        cube_marker_ids=(23, 40),
    )
    assert outcome["status"] == "landed"
    assert outcome["visible_cube_markers"] == {"23": [1.0, 3.0]}
    landing = outcome["landing"]
    assert landing["marker_id"] == 23
    assert landing["x"] == 1.0 and landing["y"] == 3.0
    assert landing["distance_m"] == pytest.approx(2.0)
    # Straight ahead of a robot facing +y.
    assert landing["bearing_rad"] == pytest.approx(0.0)
    assert landing["bearing_deg"] == pytest.approx(0.0)
    assert outcome["robot_map_pose_at_throw"] == [1.0, 1.0, math.pi / 2]


def test_measure_landing_bearing_is_wrapped():
    """A landing behind-left of the robot reads as about +135 deg, not -225."""
    outcome = measure_landing(
        robot_pose_at_throw=(0.0, 0.0, 0.0),
        targets={40: (-1.0, 1.0)},
        cube_marker_ids=(23, 40),
    )
    assert outcome["landing"]["bearing_deg"] == pytest.approx(135.0)


def test_measure_landing_first_configured_id_wins():
    """With two cube faces visible the first id in configured order is the landing;
    both are logged."""
    outcome = measure_landing(
        robot_pose_at_throw=(0.0, 0.0, 0.0),
        targets={23: (1.0, 0.0), 40: (1.1, 0.0)},
        cube_marker_ids=(40, 23),
    )
    assert outcome["landing"]["marker_id"] == 40
    assert set(outcome["visible_cube_markers"]) == {"23", "40"}


def test_measure_landing_with_no_cube_marker_visible():
    """No cube face in view is an explicit status, not an error."""
    outcome = measure_landing(
        robot_pose_at_throw=(0.0, 0.0, 0.0),
        targets={35: (1.0, 0.0)},
        cube_marker_ids=(23, 40),
    )
    assert outcome["status"] == "no_cube_marker_visible"
    assert outcome["landing"] is None
    assert outcome["visible_cube_markers"] == {}


def test_measure_landing_without_a_robot_pose():
    """Without the robot's pose the landing position is still logged, but not the
    distance and bearing."""
    outcome = measure_landing(
        robot_pose_at_throw=None,
        targets={23: (1.0, 0.0)},
        cube_marker_ids=(23,),
    )
    assert outcome["status"] == "landed"
    assert outcome["robot_map_pose_at_throw"] is None
    assert outcome["landing"]["distance_m"] is None
    assert outcome["landing"]["bearing_deg"] is None


def _capture(tmp_path, targets, robot_pose=(0.0, 0.0, 0.0)):
    renderer = CeilingCameraRenderer(client=FakeCeilingImageClient())
    client = FakeMarkerDetectorClient(robot_pose=robot_pose, targets=targets)
    sleeps: list[float] = []
    capture = TossCapture(
        renderer=renderer,
        marker_client=client,
        log_dir=tmp_path,
        cube_marker_ids=(23, 40),
        fps=10,
        post_toss_record_s=0.3,
        settle_s=0.2,
        sleep_fn=sleeps.append,
    )
    return capture, sleeps


def test_toss_capture_writes_video_and_outcome_json(tmp_path):
    """start() then finish(): the ceiling video and the landing JSON land in the log
    dir; the settle wait comes before the marker read."""
    capture, sleeps = _capture(tmp_path, targets=[[23, 1.5, 0.0]])
    capture.start()
    outcome = capture.finish(toss={"heading_deg": 0.0})
    capture.close()
    assert sleeps[0] == pytest.approx(0.2)
    assert sum(sleeps) == pytest.approx(0.3)
    assert (tmp_path / "toss.mp4").stat().st_size > 0
    written = json.loads((tmp_path / "toss_outcome.json").read_text())
    assert written == outcome
    assert written["status"] == "landed"
    assert written["landing"]["distance_m"] == pytest.approx(1.5)
    assert written["toss"] == {"heading_deg": 0.0}
    assert written["video"] == "toss.mp4"


def test_toss_capture_logs_a_missing_cube(tmp_path):
    """No cube face visible after the toss: the JSON says so and nothing raises."""
    capture, _ = _capture(tmp_path, targets=[])
    capture.start()
    outcome = capture.finish(toss={})
    capture.close()
    assert outcome["status"] == "no_cube_marker_visible"
    written = json.loads((tmp_path / "toss_outcome.json").read_text())
    assert written["landing"] is None


def test_toss_capture_reads_the_robot_pose_at_the_throw(tmp_path):
    """The robot pose is read in start(), just before the toss."""
    capture, _ = _capture(tmp_path, targets=[[23, 0.0, 2.0]], robot_pose=(0, 1, 0))
    capture.start()
    outcome = capture.finish(toss={})
    capture.close()
    assert outcome["robot_map_pose_at_throw"] == [0.0, 1.0, 0.0]
    assert outcome["landing"]["distance_m"] == pytest.approx(1.0)
