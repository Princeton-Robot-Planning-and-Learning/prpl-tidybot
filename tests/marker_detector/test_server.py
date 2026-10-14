"""Tests for marker_detector/server.py."""

import numpy as np
import pytest

from prpl_tidybot.marker_detector import server as server_module
from prpl_tidybot.marker_detector.constants import DETECTED_MARKER_IDS
from prpl_tidybot.marker_detector.server import (
    Detector,
    MarkerDetectorServer,
    _merge_target_estimates,
    _merge_target_yaws,
    _target_marker_yaw,
)


def test_merge_targets_seen_by_both_cameras_are_averaged():
    """Both cameras see marker 23: the estimates average, residual = distance."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {23: (1.0, 2.1)}])
    assert targets == {23: (1.0, 2.05)}
    assert np.isclose(residuals[23], 0.1)


def test_merge_target_seen_by_one_camera_passes_through():
    """A marker seen by a single camera is unchanged and has no residual."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {}])
    assert targets == {23: (1.0, 2.0)}
    assert not residuals


def test_merge_disjoint_targets_across_cameras():
    """Cameras seeing different markers each contribute their own estimate."""
    targets, residuals = _merge_target_estimates([{23: (1.0, 2.0)}, {24: (-1.0, 0.5)}])
    assert targets == {23: (1.0, 2.0), 24: (-1.0, 0.5)}
    assert not residuals


def test_merge_no_targets():
    """No detections at all yields empty targets and residuals."""
    assert _merge_target_estimates([{}, {}]) == ({}, {})
    assert _merge_target_estimates([]) == ({}, {})


def _square_corners(center: tuple[float, float], yaw: float, half: float = 0.05):
    """Map-frame corners of a marker whose +x (corner 0 -> 1) points along `yaw`.

    ArUco orders corners top-left, top-right, bottom-right, bottom-left; seen from
    the ceiling and projected to the floor, the marker's "up" is +90 degrees from
    its +x.
    """
    local = np.array([(-half, half), (half, half), (half, -half), (-half, -half)])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rot.T + np.array(center)


@pytest.mark.parametrize("yaw", [0.0, 0.3, np.pi / 2, -2.0, np.pi - 1e-3, -np.pi])
def test_target_marker_yaw_is_the_direction_of_the_top_edge(yaw):
    """The yaw of a projected marker is the direction of its corner-0 -> corner-1
    edge, in (-pi, pi], including either side of the +-pi cut."""
    estimated = _target_marker_yaw(_square_corners((1.0, -2.0), yaw))
    assert -np.pi < estimated <= np.pi
    assert np.isclose(np.angle(np.exp(1j * (estimated - yaw))), 0.0, atol=1e-9)


def test_target_marker_yaw_averages_noisy_corners_across_the_cut():
    """Corners perturbed to either side of +-pi still average to ~pi, not ~0."""
    corners = _square_corners((0.0, 0.0), np.pi)
    corners[0] += (0.0, 0.004)
    corners[2] += (0.0, -0.004)
    estimated = _target_marker_yaw(corners)
    assert abs(abs(estimated) - np.pi) < 0.1


def test_merge_target_yaws_is_a_circular_mean():
    """Two cameras either side of the +-pi cut average to pi, not to 0."""
    yaws = _merge_target_yaws([{40: np.pi - 0.1}, {40: -np.pi + 0.1}])
    assert np.isclose(abs(yaws[40]), np.pi)


def test_merge_target_yaws_single_camera_and_disjoint_ids_pass_through():
    """A yaw seen by one camera is unchanged; different ids stay separate."""
    yaws = _merge_target_yaws([{40: 0.5}, {41: -1.0}])
    assert yaws == {40: pytest.approx(0.5), 41: pytest.approx(-1.0)}
    assert not _merge_target_yaws([{}, {}])
    assert not _merge_target_yaws([])


class _IdentityProjector:
    """Treats pixel coordinates as map-frame coordinates."""

    def __init__(self, *args) -> None:
        del args

    def project_to_heights(self, pixels, heights):
        """Return the pixels unchanged."""
        del heights
        return np.asarray(pixels, dtype=np.float64)


def test_detector_publishes_target_yaw_beside_unchanged_position(monkeypatch):
    """A scene target's `targets` entry stays an (x, y) pair; its yaw is published
    separately under `target_yaws`."""
    monkeypatch.setattr(server_module, "get_camera_params", lambda _: (0, 0, None, 0))
    monkeypatch.setattr(server_module, "load_extrinsics", lambda _: ())
    monkeypatch.setattr(server_module, "FloorProjector", _IdentityProjector)
    monkeypatch.setattr(server_module, "CameraClient", lambda _: None)
    detector = Detector("serial", port=0, inverse_heading=True)
    slot = DETECTED_MARKER_IDS.index(40)
    corners = [_square_corners((0.7, -0.3), 1.2).reshape((1, 4, 2)).astype(np.float32)]
    data = detector.get_poses_from_markers(corners, np.array([[slot]]))
    assert data["targets"][40] == pytest.approx((0.7, -0.3))
    assert len(data["targets"][40]) == 2
    assert data["target_yaws"][40] == pytest.approx(1.2, abs=1e-5)


def test_server_payload_carries_fused_target_yaws():
    """`MarkerDetectorServer.get_data` fuses the per-camera yaws (circular mean) and
    keeps the existing `targets` / `target_residuals` fields as they were."""

    class _ScriptedDetector:
        def __init__(self, data):
            self._data = data

        def get_poses(self, debug=False):
            """Return the scripted per-camera detection."""
            del debug
            return self._data

    server = MarkerDetectorServer.__new__(MarkerDetectorServer)
    server.debug = False
    server._last_residual_warning = {}  # pylint: disable=protected-access
    # setattr keeps pylint from attributing the stand-ins' type to the real field.
    scripted = [
        _ScriptedDetector(
            {
                "poses": {},
                "targets": {40: (1.0, 2.0)},
                "target_yaws": {40: np.pi - 0.1},
                "single_marker_robots": set(),
            }
        ),
        _ScriptedDetector(
            {
                "poses": {},
                "targets": {40: (1.0, 2.02)},
                "target_yaws": {40: -np.pi + 0.1},
                "single_marker_robots": set(),
            }
        ),
    ]
    setattr(server, "detectors", scripted)
    data = server.get_data()
    assert data["targets"] == {40: pytest.approx((1.0, 2.01))}
    assert data["target_residuals"][40] == pytest.approx(0.02)
    assert np.isclose(abs(data["target_yaws"][40]), np.pi)
