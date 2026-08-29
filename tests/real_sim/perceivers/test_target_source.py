"""Tests for real_sim/perceivers/target_source.py."""

import pytest

from prpl_tidybot.real_sim.perceivers.target_source import (
    ConstantCylinderTargets,
    ConstantTargetSource,
    CylinderSpec,
    MarkerDetectorCylinderTargets,
    MarkerDetectorTargetSource,
    parse_cylinder_specs,
)


class _FakeMarkerDetectorClient:
    """In-memory stand-in for `MarkerDetectorClient`.

    Each `get_latest()` returns the next scripted payload (or the last one once the
    script is exhausted), letting tests drive detection-flicker behavior without a live
    socket.
    """

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self._idx = 0
        self.closed = False

    def get_latest(self) -> dict:
        """Return the next scripted payload (or the last one if the script is done)."""
        payload = self._payloads[min(self._idx, len(self._payloads) - 1)]
        self._idx += 1
        return payload

    def close(self) -> None:
        """Record that close() was called so tests can assert on cleanup."""
        self.closed = True


def test_constant_target_source_returns_fixed_pose():
    """`ConstantTargetSource(x, y, z)` always returns the same tuple."""
    src = ConstantTargetSource(1.5, -0.25, 0.4)
    assert src.get_target() == (1.5, -0.25, 0.4)
    assert src.get_target() == (1.5, -0.25, 0.4)


def test_constant_target_source_coerces_to_float():
    """Integer constructor args are stored as floats."""
    src = ConstantTargetSource(1, 2, 3)
    x, y, z = src.get_target()
    assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float)


def test_marker_detector_target_source_pairs_detection_with_target_z():
    """`MarkerDetectorTargetSource` zips the marker's published (x, y) with the
    configured z to build the (x, y, z) target."""
    client = _FakeMarkerDetectorClient([{"targets": {23: (0.5, -0.2)}}])
    src = MarkerDetectorTargetSource(
        marker_id=23,
        target_z=0.3,
        client=client,  # type: ignore[arg-type]
    )
    assert src.get_target() == (0.5, -0.2, 0.3)


def test_marker_detector_target_source_caches_last_detection():
    """When the marker briefly drops out, the last successful detection is reused."""
    client = _FakeMarkerDetectorClient(
        [
            {"targets": {23: (0.5, -0.2)}},
            {"targets": {}},
            {"targets": {99: (0.0, 0.0)}},  # other markers, our id missing
        ]
    )
    src = MarkerDetectorTargetSource(
        marker_id=23,
        target_z=0.3,
        client=client,  # type: ignore[arg-type]
    )
    assert src.get_target() == (0.5, -0.2, 0.3)
    assert src.get_target() == (0.5, -0.2, 0.3)
    assert src.get_target() == (0.5, -0.2, 0.3)


def test_marker_detector_target_source_raises_if_marker_never_seen():
    """Before the marker has ever been detected, `get_target` raises so the rollout
    doesn't proceed on a phantom target."""
    client = _FakeMarkerDetectorClient([{"targets": {}}, {"targets": {99: (1.0, 1.0)}}])
    src = MarkerDetectorTargetSource(
        marker_id=23,
        target_z=0.3,
        client=client,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="never been reported"):
        src.get_target()


def test_marker_detector_target_source_close_propagates_to_client():
    """`close()` closes the underlying client."""
    client = _FakeMarkerDetectorClient([{"targets": {23: (0.0, 0.0)}}])
    src = MarkerDetectorTargetSource(
        marker_id=23,
        target_z=0.3,
        client=client,  # type: ignore[arg-type]
    )
    src.close()
    assert client.closed


def test_cylinder_spec_from_mapping_and_center_z():
    """Config mappings become specs; the centre sits at half the height."""
    specs = parse_cylinder_specs(
        [
            {"marker_id": 35, "radius": 0.039, "height": 0.233, "fake_xy": [0.5, 0.0]},
            CylinderSpec(radius=0.04, height=0.21),
        ]
    )
    assert specs[0].marker_id == 35 and specs[0].fake_xy == (0.5, 0.0)
    assert specs[0].center_z == pytest.approx(0.1165)
    assert specs[1].marker_id is None and specs[1].fake_xy is None
    with pytest.raises(ValueError):
        parse_cylinder_specs([])


def test_constant_cylinder_targets_use_fake_xy_and_half_height():
    """Fake-mode targets are each spec's fake_xy at its centre height, in order."""
    targets = ConstantCylinderTargets(
        [
            {"radius": 0.039, "height": 0.233, "fake_xy": [0.5, 0.0]},
            {"radius": 0.040, "height": 0.210, "fake_xy": [0.3, -0.6]},
        ]
    )
    assert targets.get_targets() == [(0.5, 0.0, 0.1165), (0.3, -0.6, 0.105)]
    with pytest.raises(ValueError, match="fake_xy"):
        ConstantCylinderTargets([{"radius": 0.03, "height": 0.2}])


def test_marker_detector_cylinder_targets_read_each_marker():
    """Real-mode targets pair each spec's marker (x, y) with its centre height and reuse
    the last detection for a marker that drops out."""
    client = _FakeMarkerDetectorClient(
        [
            {"targets": {35: (0.5, 0.0), 36: (0.3, 0.6)}},
            {"targets": {36: (0.31, 0.6)}},
        ]
    )
    targets = MarkerDetectorCylinderTargets(
        [
            {"marker_id": 35, "radius": 0.039, "height": 0.233},
            {"marker_id": 36, "radius": 0.039, "height": 0.120},
        ],
        client=client,  # type: ignore[arg-type]
    )
    assert targets.get_targets() == [(0.5, 0.0, 0.1165), (0.3, 0.6, 0.06)]
    # Second payload: marker 35 dropped out (cached), 36 moved.
    assert targets.get_targets() == [(0.5, 0.0, 0.1165), (0.31, 0.6, 0.06)]
    with pytest.raises(ValueError, match="marker_id"):
        MarkerDetectorCylinderTargets(
            [{"radius": 0.03, "height": 0.2}], client=client  # type: ignore[arg-type]
        )


def test_marker_detector_cylinder_targets_raise_for_a_never_seen_marker():
    """If any configured marker has never been reported, get_targets raises."""
    client = _FakeMarkerDetectorClient([{"targets": {35: (0.5, 0.0)}}])
    targets = MarkerDetectorCylinderTargets(
        [
            {"marker_id": 35, "radius": 0.039, "height": 0.233},
            {"marker_id": 36, "radius": 0.039, "height": 0.120},
        ],
        client=client,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match=r"\[36\]"):
        targets.get_targets()
