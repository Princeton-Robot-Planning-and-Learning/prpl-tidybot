"""Tests for real_sim/perceivers/target_source.py."""

import pytest

from prpl_tidybot.real_sim.perceivers.target_source import (
    ConstantTargetSource,
    MarkerDetectorTargetSource,
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
