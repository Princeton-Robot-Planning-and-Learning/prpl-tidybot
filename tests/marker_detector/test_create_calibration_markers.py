"""Tests for marker_detector/create_calibration_markers.py."""

from prpl_tidybot.marker_detector.create_calibration_markers import create_marker_pdf


def test_create_marker_pdf_writes_expected_pages(tmp_path):
    """Three ids at two per page yields a two-page, non-empty PDF."""
    path = tmp_path / "markers.pdf"
    num_pages = create_marker_pdf(path, (0, 1, 2))
    assert num_pages == 2
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"%PDF")
