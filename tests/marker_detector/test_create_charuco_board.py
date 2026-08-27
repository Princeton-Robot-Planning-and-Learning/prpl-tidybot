"""Tests for marker_detector/create_charuco_board.py."""

from prpl_tidybot.marker_detector.create_charuco_board import (
    create_charuco_board_pdf,
)


def test_create_charuco_board_pdf_writes_pdf(tmp_path):
    """The board sheet is a non-empty single PDF."""
    path = tmp_path / "board.pdf"
    create_charuco_board_pdf(path)
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"%PDF")
