"""Tests for the wrist image sources that do not need hardware."""

import numpy as np
import pytest

from prpl_tidybot.visual_servo.image_sources import SequenceImageSource


def test_sequence_source_replays_then_repeats_last_frame():
    """Frames come back in order and the last one repeats afterwards."""
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]
    source = SequenceImageSource(frames)
    for i in range(3):
        image = source.get_image()
        assert image is not None and image[0, 0, 0] == i
    for _ in range(2):
        image = source.get_image()
        assert image is not None and image[0, 0, 0] == 2


def test_sequence_source_rejects_empty():
    """An empty sequence is a configuration error."""
    with pytest.raises(ValueError):
        SequenceImageSource([])
