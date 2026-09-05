"""SAM-backed drop-in for the OpenCV cylinder edge detector.

``SamEdgeDetector`` is a callable with the same contract as
:func:`prpl_tidybot.visual_servo.cylinder_edges.detect_cylinder_edges`
(image in, :class:`CylinderEdges` or None out), so anything that consumes
edges — the visual-servo executor, the arm executor's lateral correction —
can select it by config. Detection is delegated to the segmentation service
on the perception laptop; instance selection happens here: among instances
with a plausible column width, the one whose centre is nearest the image
centre wins (the servo stages the target can roughly ahead of the camera;
neighbouring cans and background objects sit off-centre).

Returning None on service failure lets the caller's existing fallback chain
run (retries, then the OpenCV detector or the marker-based compensation).
"""

import logging

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.segmentation.client import SegmentationClient, SegmentedInstance
from prpl_tidybot.segmentation.constants import SEGMENTATION_PORT
from prpl_tidybot.third_party.constants import SERVER_HOSTNAME
from prpl_tidybot.visual_servo.cylinder_edges import CylinderEdges

_logger = logging.getLogger(__name__)


class SamEdgeDetector:
    """Produce :class:`CylinderEdges` from the segmentation service."""

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = SEGMENTATION_PORT,
        prompt: str = "a can. a jar.",
        min_score: float = 0.3,
        min_width_frac: float = 0.06,
        max_width_frac: float = 0.7,
        timeout_s: float = 5.0,
    ) -> None:
        self._client = SegmentationClient(host=host, port=port, timeout_s=timeout_s)
        self._prompt = prompt
        self._min_score = min_score
        self._min_width_frac = min_width_frac
        self._max_width_frac = max_width_frac

    def __call__(self, image: NDArray[np.uint8]) -> CylinderEdges | None:
        instances = self._client.detect(image, self._prompt)
        if instances is None:
            return None
        height, width = image.shape[:2]
        chosen = self._choose(instances, width)
        if chosen is None:
            _logger.info(
                "Segmentation returned %d instance(s), none plausible.",
                len(instances),
            )
            return None
        clipped_left = chosen["left_x"] <= 1.0
        clipped_right = chosen["right_x"] >= width - 2.0
        return CylinderEdges(
            left_x=float(chosen["left_x"]),
            right_x=float(chosen["right_x"]),
            image_width=width,
            image_height=height,
            left_response=float(chosen["score"]),
            right_response=float(chosen["score"]),
            contrast=float(chosen["score"]),
            clipped_left=clipped_left,
            clipped_right=clipped_right,
        )

    def _choose(
        self, instances: list[SegmentedInstance], image_width: int
    ) -> SegmentedInstance | None:
        centre = 0.5 * (image_width - 1)
        min_width = self._min_width_frac * image_width
        max_width = self._max_width_frac * image_width
        best: SegmentedInstance | None = None
        best_offset = float("inf")
        for instance in instances:
            if instance["score"] < self._min_score:
                continue
            span = instance["right_x"] - instance["left_x"]
            touches_border = (
                instance["left_x"] <= 1.0 or instance["right_x"] >= image_width - 2.0
            )
            if span < min_width:
                continue
            if span > max_width and not touches_border:
                continue
            offset = abs(0.5 * (instance["left_x"] + instance["right_x"]) - centre)
            if offset < best_offset:
                best = instance
                best_offset = offset
        return best

    def close(self) -> None:
        """Release the service connection."""
        self._client.close()
