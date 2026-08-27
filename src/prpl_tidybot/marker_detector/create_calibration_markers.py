"""Generate a printable PDF of ArUco markers for extrinsics calibration.

Writes `conf/calibration/markers_to_print.pdf` by default: one 90 mm
`DICT_4X4_50` marker per half page (US letter), labelled with its id, plus
a 100 mm scale bar on every page. Print at 100% scale (no "fit to page")
and check a marker with a ruler — it must measure exactly 90 mm across,
including the black border. The default ids match
`conf/calibration/markers_example.yaml`.

The marker side length does not need to be entered anywhere: the
extrinsics calibration uses only marker centres, so a slightly rescaled
print merely blurs the centre estimate instead of biasing the result. The
scale check still matters for keeping detections sharp and comparable.
"""

import argparse
from pathlib import Path
from typing import Any

import cv2 as cv
import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from prpl_tidybot.marker_detector.constants import MARKER_DICT_ID

# pylint: enable=wrong-import-position

DEFAULT_MARKER_IDS = (0, 1, 2, 3, 4, 5)
DEFAULT_MARKER_LENGTH_MM = 90.0
DEFAULT_OUTPUT = (
    Path(__file__).parents[3] / "conf" / "calibration" / "markers_to_print.pdf"
)

_PAGE_SIZE_IN = (8.5, 11.0)  # US letter
_MARKERS_PER_PAGE = 2
_MM_PER_IN = 25.4
# Center-axis ticks sit this far outside the marker (preserving the white
# quiet zone the detector needs) and extend this long.
_TICK_GAP_IN = 6.0 / _MM_PER_IN
_TICK_LENGTH_IN = 6.0 / _MM_PER_IN


def create_marker_pdf(
    path: Path,
    marker_ids: tuple[int, ...] = DEFAULT_MARKER_IDS,
    marker_length_mm: float = DEFAULT_MARKER_LENGTH_MM,
) -> int:
    """Write the marker sheet to `path`; returns the number of pages."""
    aruco: Any = cv.aruco
    dictionary = aruco.getPredefinedDictionary(MARKER_DICT_ID)
    page_w, page_h = _PAGE_SIZE_IN
    side_in = marker_length_mm / _MM_PER_IN
    width_frac = side_in / page_w
    height_frac = side_in / page_h
    row_tops = (0.61, 0.15)  # bottom-left y of each marker's axes
    bar_frac = (100.0 / _MM_PER_IN) / page_w

    pages = [
        marker_ids[i : i + _MARKERS_PER_PAGE]
        for i in range(0, len(marker_ids), _MARKERS_PER_PAGE)
    ]
    with PdfPages(path) as pdf:
        for page_ids in pages:
            fig = plt.figure(figsize=(page_w, page_h))
            for row, marker_id in enumerate(page_ids):
                image = aruco.generateImageMarker(dictionary, marker_id, 600)
                axes = fig.add_axes(
                    ((1 - width_frac) / 2, row_tops[row], width_frac, height_frac)
                )
                axes.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
                axes.set_axis_off()
                # Center-line ticks: the calibration YAML positions refer to
                # marker centres, so extend each centre axis outward (clear
                # of the marker's quiet zone) for lining up with floor marks.
                center_y = row_tops[row] + height_frac / 2
                gap_w = _TICK_GAP_IN / page_w
                tick_w = _TICK_LENGTH_IN / page_w
                gap_h = _TICK_GAP_IN / page_h
                tick_h = _TICK_LENGTH_IN / page_h
                half_w = width_frac / 2
                half_h = height_frac / 2
                for x_start, x_end in (
                    (0.5 - half_w - gap_w - tick_w, 0.5 - half_w - gap_w),
                    (0.5 + half_w + gap_w, 0.5 + half_w + gap_w + tick_w),
                ):
                    fig.add_artist(
                        Line2D(
                            [x_start, x_end],
                            [center_y, center_y],
                            transform=fig.transFigure,
                            color="black",
                            linewidth=0.8,
                        )
                    )
                for y_start, y_end in (
                    (center_y - half_h - gap_h - tick_h, center_y - half_h - gap_h),
                    (center_y + half_h + gap_h, center_y + half_h + gap_h + tick_h),
                ):
                    fig.add_artist(
                        Line2D(
                            [0.5, 0.5],
                            [y_start, y_end],
                            transform=fig.transFigure,
                            color="black",
                            linewidth=0.8,
                        )
                    )
                fig.text(
                    0.5,
                    row_tops[row] - 0.055,
                    f"ArUco DICT_4X4_50  id {marker_id}  —  "
                    f"{marker_length_mm:.0f} mm across (incl. black border); "
                    "ticks mark the centre axes",
                    ha="center",
                    fontsize=11,
                )
            fig.add_artist(
                Line2D(
                    [0.5 - bar_frac / 2, 0.5 + bar_frac / 2],
                    [0.055, 0.055],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=2,
                )
            )
            fig.text(
                0.5,
                0.035,
                "scale check: this bar must measure exactly 100 mm "
                "(print at 100% scale, not fit-to-page)",
                ha="center",
                fontsize=9,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return len(pages)


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...` and `scripts/`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_MARKER_IDS),
        help="ArUco ids to include (avoid the robot/target ids).",
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=DEFAULT_MARKER_LENGTH_MM,
        help="Printed side length of each marker, black border included.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    args = parser.parse_args()
    num_pages = create_marker_pdf(
        args.out, tuple(args.ids), marker_length_mm=args.marker_length_mm
    )
    print(f"Wrote {num_pages} page(s) to {args.out}")


if __name__ == "__main__":
    cli()
