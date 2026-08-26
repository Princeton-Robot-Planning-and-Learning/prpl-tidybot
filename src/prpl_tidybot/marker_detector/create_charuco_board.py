"""Generate the printable ChArUco board for intrinsics calibration.

Writes `conf/calibration/charuco_board.pdf`: the 10x7 board from
`CHARUCO_BOARD_PARAMS` at its exact physical size (240 x 168 mm) on
landscape US letter, plus a 100 mm scale bar. Print at 100% scale (no
"fit to page") and verify the bar with a ruler; mount the print as flat
as possible (e.g. on a clipboard).

Adapted from `yixuanhuang98/tidybot_server/server/create_charuco_board.py`,
rewritten for the OpenCV 4.7+ aruco API.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from prpl_tidybot.marker_detector.calibrate_intrinsics import make_board
from prpl_tidybot.marker_detector.constants import CHARUCO_BOARD_PARAMS

# pylint: enable=wrong-import-position

DEFAULT_OUTPUT = (
    Path(__file__).parents[3] / "conf" / "calibration" / "charuco_board.pdf"
)

_PAGE_SIZE_IN = (11.0, 8.5)  # US letter, landscape
_MM_PER_IN = 25.4
_PIXELS_PER_SQUARE = 60


def create_charuco_board_pdf(path: Path) -> None:
    """Write the board sheet to `path`."""
    squares_x = CHARUCO_BOARD_PARAMS["squares_x"]
    squares_y = CHARUCO_BOARD_PARAMS["squares_y"]
    board_image = make_board().generateImage(
        (squares_x * _PIXELS_PER_SQUARE, squares_y * _PIXELS_PER_SQUARE)
    )
    board_w_in = squares_x * CHARUCO_BOARD_PARAMS["square_length"] * 1000 / _MM_PER_IN
    board_h_in = squares_y * CHARUCO_BOARD_PARAMS["square_length"] * 1000 / _MM_PER_IN
    page_w, page_h = _PAGE_SIZE_IN
    width_frac = board_w_in / page_w
    height_frac = board_h_in / page_h
    bar_frac = (100.0 / _MM_PER_IN) / page_w

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(page_w, page_h))
        axes = fig.add_axes(
            (
                (1 - width_frac) / 2,
                (1 - height_frac) / 2 + 0.03,
                width_frac,
                height_frac,
            )
        )
        axes.imshow(board_image, cmap="gray", interpolation="nearest", aspect="auto")
        axes.set_axis_off()
        fig.text(
            0.5,
            (1 - height_frac) / 2 - 0.02,
            f"ChArUco {squares_x}x{squares_y}, "
            f"{CHARUCO_BOARD_PARAMS['square_length'] * 1000:.0f} mm squares — "
            "keep flat; used for camera intrinsics calibration",
            ha="center",
            fontsize=10,
        )
        fig.add_artist(
            Line2D(
                [0.5 - bar_frac / 2, 0.5 + bar_frac / 2],
                [0.06, 0.06],
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


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...` and `scripts/`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path."
    )
    args = parser.parse_args()
    create_charuco_board_pdf(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    cli()
