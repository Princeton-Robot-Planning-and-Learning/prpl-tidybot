"""Generate the printable ChArUco board for intrinsics calibration.

Writes conf/calibration/charuco_board.pdf (240 x 168 mm board on landscape
US letter with a 100 mm scale bar). Print at 100% scale and keep it flat.
See `prpl_tidybot.marker_detector.create_charuco_board` for details.
"""

from prpl_tidybot.marker_detector.create_charuco_board import cli

if __name__ == "__main__":
    cli()
