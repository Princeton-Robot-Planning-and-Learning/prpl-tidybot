"""Generate a printable PDF of ArUco markers for extrinsics calibration.

Writes conf/calibration/markers_to_print.pdf: 90 mm DICT_4X4_50 markers,
one per half page, each labelled with its id, plus a 100 mm scale bar per
page. Print at 100% scale and verify the bar with a ruler. See
`prpl_tidybot.marker_detector.create_calibration_markers` for options.
"""

from prpl_tidybot.marker_detector.create_calibration_markers import cli

if __name__ == "__main__":
    cli()
