"""Re-annotate a ceiling camera's floor-alignment corners.

Interactive: shows a live undistorted view from the chosen camera with the
current corner quad overlaid; double-clicks move the nearest corner and are
saved immediately to the camera's alignment JSON under
`src/prpl_tidybot/marker_detector/camera_params/`. Run on the perception PC
with a display and the camera servers stopped. See
`prpl_tidybot.marker_detector.annotate_alignment` for details.
"""

from prpl_tidybot.marker_detector.annotate_alignment import cli

if __name__ == "__main__":
    cli()
