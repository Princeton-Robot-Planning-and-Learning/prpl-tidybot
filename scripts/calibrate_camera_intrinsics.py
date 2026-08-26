"""Interactive ChArUco intrinsics calibration for a ceiling camera.

Shows a live view; press 'c' to capture board views, Esc to calibrate.
Overwrites `camera_params/<lab>/<serial>.yml` — re-run the extrinsics
calibration afterwards. Run on the perception PC with a display and the
camera servers stopped. See
`prpl_tidybot.marker_detector.calibrate_intrinsics` for details.
"""

from prpl_tidybot.marker_detector.calibrate_intrinsics import cli

if __name__ == "__main__":
    cli()
