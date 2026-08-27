"""Calibrate the ceiling cameras' extrinsics from markers on the floor.

Tape ArUco markers at known floor-tile intersections, list their positions
in a YAML file, and run this on the perception PC. Writes
`camera_params/<lab>/<serial>_extrinsics.json` for each camera and reports
reprojection, per-marker, and cross-camera residuals. See
`prpl_tidybot.marker_detector.calibrate_extrinsics` for details.
"""

from prpl_tidybot.marker_detector.calibrate_extrinsics import cli

if __name__ == "__main__":
    cli()
