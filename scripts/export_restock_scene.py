"""Build or verify an alphatamp Restock3D scene.yaml from marker detections.

Reads the ceiling marker detector once (or a recorded payload via
--from-json), transforms each object's floor-marker position into
alphatamp's home frame using the shelf staging in the lab config, and
writes the scene.yaml that alphatamp's deploy.py plans from. With
--verify it instead reports each object's distance from a previously
written scene — the lab-day staging check. See
`prpl_tidybot.restock_scene` for options and the config format.
"""

import sys

from prpl_tidybot.restock_scene import cli

if __name__ == "__main__":
    sys.exit(cli())
