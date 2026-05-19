"""Launch the marker-detector server (port 6002).

Spawns the ceiling-camera capture subprocesses, then publishes fused
map-frame robot poses for `RealBaseInterface.get_map_base_state` to consume.
Run this on the perception PC with the two Logitech C930e cameras attached
(see `src/prpl_tidybot/marker_detector/99-webcam.rules` for udev setup).
"""

from prpl_tidybot.marker_detector.server import cli

if __name__ == "__main__":
    cli()
