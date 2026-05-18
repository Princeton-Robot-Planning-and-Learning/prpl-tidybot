"""Shared constants for the prpl_tidybot package.

Constants that are tied to specific hardware (RPC ports, encoder offsets,
camera serials, the retract arm pose, etc.) intentionally do NOT live here;
they stay behind in prpl-mono/prpl-tidybot until that hardware-controller
layer is migrated.
"""

# Camera frame shapes returned by the wrist (Kinova) and base (Logitech)
# cameras.
BASE_CAMERA_DIMS = (360, 640, 3)
WRIST_CAMERA_DIMS = (480, 640, 3)

# Default delay (seconds) between consecutive polls inside the
# control/*_movement.py convergence helpers. In production this paces polling
# so the real robot has time to respond between reads; in tests it can be
# overridden to 0.0 for instant checks.
DEFAULT_CONTROL_PERIOD = 0.1
