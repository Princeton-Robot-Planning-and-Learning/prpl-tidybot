"""Retract the real arm and verify completion before allowing shutdown.

Default: use the running arm RPC server. With --shutdown, a refused RPC
connection selects the stored Kinova Retract action instead. Neither path
opens the gripper or runs an implicit reset/home action.
"""

import argparse

from prpl_tidybot.sprout.shutdown_arm import main as retract_main


def main() -> int:
    """Run verified retraction using the selected control connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Allow direct Retract when the arm server is not running",
    )
    args = parser.parse_args()
    retract_main(server=True, auto=args.shutdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
