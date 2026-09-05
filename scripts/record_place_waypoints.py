"""Record a teleoperated place demonstration as key waypoints.

Run on the NUC while the arm is under the Kinova gamepad's own control (the
arm server must be DOWN — this script opens its own read-only kortex session)
and the marker detector is up (for the base map pose). The base should stand
where it ought to stand for the place; drive the arm through the placement
and record a handful of key waypoints:

    <Enter>   record a waypoint at the current configuration
    o<Enter>  record a waypoint AND mark it as the release (gripper opens
              after reaching it)
    u<Enter>  undo the last waypoint
    q<Enter>  save and quit

Output JSON: {"base_map": [x, y, rot], "waypoints": [[7 joints (rad)], ...],
"release_index": int}. The joints are kortex positions converted to radians
and wrapped to (-pi, pi], matching the executor's perceived-joint convention.

    python scripts/record_place_waypoints.py --out ~/place_demo.json
"""

import argparse
import json
import math
from pathlib import Path

from prpl_tidybot.marker_detector.client import MarkerDetectorClient
from prpl_tidybot.third_party import kinova


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> int:
    """Interactive waypoint recording loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="~/place_demo.json")
    args = parser.parse_args()
    out_path = Path(args.out).expanduser()

    # Read-only kortex session: TCP base client (gripper reads) + UDP cyclic
    # (joint feedback). Safe alongside the Kinova gamepad's own control; do
    # NOT run while the arm server is up (it owns the arm then).
    kinova._import_kortex()  # pylint: disable=protected-access
    tcp = kinova.DeviceConnection.createTcpConnection()
    udp = kinova.DeviceConnection.createUdpConnection()
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import (  # pylint: disable=import-outside-toplevel,import-error
        BaseCyclicClient,
    )

    tcp.__enter__()
    base_cyclic = BaseCyclicClient(udp.__enter__())

    marker_client = MarkerDetectorClient()

    def read_joints() -> list[float]:
        feedback = base_cyclic.RefreshFeedback()
        return [
            _wrap(math.radians(actuator.position))
            for actuator in feedback.actuators[:7]
        ]

    def read_base() -> list[float] | None:
        data = marker_client.get_latest()
        if "poses" in data and 0 in data["poses"]:
            pose = data["poses"][0]
            return [float(pose[0]), float(pose[1]), float(pose[2])]
        return None

    waypoints: list[list[float]] = []
    release_index: int | None = None
    base_map = None
    print(__doc__)
    try:
        while True:
            joints = read_joints()
            print(
                "current joints:",
                " ".join(f"{j:+.3f}" for j in joints),
                f"| {len(waypoints)} recorded, release at {release_index}",
            )
            command = input("[Enter]=record  o=record+release  u=undo  q=save > ")
            command = command.strip().lower()
            if command == "q":
                break
            if command == "u":
                if waypoints:
                    removed = len(waypoints) - 1
                    waypoints.pop()
                    if release_index == removed:
                        release_index = None
                    print("undid last waypoint")
                continue
            joints = read_joints()
            waypoints.append(joints)
            if base_map is None:
                base_map = read_base()
                print("base map pose:", base_map)
            if command == "o":
                release_index = len(waypoints) - 1
                print(f"waypoint {release_index} marked as the RELEASE")
            else:
                print(f"recorded waypoint {len(waypoints) - 1}")
    finally:
        marker_client.close()
        tcp.__exit__(None, None, None)
        udp.__exit__(None, None, None)

    if not waypoints or release_index is None or base_map is None:
        print(
            "NOT SAVED: need at least one waypoint, a release mark (o), and "
            "a base pose from the marker detector."
        )
        return 1
    out_path.write_text(
        json.dumps(
            {
                "base_map": base_map,
                "waypoints": waypoints,
                "release_index": release_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"saved {len(waypoints)} waypoints (release at {release_index}) to {out_path}"
    )
    return 0


if __name__ == "__main__":
    main()
