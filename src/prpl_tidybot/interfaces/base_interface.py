"""Base interface."""

import abc
import time

import numpy as np
from spatialmath import SE2

from prpl_tidybot.marker_detector.client import MarkerDetectorClient
from prpl_tidybot.third_party.base_server import BaseManager
from prpl_tidybot.third_party.constants import (
    BASE_RPC_HOST,
    BASE_RPC_PORT,
    RPC_AUTHKEY,
    SERVER_HOSTNAME,
)


class BaseInterface(abc.ABC):
    """Base interface."""

    @abc.abstractmethod
    def get_base_state(self) -> SE2:
        """Get the current base pose in the odom frame."""

    @abc.abstractmethod
    def get_map_base_state(self) -> SE2:
        """Get the current base pose in the map frame."""

    @abc.abstractmethod
    def execute_action(self, action: SE2) -> None:
        """Execute a local-frame base action."""

    def close(self) -> None:
        """Tear down hardware connections; real-resource subclasses override."""


class FakeBaseInterface(BaseInterface):
    """Fake base interface that stores commanded poses in memory."""

    def __init__(self) -> None:
        self.base_state: SE2 = SE2(x=0, y=0, theta=0)
        self.map_base_state: SE2 = SE2(x=0, y=0, theta=0)

    def get_base_state(self) -> SE2:
        return self.base_state

    def get_map_base_state(self) -> SE2:
        return self.map_base_state

    def execute_action(self, action: SE2) -> None:
        self.base_state = action
        self.map_base_state = action


class RealBaseInterface(BaseInterface):
    """Real base interface.

    State reading is wired to the TidyBot base controller (odom frame) and the marker
    detector (map frame).
    """

    _ROBOT_MARKER_ID = 0

    def __init__(self, marker_detector_host: str = SERVER_HOSTNAME) -> None:
        self.base_manager = BaseManager(
            address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.base_manager.connect()
        self.base = self.base_manager.Base()  # type: ignore # pylint: disable=no-member
        self.base.reset()

        self.marker_detector_client = MarkerDetectorClient(host=marker_detector_host)
        self.last_pose_map = self._wait_for_initial_map_pose()

    def _wait_for_initial_map_pose(self, deadline_s: float = 10.0) -> SE2:
        """Block until the detector reports the robot marker once.

        `get_map_base_state` reads are non-blocking cached lookups, so the
        first payload has to be guaranteed here — otherwise a rollout could
        start from a bogus identity map pose and mis-calibrate the map->odom
        converter.
        """
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            data = self.marker_detector_client.get_latest()
            if "poses" in data and self._ROBOT_MARKER_ID in data["poses"]:
                pose_map = data["poses"][self._ROBOT_MARKER_ID]
                return SE2(pose_map[0], pose_map[1], pose_map[2])
        raise RuntimeError(
            "Marker detector never reported the robot marker "
            f"(id {self._ROBOT_MARKER_ID}) within {deadline_s:.0f}s. Is the "
            "marker detector server running and the robot marker visible to "
            "the ceiling camera?"
        )

    def get_base_state(self) -> SE2:
        base_pose = self.base.get_state()["base_pose"]
        return SE2(base_pose[0], base_pose[1], base_pose[2])

    def get_map_base_state(self) -> SE2:
        # Non-blocking read: return the freshest already-delivered payload
        # instead of waiting for the publisher's next refresh. This keeps a
        # 10 Hz control loop's tick near its nominal period — a blocking
        # read stalls each tick by up to one publisher period, which
        # stretched the gap between consecutive arm/base commands past the
        # controllers' "no command in 2.5 * POLICY_CONTROL_PERIOD" watchdog
        # and made the OTG brake between commands. The pose returned here is
        # at most one publisher period staler than the blocking read's.
        detector_data = self.marker_detector_client.get_latest(timeout_s=0.0)
        if "poses" in detector_data and self._ROBOT_MARKER_ID in detector_data["poses"]:
            pose_map = detector_data["poses"][self._ROBOT_MARKER_ID]
            self.last_pose_map = SE2(pose_map[0], pose_map[1], pose_map[2])
        return self.last_pose_map

    def execute_action(self, action: SE2) -> None:
        self.base.execute_action(
            {"base_pose": np.array([action.x, action.y, action.theta()])}
        )

    def close(self) -> None:
        """Stop the low-level base control loop."""
        self.base.close()
        self.marker_detector_client.close()
