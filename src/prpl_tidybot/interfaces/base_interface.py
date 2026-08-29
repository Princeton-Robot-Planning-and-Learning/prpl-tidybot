"""Base interface."""

import abc
import math

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


def unwrap_heading(target: float, reference: float) -> float:
    """``target`` expressed on the same branch of the angle as ``reference``: the
    value within pi of ``reference`` that is equivalent to ``target`` modulo 2 pi.

    The base controller integrates its odometry heading continuously (it can
    read 3.5 rad after a few turns) and tracks a target heading as a plain
    number, while an :class:`SE2` heading is always wrapped to (-pi, pi]. A
    wrapped target handed to the controller when its heading sits past the
    wrap is up to 2 pi away, and the base spins a full turn to reach it.
    """
    return reference + math.atan2(
        math.sin(target - reference), math.cos(target - reference)
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
    """Real base interface. State reading is wired to the TidyBot base
    controller (odom frame) and the marker detector (map frame).
    Action execution is not yet implemented."""

    def __init__(self, marker_detector_host: str = SERVER_HOSTNAME) -> None:
        self.base_manager = BaseManager(
            address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.base_manager.connect()
        self.base = self.base_manager.Base()  # type: ignore # pylint: disable=no-member
        self.base.reset()

        self.marker_detector_client = MarkerDetectorClient(host=marker_detector_host)
        self.last_pose_map = SE2(0, 0, 0)
        # The controller's continuous odometry heading as of the last state
        # read; commanded headings are unwrapped onto its branch.
        self._last_odom_heading = 0.0

    def get_base_state(self) -> SE2:
        base_pose = self.base.get_state()["base_pose"]
        self._last_odom_heading = float(base_pose[2])
        return SE2(base_pose[0], base_pose[1], base_pose[2])

    def get_map_base_state(self) -> SE2:
        detector_data = self.marker_detector_client.get_latest()
        if "poses" in detector_data and 0 in detector_data["poses"]:
            pose_map = detector_data["poses"][0]
            self.last_pose_map = SE2(pose_map[0], pose_map[1], pose_map[2])
        return self.last_pose_map

    def execute_action(self, action: SE2) -> None:
        heading = unwrap_heading(action.theta(), self._last_odom_heading)
        self.base.execute_action({"base_pose": np.array([action.x, action.y, heading])})

    def close(self) -> None:
        """Stop the low-level base control loop."""
        self.base.close()
        self.marker_detector_client.close()
