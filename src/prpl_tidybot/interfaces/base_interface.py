"""Base interface."""

import abc
from multiprocessing.connection import Client

import numpy as np
from spatialmath import SE2

from prpl_tidybot.marker_detector.constants import MARKER_DETECTOR_PORT
from prpl_tidybot.third_party.base_server import BaseManager
from prpl_tidybot.third_party.constants import (
    BASE_RPC_HOST,
    BASE_RPC_PORT,
    CONN_AUTHKEY,
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

    def __init__(self) -> None:
        self.base_manager = BaseManager(
            address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.base_manager.connect()
        self.base = self.base_manager.Base()  # type: ignore # pylint: disable=no-member
        self.base.reset()

        self.marker_detector_conn = Client(
            (SERVER_HOSTNAME, MARKER_DETECTOR_PORT), authkey=CONN_AUTHKEY
        )
        self.marker_detector_conn.send(None)
        self.last_pose_map = SE2(0, 0, 0)

    def get_base_state(self) -> SE2:
        base_pose = self.base.get_state()["base_pose"]
        return SE2(base_pose[0], base_pose[1], base_pose[2])

    def get_map_base_state(self) -> SE2:
        # poll() returns as soon as data is ready, so the timeout only adds
        # latency on the first call (before the detector has published) or if
        # the detector stalls.
        if self.marker_detector_conn.poll(timeout=1.0):
            detector_data = self.marker_detector_conn.recv()
            self.marker_detector_conn.send(None)
            robot_idx = 0
            pose_map = detector_data["poses"][robot_idx]
            self.last_pose_map = SE2(pose_map[0], pose_map[1], pose_map[2])
            return self.last_pose_map
        print("warning: no marker detector data received")
        return self.last_pose_map

    def execute_action(self, action: SE2) -> None:
        self.base.execute_action(
            {"base_pose": np.array([action.x, action.y, action.theta()])}
        )

    def close(self) -> None:
        """Stop the low-level base control loop."""
        self.base.close()
