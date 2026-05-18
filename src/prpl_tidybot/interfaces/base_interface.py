"""Base interface."""

import abc

from spatialmath import SE2


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
