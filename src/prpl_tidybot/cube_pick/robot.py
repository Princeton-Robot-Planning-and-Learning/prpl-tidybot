"""The robot port of the pick policy: what it reads and what it commands, per tick.

The policy (:mod:`prpl_tidybot.cube_pick.policy`) talks to a :class:`PickRobot` and
nothing else, so the same policy object drives the MuJoCo simulator
(:class:`prpl_tidybot.cube_pick.kinder_sim.KinderPickRobot`) and the real TidyBot
(:class:`prpl_tidybot.cube_pick.real.RealPickRobot`). Commands are absolute targets
(base pose in the map frame, arm joints, gripper opening in metres), which is the
real arm's interface (`TidyBotAction`); the simulator backend turns them into
KINDER's delta actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_pick.camera import RGBDFrame

FloatArray = NDArray[np.float64]

# Hand-eye calibration: the wrist colour camera (OpenCV axes) in the arm model's
# tool frame. Measured from KINDER's MuJoCo model (camera "wrist" on the bracelet and
# the PyBullet tool frame, which sits 5.2 mm behind MuJoCo's pinch site): the optical
# axis is the approach axis, 11.8 cm behind the tool frame and 5.6 cm off it along the
# tool's y axis. The real robot's value comes from its own hand-eye calibration.
TOOL_FROM_CAMERA = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0559],
        [0.0, 0.0, 1.0, -0.1179],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

GRIPPER_CLOSE = "close"


@dataclass(frozen=True)
class Proprio:
    """The robot's own state, from its encoders and odometry."""

    base: tuple[float, float, float]
    arm: tuple[float, ...]
    gripper_opening: float


@dataclass(frozen=True)
class RobotCommand:
    """One tick's absolute targets; None holds the current value."""

    base: tuple[float, float, float] | None = None
    arm: Sequence[float] | None = None
    gripper: float | str | None = None


class PickRobot(Protocol):
    """What the pick policy needs from a robot, sim or real."""

    def proprio(self) -> Proprio:
        """The current configuration."""

    def wrist_frame(self) -> RGBDFrame:
        """A fresh wrist RGB-D observation (depth registered to colour)."""

    def command(self, cmd: RobotCommand) -> None:
        """Apply one tick of targets and advance one control period."""
