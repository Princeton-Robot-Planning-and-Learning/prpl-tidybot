"""Action grounder for kinder kinematic3d envs (PrplLab3D, BaseMotion3D, …).

The 11-d action layout is shared across kinematic3d envs (they all use
`Kinematic3DRobotActionSpace`), so a single grounder works for any of
them — pick this class regardless of which kinematic3d env you're driving.
"""

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import ActionGrounder
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.structs import TidyBotAction


class Kinematic3DActionGrounder(
    ActionGrounder[NDArray[np.floating], TidyBotAction, ObjectCentricState]
):
    """Map a kinematic3d 11-d sim action to an absolute TidyBotAction.

    The kinematic3d action layout is [dx, dy, drot, dj1..dj7, gripper_cmd].
    Base and arm components are componentwise (world-frame) deltas that
    `kinder.envs.kinematic3d.base_env.step` adds to the current robot
    state via `SE2Pose.__add__` / `current_joints + delta_joints`. The
    gripper command is in [-1, 1] with `<-0.5` = close / `>0.5` = open /
    else = no change.

    The grounder reads the current robot features from the sim state
    (which the perceiver populated from the real observation) and emits a
    TidyBotAction whose base/arm components are absolute targets — that
    matches what `Interface.execute_action` expects from the old odom-
    frame controller path, and what the Fake variants already store.

    Gripper convention: kinder uses bipolar `<-0.5 close / >0.5 open`;
    TidyBotAction uses absolute 0..1 with 1 = closed (per
    TidyBotObservation.gripper). The "no change" branch passes through
    the perceiver-written `finger_state`, which is in the same convention
    as TidyBotAction.gripper_goal.
    """

    def __init__(self, robot_name: str = "robot") -> None:
        self._robot_name = robot_name

    def __call__(
        self,
        sim_action: NDArray[np.floating],
        sim_state: ObjectCentricState,
    ) -> TidyBotAction:
        robot = sim_state.get_object_from_name(self._robot_name)

        base_goal = SE2(
            x=sim_state.get(robot, "pos_base_x") + float(sim_action[0]),
            y=sim_state.get(robot, "pos_base_y") + float(sim_action[1]),
            theta=sim_state.get(robot, "pos_base_rot") + float(sim_action[2]),
        )

        arm_goal = [
            sim_state.get(robot, f"joint_{j + 1}") + float(sim_action[3 + j])
            for j in range(7)
        ]

        gripper_cmd = float(sim_action[10])
        if gripper_cmd < -0.5:
            gripper_goal = 1.0  # close
        elif gripper_cmd > 0.5:
            gripper_goal = 0.0  # open
        else:
            gripper_goal = sim_state.get(robot, "finger_state")

        return TidyBotAction(
            arm_goal=arm_goal,
            base_local_goal=base_goal,
            gripper_goal=gripper_goal,
        )
