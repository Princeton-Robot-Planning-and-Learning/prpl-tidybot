"""Hardware integration test: extend / retract the arm via hand-built TidyBotAction
commands, logging base drift.

Bypasses the plan executor entirely. The goal is to validate the underlying
``RealTidyBotEnv.step`` path: handing it a sequence of arm-only commands (where
``base_pose_target_map`` is held at the perceived map pose from reset) should
not cause the real base to move beyond perception / controller noise.

Two phases, each ``N_STEPS_PER_PHASE`` ticks at policy rate:

* extend: ``arm_goal = HOME_ARM_CONF`` (solved via IK on a fixed EE pose, same
  as ``test_arm_ik_home``)
* retract: ``arm_goal = RETRACT_ARM_CONF``

In both phases ``base_pose_target_map`` is held at the initial map pose from
``env.reset()``, ``gripper_goal`` is held at the current perceived value. Per
tick we print the arm joints, the arm error vs. the phase target, the perceived
base pose, and its drift from the initial pose. The arm should reach each phase
target within a few ticks; base drift should stay within marker-detector noise
(a few millimeters in xy, milliradians in theta) throughout.

Requires the base server (on the NUC), arm server (on the NUC), and marker
detector (on the laptop) to be up — easiest via ``./scripts/launch.sh``. The
cameras are stubbed because get_observation otherwise hits the real wrist /
base cameras.

python hardware_tests/test_arm_extend_via_env.py
"""

import sys
from typing import Sequence

import numpy as np

from prpl_tidybot.interfaces.base_interface import RealBaseInterface
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.structs import TidyBotAction
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.third_party.ik_solver import IKSolver

HOME_POS = np.array([0.456, 0.0, 0.434])
HOME_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # (x, y, z, w)

N_STEPS_PER_PHASE = 30


def _run_phase(
    label: str,
    arm_target: Sequence[float],
    env: RealTidyBotEnv,
    obs,
    initial_base_pose,
) -> object:
    """Send ``N_STEPS_PER_PHASE`` arm-only commands with the base target held at the
    initial map pose.

    Returns the final obs.
    """
    target = np.array(arm_target)
    print(f"\n=== {label}: target arm = [{'  '.join(f'{j:+.3f}' for j in target)}]")
    for i in range(N_STEPS_PER_PHASE):
        action = TidyBotAction(
            arm_goal=list(arm_target),
            base_pose_target_map=initial_base_pose,
            gripper_goal=obs.gripper,
        )
        obs, _, _, _, _ = env.step(action)
        arm_err = float(np.linalg.norm(np.array(obs.arm_conf) - target))
        dx = obs.map_base_pose.x - initial_base_pose.x
        dy = obs.map_base_pose.y - initial_base_pose.y
        dtheta = obs.map_base_pose.theta() - initial_base_pose.theta()
        joints_str = "  ".join(f"{j:+.3f}" for j in obs.arm_conf)
        print(
            f"{label} step {i + 1:02d}/{N_STEPS_PER_PHASE}  "
            f"arm=[{joints_str}]  arm_err={arm_err:.4f}  "
            f"base_drift=({dx:+.4f}, {dy:+.4f}, {dtheta:+.4f})"
        )
    return obs


def main() -> int:
    """Extend to HOME, then retract to RETRACT_ARM_CONF, via hand-built
    TidyBotActions."""
    print("Solving IK for home pose (seed = RETRACT_ARM_CONF)...")
    home_arm_conf = (
        IKSolver()
        .solve(HOME_POS, HOME_QUAT, RETRACT_ARM_CONF)  # type: ignore[no-untyped-call]
        .tolist()
    )
    home_str = "  ".join(f"{j:+.3f}" for j in home_arm_conf)
    print(f"HOME joint angles: [{home_str}]")

    print("Bringing up RealTidyBotEnv (real arm + real base; fake cameras)...")
    interface = RealInterface(
        base_interface=RealBaseInterface(),
        camera_interface=FakeCameraInterface(),
    )
    env = RealTidyBotEnv(interface=interface)
    try:
        obs, _ = env.reset()
        initial_base_pose = obs.map_base_pose
        print(
            f"Initial base map pose: "
            f"x={initial_base_pose.x:+.4f}  "
            f"y={initial_base_pose.y:+.4f}  "
            f"theta={initial_base_pose.theta():+.4f}"
        )
        obs = _run_phase("extend", home_arm_conf, env, obs, initial_base_pose)
        _run_phase("retract", RETRACT_ARM_CONF.tolist(), env, obs, initial_base_pose)
        return 0
    finally:
        interface.arm_interface.close()
        interface.base_interface.close()


if __name__ == "__main__":
    sys.exit(main())
