"""Wait for the gripper to reach a commanded value.

`RealTidyBotEnv.step()` issues a gripper action once via
`Interface.execute_gripper_action()` and then calls `reach_target_gripper()`
below to block until the gripper actually arrives at the commanded value
(or a max-iteration budget elapses, in which case the env reports the gym
step as truncated). The helper itself only polls — the env is responsible
for the execute call.
"""

import time

from prpl_tidybot.constants import DEFAULT_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import Interface


def reach_target_gripper(
    interface: Interface,
    target: float,
    *,
    tolerance: float = 0.01,
    max_iter: int = 100,
    control_period: float = DEFAULT_CONTROL_PERIOD,
) -> bool:
    """Poll the gripper state until it is within tolerance of target.

    Args:
        interface: TidyBot Interface used to read the current gripper state.
        target: Gripper value to wait for (0 = closed, 1 = open).
        tolerance: Absolute convergence threshold.
        max_iter: Maximum number of polls before giving up.
        control_period: Seconds to sleep between polls. Paces the polling so
            the real gripper has time to move between reads; tests can set
            this to 0.0 for instant checks.

    Returns:
        True if the gripper converged within max_iter polls; False otherwise.
    """
    for _ in range(max_iter):
        time.sleep(control_period)
        if abs(interface.get_gripper_state() - target) < tolerance:
            return True
    return False
