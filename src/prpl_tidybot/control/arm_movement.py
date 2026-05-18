"""Wait for the arm to reach a commanded joint configuration.

`RealTidyBotEnv.step()` issues an arm action once via
`Interface.execute_arm_action()` and then calls `reach_target_arm_conf()`
below to block until the arm actually arrives at the commanded joints (or a
max-iteration budget elapses, in which case the env reports the gym step as
truncated). The helper itself only polls — the env is responsible for the
execute call, and the polling-only design is sufficient while the env
operates in a single world frame.
"""

import time

from prpl_tidybot.constants import DEFAULT_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import Interface


def reach_target_arm_conf(
    interface: Interface,
    target_conf: list[float],
    *,
    tolerance: float = 0.01,
    max_iter: int = 100,
    control_period: float = DEFAULT_CONTROL_PERIOD,
) -> bool:
    """Poll the arm joint state until it is within tolerance of target_conf.

    Args:
        interface: TidyBot Interface used to read the current arm state.
        target_conf: 7-D joint configuration to wait for (radians).
        tolerance: Per-joint L-infinity tolerance for convergence.
        max_iter: Maximum number of polls before giving up.
        control_period: Seconds to sleep between polls. Paces the polling so
            the real robot has time to move between reads; tests can set
            this to 0.0 for instant checks.

    Returns:
        True if the arm converged within max_iter polls; False otherwise.
    """
    for _ in range(max_iter):
        time.sleep(control_period)
        cur = interface.get_arm_state()
        if all(abs(c - t) < tolerance for c, t in zip(cur, target_conf)):
            return True
    return False
