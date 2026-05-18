"""Wait for the base to reach a commanded SE2 pose.

`RealTidyBotEnv.step()` issues a base action once via
`Interface.execute_base_action()` and then calls `reach_target_pose()` below
to block until the base actually arrives at the commanded pose (or a
max-iteration budget elapses, in which case the env reports the gym step as
truncated). The helper itself only polls — the env is responsible for the
execute call, and the polling-only design is sufficient while the env
operates in a single world frame.
"""

import math
import time

from spatialmath import SE2

from prpl_tidybot.constants import DEFAULT_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import Interface


def reach_target_pose(
    interface: Interface,
    target_pose: SE2,
    *,
    tolerance: float = 0.01,
    max_iter: int = 100,
    control_period: float = DEFAULT_CONTROL_PERIOD,
) -> bool:
    """Poll the base pose until it is within tolerance of target_pose.

    Args:
        interface: TidyBot Interface used to read the current base pose.
        target_pose: SE2 pose to wait for (same frame as
            `interface.get_base_state()`).
        tolerance: Convergence threshold; applies independently to the
            (x, y) Euclidean distance and the heading-angle error.
        max_iter: Maximum number of polls before giving up.
        control_period: Seconds to sleep between polls. Paces the polling so
            the real robot has time to move between reads; tests can set
            this to 0.0 for instant checks.

    Returns:
        True if the base converged within max_iter polls; False otherwise.
    """
    for _ in range(max_iter):
        time.sleep(control_period)
        cur = interface.get_base_state()
        pos_err = math.hypot(cur.x - target_pose.x, cur.y - target_pose.y)
        ang_err = abs(cur.theta() - target_pose.theta())
        if pos_err < tolerance and ang_err < tolerance:
            return True
    return False
