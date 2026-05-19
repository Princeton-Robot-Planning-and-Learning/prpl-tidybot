"""Hardware integration test: drive the real base 20 cm forward and back.

Run on the robot (the base controller server must already be up). Mirrors the upstream
test pattern: a 50-step ramp at policy rate, sending an absolute odom-frame target each
step. Prints the per-step tracking error, then runs the same ramp in reverse to return
to the start pose.

Before running, confirm ~40 cm of clear floor in front of the robot.

python hardware_tests/test_base_forward.py
"""

import sys
import time

from spatialmath import SE2

from prpl_tidybot.interfaces.base_interface import RealBaseInterface
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD

DISTANCE_M = 0.20
N_STEPS = 50


def _ramp(base: RealBaseInterface, start: SE2, distance: float) -> None:
    """Send N_STEPS targets ramping x from start.x to start.x + distance."""
    for i in range(1, N_STEPS + 1):
        progress = i / N_STEPS
        target = SE2(start.x + progress * distance, start.y, start.theta())
        base.execute_action(target)
        current = base.get_base_state()
        print(
            f"  step {i:02d}/{N_STEPS}  "
            f"target_x={target.x:+.3f}  current_x={current.x:+.3f}  "
            f"err={current.x - target.x:+.3f}"
        )
        time.sleep(POLICY_CONTROL_PERIOD)


def main() -> int:
    """Drive 20 cm forward, then 20 cm back to the start pose."""
    print("Connecting to the real base interface...")
    base = RealBaseInterface()
    try:
        start = base.get_base_state()
        print(
            f"Start pose:  x={start.x:.3f}  y={start.y:.3f}  "
            f"theta={start.theta():.3f}"
        )

        print(f"Driving forward {DISTANCE_M:.2f} m...")
        _ramp(base, start, +DISTANCE_M)
        time.sleep(0.5)  # let the OTG settle

        mid = base.get_base_state()
        print(
            f"After forward leg: x={mid.x:.3f}  y={mid.y:.3f}  "
            f"theta={mid.theta():.3f}"
        )
        print(f"  forward error vs target: {(mid.x - start.x) - DISTANCE_M:+.3f} m")

        print("Returning to start...")
        _ramp(base, mid, -DISTANCE_M)
        time.sleep(0.5)

        end = base.get_base_state()
        print(f"Final pose:  x={end.x:.3f}  y={end.y:.3f}  " f"theta={end.theta():.3f}")
        print(f"  total error vs start: {end.x - start.x:+.3f} m")
        return 0
    finally:
        base.close()


if __name__ == "__main__":
    sys.exit(main())
