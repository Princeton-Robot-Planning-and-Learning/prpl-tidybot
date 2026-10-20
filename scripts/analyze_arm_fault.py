"""Summarize an arm fault ring-buffer dump written by the cyclic loop.

When the Kinova red-lights, ``TorqueControlledArm.run_cyclic`` writes the last
~0.5 s of per-joint state to ``/tmp/arm_fault_<timestamp>.json`` (see
``third_party/kinova.py``). This reads that file and prints, per joint, the
state in the window leading up to the fault so we can tell a tracking surge
(desired position leading measured while current saturates) from an
acceleration the controller could not deliver.

    python scripts/analyze_arm_fault.py [path]

With no path it analyzes the newest ``/tmp/arm_fault_*.json``.
"""

import glob
import json
import math
import os
import sys

# Current limits the torque loop clamps to (third_party/kinova.py); a joint at
# its limit right before the fault means the controller asked for more torque
# than the actuator could deliver.
_CURRENT_LIMIT = [10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0]


def _latest_dump() -> str:
    matches = sorted(glob.glob("/tmp/arm_fault_*.json"))
    if not matches:
        raise SystemExit("No /tmp/arm_fault_*.json found; pass a path explicitly.")
    return matches[-1]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else _latest_dump()
    records = json.loads(open(path, encoding="utf-8").read())
    if not records:
        raise SystemExit(f"{path} is empty")
    n = len(records)
    dur = records[-1]["t"] - records[0]["t"]
    print(f"{os.path.basename(path)}: {n} ticks over {dur * 1000:.0f} ms")

    last = records[-1]
    faulted = [
        j + 1
        for j in range(len(last["fault_bank_a"]))
        if last["fault_bank_a"][j] or last["fault_bank_b"][j]
    ]
    print(f"Faulted joints: {faulted}")
    print(
        "  bank A bit 1 = FOLLOWING_ERROR, 2 = MAXIMUM_VELOCITY, 4/8 = "
        "JOINT_LIMIT_HIGH/LOW, 32 = MAXIMUM_TORQUE"
    )

    njoints = len(last["q_meas"])
    print()
    print(
        f"{'joint':>5} {'q_meas':>9} {'q_des':>9} {'des-meas':>9} "
        f"{'peak|des-meas|':>14} {'vel':>8} {'peak|vel|':>10} "
        f"{'cur':>7} {'peak|cur|':>10} {'lim':>5}"
    )
    for j in range(njoints):
        q_meas = math.degrees(last["q_meas"][j])
        has_des = last["q_des"] is not None
        q_des = math.degrees(last["q_des"][j]) if has_des else float("nan")
        err = q_des - q_meas if has_des else float("nan")
        peak_err = (
            max(
                abs(r["q_des"][j] - r["q_meas"][j])
                for r in records
                if r["q_des"] is not None
            )
            if has_des
            else float("nan")
        )
        peak_err = math.degrees(peak_err) if has_des else float("nan")
        vel = math.degrees(last["dq_meas"][j])
        peak_vel = math.degrees(max(abs(r["dq_meas"][j]) for r in records))
        cur = last["current_cmd"][j]
        peak_cur = max(abs(r["current_cmd"][j]) for r in records)
        lim = "*" if peak_cur >= 0.98 * _CURRENT_LIMIT[j] else ""
        star = " <<" if (j + 1) in faulted else ""
        print(
            f"{j + 1:>5} {q_meas:>9.1f} {q_des:>9.1f} {err:>9.1f} "
            f"{peak_err:>14.1f} {vel:>8.1f} {peak_vel:>10.1f} "
            f"{cur:>7.2f} {peak_cur:>10.2f} {lim:>5}{star}"
        )
    print()
    print("Angles in degrees, velocities deg/s, current in amps. 'lim' marks a")
    print("joint whose commanded current hit its limit in the window.")


if __name__ == "__main__":
    main()
