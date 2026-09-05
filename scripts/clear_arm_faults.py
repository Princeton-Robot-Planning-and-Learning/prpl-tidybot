"""Clear a faulted Kinova arm (the red-light condition) from the command line.

Run this on the NUC (which is on the arm's 192.168.1.10 network) when the arm
red-lights mid-run and the Kinova web app is not reachable from the laptop:

    python -m scripts.clear_arm_faults

It opens its own TCP session to the base, prints the arm state and the
per-actuator fault flags (so we can see *which* joint tripped and whether it is
a position/limit fault versus a velocity/effort one), calls ClearFaults, and
waits for the arm to return to SERVOING_READY. Stop the arm server first if it
still holds the connection.
"""

import time

from prpl_tidybot.third_party.kinova import DeviceConnection, _import_kortex

_import_kortex()

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


def _state_name(state: int) -> str:
    return Base_pb2.ArmState.Name(state)


def main() -> None:
    with DeviceConnection.createTcpConnection() as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        before = base.GetArmState().active_state
        print(f"Arm state: {_state_name(before)}")

        feedback = base_cyclic.RefreshFeedback()
        faulted = False
        for i, actuator in enumerate(feedback.actuators):
            bank_a = actuator.fault_bank_a
            bank_b = actuator.fault_bank_b
            if bank_a or bank_b:
                faulted = True
                print(
                    f"  joint {i + 1}: fault_bank_a=0x{bank_a:08x} "
                    f"fault_bank_b=0x{bank_b:08x} "
                    f"position={actuator.position:.2f}deg "
                    f"velocity={actuator.velocity:.2f}deg/s "
                    f"torque={actuator.torque:.2f}Nm"
                )
        if not faulted:
            print("  no per-actuator fault flags set")

        if before == Base_pb2.ARMSTATE_IN_FAULT:
            print("Clearing faults...")
            base.ClearFaults()
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if base.GetArmState().active_state == Base_pb2.ARMSTATE_SERVOING_READY:
                    print("Arm is SERVOING_READY.")
                    return
                time.sleep(0.1)
            print(f"Timed out; arm state: {_state_name(base.GetArmState().active_state)}")
        else:
            print("Arm is not in fault; nothing to clear.")


if __name__ == "__main__":
    main()
