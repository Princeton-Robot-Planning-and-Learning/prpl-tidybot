# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the mobile base
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

# mypy: ignore-errors
# pylint: disable=all

import threading
import time
from multiprocessing.managers import BaseManager as MPBaseManager

from prpl_tidybot.third_party.base_controller import Vehicle
from prpl_tidybot.third_party.constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY


class Base:
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None
        self._loop_lock = threading.Lock()

    def reset(self):
        # Locked so two clients' RPC threads cannot both start a loop.
        with self._loop_lock:
            # Stop low-level control
            if self.vehicle is not None:
                if self.vehicle.control_loop_running:
                    self.vehicle.stop_control()

            # Create new instance of vehicle
            self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)

            # Start low-level control
            self.vehicle.start_control()
            while not self.vehicle.control_loop_running:
                time.sleep(0.01)

    def execute_action(self, action):
        self.vehicle.set_target_position(action["base_pose"])

    def get_state(self):
        state = {"base_pose": self.vehicle.x}
        return state

    def close(self):
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()


class BaseManager(MPBaseManager):
    pass


# Every client gets this one Base, so a client's reset() stops the previous client's
# Vehicle control loop instead of starting a second one beside it. Registering the
# class gave each client its own Base, and a client that died without close() left
# its loop running (two loops found on the robot, 2026-10-16).
_shared_base = None
_shared_base_lock = threading.Lock()


def _get_shared_base():
    global _shared_base
    with _shared_base_lock:
        if _shared_base is None:
            _shared_base = Base()
        return _shared_base


BaseManager.register("Base", _get_shared_base)

if __name__ == "__main__":
    manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f"Base manager server started at {BASE_RPC_HOST}:{BASE_RPC_PORT}")
    server.serve_forever()
