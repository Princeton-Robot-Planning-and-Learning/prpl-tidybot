# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

# mypy: ignore-errors
# pylint: disable=all

import queue
import threading
import time
from multiprocessing.managers import BaseManager as MPBaseManager

import numpy as np

from prpl_tidybot.third_party.arm_controller import JointCompliantController
from prpl_tidybot.third_party.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from prpl_tidybot.third_party.ik_solver import IKSolver
from prpl_tidybot.third_party.kinova import TorqueControlledArm
from prpl_tidybot.toss import (
    TOSS_GRIPPER_RELEASE_MS,
    TOSS_MAX_VEL,
    TOSS_WINDUP_TOLERANCE_DEG,
    max_joint_error_deg,
    plan_toss_swing,
    toss_windup_pose,
)

# Guards starting and stopping the torque loop, so two clients' RPC threads cannot
# both see it stopped and both start one.
_LOOP_LOCK = threading.RLock()


class Arm:
    def __init__(self):
        self._connect()
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)

    def _connect(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(
            speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,))
        )

    def reset(self, reset_arm: bool = True):
        if self.arm is None:  # A previous client close()d the shared Arm
            self._connect()
        self.release()

        # Clear faults
        self.arm.clear_faults()

        # Reset arm configuration

        if reset_arm:
            self.arm.open_gripper()
            self.arm.retract()

        self.resume()

    def release(self):
        # Stop low-level control. The arm returns to high-level servoing,
        # where its onboard controller (and any gamepad plugged into the
        # Kinova base) is in charge.
        with _LOOP_LOCK:
            if self.arm.cyclic_running:
                time.sleep(0.75)  # Wait for arm to stop moving
                self.arm.stop_cyclic()

    def is_cyclic_running(self):
        return self.arm is not None and self.arm.cyclic_running

    def resume(self):
        with _LOOP_LOCK:
            # Stop a loop that is already running rather than starting a second one
            # beside it: two 1 kHz loops commanding one arm with different targets
            # is what made it flip out.
            self.release()
            self._resume()

    def _resume(self):
        # Drain any leftover commands from the previous session. The
        # command_queue is reused across sessions (passed into each new
        # JointCompliantController), so a stale (qpos, gripper) item from
        # the previous session would otherwise be the first thing the new
        # controller reads, briefly steering the OTG toward the previous
        # session's target before the current session's first command
        # arrives. Mirrors issue #54's pattern for the base.
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

        # Create new instance of controller; it initialises at the arm's
        # current configuration, so the arm holds wherever it is.
        self.controller = JointCompliantController(self.command_queue)

        # Start low-level control
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action(self, action):
        qpos = self.ik_solver.solve(action["arm_pos"], action["arm_quat"], self.arm.q)
        self.command_queue.put((qpos, action["gripper_pos"].item()))

    def execute_action_angular(self, qpos, gripper_pos):
        self.command_queue.put((qpos, gripper_pos))

    def toss(
        self,
        heading=0.0,
        max_vel=TOSS_MAX_VEL,
        gripper_release_ms=TOSS_GRIPPER_RELEASE_MS,
    ):
        # Jimmy Wu's KinovaArm.toss: a high-level move to the wind-up pose,
        # then a trapezoidal position-servo swing at 1 kHz that opens the
        # gripper at cycle gripper_release_ms. The torque loop is stopped for
        # the duration (release() leaves the actuators in position mode under
        # high-level servoing, which is where Jimmy's toss starts) and
        # restarted afterwards from wherever the swing ended.
        windup = toss_windup_pose(heading)
        # Validate the parameters before anything moves.
        plan_toss_swing(
            windup, heading, max_vel=max_vel, gripper_release_ms=gripper_release_ms
        )
        self.release()
        try:
            self.arm.move_angular(windup)
            error = max_joint_error_deg(self.arm.get_measured_joint_angles(), windup)
            if error >= TOSS_WINDUP_TOLERANCE_DEG:
                raise RuntimeError(
                    f"Arm stopped {error:.2f} deg from the toss wind-up pose "
                    f"(tolerance {TOSS_WINDUP_TOLERANCE_DEG} deg); not swinging."
                )
            trajectory = plan_toss_swing(
                self.arm.get_feedback_joint_positions(),
                heading,
                max_vel=max_vel,
                gripper_release_ms=gripper_release_ms,
            )
            swing_start = time.time()
            self.arm.run_position_trajectory(
                trajectory, gripper_release_step=gripper_release_ms
            )
        finally:
            self.resume()
        # The wind-up takes seconds; callers align video to the swing itself.
        return {"swing_start_time": swing_start}

    def move_joints_angular(self, qpos):
        # One blocking move under the arm's onboard position controller (as Jimmy
        # Wu's original pick_object), between release() and resume(). Radians in,
        # wrapped radians of the measured pose out.
        if self.arm.cyclic_running:
            raise RuntimeError("release() the torque loop before an angular move")
        self.arm.move_angular([float(v) for v in np.degrees(qpos) % 360.0])
        measured = np.radians(self.arm.get_measured_joint_angles())
        return [float(v) for v in self.radians_clip(measured)]

    def gripper_angular(self, value, timeout=4.0):
        # Gripper move under the onboard controller; returns where it stopped.
        if self.arm.cyclic_running:
            raise RuntimeError("release() the torque loop before an angular move")
        return float(self.arm.gripper_to(value, timeout))

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        return {
            "arm_pos": arm_pos,
            "arm_quat": arm_quat,
            "gripper_pos": np.array([self.arm.gripper_pos]),
        }

    def get_gripper_position(self):
        return self.arm.gripper_pos

    def radians_clip(self, radians):
        return (radians + np.pi) % (2 * np.pi) - np.pi

    def get_joint_angles(self):
        return self.radians_clip(self.arm.q)

    def close(self):
        if self.arm is None:
            return
        self.release()
        self.arm.disconnect()
        # The Arm is shared, so the next client's reset() reconnects (as each
        # client used to get a fresh session).
        self.arm = None


class ArmManager(MPBaseManager):
    pass


# Every client gets this one Arm. Registering the class itself built a new Arm (a new
# Kortex session and torque loop) per client, all in this process, so the PID-file
# guard never fired; a client that died without close() left its loop running
# beside the next client's, and the arm flipped out (2026-10-16, runs 21c/21d).
_shared_arm = None
_shared_arm_lock = threading.Lock()


def _get_shared_arm():
    global _shared_arm
    with _shared_arm_lock:
        if _shared_arm is None:
            _shared_arm = Arm()
        return _shared_arm


ArmManager.register("Arm", _get_shared_arm)


if __name__ == "__main__":
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f"Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}")
    server.serve_forever()
