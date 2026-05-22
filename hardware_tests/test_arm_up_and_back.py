"""Hardware integration test: drive the arm to HOME and back to RETRACT via the
streaming arm plan executor.

The motion is split into two trajectories:

  * extend: current joint configuration → HOME (solved via IK on a fixed
    end-effector pose; same as ``test_arm_ik_home``)
  * retract: HOME → ``RETRACT_ARM_CONF``

Each trajectory is discretised into ``--n-waypoints`` evenly spaced joint
configurations (wrap-aware via pybullet-helpers' ``interpolate_joints``)
and converted into kinder ``(state, delta)`` pairs. The two trajectories
are handed to the same ``StreamingArmMotion3DPlanExecutor`` instance
back-to-back via separate ``set_trajectory`` calls — there's no
within-trajectory direction change, so the executor's discrete cursor
visits each waypoint within ``advance_radius`` before advancing.

The point of the test is to validate the executor at two granularities
with one motion:

  * ``--n-waypoints 2`` (coarse): each trajectory is essentially a single
    segment endpoint, and the executor behaves like settle-then-advance.
  * ``--n-waypoints 50`` (fine): the cursor crosses many intermediate
    waypoints under the same threshold, and the arm flows through them
    in cruise without per-waypoint accel/decel cycles.

Trajectory pairs use **cumulative deltas from the starting waypoint** —
every pair shares the same template state (the one perceived at the
start of the trajectory) and the i-th pair's ``action.arm_delta`` is
``waypoint[i+1] - start`` rather than ``waypoint[i+1] - waypoint[i]``.
This is equivalent for the executor's target computation
(``state.joints + action.delta`` is the same absolute target either
way) and avoids having to clone a fresh ObjectCentricState per pair.

Run on the robot (the arm server must already be up):

    python hardware_tests/test_arm_up_and_back.py --n-waypoints 50
"""

import argparse
import sys
from typing import Callable, Sequence

import numpy as np
import pybullet as p
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_jointwise_difference,
    interpolate_joints,
)
from pybullet_helpers.motion_planning import create_joint_distance_fn
from pybullet_helpers.robots.kinova import KinovaGen3NoGripperPyBulletRobot
from relational_structs import ObjectCentricState

from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.interfaces.camera_interface import FakeCameraInterface
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    StreamingArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.third_party.ik_solver import IKSolver

HOME_POS = np.array([0.456, 0.0, 0.434])
HOME_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # (x, y, z, w)


def _build_arm_trajectory(
    joint_infos: list[JointInfo],
    q_start: Sequence[float],
    q_end: Sequence[float],
    n_waypoints: int,
    template_state: ObjectCentricState,
) -> list[tuple[ObjectCentricState, np.ndarray]]:
    """Discretise ``q_start → q_end`` into ``n_waypoints`` kinder (state, action) pairs.

    Every pair shares ``template_state`` (whose robot joints are ``q_start``). The i-th
    pair's ``action.arm_delta`` is the wrap-aware difference from ``q_start`` to
    ``waypoint[i+1]``, so the executor's per-pair absolute target ``state.joints +
    delta`` resolves to ``waypoint[i+1]``.
    """
    waypoints: list[JointPositions] = [
        list(interpolate_joints(joint_infos, list(q_start), list(q_end), t))
        for t in np.linspace(0.0, 1.0, n_waypoints + 1)
    ]
    pairs: list[tuple[ObjectCentricState, np.ndarray]] = []
    for i in range(n_waypoints):
        delta = get_jointwise_difference(joint_infos, waypoints[i + 1], list(q_start))
        action = np.zeros(11)
        action[3:10] = delta
        pairs.append((template_state, action))
    return pairs


def _run_trajectory(
    label: str,
    pairs: list[tuple[ObjectCentricState, np.ndarray]],
    env: RealTidyBotEnv,
    perceiver: PrplLab3DPerceiver,
    executor: Kinematic3DPlanExecutor,
    obs,
    state: ObjectCentricState,
    target: Sequence[float],
    distance_fn: Callable[[Sequence[float], Sequence[float]], float],
) -> tuple[object, ObjectCentricState]:
    """Run one trajectory through the env/executor loop.

    ``err_to_target`` is logged in the same weighted-L1 metric the executor uses for its
    arrival check, so the printed value can be compared directly against
    ``arrival_tolerance``.

    Returns final (obs, state).
    """
    print(f"\n=== {label}: {len(pairs)} pairs ===")
    executor.set_trajectory(pairs)
    step = 0
    while not executor.done(state):
        real_action, _ = executor.step(state)
        cmd_str = "  ".join(f"{j:+.3f}" for j in real_action.arm_goal)
        joints_str = "  ".join(f"{j:+.3f}" for j in obs.arm_conf)
        err = float(distance_fn(list(obs.arm_conf), list(target)))
        print(
            f"step {step + 1:03d}  joints=[{joints_str}]  "
            f"cmd=[{cmd_str}]  err_to_target={err:.4f}"
        )
        obs, _, _, _, info = env.step(real_action)
        state = perceiver.step(obs, info)
        step += 1
    return obs, state


def main() -> int:
    """Move the arm to HOME, then back to RETRACT_ARM_CONF, via the streaming arm
    executor with the given waypoint count."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-waypoints",
        type=int,
        default=50,
        help="Number of waypoints per leg (coarse=2, fine=50).",
    )
    args = parser.parse_args()
    if args.n_waypoints < 1:
        parser.error("--n-waypoints must be >= 1")

    print("Solving IK for home pose (seed = RETRACT_ARM_CONF)...")
    home_arm_conf = (
        IKSolver()
        .solve(HOME_POS, HOME_QUAT, RETRACT_ARM_CONF)  # type: ignore[no-untyped-call]
        .tolist()
    )
    home_str = "  ".join(f"{j:+.3f}" for j in home_arm_conf)
    print(f"HOME joint angles: [{home_str}]")

    print("Building pybullet Kinova arm for the joint distance function...")
    # Standalone 7-DOF Kinova arm (no Robotiq gripper) — its arm_joints is the 7-joint
    # kinematic chain to the end effector, matching the 7-element configs we get from
    # obs.arm_conf. Using TidyBotKinova / KinovaGen3RobotiqGripperPyBulletRobot here
    # would extend arm_joints with the 6 Robotiq finger ids → 13 entries, and the
    # length check in get_jointwise_difference would reject our 7-element configs.
    client_id = p.connect(p.DIRECT)
    pb_robot = KinovaGen3NoGripperPyBulletRobot(physics_client_id=client_id)
    distance_fn = create_joint_distance_fn(
        pb_robot, metric="weighted_joints", weight_base=0.9
    )
    joint_infos = pb_robot.get_arm_joint_infos()

    print("Bringing up RealTidyBotEnv (real arm; fake base/cameras)...")
    interface = RealInterface(
        base_interface=FakeBaseInterface(),
        camera_interface=FakeCameraInterface(),
    )
    env = RealTidyBotEnv(interface=interface)
    perceiver = PrplLab3DPerceiver()
    # arrival_tolerance matches advance_radius: if the cursor was willing to
    # advance through this waypoint mid-trajectory, it's willing to terminate
    # on it as the final. Tighter values (e.g. 0.1) are unreachable against
    # the compliant controller's steady-state lag — joint 5 alone can sit
    # ~0.1 rad off target indefinitely.
    arm_executor = StreamingArmMotion3DPlanExecutor(
        distance_fn=distance_fn,
        advance_radius=0.2,
        arrival_tolerance=0.2,
        max_iter_total=300,
    )
    executor = Kinematic3DPlanExecutor(arm_executor=arm_executor)

    try:
        obs, info = env.reset()
        state = perceiver.reset(obs, info)
        current_arm = list(obs.arm_conf)

        extend_pairs = _build_arm_trajectory(
            joint_infos,
            current_arm,
            home_arm_conf,
            args.n_waypoints,
            template_state=state,
        )
        obs, state = _run_trajectory(
            "extend (current → HOME)",
            extend_pairs,
            env,
            perceiver,
            executor,
            obs,
            state,
            target=home_arm_conf,
            distance_fn=distance_fn,
        )

        # q_start is the perceived arm conf, not home_arm_conf — the OTG may have
        # settled within arrival_tolerance but not exactly on home, and the
        # cumulative-delta trajectory math assumes template_state.joints == q_start.
        retract_pairs = _build_arm_trajectory(
            joint_infos,
            list(obs.arm_conf),
            RETRACT_ARM_CONF.tolist(),
            args.n_waypoints,
            template_state=state,
        )
        obs, state = _run_trajectory(
            "retract (HOME → RETRACT)",
            retract_pairs,
            env,
            perceiver,
            executor,
            obs,
            state,
            target=RETRACT_ARM_CONF.tolist(),
            distance_fn=distance_fn,
        )
        return 0
    finally:
        interface.arm_interface.close()
        p.disconnect(client_id)


if __name__ == "__main__":
    sys.exit(main())
