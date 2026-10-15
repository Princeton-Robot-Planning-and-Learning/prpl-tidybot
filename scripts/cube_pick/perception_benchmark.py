"""Perception accuracy: SAM 2 + depth fit vs SAM 2 + FoundationPose, against ground
truth.

For each recorded state the robot is placed (kinematically: base pose and arm joints
written into the state, no dynamics) at a base pose facing the cube and an arm
configuration whose wrist camera looks at the cube from one of a few viewpoints; the
wrist RGB-D frame is rendered exactly as the pick sees it, and both backends estimate
the cube's pose from the same frame and the same prior. Errors are measured against the
simulator's cube pose: centre position (mm) and yaw modulo the cube's 90-degree symmetry
(deg). A backend that returns no estimate counts as a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402  pylint: disable=wrong-import-position

from prpl_tidybot.cube_pick.cube_fit import (  # noqa: E402
    cube_yaw_and_up,
    yaw_error_mod_90,
)
from prpl_tidybot.cube_pick.kinder_sim import KinderPickRobot  # noqa: E402
from prpl_tidybot.cube_pick.perception import CubeEstimate, CubePrior  # noqa: E402
from prpl_tidybot.cube_pick.planning import PlanningScene  # noqa: E402
from prpl_tidybot.cube_pick.robot import TOOL_FROM_CAMERA  # noqa: E402

# (tilt from vertical, azimuth relative to the base->cube direction, camera distance)
VIEWS = {
    "overhead": (0.0, 0.0, 0.36),
    "oblique25": (np.radians(25), np.pi, 0.32),
    "oblique45": (np.radians(45), np.pi, 0.28),
}


def camera_pose(
    target: np.ndarray,
    base_heading: float,
    tilt: float,
    azimuth: float,
    distance: float,
) -> np.ndarray:
    direction = np.array(
        [
            np.sin(tilt) * np.cos(base_heading + azimuth),
            np.sin(tilt) * np.sin(base_heading + azimuth),
            np.cos(tilt),
        ]
    )
    z_axis = -direction
    x_axis = np.array(
        [np.cos(base_heading + np.pi / 2), np.sin(base_heading + np.pi / 2), 0.0]
    )
    x_axis -= (x_axis @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis], axis=1)
    pose[:3, 3] = target + direction * distance
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--rows", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=50111)
    parser.add_argument("--prior-noise", type=float, default=0.02)
    args = parser.parse_args()
    rows = json.load(open(args.states))
    sim = harness.TossingSim(perception_port=args.port)
    estimators = {
        "sam2+depthfit": sim.estimator("sam2"),
        "sam2+foundationpose": sim.estimator("foundationpose"),
    }
    with open(args.out, "a") as out:
        for index in harness.parse_rows(args.rows, len(rows)):
            row = rows[index]
            state = sim.restore(row["state"])
            cube = state.get_object_from_name(harness.CUBE)
            robot = state.get_object_from_name(sim.robot_obj.name)
            truth = np.array([state.get(cube, k) for k in ("x", "y", "z")])
            from scipy.spatial.transform import (  # pylint: disable=import-outside-toplevel
                Rotation,
            )

            true_rot = Rotation.from_quat(
                [state.get(cube, k) for k in ("qx", "qy", "qz", "qw")]
            ).as_matrix()
            true_yaw, true_up = cube_yaw_and_up(true_rot)
            rng = np.random.default_rng(zlib.crc32(row["id"].encode()))
            prior_pos = truth.copy()
            prior_pos[:2] += rng.normal(scale=args.prior_noise, size=2)
            scene = PlanningScene(
                state, cube_name=harness.CUBE, containers=sim.containers(state)
            )
            try:
                # A collision-free base pose 0.55 m from the cube, facing it.
                for k in range(12):
                    heading = (
                        np.arctan2(
                            truth[1] - state.get(robot, "pos_base_y"),
                            truth[0] - state.get(robot, "pos_base_x"),
                        )
                        + k * np.pi / 6
                    )
                    base = (
                        truth[0] - 0.55 * np.cos(heading),
                        truth[1] - 0.55 * np.sin(heading),
                        heading,
                    )
                    if not scene.base_pose_collides(base):
                        break
                scene.set_robot(base, scene.arm_joints())
                for view, (tilt, azimuth, distance) in VIEWS.items():
                    cam = camera_pose(prior_pos, base[2], tilt, azimuth, distance)
                    joints = scene.ik(
                        cam @ np.linalg.inv(TOOL_FROM_CAMERA),
                        seed=[0, -0.35, 3.14, -2.5, 0, -0.9, 1.57],
                    )
                    record = {
                        "row": index,
                        "id": row["id"],
                        "d4": row["d4"],
                        "view": view,
                    }
                    if joints is None:
                        record["skipped"] = "no ik for the view"
                        out.write(json.dumps(record) + "\n")
                        continue
                    placed = state.copy()
                    placed.set(robot, "pos_base_x", base[0])
                    placed.set(robot, "pos_base_y", base[1])
                    placed.set(robot, "pos_base_rot", base[2])
                    for i, q in enumerate(joints):
                        placed.set(robot, f"pos_arm_joint{i + 1}", q)
                    sim.oc.set_state(placed)
                    pick_robot = KinderPickRobot(
                        sim.env,
                        camera_noise_seed=zlib.crc32(f"{row['id']}{view}".encode()),
                    )
                    frame = pick_robot.wrist_frame()
                    live = sim.live()
                    cube_live = live.get_object_from_name(harness.CUBE)
                    truth_live = np.array(
                        [live.get(cube_live, k) for k in ("x", "y", "z")]
                    )
                    record["camera_distance"] = float(
                        np.linalg.norm(frame.world_from_camera[:3, 3] - truth_live)
                    )
                    prior = CubePrior(
                        position=prior_pos,
                        size=float(state.get(cube, "bb_x")),
                        sigma=0.05,
                        support_heights=sim.support_heights(state),
                    )
                    for name, est in estimators.items():
                        result = est.estimate(frame, prior)
                        if isinstance(result, CubeEstimate):
                            record[name] = {
                                "ok": True,
                                "position_error_mm": float(
                                    np.linalg.norm(result.position - truth_live) * 1000
                                ),
                                "yaw_error_deg": float(
                                    np.degrees(yaw_error_mod_90(result.yaw, true_yaw))
                                ),
                            }
                        else:
                            record[name] = {"ok": False, "reason": result.reason}
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(index, view, {k: record[k] for k in estimators}, flush=True)
            finally:
                scene.close()


if __name__ == "__main__":
    main()
