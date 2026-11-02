"""Sprout targets planned with existing PRPL real_sim URDF/RRT machinery."""

from __future__ import annotations

import itertools
import math

import mujoco
import numpy as np

from prpl_tidybot.sprout.live_kinematics import top_down
from prpl_tidybot.sprout.sequence import make_plan
from prpl_tidybot.sprout.transit_geometry import (
    DEFAULT_NAVIGATION,
    TransitChecker,
    build_geometry,
)


def probe(sim, grasp, recipe, failures=None):
    """Test the same Cartesian approach from the measured arm configuration."""

    plan = make_plan(grasp, recipe)
    saved = sim.data.qpos.copy()
    phase = "pregrasp"
    try:
        sim.data.qpos[sim.arm_qpos] = sim.pregrasp_joints(plan.pregrasp)
        mujoco.mj_forward(sim.model, sim.data)
        for phase, pose in zip(
            ("grasp", "lift", "put_down", "retreat"),
            (plan.grasp, plan.lifted, plan.put_down, plan.retreat),
        ):
            start = sim.data.site_xpos[sim.pinch].copy()
            qa, qb = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qa, sim.data.site_xmat[sim.pinch])
            mujoco.mju_mat2Quat(qb, top_down(pose.yaw).ravel())
            if qa @ qb < 0:
                qb *= -1
            for f in np.linspace(0, 1, 12):
                q = qa * (1 - f) + qb * f
                q /= np.linalg.norm(q)
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, q)
                sim.ik(
                    sim.pinch,
                    sim.arm_joints,
                    start + f * (np.array([pose.x, pose.y, pose.z]) - start),
                    rot.reshape(3, 3),
                )
        return True
    except ValueError as error:
        if failures is not None:
            failures.append(f"{phase}: {error}")
        return False
    finally:
        sim.data.qpos[:] = saved
        mujoco.mj_forward(sim.model, sim.data)


def plan_base(sim, model, grasp, recipe, obstacles, settings, navigation=None):
    """Find a reachable base pose with model geometry and stopping clearance."""
    sim.validate_initial_joints()
    start = np.asarray(sim.scene["tidybot_base_xyyaw"], float)
    addresses = [
        sim.model.joint("tidy/" + n).qposadr[0]
        for n in ("joint_x", "joint_y", "joint_th")
    ]
    geometry = build_geometry(
        model,
        sim.scene["tidybot_arm_joints_rad"],
        settings["clearance_m"],
        DEFAULT_NAVIGATION if navigation is None else navigation,
    )
    bounds = settings["bounds_xy"]
    checker = TransitChecker(geometry, obstacles, bounds)
    common = {
        "transit_geometry": geometry,
        "obstacles": obstacles,
        "bounds_xy": bounds,
    }
    try:
        if not checker.free(start):
            raise ValueError(
                "Current robot model/stopping envelope overlaps keep-out/boundary"
            )
        if probe(sim, grasp, recipe):
            return dict(common, required=False, path_xyyaw=[start.tolist()])
        candidates = []
        goal = np.array([grasp.x, grasp.y])
        for distance in (0.55, 0.65, 0.75, 0.85):
            for angle in np.linspace(-math.pi, math.pi, 20, endpoint=False):
                xy = goal + distance * np.array([math.cos(angle), math.sin(angle)])
                for yaw in (
                    start[2],
                    angle + math.pi,
                    angle + math.pi / 2,
                    angle - math.pi / 2,
                ):
                    candidate = [*xy, float(yaw)]
                    if checker.free(candidate):
                        candidates.append(
                            (
                                float(np.linalg.norm(xy - start[:2]))
                                + 0.08
                                * abs(math.remainder(yaw - start[2], 2 * math.pi)),
                                candidate,
                            )
                        )
        reachable = 0
        failures = []
        try:
            for _, candidate in sorted(candidates, key=lambda c: c[0]):
                sim.data.qpos[addresses] = candidate
                mujoco.mj_forward(sim.model, sim.data)
                if not probe(sim, grasp, recipe, failures):
                    continue
                reachable += 1
                try:
                    poses = checker.route(start, candidate)
                except ValueError:
                    continue
                result = dict(
                    common,
                    required=True,
                    path_xyyaw=poses,
                    goal_xyyaw=candidate,
                    method="PRPL URDF collision + mobile-base BiRRT + Kinova IKFast",
                )
                validate_base_plan(result)
                return result
        finally:
            sim.data.qpos[addresses] = start
            mujoco.mj_forward(sim.model, sim.data)
        raise ValueError(
            "No reachable base candidate with a collision-free transit route "
            f"({len(candidates)} collision-free poses, {reachable} pass arm sequence). "
            + (failures[0] if failures else "No admissible base route")
        )

    finally:
        checker.close()


def validate_base_plan(plan: dict) -> None:
    """Reject malformed or obstructed routes before base execution."""
    bounds = np.asarray(plan["bounds_xy"], float)
    if (
        bounds.shape != (2, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[0] >= bounds[1])
    ):
        raise ValueError("Invalid operating bounds")
    if not isinstance(plan.get("required"), bool) or not plan.get("obstacles"):
        raise ValueError("Base plan must declare motion and body obstacles")
    for obstacle in plan["obstacles"]:
        c, h = (
            np.asarray(obstacle["center"], float),
            np.asarray(obstacle["half_size"], float),
        )
        if (
            c.shape != (3,)
            or h.shape != (3,)
            or not np.all(np.isfinite([c, h]))
            or np.any(h <= 0)
        ):
            raise ValueError("Invalid base obstacle bounds")
    path = np.asarray(plan["path_xyyaw"], float)
    if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
        raise ValueError("Invalid base path")
    if len(path) < (2 if plan["required"] else 1):
        raise ValueError("Missing base waypoints")
    geometry = plan.get("transit_geometry")
    if not geometry:
        raise ValueError("Regenerate preview: PRPL URDF collision review required")
    with TransitChecker(geometry, plan["obstacles"], plan["bounds_xy"]) as checker:
        if not all(checker.free(p) for p in path):
            raise ValueError("Base URDF waypoint intersects keep-out/boundary")
        for a, b in itertools.pairwise(path):
            if not checker.segment(a, b):
                raise ValueError("Reviewed URDF sweep intersects keep-out/boundary")
