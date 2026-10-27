"""Bounded base repositioning with inflated boxes and full-robot transit radius.

A* searches a local XY grid. A circumscribed radius bounds base, arm and fingers
at the measured transit configuration for all headings. This is geometric
planning; a verified free operating area/obstacle survey is still required.
"""

from __future__ import annotations

import heapq
import itertools
import math
from pathlib import Path

import numpy as np

from prpl_tidybot.sprout.live_config import model_digest
from prpl_tidybot.sprout.live_kinematics import LiveKinematics, top_down


def transit_radius(model: Path, joints: list[float]) -> float:
    kin = LiveKinematics(model, [0, 0, 0], {"model_sha256": model_digest(model)})
    kin.set(joints)
    return max(
        float(np.linalg.norm(np.abs(c[:2]) + h[:2]))
        for c, h in (kin.bounds(g) for g in kin.collision_geoms)
    )


def free(point, obstacles, radius, bounds) -> bool:
    p = np.asarray(point)[:2]
    if (
        not np.all(np.isfinite(p))
        or np.any(p < np.asarray(bounds[0]) + radius)
        or np.any(p > np.asarray(bounds[1]) - radius)
    ):
        return False
    for box in obstacles:
        distance = np.maximum(
            np.abs(p - np.asarray(box["center"])[:2])
            - np.asarray(box["half_size"])[:2],
            0,
        )
        if np.linalg.norm(distance) <= radius:
            return False
    return True


def segment_free(a, b, obstacles, radius, bounds, resolution=0.02):
    # Add half a sampling interval to cover every point between samples.
    n = max(
        2,
        int(np.ceil(np.linalg.norm(np.asarray(b)[:2] - np.asarray(a)[:2]) / resolution))
        + 1,
    )
    return all(
        free(p, obstacles, radius + resolution / 2, bounds)
        for p in np.linspace(np.asarray(a)[:2], np.asarray(b)[:2], n)
    )


def route(start, goal, obstacles, radius, bounds, step=0.05):
    start, goal = np.asarray(start)[:2], np.asarray(goal)[:2]
    if not free(start, obstacles, radius, bounds) or not free(
        goal, obstacles, radius, bounds
    ):
        raise ValueError(
            "Base start/goal overlaps inflated keep-out or operating boundary"
        )
    if segment_free(start, goal, obstacles, radius, bounds):
        return [start.tolist(), goal.tolist()]
    origin = start
    queue = [(float(np.linalg.norm(goal - start)), 0.0, (0, 0))]
    costs, parent = {(0, 0): 0.0}, {}
    found = None
    while queue and len(costs) < 20000:
        _, cost, key = heapq.heappop(queue)
        if cost > costs[key]:
            continue
        here = origin + np.asarray(key) * step
        if np.linalg.norm(here - goal) < 2 * step and segment_free(
            here, goal, obstacles, radius, bounds
        ):
            found = key
            break
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            other = (key[0] + dx, key[1] + dy)
            there = origin + np.asarray(other) * step
            new = cost + math.hypot(dx, dy) * step
            if new >= costs.get(other, math.inf) or not segment_free(
                here, there, obstacles, radius, bounds
            ):
                continue
            costs[other], parent[other] = new, key
            heapq.heappush(
                queue, (new + float(np.linalg.norm(goal - there)), new, other)
            )
    if found is None:
        raise ValueError("No collision-free base route within local search bounds")
    points = [goal.tolist()]
    while found != (0, 0):
        points.append((origin + np.asarray(found) * step).tolist())
        found = parent[found]
    points.append(start.tolist())
    points.reverse()
    simplified = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not segment_free(
            points[i], points[j], obstacles, radius, bounds
        ):
            j -= 1
        simplified.append(points[j])
        i = j
    return simplified


def probe(sim, grasp, recipe):
    """Test the same Cartesian approach from the measured arm configuration."""
    import mujoco

    from prpl_tidybot.sprout.sequence import make_plan

    plan = make_plan(grasp, recipe)
    saved = sim.data.qpos.copy()
    try:
        sim.data.qpos[sim.arm_qpos] = sim.pregrasp_joints(plan.pregrasp)
        mujoco.mj_forward(sim.model, sim.data)
        for pose in (
            plan.grasp,
            plan.lifted,
            plan.put_down,
            plan.retreat,
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
    except ValueError:
        return False
    finally:
        sim.data.qpos[:] = saved
        mujoco.mj_forward(sim.model, sim.data)


def plan_base(sim, model, grasp, recipe, obstacles, settings):
    import mujoco

    sim.validate_initial_joints()
    start = np.asarray(sim.scene["tidybot_base_xyyaw"], float)
    addresses = [
        sim.model.joint("tidy/" + n).qposadr[0]
        for n in ("joint_x", "joint_y", "joint_th")
    ]
    radius = (
        transit_radius(model, sim.scene["tidybot_arm_joints_rad"])
        + settings["clearance_m"]
    )
    bounds = settings["bounds_xy"]
    if not free(start, obstacles, radius, bounds):
        raise ValueError("Current robot transit envelope overlaps keep-out/boundary")
    if probe(sim, grasp, recipe):
        return {
            "required": False,
            "path_xyyaw": [start.tolist()],
            "radius_m": radius,
            "obstacles": obstacles,
            "bounds_xy": bounds,
        }
    candidates = []
    goal = np.array([grasp.x, grasp.y])
    for distance in (0.55, 0.65, 0.75, 0.85):
        for angle in np.linspace(-math.pi, math.pi, 20, endpoint=False):
            xy = goal + distance * np.array([math.cos(angle), math.sin(angle)])
            if not free(xy, obstacles, radius, bounds):
                continue
            for yaw in (
                start[2],
                angle + math.pi,
                angle + math.pi / 2,
                angle - math.pi / 2,
            ):
                candidates.append(
                    (
                        float(np.linalg.norm(xy - start[:2]))
                        + 0.08 * abs(math.remainder(yaw - start[2], 2 * math.pi)),
                        [*xy, float(yaw)],
                    )
                )
    for _, candidate in sorted(candidates, key=lambda c: c[0]):
        try:
            path = route(start, candidate, obstacles, radius, bounds)
        except ValueError:
            continue
        sim.data.qpos[addresses] = candidate
        mujoco.mj_forward(sim.model, sim.data)
        if probe(sim, grasp, recipe):
            sim.data.qpos[addresses] = start
            mujoco.mj_forward(sim.model, sim.data)
            poses = [[*p, float(start[2])] for p in path]
            poses.append(candidate)
            return {
                "required": True,
                "path_xyyaw": poses,
                "radius_m": radius,
                "obstacles": obstacles,
                "bounds_xy": bounds,
                "goal_xyyaw": candidate,
                "method": "inflated-box A* + joint-space approach and Cartesian lift IK",
            }
    sim.data.qpos[addresses] = start
    mujoco.mj_forward(sim.model, sim.data)
    raise ValueError("No reachable base candidate with a collision-free transit route")


def validate_base_plan(plan):
    bounds = np.asarray(plan["bounds_xy"], float)
    if (
        bounds.shape != (2, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[0] >= bounds[1])
    ):
        raise ValueError("Invalid operating bounds")
    if type(plan.get("required")) is not bool or not plan.get("obstacles"):
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
    if not math.isfinite(plan["radius_m"]) or plan["radius_m"] <= 0:
        raise ValueError("Invalid base collision radius")
    if len(path) < (2 if plan["required"] else 1):
        raise ValueError("Missing base waypoints")
    if not all(
        free(p, plan["obstacles"], plan["radius_m"], plan["bounds_xy"]) for p in path
    ):
        raise ValueError("Base waypoint overlaps keep-out")
    for a, b in itertools.pairwise(path):
        if not segment_free(
            a, b, plan["obstacles"], plan["radius_m"], plan["bounds_xy"]
        ):
            raise ValueError("Reviewed base route intersects keep-out/boundary")
