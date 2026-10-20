"""Top-down cube pick, ported from Jimmy Wu's TidyBot ``pick_object``.

The original (``tidybot_real/robot/controller.py``, as adapted by Yixuan Huang for the
Princeton robots) parks the base so the object is a fixed reach away, then moves the
arm to a point 10 cm above the object with the tool pointing straight down and the
wrist turned to the object's orientation, descends vertically, closes the gripper and
checks the gripper position to tell a grasp from a miss. A vertical approach turns a
centimetre of localization or parking error into an off-centre grasp rather than a
push, which is why it works open-loop.

Here the object is the floor cube, localized by the ceiling cameras
(:mod:`prpl_tidybot.cube_localizer`), and the arm is driven through prpl-tidybot's
normal arm path (streamed joint targets) instead of Kortex's high-level moves. The base
faces the cube from the standoff, so the cube lies straight ahead of the arm mount.

Frames: the IK solver works in the arm's base frame (+x along the robot's heading when
joint 1 is at 0, +z up). The arm mount sits ``mounting_offset`` ahead of the base
centre (``MOUNTING_OFFSET`` in ``third_party/constants.py``) and the floor is at
``floor_z`` in the arm frame.
"""

# mujoco is a compiled extension; pylint cannot see its members.
# pylint: disable=no-member

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.third_party.constants import MOUNTING_OFFSET, RETRACT_ARM_CONF
from prpl_tidybot.third_party.ik_solver import IKSolver

# The original's carry pose after a grasp, [h, 340, 180, 214, 0, 320, 90] in Kortex
# degrees with h = 0 (the arm faces the base's heading here).
CARRY_ARM_CONF = np.deg2rad([0.0, -20.0, 180.0, -146.0, 0.0, -40.0, 90.0])


@dataclass(frozen=True)
class TopDownPickGeometry:
    """Distances (m) and angles (rad) for the top-down pick; the defaults follow the
    original primitive and the lab's 5 cm cube."""

    # Gripper reach from the arm mount at the grasp (the original's open-gripper 0.55).
    reach: float = 0.55
    mounting_offset: float = MOUNTING_OFFSET
    # KINDER's TidyBot mounts the arm 0.3948 m above the floor.
    floor_z: float = -0.395
    # The open 2F-85's finger pads reach this far below the pinch point (KINDER's
    # gen3_2f85 model), so the pinch point must stay this high plus a clearance.
    pad_drop: float = 0.0295
    pad_clearance: float = 0.012
    # The original hovers 10 cm above the grasp before descending.
    hover_height: float = 0.10
    # Added to the cube's yaw to align the fingers' closing axis with a pair of faces.
    wrist_yaw_offset: float = 0.0
    # Added after the 90-degree fold: pi/2 closes on the cube's other pair of faces
    # (inside the bin, the fingers then spread along the other wall), the wrist still
    # turning at most 90 degrees.
    closing_turn: float = 0.0
    # Pinch point beyond the arm's flange (the arm server's IK uses 0.12).
    ee_offset: float = 0.12
    # Gripper readings: closing on nothing reaches ~1.0; on the 5 cm cube it read 0.49.
    grasp_min: float = 0.2
    grasp_max: float = 0.9

    @property
    def grasp_height(self) -> float:
        """Pinch point above the floor at the grasp: the pads just clear the floor."""
        return self.pad_drop + self.pad_clearance

    @property
    def standoff(self) -> float:
        """Base centre to cube when parked to pick."""
        return self.reach + self.mounting_offset


def standoff_pose(
    robot: Sequence[float],
    cube_xy: Sequence[float],
    geom: TopDownPickGeometry,
    floor_bounds: Sequence[float],
    is_valid: Callable[[tuple[float, float, float]], bool] | None = None,
) -> tuple[float, float, float]:
    """Map-frame base pose `standoff` from the cube and facing it.

    Prefers approaching along the line from the robot's current position; if that
    standoff is off the floor (or fails ``is_valid``, e.g. the barrier's clearance),
    swings the approach round the cube in 15-degree steps (nearest first) until it is on
    the floor.
    """
    cx, cy = cube_xy
    direct = math.atan2(cy - robot[1], cx - robot[0])  # direction of travel to the cube
    for k in range(25):
        for sign in ((1,) if k == 0 else (1, -1)):
            heading = direct + sign * math.radians(15 * k)
            x = cx - geom.standoff * math.cos(heading)
            y = cy - geom.standoff * math.sin(heading)
            pose = (x, y, math.atan2(math.sin(heading), math.cos(heading)))
            if (
                floor_bounds[0] <= x <= floor_bounds[1]
                and floor_bounds[2] <= y <= floor_bounds[3]
                and (is_valid is None or is_valid(pose))
            ):
                return pose
    raise ValueError(
        f"No standoff for a cube at {tuple(cube_xy)} inside {tuple(floor_bounds)}."
    )


def grasp_search_bounds(
    floor_bounds: Sequence[float],
    geom: TopDownPickGeometry,
    base_front: float = 0.274,
    clearance: float = 0.05,
) -> tuple[float, float, float, float]:
    """Where a cube can be and still be picked with the base inside `floor_bounds`:
    the drivable box grown on every side by standoff - (base front + clearance), so a
    cube just past the box edge (e.g. in a bin there) is still searched for."""
    margin = geom.standoff - base_front - clearance
    return (
        float(floor_bounds[0]) - margin,
        float(floor_bounds[1]) + margin,
        float(floor_bounds[2]) - margin,
        float(floor_bounds[3]) + margin,
    )


def arm_frame_target(
    base_pose: Sequence[float], cube_xy: Sequence[float], geom: TopDownPickGeometry
) -> tuple[float, float]:
    """The cube's (x, y) in the arm's base frame, given the base's map pose."""
    x, y, theta = base_pose
    mount = (
        x + geom.mounting_offset * math.cos(theta),
        y + geom.mounting_offset * math.sin(theta),
    )
    dx, dy = cube_xy[0] - mount[0], cube_xy[1] - mount[1]
    c, s = math.cos(theta), math.sin(theta)
    return (c * dx + s * dy, -s * dx + c * dy)


def _down_quat(closing_yaw: float) -> NDArray[np.float64]:
    """(x, y, z, w) of a tool frame whose approach axis (z) points down and whose
    closing axis (y) lies along `closing_yaw` in the horizontal plane."""
    c, s = math.cos(closing_yaw), math.sin(closing_yaw)
    rot = np.column_stack([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])
    quat_wxyz = np.empty(4)
    mujoco.mju_mat2Quat(quat_wxyz, rot.flatten())
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def cube_closing_yaw(cube_yaw_rel: float, geom: TopDownPickGeometry) -> float:
    """The cube is 90-degree symmetric: pick the equivalent yaw nearest straight
    ahead, so the wrist turns as little as possible."""
    yaw = cube_yaw_rel + geom.wrist_yaw_offset
    folded = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    if not geom.closing_turn:
        return folded
    # The two fingers are symmetric too: keep the turned grasp within 90 degrees.
    return (folded + geom.closing_turn + math.pi / 2) % math.pi - math.pi / 2


_SOLVERS: dict[float, IKSolver] = {}


def _solver(geom: TopDownPickGeometry) -> IKSolver:
    if geom.ee_offset not in _SOLVERS:
        solver = IKSolver(ee_offset=geom.ee_offset)  # type: ignore[no-untyped-call]
        _SOLVERS[geom.ee_offset] = solver
    return _SOLVERS[geom.ee_offset]


def forward_kinematics(
    joints: Sequence[float], geom: TopDownPickGeometry
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pinch-point position, approach axis and closing axis in the arm frame."""
    solver = _solver(geom)
    solver.data.qpos[:] = joints
    mujoco.mj_kinematics(solver.model, solver.data)
    mat = solver.data.site(solver.site_id).xmat.reshape(3, 3)
    return (
        solver.data.site(solver.site_id).xpos.copy(),
        mat[:, 2].copy(),
        mat[:, 1].copy(),
    )


def arm_link_points(
    joints: Sequence[float], geom: TopDownPickGeometry
) -> dict[str, NDArray[np.float64]]:
    """Every collision mesh vertex of each arm link at ``joints``, in the arm frame
    (the IK model has no gripper mesh: the 2F-85 is only its pinch-point offset)."""
    solver = _solver(geom)
    model, data = solver.model, solver.data
    data.qpos[:] = joints
    mujoco.mj_kinematics(model, data)
    links: dict[str, NDArray[np.float64]] = {}
    for g in range(model.ngeom):
        mesh = int(model.geom_dataid[g])
        if mesh < 0 or not model.geom_contype[g]:
            continue
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start : start + count], np.float64)
        rot = data.geom_xmat[g].reshape(3, 3)
        name = str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        )
        links[name] = data.geom_xpos[g] + local @ rot.T
    return links


def top_down_joint_targets(
    ax: float,
    ay: float,
    cube_yaw_rel: float,
    geom: TopDownPickGeometry,
    seed: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Joint targets above the cube and at the grasp (straight below), pointing down.

    Raises if the IK does not converge to within 5 mm and 2 degrees of either pose.
    """
    closing = cube_closing_yaw(cube_yaw_rel, geom)
    hover = pointing_down_joints(
        (ax, ay, geom.floor_z + geom.grasp_height + geom.hover_height),
        closing,
        geom,
        seed,
    )
    grasp = pointing_down_joints(
        (ax, ay, geom.floor_z + geom.grasp_height), closing, geom, hover
    )
    return hover, grasp


def pointing_down_joints(
    position: Sequence[float],
    closing_yaw: float,
    geom: TopDownPickGeometry,
    seed: Sequence[float] | None = None,
) -> list[float]:
    """Joints putting the pinch point at `position` (arm frame), tool pointing straight
    down, fingers closing along `closing_yaw`. Raises if the IK does not get within 5
    mm and 2 degrees."""
    quat = _down_quat(closing_yaw)
    q = np.asarray(seed if seed is not None else RETRACT_ARM_CONF, float)
    goal = np.asarray(position, float)
    for _ in range(10):  # the solver takes 20 steps per call; restart from its result
        q = _solver(geom).solve(goal, quat, q)  # type: ignore[no-untyped-call]
        pos, approach, _ = forward_kinematics(q, geom)
        if np.linalg.norm(pos - goal) < 0.005 and approach[2] < -0.9994:
            return [float(v) for v in q]
    raise ValueError(f"IK did not reach {goal.round(3).tolist()} pointing down.")


def grasp_succeeded(gripper_position: float, geom: TopDownPickGeometry) -> bool:
    """Closed on something: neither fully closed (nothing there) nor still open."""
    return geom.grasp_min < gripper_position < geom.grasp_max


def geometry_from_config(cfg: Any) -> TopDownPickGeometry:
    """Build the geometry from a ``pick.top_down`` config block (defaults fill gaps)."""
    names = [f.name for f in fields(TopDownPickGeometry)]
    kwargs = {k: float(cfg[k]) for k in names if cfg is not None and k in cfg}
    if "wrist_yaw_offset_deg" in (cfg or {}):
        kwargs["wrist_yaw_offset"] = math.radians(float(cfg["wrist_yaw_offset_deg"]))
    return TopDownPickGeometry(**kwargs)
