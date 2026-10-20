"""Film a :class:`~prpl_tidybot.sim_robot.SimWorld` and check its paths.

The robot is KINDER's MuJoCo model of the Stanford TidyBot (Jimmy Wu's base, Kinova
Gen3 and Robotiq 2F-85: the same Gen3 chain as the IK model the pick plans with),
posed kinematically from the sim's record. Two views go side by side: overhead over
the whole floor, and oblique following the base. The floor bounds, the bin
(``bin_stop_box``) and the toss pose are drawn on the floor.

``safety_summary`` poses the same model at every recorded instant and reports, step
by step, how close the arm and gripper came to the floor, any contact between the
arm and the base or itself, and how close joints 2, 4 and 6 came to their stops.
"""

# mujoco is a compiled extension; pylint cannot see its members.
# pylint: disable=no-member

from __future__ import annotations

import functools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2 as cv
import kinder
import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from prpl_tidybot.sim_robot import SimWorld

TIDYBOT_XML = (
    Path(kinder.__file__).parent
    / "envs"
    / "dynamic3d"
    / "models"
    / "stanford_tidybot"
    / "tidybot.xml"
)
_ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
# The 2F-85's linkage angles per unit of closure, from settling the model's own
# closed chain (its equality constraints) at a range of closures.
_GRIPPER_JOINTS = {
    "right_driver_joint": 0.787,
    "left_driver_joint": 0.787,
    "right_spring_link_joint": 0.785,
    "left_spring_link_joint": 0.785,
    "right_follower_joint": -0.766,
    "left_follower_joint": -0.766,
}
_GREEN = (0.1, 0.65, 0.25, 1.0)
_RED = (0.85, 0.15, 0.1, 1.0)
_BLUE = (0.1, 0.35, 0.85, 1.0)
_SAFETY_PERIOD_S = 0.02


@dataclass(frozen=True)
class SimMarkers:
    """What is drawn on the floor (map frame, m and rad)."""

    toss_pose: tuple[float, float, float]
    floor_bounds: tuple[float, float, float, float] | None = None
    bin_box: tuple[float, float, float, float] | None = None
    # A bin with walls (x, y, yaw, length, width, height, wall): drawn, and the arm's
    # clearance to its walls reported per step.
    bin: tuple[float, float, float, float, float, float, float] | None = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SimMarkers:
        """From a toss_cycles config (``sim.bin``: x, y, yaw in degrees)."""
        floor = cfg.preview.get("floor_bounds")
        box = cfg.get("bin_stop_box")
        sim_bin = cfg.get("sim", {}).get("bin")
        walls = None
        if sim_bin is not None:
            b = cfg.bin
            walls = (
                float(sim_bin[0]),
                float(sim_bin[1]),
                math.radians(float(sim_bin[2])),
                float(b.length),
                float(b.width),
                float(b.height),
                float(b.wall),
            )
        return cls(
            toss_pose=tuple(float(v) for v in cfg.toss_pose),  # type: ignore[arg-type]
            floor_bounds=None if floor is None else tuple(float(v) for v in floor),  # type: ignore[arg-type]
            bin_box=None if box is None else tuple(float(v) for v in box),  # type: ignore[arg-type]
            bin=walls,
        )

    def bin_bounds(self) -> tuple[float, float, float, float] | None:
        """The axis-aligned floor box around the walled bin."""
        if self.bin is None:
            return None
        x, y, yaw, length, width = self.bin[:5]
        hx = 0.5 * (abs(length * math.cos(yaw)) + abs(width * math.sin(yaw)))
        hy = 0.5 * (abs(length * math.sin(yaw)) + abs(width * math.cos(yaw)))
        return (x - hx, x + hx, y - hy, y + hy)


class SimScene:
    """The TidyBot model plus the floor, the markers and the cube."""

    def __init__(self, markers: SimMarkers, width: int = 640, height: int = 480):
        spec = mujoco.MjSpec.from_file(str(TIDYBOT_XML))
        spec.visual.global_.offwidth = width
        spec.visual.global_.offheight = height
        spec.add_texture(
            name="grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.86, 0.86, 0.84],
            rgb2=[0.78, 0.78, 0.76],
            width=256,
            height=256,
        )
        spec.add_material(name="grid", textures=["grid"], texrepeat=[1, 1])
        spec.materials[-1].texuniform = True
        world = spec.worldbody
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.05],
            material="grid",
        )
        world.add_light(
            pos=[0, 0, 4],
            dir=[0.3, 0.2, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=True,
            diffuse=[0.45, 0.45, 0.45],
        )
        self._add_markers(world, markers)
        cube = world.add_body(name="cube", mocap=True)
        cube.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025] * 3,
            rgba=[0.95, 0.55, 0.05, 1],
            contype=0,
            conaffinity=0,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._base_adr = [m.joint(n).qposadr[0] for n in ("joint_x", "joint_y")]
        self._th_adr = m.joint("joint_th").qposadr[0]
        self._arm_adr = [m.joint(n).qposadr[0] for n in _ARM_JOINTS]
        self._grip_adr = {m.joint(n).qposadr[0]: k for n, k in _GRIPPER_JOINTS.items()}
        self.pinch_site = m.site("pinch_site").id
        self.base_body_geom = next(
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == m.body("base_link").id and m.geom_group[g] == 2
        )
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.3
        self.floor = m.geom("floor").id
        base_link = m.body("base_link").id
        arm_root = m.body("gen3/base_link").id
        gripper_root = m.body("base").id
        self.gripper_bodies = _subtree(m, gripper_root)
        self.arm_bodies = _subtree(m, arm_root) - {arm_root}
        self.base_bodies = {base_link, arm_root}
        robot = self.arm_bodies | self.base_bodies
        self.robot_geoms = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in robot and m.geom_contype[g] and m.geom_group[g] == 3
        ]
        self.joint_ranges = {
            j: tuple(m.jnt_range[m.joint(_ARM_JOINTS[j]).id]) for j in (1, 3, 5)
        }
        self.bin_walls = [
            g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("bin_wall")
        ]

    @staticmethod
    def _add_markers(world: Any, markers: SimMarkers) -> None:
        def rect(box: Sequence[float], rgba: Sequence[float], width: float) -> None:
            x0, x1, y0, y1 = box
            for (ax, ay), (bx, by) in (
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ):
                _flat_bar(world, (ax, ay), (bx, by), width, rgba)

        if markers.floor_bounds is not None:
            rect(markers.floor_bounds, _GREEN, 0.03)
        if markers.bin_box is not None:
            rect(markers.bin_box, _RED, 0.03)
            x0, x1, y0, y1 = markers.bin_box
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, 0.0015],
                size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.001],
                rgba=[0.85, 0.15, 0.1, 0.25],
                contype=0,
                conaffinity=0,
            )
        if markers.bin is not None:
            bx, by, yaw, length, width, height, wall = markers.bin
            c, s = math.cos(yaw), math.sin(yaw)
            quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            walls = [
                ((0.0, sign * (width - wall) / 2), (length / 2, wall / 2))
                for sign in (1, -1)
            ] + [
                ((sign * (length - wall) / 2, 0.0), (wall / 2, width / 2))
                for sign in (1, -1)
            ]
            for k, ((lx, ly), (hx, hy)) in enumerate(walls):
                world.add_geom(
                    name=f"bin_wall_{k}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[bx + c * lx - s * ly, by + s * lx + c * ly, height / 2],
                    quat=quat,
                    size=[hx, hy, height / 2],
                    rgba=[0.72, 0.55, 0.35, 1.0],
                    contype=0,
                    conaffinity=0,
                )
        x, y, theta = markers.toss_pose
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x, y, 0.002],
            size=[0.07, 0.002, 0],
            rgba=list(_BLUE),
            contype=0,
            conaffinity=0,
        )
        tip = (x + 0.45 * math.cos(theta), y + 0.45 * math.sin(theta))
        _flat_bar(world, (x, y), tip, 0.03, _BLUE)
        for side in (1, -1):
            back = theta + math.pi - side * math.radians(30)
            end = (tip[0] + 0.12 * math.cos(back), tip[1] + 0.12 * math.sin(back))
            _flat_bar(world, tip, end, 0.03, _BLUE)

    def set(self, sample: tuple[Any, ...]) -> None:
        """Pose the model as in one of the world's samples."""
        _, base, q, gripper, cube, *_ = sample
        qpos = self.data.qpos
        qpos[self._base_adr[0]], qpos[self._base_adr[1]] = base[0], base[1]
        qpos[self._th_adr] = base[2]
        qpos[self._arm_adr] = q
        for adr, k in self._grip_adr.items():
            qpos[adr] = k * gripper
        mujoco.mj_kinematics(self.model, self.data)
        if cube is None:
            pos = self.data.site_xpos[self.pinch_site].copy()
            mat = self.data.site_xmat[self.pinch_site].reshape(3, 3)
            yaw = math.atan2(mat[1, 0], mat[0, 0])
        else:
            pos, yaw = np.asarray(cube[:3]), float(cube[3])
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)


def _wall_gap(model: Any, data: Any, wall: int, geom: int, fromto: Any) -> float:
    """The robot geom's least distance to a wall box (negative inside it), up to 0.5
    m. A mesh is measured from its vertices and a box from its corners:
    mj_geomDistance returned 0.0 for such pairs ~10 cm apart (MuJoCo 3.3.7)."""
    mesh = int(model.geom_dataid[geom])
    rot = data.geom_xmat[geom].reshape(3, 3)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and mesh >= 0:
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local_points = model.mesh_vert[start : start + count]
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        signs = np.array(
            [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float
        )
        local_points = signs * model.geom_size[geom]
    else:
        return float(mujoco.mj_geomDistance(model, data, wall, geom, 0.5, fromto))
    points = data.geom_xpos[geom] + local_points @ rot.T
    wall_rot = data.geom_xmat[wall].reshape(3, 3)
    local = np.abs((points - data.geom_xpos[wall]) @ wall_rot) - model.geom_size[wall]
    outside = np.linalg.norm(np.maximum(local, 0.0), axis=1)
    inside = np.minimum(local.max(axis=1), 0.0)
    return float(min(0.5, (outside + inside).min()))


def _subtree(model: Any, root: int) -> set[int]:
    bodies = {root}
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if parent == root:
                bodies.add(b)
                break
            parent = model.body_parentid[parent]
    return bodies


def _flat_bar(
    world: Any,
    a: Sequence[float],
    b: Sequence[float],
    width: float,
    rgba: Sequence[float],
) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.003],
        size=[length / 2 + width / 2, width / 2, 0.002],
        quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
    )


def _steps(world: SimWorld) -> list[tuple[str, list[int]]]:
    """The samples grouped into consecutive runs of one step (label and sub-step)."""
    steps: list[tuple[str, list[int]]] = []
    for i, sample in enumerate(world.samples):
        name = sample[5] + (f" / {sample[6]}" if sample[6] else "")
        if not steps or steps[-1][0] != name:
            steps.append((name, []))
        steps[-1][1].append(i)
    return steps


def safety_summary(world: SimWorld, markers: SimMarkers) -> dict[str, Any]:
    """Per step: the arm's and the gripper's least clearance above the floor, any
    contact of the arm with the base or itself, the least margin of joints 2, 4 and 6
    to their stops, peak joint and base speeds, and whether the base centre left the
    floor bounds."""
    scene = SimScene(markers)
    m, d = scene.model, scene.data
    fromto = np.zeros(6)
    out: list[dict[str, Any]] = []
    times = np.array([s[0] for s in world.samples])
    for name, idx in _steps(world):
        chosen, last_t = [], -math.inf
        for i in idx:
            if times[i] - last_t >= _SAFETY_PERIOD_S or i == idx[-1]:
                chosen.append(i)
                last_t = times[i]
        arm_clear, grip_clear = math.inf, math.inf
        arm_bin, grip_bin = math.inf, math.inf
        margin, margin_joint = math.inf, 0
        contacts: set[tuple[str, str]] = set()
        outside = 0.0
        for i in chosen:
            sample = world.samples[i]
            scene.set(sample)
            for g in scene.robot_geoms:
                body = m.geom_bodyid[g]
                if body in scene.base_bodies:
                    continue
                dist = mujoco.mj_geomDistance(m, d, scene.floor, g, 0.5, fromto)
                if body in scene.gripper_bodies:
                    grip_clear = min(grip_clear, dist)
                else:
                    arm_clear = min(arm_clear, dist)
                for wall in scene.bin_walls:
                    gap = _wall_gap(m, d, wall, g, fromto)
                    if body in scene.gripper_bodies:
                        grip_bin = min(grip_bin, gap)
                    else:
                        arm_bin = min(arm_bin, gap)
            for c in d.contact[: d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if {b1, b2} <= scene.gripper_bodies or c.dist > 0:
                    continue
                if (b1 in scene.arm_bodies) or (b2 in scene.arm_bodies):
                    contacts.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
            q = sample[2]
            for j, (lo, hi) in scene.joint_ranges.items():
                signed = (q[j] + math.pi) % (2 * math.pi) - math.pi
                room = math.degrees(min(signed - lo, hi - signed))
                if room < margin:
                    margin, margin_joint = room, j + 1
            if markers.floor_bounds is not None:
                x0, x1, y0, y1 = markers.floor_bounds
                bx, by = sample[1][0], sample[1][1]
                outside = max(outside, x0 - bx, bx - x1, y0 - by, by - y1)
        qs = np.array([world.samples[i][2] for i in idx])
        bases = np.array([world.samples[i][1][:2] for i in idx])
        ts = times[idx]
        dt = np.diff(ts)
        ok = dt > 0
        joint_speed = (
            float(np.degrees(np.abs(np.diff(qs, axis=0))[ok] / dt[ok, None]).max())
            if ok.any()
            else 0.0
        )
        base_speed = (
            float((np.linalg.norm(np.diff(bases, axis=0), axis=1)[ok] / dt[ok]).max())
            if ok.any()
            else 0.0
        )
        out.append(
            {
                "step": name,
                "t_start_s": round(float(ts[0]), 3),
                "duration_s": round(float(ts[-1] - ts[0]), 3),
                "min_floor_clearance_m": {
                    "arm": _round(arm_clear),
                    "gripper": _round(grip_clear),
                },
                "contacts": [list(c) for c in sorted(contacts)],
                # Negative: the arm or gripper inside a wall (within 0.5 m).
                "min_bin_wall_clearance_m": (
                    {"arm": _round(arm_bin), "gripper": _round(grip_bin)}
                    if scene.bin_walls
                    else None
                ),
                "min_joint_limit_margin_deg": {
                    "joint": margin_joint,
                    "margin": _round(margin, 1),
                },
                "max_joint_speed_deg_s": round(joint_speed, 1),
                "max_base_speed_m_s": round(base_speed, 3),
                "base_outside_floor_bounds_m": round(max(0.0, outside), 3),
            }
        )
    return {
        "steps": out,
        "warnings": list(world.warnings),
        "swings": [
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in s.items()
                if k not in ("executed_deg", "windup_end_deg")
            }
            for s in world.swings
        ],
    }


def _round(value: float, digits: int = 4) -> float | None:
    return None if not math.isfinite(value) else round(float(value), digits)


def _make_context(width: int, height: int) -> Any:
    """An offscreen OpenGL context: EGL (GPU), else OSMesa (CPU)."""
    errors = []
    # kinder.register_all_environments() sets PYOPENGL_PLATFORM=osmesa without a
    # display; PyOpenGL fixes its platform at first import, so until then EGL can
    # still be chosen (OSMesa renders on the CPU, ~100x slower).
    if "OpenGL" not in sys.modules:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    for backend in ("egl", "osmesa"):
        if backend == "osmesa" and "OpenGL" not in sys.modules:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        try:
            module = __import__(f"mujoco.{backend}", fromlist=["GLContext"])
            ctx = module.GLContext(width, height)
            ctx.make_current()
            return ctx
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"{backend}: {e!r}")
    raise RuntimeError(f"No offscreen OpenGL for MuJoCo ({'; '.join(errors)}).")


@functools.cache
def gl_available() -> bool:
    """Whether an offscreen OpenGL context can be made (for the video)."""
    try:
        ctx = _make_context(16, 16)
    except RuntimeError:
        return False
    ctx.free()
    return True


def _frame_times(world: SimWorld, fps: int, slowdown: float) -> list[float]:
    end = world.samples[-1][0]
    spans = world.slow_spans
    times, t = [], 0.0
    while t <= end:
        times.append(t)
        slow = any(a <= t <= b for a, b in spans)
        step = t + 1.0 / (fps * (slowdown if slow else 1.0))
        # Land a frame on the start of each swing, so all of it plays slowly.
        t = min([step] + [a for a, _ in spans if t < a < step])
    return times


class _Views:
    """Renders the overhead and oblique views of a posed :class:`SimScene`."""

    def __init__(self, scene: SimScene, markers: SimMarkers, world: SimWorld):
        self.scene = scene
        m = scene.model
        self.width, self.height = m.vis.global_.offwidth, m.vis.global_.offheight
        self.ctx = _make_context(self.width, self.height)
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(m, maxgeom=8000)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.overhead = self._overhead_camera(markers, world)
        self.oblique = mujoco.MjvCamera()
        self.oblique.distance = 3.2
        self.oblique.elevation = -22
        # Side-on to the throw, slightly from behind, so the swing is in profile.
        self.oblique.azimuth = math.degrees(markers.toss_pose[2]) - 115

    def _overhead_camera(self, markers: SimMarkers, world: SimWorld) -> Any:
        xs = [markers.toss_pose[0]] + [s[1][0] for s in world.samples[::50]]
        ys = [markers.toss_pose[1]] + [s[1][1] for s in world.samples[::50]]
        for box in (markers.floor_bounds, markers.bin_box, markers.bin_bounds()):
            if box is not None:
                xs += [box[0], box[1]]
                ys += [box[2], box[3]]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_h = max((max(ys) - min(ys)) / 2, (max(xs) - min(xs)) / 2 * 0.75) + 0.35
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [cx, cy, 0.0]
        cam.azimuth = 90
        cam.elevation = -89.9
        cam.distance = half_h / math.tan(math.radians(22.5))
        return cam

    def _add_footprint(self) -> None:
        """The base's body as a flat box: from straight above its mesh does not
        render (one-sided faces)."""
        m, d = self.scene.model, self.scene.data
        g = self.scene.base_body_geom
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mat = d.geom_xmat[g].reshape(3, 3)
        center = d.geom_xpos[g] + mat @ m.geom_aabb[g][:3]
        size = m.geom_aabb[g][3:].copy()
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            center,
            mat.flatten(),
            np.array([0.35, 0.35, 0.38, 0.85], dtype=np.float32),
        )
        self.scn.ngeom += 1

    def render(
        self,
        cam: Any,
        trails: list[tuple[list[NDArray[np.float64]], float, Any]],
        footprint: bool = False,
    ) -> NDArray[np.uint8]:
        """One view, with `trails` (points, radius, rgba) drawn as polylines."""
        m, d = self.scene.model, self.scene.data
        mujoco.mjv_updateScene(
            m, d, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        for points, radius, rgba in trails:
            for a, b in zip(points[:-1], points[1:]):
                if self.scn.ngeom >= self.scn.maxgeom:
                    break
                geom = self.scn.geoms[self.scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.asarray(rgba, dtype=np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
                self.scn.ngeom += 1
        if footprint:
            self._add_footprint()
        mujoco.mjr_render(self.viewport, self.scn, self.con)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.con)
        return np.flipud(self.rgb).copy()

    def close(self) -> None:
        """Release the GL resources."""
        self.con.free()
        self.ctx.free()


def render_video(  # pylint: disable=too-many-arguments,too-many-locals
    world: SimWorld,
    markers: SimMarkers,
    path: Path,
    fps: int = 30,
    slowdown: float = 4.0,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Write the overhead | oblique video of the whole run to `path` (H.264)."""
    import imageio_ffmpeg  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    scene = SimScene(markers, width, height)
    views = _Views(scene, markers, world)
    times = np.array([s[0] for s in world.samples])
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (2 * width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-crf", "23"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    base_trail: list[NDArray[np.float64]] = []
    pinch_trail: list[NDArray[np.float64]] = []
    step = None
    try:
        for t in _frame_times(world, fps, slowdown):
            sample = world.samples[max(0, int(np.searchsorted(times, t, "right")) - 1)]
            scene.set(sample)
            base_xy = np.array([sample[1][0], sample[1][1], 0.006])
            if not base_trail or np.linalg.norm(base_trail[-1] - base_xy) > 0.05:
                base_trail.append(base_xy)
            if (sample[5], sample[6]) != step:
                step, pinch_trail = (sample[5], sample[6]), []
            pinch = scene.data.site_xpos[scene.pinch_site].copy()
            if not pinch_trail or np.linalg.norm(pinch_trail[-1] - pinch) > 0.005:
                pinch_trail.append(pinch)
            pinch_trail = pinch_trail[-400:]
            views.oblique.lookat[:] = [sample[1][0], sample[1][1], 0.6]
            trail_pinch = (pinch_trail, 0.006, (1.0, 0.45, 0.0, 0.9))
            top = views.render(
                views.overhead,
                [(base_trail[-1500:], 0.012, (0.1, 0.35, 0.85, 0.6)), trail_pinch],
                footprint=True,
            )
            side = views.render(views.oblique, [trail_pinch])
            frame = np.concatenate([top, side], axis=1)
            slow = any(a <= t <= b for a, b in world.slow_spans)
            _overlay(frame, sample, t, slow, slowdown, width, markers)
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
        views.close()
    return path


def _overlay(  # pylint: disable=too-many-arguments
    frame: NDArray[np.uint8],
    sample: tuple[Any, ...],
    t: float,
    slow: bool,
    slowdown: float,
    width: int,
    markers: SimMarkers,
) -> None:
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
    cv.putText(
        frame, f"{t:7.2f} s  {sample[5][:120]}", (8, 18), font, 0.5, (255, 255, 255), 1
    )
    sub = sample[6] or ""
    if slow:
        sub += f"   [{1 / slowdown:.2f}x slow motion]"
    held = " (holding the cube)" if sample[4] is None else ""
    sub = f"gripper {sample[3]:.2f}{held}   {sub}"
    cv.putText(frame, sub[:140], (8, 38), font, 0.5, (120, 220, 255), 1)
    h = frame.shape[0]
    legend = [
        ("floor bounds (base centre)", _GREEN),
        ("bin_stop_box" if markers.bin_box is not None else "no bin_stop_box", _RED),
        ("toss pose", _BLUE),
        ("orange: gripper path this step", (1.0, 0.45, 0.0, 1)),
    ]
    for k, (text, rgba) in enumerate(legend):
        color = tuple(int(255 * c) for c in rgba[:3])
        y = h - 12 - 16 * (len(legend) - 1 - k)
        cv.rectangle(frame, (8, y - 9), (20, y + 1), color, -1)
        cv.putText(frame, text, (26, y), font, 0.42, (30, 30, 30), 1)
    cv.putText(frame, "overhead", (8, 62), font, 0.5, (30, 30, 30), 1)
    cv.putText(
        frame,
        "oblique (follows the base)",
        (width + 8, 62),
        font,
        0.5,
        (30, 30, 30),
        1,
    )
