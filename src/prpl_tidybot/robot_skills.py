"""The robot's motions behind the skill server (:mod:`prpl_tidybot.skill_server`), its
only caller: driving (straight, checked against the drivable box, the barrier and the
bin), retracting, the top-down pick (off the floor, or out of the bin through a waypoint
above its rim, every arm pose checked against its walls), the wrist servo, the place,
and the ceiling and wrist recorders.

``scripts/toss_cycles.py`` and hitl-pmp both reach these through the skills
(:class:`prpl_tidybot.skill_server.TossingSkills`), never directly: one robot code path.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from prpl_tidybot.aimed_toss import Workspace
from prpl_tidybot.bin_localizer import BinFit, cube_in_bin
from prpl_tidybot.gripper_shape import (
    FINGER_LINKS,
    GRIPPER_HOUSING,
    gripper_link_points,
    gripper_link_spheres,
    opening_command,
)
from prpl_tidybot.interfaces.arm_interface import ArmInterface, ArmMotionAborted
from prpl_tidybot.real_sim.perceivers.kinematic3d import PrplLab3DPerceiver
from prpl_tidybot.real_sim.plan_executors.kinematic3d import Kinematic3DPlanExecutor
from prpl_tidybot.third_party.constants import POLICY_CONTROL_PERIOD, RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    CARRY_ARM_CONF,
    TopDownPickGeometry,
    arm_frame_target,
    arm_link_points,
    cube_closing_yaw,
    forward_kinematics,
    geometry_from_config,
    grasp_search_bounds,
    grasp_succeeded,
    pointing_down_joints,
    standoff_pose,
    top_down_joint_targets,
)
from prpl_tidybot.visual_servo.image_sources import ImageSource
from prpl_tidybot.visual_servo.top_down_cube import (
    LateralServoParams,
    TopDownLateralServo,
    TopDownServoResult,
)

_logger = logging.getLogger(__name__)

_MAX_DRIVE_TICKS = 600
# Slack (m) on the wall checks' shortcuts (a link's bounding sphere, a point inside the
# outer footprint), for float rounding: a shortcut only ever gives the full check's
# answer.
_SHORTCUT_EPS = 1e-9


def _bin_box(bin_fit: BinFit) -> list[float]:
    """The axis-aligned floor box (x_min, x_max, y_min, y_max) around the bin."""
    c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    hx = 0.5 * (abs(bin_fit.length * c) + abs(bin_fit.width * s))
    hy = 0.5 * (abs(bin_fit.length * s) + abs(bin_fit.width * c))
    return [bin_fit.x - hx, bin_fit.x + hx, bin_fit.y - hy, bin_fit.y + hy]


def _in_box(xy: Sequence[float], box: Sequence[float]) -> bool:
    return bool(box[0] <= xy[0] <= box[1] and box[2] <= xy[1] <= box[3])


def _find_in(source: Any, box: Sequence[float]) -> tuple[float, float, float] | None:
    """The cube's (x, y, yaw) inside the floor box, from the source's own in-box search
    (``find_pose_in_box``, else ``find_in_box`` with yaw 0), or None."""
    finder = getattr(source, "find_pose_in_box", None)
    if callable(finder):
        found = finder(list(box))
        return None if found is None else (found[0], found[1], found[2])
    finder = getattr(source, "find_in_box", None)
    if callable(finder):
        xy = finder(list(box))
        return None if xy is None else (xy[0], xy[1], 0.0)
    return None


def _find_cube(
    source: Any, bin_fit: BinFit | None, box: Sequence[float] | None = None
) -> dict[str, Any]:
    """The cube's pose and whether it lies inside the bin (``in_bin``: its centre inside
    the walls): inside the bin it is in shadow and beyond the floor search, so the bin
    is looked in first when the source can (run 31). ``box`` (x_min, x_max, y_min,
    y_max; a bin's fixed floor box, ``bin_stop_box``) counts as the bin too.

    ``iou`` is the floor localizer's flat-cube fit score when it has one (None for a
    cube found in the bin, whose search fits none). Raises ``RuntimeError`` when the
    cube is not found.
    """

    def inside(xy: Sequence[float]) -> bool:
        return bool(
            (bin_fit is not None and cube_in_bin(xy, bin_fit, margin=0.0))
            or (box is not None and _in_box(xy, box))
        )

    searches = ([] if bin_fit is None else [_bin_box(bin_fit)]) + (
        [] if box is None else [list(box)]
    )
    for search in searches:
        found = _find_in(source, search)
        if found is not None and inside(found[:2]):
            return {
                "x": float(found[0]),
                "y": float(found[1]),
                "yaw_deg": math.degrees(found[2]),
                "in_bin": True,
                "iou": None,
            }
    cube: dict[str, Any] = dict(_localize(source))
    cube["in_bin"] = inside((cube["x"], cube["y"]))
    fit = getattr(source, "last_fit", None)
    cube["iou"] = None if fit is None else float(getattr(fit, "iou", 1.0))
    return cube


def _overlaps(poly: NDArray[np.float64], half: tuple[float, float]) -> bool:
    """Whether the convex polygon ``poly`` (N x 2) overlaps the rectangle centred on the
    origin with half extents ``half`` (separating axes)."""
    rect = np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ]
    )
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for k in range(len(poly)):
        edge = poly[(k + 1) % len(poly)] - poly[k]
        axes.append(np.array([-edge[1], edge[0]]))
    for axis in axes:
        a, b = poly @ axis, rect @ axis
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def arm_bin_collision(
    joints: Sequence[float],
    geom: TopDownPickGeometry,
    base: Sequence[float],
    bin_fit: BinFit,
    in_bin: Any,
) -> str | None:
    """Why the arm at ``joints``, its base at map pose ``base``, would touch the bin's
    walls, or None.

    Each arm link's collision mesh, and the 2F-85 (the IK model has no gripper mesh),
    is cut below the rim (plus ``wall_margin``); a part touches a wall unless that
    cut lies wholly inside the bin's inner footprint or wholly outside its outer one,
    both by ``wall_margin`` (:func:`part_wall_problem`). The 2F-85 is one box
    (``gripper_model: box``), long along the fingers' travel, which is the tool's x
    axis (the IK site's 'closing' axis is across the fingers); or, with
    ``gripper_model: mesh``, each of its links as KINDER's gen3_2f85 shapes them at
    the grasp's opening (:func:`prpl_tidybot.gripper_shape.gripper_link_points`).
    The arm's links are checked first, then the gripper's.
    """
    frame = _BinFrame(geom, base, bin_fit, in_bin)
    return frame.arm_problem(joints) or frame.gripper_problem(
        forward_kinematics(joints, geom), geom.open_position
    )


class _BinFrame:
    """The bin's walls seen from the arm, its base at map pose ``base``: what
    :func:`arm_bin_collision` checks each part against."""

    def __init__(
        self,
        geom: TopDownPickGeometry,
        base: Sequence[float],
        bin_fit: BinFit,
        in_bin: Any,
    ) -> None:
        self.geom, self.bin_fit, self.in_bin = geom, bin_fit, in_bin
        margin = float(in_bin.wall_margin)
        x, y, theta = base
        self.mx = x + geom.mounting_offset * math.cos(theta)
        self.my = y + geom.mounting_offset * math.sin(theta)
        self.rim = geom.floor_z + bin_fit.height + margin
        self.ct, self.st = math.cos(theta), math.sin(theta)
        self.cb, self.sb = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
        self.inner = (
            0.5 * bin_fit.length - bin_fit.wall - margin,
            0.5 * bin_fit.width - bin_fit.wall - margin,
        )
        self.mesh = str(_in_bin_option(in_bin, "gripper_model")) == "mesh"

    def local(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Arm-frame points (N x >=2) in the bin's own frame (N x 2, its x along its
        length)."""
        wx = self.mx + self.ct * points[:, 0] - self.st * points[:, 1] - self.bin_fit.x
        wy = self.my + self.st * points[:, 0] + self.ct * points[:, 1] - self.bin_fit.y
        return np.c_[self.cb * wx + self.sb * wy, -self.sb * wx + self.cb * wy]

    def part_problem(self, name: str, points: NDArray[np.float64]) -> str | None:
        """Why one part (arm-frame points) touches a wall below the rim, or None."""
        low = points[points[:, 2] < self.rim]
        if len(low) == 0:
            return None
        problem = part_wall_problem(name, self.local(low), self.bin_fit, self.in_bin)
        if problem is None:
            return None
        depth = (self.bin_fit.height + self.geom.floor_z) - float(low[:, 2].min())
        return f"{problem} ({depth:.3f} m below the rim)"

    def arm_problem(self, joints: Sequence[float]) -> str | None:
        """The first arm link at ``joints`` that touches a wall, as why."""
        for name, points in arm_link_points(joints, self.geom).items():
            problem = self.part_problem(name, points)
            if problem is not None:
                return problem
        return None

    def gripper_problem(
        self,
        tool_pose: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        open_position: float,
        links: str = "all",
    ) -> str | None:
        """The first gripper part at the tool pose (pinch point, approach, closing:
        :func:`forward_kinematics`) that touches a wall, as why: the housing (or the
        box), then the finger links. ``links``: ``housing`` or ``fingers`` checks only
        those (the housing does not depend on the opening).

        A mesh link whose bounding sphere lies wholly above the rim or wholly inside
        the inner footprint less the margin is passed without transforming its
        points: its points would give the same answer (no problem).
        """
        pos, approach, closing = tool_pose
        fingers = np.cross(closing, approach)
        if not self.mesh:
            if links == "fingers":
                return None
            in_bin = self.in_bin
            corners = [
                pos
                + a * approach
                + b * float(in_bin.finger_half_span) * fingers
                + c * float(in_bin.half_width) * closing
                for a in (float(in_bin.fingertip_drop), -float(in_bin.gripper_length))
                for b in (-1.0, 1.0)
                for c in (-1.0, 1.0)
            ]
            return self.part_problem("gripper", np.asarray(corners))
        tool = np.column_stack([fingers, closing, approach])
        spheres = gripper_link_spheres(open_position)
        for name, local in gripper_link_points(open_position).items():
            if (links == "housing" and name != GRIPPER_HOUSING) or (
                links == "fingers" and name == GRIPPER_HOUSING
            ):
                continue
            centre, radius = spheres[name]
            c = pos + tool @ centre
            if c[2] - radius >= self.rim + _SHORTCUT_EPS:
                continue
            u, v = self.local(c[None, :])[0]
            if (
                abs(u) + radius <= self.inner[0] - _SHORTCUT_EPS
                and abs(v) + radius <= self.inner[1] - _SHORTCUT_EPS
            ):
                continue
            # einsum, not @: a ~100k-point matmul goes multithreaded in OpenBLAS, which
            # spun forever beside the skill server's other threads (a hung test run).
            problem = self.part_problem(
                f"gripper {name}", pos + np.einsum("ij,kj->ik", local, tool)
            )
            if problem is not None:
                return problem
        return None


def part_wall_problem(
    name: str, local: NDArray[np.float64], bin_fit: BinFit, in_bin: Any
) -> str | None:
    """Why one part's cut below the rim (``local``: N x 2, the bin's frame) touches a
    wall, or None: it touches unless it lies wholly inside the inner footprint or wholly
    outside the outer one, both by ``wall_margin``.

    A finger link (:data:`prpl_tidybot.gripper_shape.FINGER_LINKS`) may instead reach up
    to ``finger_allowance`` past the inner wall face, when that is set: light contact
    the cardboard is assumed to give under. The housing and the arm keep the margin.
    """
    margin = float(in_bin.wall_margin)
    inner = (
        0.5 * bin_fit.length - bin_fit.wall - margin,
        0.5 * bin_fit.width - bin_fit.wall - margin,
    )
    if np.all(np.abs(local[:, 0]) <= inner[0]) and np.all(
        np.abs(local[:, 1]) <= inner[1]
    ):
        return None
    allowance = float(_in_bin_option(in_bin, "finger_allowance"))
    link = name.removeprefix("gripper ")
    if allowance > 0 and link in FINGER_LINKS:
        past = np.maximum(
            np.abs(local[:, 0]) - (0.5 * bin_fit.length - bin_fit.wall),
            np.abs(local[:, 1]) - (0.5 * bin_fit.width - bin_fit.wall),
        )
        if float(past.max()) <= allowance:
            return None
    outer = (0.5 * bin_fit.length + margin, 0.5 * bin_fit.width + margin)
    flat = local.astype(np.float32)
    # A point inside the outer footprint puts the hull over it: no hull needed.
    inside = (np.abs(flat[:, 0]) < outer[0] - _SHORTCUT_EPS) & (
        np.abs(flat[:, 1]) < outer[1] - _SHORTCUT_EPS
    )
    if bool(np.any(inside)) or _overlaps(
        cv.convexHull(flat).reshape(-1, 2).astype(np.float64), outer
    ):
        label = f"the gripper's {link}" if link != name else f"the {name}"
        return f"{label} would touch the bin wall"
    return None


def _bin_pick_targets(
    geom: TopDownPickGeometry,
    target: tuple[float, float, float],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> list[float]:
    """The waypoint above the cube with the fingertips ``rim_clearance`` above the bin's
    rim: where the pick into the bin starts and ends."""
    above_hover = (
        bin_fit.height
        + float(in_bin.rim_clearance)
        + float(in_bin.fingertip_drop)
        - geom.grasp_height
    )
    high = dataclasses.replace(geom, hover_height=max(above_hover, geom.hover_height))
    return top_down_joint_targets(*target, high, seed=seed)[0]


def bin_grasp_candidates(
    geom: TopDownPickGeometry, in_bin: Any
) -> list[TopDownPickGeometry]:
    """Every grasp the pick out of the bin may try, in order.

    First the usual grasp on each pair of faces, as listed first (face-aligned,
    untilted, fully open), so a pick that cleared with them keeps its grasp. Then the
    rest of the first set: gripper yaws off the faces (``yaw_offsets_deg``, as listed)
    outermost; within a yaw, kinder-baselines #178's order: tilts as listed
    (``tilts_deg``), both pairs of faces, each opening as listed (``openings_mm``).
    Then each set of ``more_grasps`` (the same three lists) in the same order, less the
    grasps already listed, so a pick the first set clears keeps its grasp. An opening
    narrower than the cube's span across the closing axis at that yaw, plus 5 mm, is
    skipped (at 25 degrees off the faces the 5 cm cube spans 6.6 cm, at 40 degrees 7.0
    cm).
    """
    width = float(_in_bin_option(in_bin, "cube_width"))
    lists = ("yaw_offsets_deg", "tilts_deg", "openings_mm")
    sets = [tuple(_in_bin_option(in_bin, n) for n in lists)] + [
        tuple(_grasp_option(more, n) for n in lists)
        for more in _in_bin_option(in_bin, "more_grasps")
    ]
    out: list[TopDownPickGeometry] = []
    seen: set[tuple[float, float, float, float]] = set()
    for yaws, tilts, openings in sets:
        for offset in (float(o) for o in yaws):
            off = math.radians(offset)
            span = width * (abs(math.cos(off)) + abs(math.sin(off)))
            for tilt in (float(t) for t in tilts):
                for turn in (0.0, math.pi / 2):
                    for opening in (float(o) for o in openings):
                        if opening / 1000 < span + 0.005 and offset:
                            continue
                        if (offset, tilt, turn, opening) in seen:
                            continue
                        seen.add((offset, tilt, turn, opening))
                        out.append(
                            dataclasses.replace(
                                geom,
                                closing_turn=turn,
                                tilt=math.radians(tilt),
                                open_position=opening_command(opening / 1000),
                                wrist_yaw_offset=geom.wrist_yaw_offset + off,
                            )
                        )

    def first(g: TopDownPickGeometry) -> bool:
        return (g.wrist_yaw_offset, g.tilt, g.open_position) == (
            out[0].wrist_yaw_offset,
            out[0].tilt,
            out[0].open_position,
        )

    return [g for g in out if first(g)] + [g for g in out if not first(g)]


def _grasp_option(more: Any, name: str) -> Any:
    """``more.<name>`` for an entry of ``more_grasps`` (config or :class:`GraspSet`), or
    the oblique extension's list when the entry leaves it out."""
    value = more.get(name) if hasattr(more, "get") else getattr(more, name, None)
    return getattr(_GRASP_SET_DEFAULTS, name) if value is None else value


def _bin_grasp_geometry(
    geom: TopDownPickGeometry,
    base: Sequence[float],
    cube: dict[str, Any],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> tuple[TopDownPickGeometry | None, str | None]:
    """The pick geometry for a cube inside the bin: of the grasps on its two pairs of
    faces (``closing_turn`` 0 and pi/2), the one that touches no wall, the usual one
    first; else None and why the usual one would touch.

    With more candidates configured (:func:`bin_grasp_candidates`) the first that clears
    is used; a refusal then says where the cube sits against the walls (a corner when
    two walls are within the open fingers' reach), and for each pair of faces the part
    that binds most often, how often, and the first reason; and how many candidates were
    tried. Nothing moves either way.
    """
    candidates = bin_grasp_candidates(geom, in_bin)
    search = _BinGraspSearch(base, cube, bin_fit, in_bin, seed)
    first: dict[float, str] = {}
    binds: dict[float, collections.Counter[str]] = {}
    for g in candidates:
        problem = search.problem(g)
        if problem is None:
            return g, None
        first.setdefault(g.closing_turn, problem)
        binds.setdefault(g.closing_turn, collections.Counter())[_binding(problem)] += 1
    if len(candidates) == 2:
        _logger.info("No grasp clears the bin's walls: %s.", first[0.0])
        return None, first[0.0]
    faces = []
    for turn, problem in first.items():
        body, n = binds[turn].most_common(1)[0]
        total = sum(binds[turn].values())
        faces.append(
            f"faces {math.degrees(turn):.0f} deg: {body} binds in {n}/{total} "
            f"grasps (first: {problem})"
        )
    why = (
        f"{_wall_situation(base, cube, bin_fit, in_bin)}; {'; '.join(faces)}; "
        f"{len(candidates)} candidates"
    )
    _logger.info("No grasp clears the bin's walls: %s.", why)
    return None, why


_PHASES = {"above": "above the rim", "hover": "hover", "grasp": "grasp"}


def _binding(problem: str) -> str:
    """The part and pose a candidate's refusal names, e.g. "the gripper's base at the
    hover" (or "no IK")."""
    phase, _, rest = problem.partition(": ")
    if phase not in _PHASES:
        return "no IK"
    part = rest.split(" would touch")[0]
    return f"{part} at the {_PHASES[phase]}"


def _wall_situation(
    base: Sequence[float], cube: dict[str, Any], bin_fit: BinFit, in_bin: Any
) -> str:
    """Where the cube sits against the bin's walls, each wall named as the robot at
    ``base`` sees it: in a corner when two walls are within the open fingers' reach
    along their travel plus ``wall_margin``, else the nearest wall."""
    dx, dy = cube["x"] - bin_fit.x, cube["y"] - bin_fit.y
    cb, sb = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
    u, v = cb * dx + sb * dy, -sb * dx + cb * dy
    hl, hw = 0.5 * bin_fit.length - bin_fit.wall, 0.5 * bin_fit.width - bin_fit.wall
    walls = [  # (distance to the inner face, outward normal in the map)
        (hl - u, (cb, sb)),
        (hl + u, (-cb, -sb)),
        (hw - v, (-sb, cb)),
        (hw + v, (sb, -cb)),
    ]
    ahead = (math.cos(base[2]), math.sin(base[2]))

    def name(normal: tuple[float, float]) -> str:
        along = normal[0] * ahead[0] + normal[1] * ahead[1]
        left = -normal[0] * ahead[1] + normal[1] * ahead[0]
        if abs(along) >= abs(left):
            return "far wall" if along > 0 else "near wall"
        return "left wall" if left > 0 else "right wall"

    reach = _finger_reach(in_bin) + float(in_bin.wall_margin)
    walls.sort(key=lambda w: w[0])
    (d1, n1), (d2, n2) = walls[0], walls[1]
    if d2 < reach:
        return (
            f"cube in a corner: {100 * d1:.1f} cm from the {name(n1)} and "
            f"{100 * d2:.1f} cm from the {name(n2)}, both within the open fingers' "
            f"reach ({100 * reach:.1f} cm)"
        )
    return f"cube {100 * d1:.1f} cm from the {name(n1)}"


def _finger_reach(in_bin: Any) -> float:
    """How far the fully open 2F-85 reaches along its fingers' travel from the pinch
    point (m): the mesh links' farthest point, or the box's half span."""
    if str(_in_bin_option(in_bin, "gripper_model")) != "mesh":
        return float(in_bin.finger_half_span)
    return max(float(np.abs(p[:, 0]).max()) for p in gripper_link_points(0.0).values())


class _BinGraspSearch:
    """:func:`_bin_pick_problem` for many candidates on one cube, the same answers
    cheaper: the joints (IK), the arm links' and the housing's check of a grasp do
    not depend on its opening, so they are found once per yaw, tilt and pair of faces
    and reused across openings, and only the finger links are checked per opening
    (bounding spheres first, :meth:`_BinFrame.gripper_problem`)."""

    def __init__(
        self,
        base: Sequence[float],
        cube: dict[str, Any],
        bin_fit: BinFit,
        in_bin: Any,
        seed: Sequence[float],
    ) -> None:
        self.base, self.cube, self.bin_fit = base, cube, bin_fit
        self.in_bin, self.seed = in_bin, seed
        self._poses: dict[TopDownPickGeometry, _PlannedGrasp | str] = {}

    def problem(self, geom: TopDownPickGeometry) -> str | None:
        """Why the grasp ``geom`` would touch a wall, or None (as
        :func:`_bin_pick_problem`)."""
        key = dataclasses.replace(geom, open_position=0.0)
        planned = self._poses.get(key)
        if planned is None:
            planned = self._plan(key)
            self._poses[key] = planned
        if isinstance(planned, str):
            return planned
        for k, phase in enumerate(("above", "hover", "grasp")):
            problem = planned.problem(k) or planned.frame.gripper_problem(
                planned.tool_poses[k], geom.open_position, links="fingers"
            )
            if problem is not None:
                return f"{phase}: {problem}"
        return None

    def _plan(self, geom: TopDownPickGeometry) -> _PlannedGrasp | str:
        ax, ay = arm_frame_target(self.base, (self.cube["x"], self.cube["y"]), geom)
        yaw = math.radians(self.cube["yaw_deg"]) - self.base[2]
        try:
            above = _bin_pick_targets(
                geom, (ax, ay, yaw), self.bin_fit, self.in_bin, self.seed
            )
            hover, grasp = top_down_joint_targets(ax, ay, yaw, geom, seed=above)
        except ValueError as e:
            return f"no top-down pick into the bin: {e}"
        return _PlannedGrasp(
            _BinFrame(geom, self.base, self.bin_fit, self.in_bin),
            [above, hover, grasp],
        )


class _PlannedGrasp:
    """One grasp's planned joints (above the rim, hover, grasp) and their tool poses,
    with each pose's check of the arm links and the gripper's housing, which no opening
    changes, done at most once."""

    def __init__(self, frame: _BinFrame, joints: list[list[float]]) -> None:
        self.frame, self.joints = frame, joints
        self.tool_poses = [forward_kinematics(q, frame.geom) for q in joints]
        self._arm: dict[int, str | None] = {}

    def problem(self, k: int) -> str | None:
        """The arm links' or the housing's problem at pose ``k``."""
        if k not in self._arm:
            self._arm[k] = self.frame.arm_problem(
                self.joints[k]
            ) or self.frame.gripper_problem(self.tool_poses[k], 0.0, links="housing")
        return self._arm[k]


def _bin_pick_problem(
    geom: TopDownPickGeometry,
    base: Sequence[float],
    cube: dict[str, Any],
    bin_fit: BinFit,
    in_bin: Any,
    seed: Sequence[float],
) -> str | None:
    """Why the pick of the cube inside the bin, from ``base``, would touch a wall at the
    waypoint above the rim, the hover or the grasp, or None."""
    ax, ay = arm_frame_target(base, (cube["x"], cube["y"]), geom)
    yaw = math.radians(cube["yaw_deg"]) - base[2]
    try:
        above = _bin_pick_targets(geom, (ax, ay, yaw), bin_fit, in_bin, seed)
        hover, grasp = top_down_joint_targets(ax, ay, yaw, geom, seed=above)
    except ValueError as e:
        return f"no top-down pick into the bin: {e}"
    for name, joints in (("above", above), ("hover", hover), ("grasp", grasp)):
        problem = arm_bin_collision(joints, geom, base, bin_fit, in_bin)
        if problem is not None:
            return f"{name}: {problem}"
    return None


def _move_arm(
    arm: ArmInterface,
    goal: list[float],
    gripper: float,
    arm_cfg: DictConfig,
    stop: threading.Event | None = None,
) -> bool:
    """Stream ``goal`` until every joint is within tolerance (wrapped) or time runs out;
    returns whether it got there.

    A set ``stop`` ends the move at once (``ArmMotionAborted``), the arm held where it
    is.
    """
    for _ in range(int(arm_cfg.max_ticks)):
        _check_stop(stop, arm)
        arm.execute_action(goal, gripper)
        err = np.array(arm.get_arm_state()) - np.array(goal)
        if np.all(
            np.abs((err + np.pi) % (2 * np.pi) - np.pi) < float(arm_cfg.tolerance)
        ):
            return True
        time.sleep(POLICY_CONTROL_PERIOD)
    _logger.warning(
        "Arm did not reach its goal within %d ticks.", int(arm_cfg.max_ticks)
    )
    return False


def _cube_source_kwargs(
    cfg: DictConfig, target_cfg: Any, log_dir: Path | None = None
) -> dict[str, Any]:
    """The ceiling localizer searches everywhere the arm can grasp from inside the
    drivable box, not just the box itself (a bin just past its edge), and saves the
    frames of any missed localization beside the run's logs."""
    if not str(target_cfg["_target_"]).endswith(
        ("CeilingCubeTargetSource", "CeilingCubeDetectorSource")
    ):
        return {}
    bounds = grasp_search_bounds(
        tuple(cfg.preview.floor_bounds), geometry_from_config(cfg.pick.get("top_down"))
    )
    kwargs: dict[str, Any] = {"floor_bounds": list(bounds)}
    if log_dir is not None:
        kwargs["failure_dir"] = str(Path(log_dir) / "localization_failures")
    return kwargs


def _retract(
    cfg: DictConfig, arm: ArmInterface, stop: threading.Event | None = None
) -> None:
    """Fold the arm to its retract pose under the compliant loop: it needs no precision,
    and only the grasp itself runs under the stiff onboard controller."""
    arm.note_step("retract")
    _move_arm(arm, list(RETRACT_ARM_CONF), arm.get_gripper_state(), cfg.arm_home, stop)


def _check_stop(stop: threading.Event | None, arm: ArmInterface | None = None) -> None:
    """Raise ``ArmMotionAborted`` once ``stop`` is set (the skill server's stop()),
    holding the arm where it is first."""
    if stop is None or not stop.is_set():
        return
    if arm is not None:
        arm.execute_action(list(arm.get_arm_state()), arm.get_gripper_state())
    raise ArmMotionAborted("stopped")


def _drive_to(
    real_env: Any,
    pose: list[float],
    workspace: Workspace | None = None,
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
) -> tuple[float, float, float]:
    """Drive the base to the map-frame pose with the kinematic3d executor (as
    hardware_tests/test_base_map_target.py does); returns the reached pose.

    Uses a robot-only perceiver: driving needs no cube, and must still work when the
    cube cannot be found (the reset after a failed localization). With a
    ``workspace`` the drive is checked first (drivable box, barrier;
    :meth:`Workspace.plan_path`, which may back straight off the barrier before
    turning) and a drive that would leave it raises ``ValueError`` before the base
    moves. A set ``stop`` ends the drive (``ArmMotionAborted``) without another
    command.
    """
    legs = [(float(pose[0]), float(pose[1]), float(pose[2]))]
    if workspace is not None:
        obs, _ = real_env.reset()
        here = obs.map_base_pose
        legs, problem = workspace.plan_path(
            (float(here.x), float(here.y), float(here.theta())), pose, bin_fit
        )
        if problem is not None:
            raise ValueError(f"Drive refused: {problem}.")
        if len(legs) > 1:
            _logger.info(
                "Backing straight off the barrier to (%+.3f, %+.3f) before the drive.",
                legs[0][0],
                legs[0][1],
            )
    reached: tuple[float, float, float] | None = None
    for leg in legs:
        reached = _drive_straight(real_env, leg, stop)
    assert reached is not None
    return reached


def _drive_straight(
    real_env: Any, pose: Sequence[float], stop: threading.Event | None
) -> tuple[float, float, float]:
    """One straight leg of :func:`_drive_to`, unchecked."""
    perceiver = PrplLab3DPerceiver()
    obs, info = real_env.reset()
    state = perceiver.step(obs, info)
    robot = state.get_object_from_name("robot")
    delta = np.zeros(11)
    delta[0] = pose[0] - state.get(robot, "pos_base_x")
    delta[1] = pose[1] - state.get(robot, "pos_base_y")
    dtheta = pose[2] - state.get(robot, "pos_base_rot")
    delta[2] = (dtheta + math.pi) % (2 * math.pi) - math.pi
    if math.hypot(delta[0], delta[1]) < 0.01 and abs(delta[2]) < 0.02:
        reached = (
            obs.map_base_pose
        )  # already there; an empty move is not a base segment
        return (float(reached.x), float(reached.y), float(reached.theta()))
    executor = Kinematic3DPlanExecutor()
    executor.set_trajectory([(state, delta)])
    for _ in range(_MAX_DRIVE_TICKS):
        _check_stop(stop)
        if executor.done(state):
            break
        real_action, _ = executor.step(state)
        obs, _, _, _, info = real_env.step(real_action)
        state = perceiver.step(obs, info)
    reached = obs.map_base_pose
    return (float(reached.x), float(reached.y), float(reached.theta()))


def _localize(source: Any) -> dict[str, float]:
    x, y, _ = source.get_target()
    return {"x": float(x), "y": float(y), "yaw_deg": math.degrees(source.get_yaw())}


def pick_problem(
    robot: Sequence[float],
    cube: dict[str, float],
    geom: TopDownPickGeometry,
    floor: Sequence[float],
    workspace: Workspace | None,
    bin_fit: BinFit | None = None,
) -> str | None:
    """Why the top-down pick of ``cube`` cannot even start from ``robot``, or None:

    the cube past or on the barrier (the robot never crosses it), no standoff that keeps
    the base in the drivable box and on its side, or no straight drive there.
    """
    if workspace is not None:
        yaw = cube.get("yaw")
        if yaw is None and cube.get("yaw_deg") is not None:
            yaw = math.radians(cube["yaw_deg"])
        problem = workspace.cube_problem((cube["x"], cube["y"]), yaw)
        if problem is not None:
            return problem
    try:
        stand = standoff_pose(
            robot,
            (cube["x"], cube["y"]),
            geom,
            floor,
            None if workspace is None else workspace.valid(bin_fit),
        )
    except ValueError as e:
        return str(e)
    if workspace is not None:
        problem = workspace.path_problem(robot, stand, bin_fit)
        if problem is not None:
            return f"pick standoff unreachable: {problem}"
    return None


def _plus_minus(step: float, top: float) -> tuple[float, ...]:
    """0, +step, -step, +2 step, ...

    up to +-top.
    """
    n = int(round(top / step))
    return (0.0, *(s * k * step for k in range(1, n + 1) for s in (1.0, -1.0)))


@dataclasses.dataclass(frozen=True)
class GraspSet:
    """One set of grasps the pick out of the bin tries after the first
    (``pick.top_down.in_bin.more_grasps``): every combination of the three lists
    (:func:`bin_grasp_candidates`).

    The default is the oblique extension: yaws up to 45
    deg off the faces, tilts up to 30 deg, 85, 80, 75 and 70 mm.
    """

    yaw_offsets_deg: tuple[float, ...] = _plus_minus(5.0, 45.0)
    tilts_deg: tuple[float, ...] = _plus_minus(5.0, 30.0)
    openings_mm: tuple[float, ...] = (85.0, 80.0, 75.0, 70.0)


@dataclasses.dataclass(frozen=True)
class InBinPick:  # pylint: disable=too-many-instance-attributes
    """The pick out of the bin's clearances (``pick.top_down.in_bin``; m): the
    fingertips ``rim_clearance`` above the rim at the waypoint, the open 2F-85 as a box
    ``finger_half_span`` along its fingers' travel and ``half_width`` across, from
    ``gripper_length`` above the pinch point to ``fingertip_drop`` below it, and no
    planned pose within ``wall_margin`` of a wall. The defaults are the measured ones
    (conf/same_side_practice.yaml).

    The grasps tried (:func:`bin_grasp_candidates`): first the near-wall set,
    ``yaw_offsets_deg`` (the gripper's yaw off the cube's faces, up to 25 deg),
    ``tilts_deg`` and ``openings_mm`` (kinder-baselines #178's tilted, partly opened
    grasp); then ``more_grasps``, by default the oblique extension (:class:`GraspSet`).
    Each is checked as ``gripper_model`` (``mesh``: the real 2F-85's links at the
    grasp's opening; ``box``: the box above). ``finger_allowance`` (how far past the
    inner wall face the finger links alone may reach; mesh only) is off: every link
    keeps ``wall_margin``.
    """

    rim_clearance: float = 0.04
    finger_half_span: float = 0.074
    half_width: float = 0.038
    gripper_length: float = 0.124
    fingertip_drop: float = 0.029
    wall_margin: float = 0.01
    tilts_deg: tuple[float, ...] = (0.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0)
    openings_mm: tuple[float, ...] = (85.0, 70.0)
    gripper_model: str = "mesh"
    finger_allowance: float = 0.0
    yaw_offsets_deg: tuple[float, ...] = _plus_minus(5.0, 25.0)
    more_grasps: tuple[GraspSet, ...] = (GraspSet(),)
    cube_width: float = 0.05


_IN_BIN_DEFAULTS = InBinPick()
_GRASP_SET_DEFAULTS = GraspSet()


def _in_bin_option(in_bin: Any, name: str) -> Any:
    """``in_bin.<name>``, or the default when the block predates the option."""
    value = in_bin.get(name) if hasattr(in_bin, "get") else getattr(in_bin, name, None)
    return getattr(_IN_BIN_DEFAULTS, name) if value is None else value


# Why a pick may not start because the cube is in the bin and the pick out of it is off.
IN_BIN_PICK_OFF = "cube in the bin (the pick out of the bin is off)"


def plan_pick(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    robot: Sequence[float],
    cube: dict[str, Any],
    geom: TopDownPickGeometry,
    floor: Sequence[float],
    workspace: Workspace | None,
    bin_fit: BinFit | None,
    in_bin: Any,
    seed: Sequence[float],
) -> tuple[tuple[float, float, float] | None, str | None]:
    """Where the base stands to pick ``cube`` (``_find_cube``'s dict) from ``robot``,
    or why the pick cannot start: everything checked before anything moves.

    The cube past or on the barrier, no standoff in the drivable box on the robot's
    side clear of the bin, no straight drive there (:func:`pick_problem`); for a cube
    inside the bin, the pick out of it off (``in_bin`` None) or no grasp on either pair
    of its faces clear of the walls; otherwise no IK for the hover and the grasp.
    Returns (standoff, None) or (None, why)."""
    if cube.get("in_bin") and in_bin is None:
        return None, IN_BIN_PICK_OFF
    problem = pick_problem(robot, cube, geom, floor, workspace, bin_fit)
    if problem is not None:
        return None, problem
    xy = (cube["x"], cube["y"])
    stand = standoff_pose(
        robot, xy, geom, floor, None if workspace is None else workspace.valid(bin_fit)
    )
    if cube.get("in_bin") and bin_fit is not None:
        chosen, wall = _bin_grasp_geometry(geom, stand, cube, bin_fit, in_bin, seed)
        if chosen is None:
            return None, f"cube too close to the bin wall to grasp ({wall})"
        return stand, None
    ax, ay = arm_frame_target(stand, xy, geom)
    try:
        top_down_joint_targets(
            ax, ay, math.radians(cube["yaw_deg"]) - stand[2], geom, seed=seed
        )
    except ValueError as e:
        return None, f"pick IK: {e}"
    return stand, None


def _place_top_down(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: DictConfig,
    real_env: Any,
    arm: ArmInterface,
    geom: TopDownPickGeometry,
    step: Callable[[str], None],
    target: tuple[float, float],
    record: dict[str, Any],
    workspace: Workspace | None = None,
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
) -> str | None:
    """The top-down pick in reverse: carry the held cube to the map point `target` and
    set it down.

    The closed gripper is lowered to the grasp height, where it lifted the cube off the
    floor, so the cube rests on the floor before the gripper opens (the foam cube
    bounces if dropped). Returns a stop reason, or None once placed.
    """
    obs, _ = real_env.reset()
    robot = (obs.map_base_pose.x, obs.map_base_pose.y, obs.map_base_pose.theta())
    stand = standoff_pose(
        robot,
        target,
        geom,
        tuple(cfg.preview.floor_bounds),
        None if workspace is None else workspace.valid(bin_fit),
    )
    step(
        f"Carry the cube to the standoff ({stand[0]:+.3f}, {stand[1]:+.3f}, "
        f"{math.degrees(stand[2]):+.0f} deg) and set it down at "
        f"({target[0]:+.3f}, {target[1]:+.3f})"
    )
    base = _drive_to(real_env, list(stand), workspace, stop, bin_fit)
    ax, ay = arm_frame_target(base, target, geom)
    moves: list[dict[str, Any]] = []
    record["place"] = {
        "target": list(target),
        "base": list(base),
        "arm_target_xy": [ax, ay],
        "moves": moves,
    }
    try:
        hover, down = top_down_joint_targets(
            ax, ay, 0.0, geom, seed=arm.get_arm_state()
        )
    except ValueError as e:
        return f"place_failure: {e}"
    td = cfg.pick.top_down
    mover: _Mover = (
        _AngularMover(arm, td, geom, moves)
        if td.get("arm_controller", "compliant") == "angular"
        else _CompliantMover(arm, td, geom, moves, cfg.arm_home)
    )
    mover.stop = stop
    mover.begin()
    try:
        if not mover.move("place_hover", hover):
            return "place: the hover did not arrive"
        if not mover.move("place_down", down, correct=True):
            mover.move("place_up", hover)
            return "place: the set-down did not arrive (the cube is still held)"
        mover.gripper(0.0)
        mover.move("place_up", hover)
    finally:
        mover.end()
    return None


# An onboard-controller move that ends further than this from its goal has hit
# something (the floor, the cube) or faulted; the toss uses the same check.
_ANGULAR_ARRIVAL_DEG = 1.0


class _Mover:
    """How the grasp sequence moves the arm and gripper: under the onboard position
    controller (`_AngularMover`) or the compliant torque loop (`_CompliantMover`).

    One controller is used for a whole run; nothing switches mid-episode.
    """

    name = ""
    # Set by the skill server: once set, the next move or gripper command raises
    # ArmMotionAborted instead of moving.
    stop: threading.Event | None = None

    def __init__(
        self,
        arm: ArmInterface,
        td: Any,
        geom: TopDownPickGeometry,
        moves: list[dict[str, Any]],
    ) -> None:
        self.arm = arm
        self.td = td
        self.geom = geom
        self.moves = moves

    def check_stop(self) -> None:
        """Raise ``ArmMotionAborted`` if ``stop`` is set (nothing more is commanded)."""
        if self.stop is not None and self.stop.is_set():
            raise ArmMotionAborted("stopped")

    def joints(self) -> list[float]:
        """Where the arm is now (IK seeds, the servo's jump check and position)."""
        return list(self.arm.get_arm_state())

    def begin(self) -> None:
        """Take whatever control the controller needs."""

    def end(self) -> None:
        """Hand the arm back to the compliant loop, holding where it is."""

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        """Move to `goal`; record it; return whether it arrived.

        `correct` asks a controller that can settle short of its goal to correct onto
        it.
        """
        raise NotImplementedError

    def gripper(self, position: float) -> float:
        """Move the gripper; return where it stopped."""
        raise NotImplementedError

    def _record(
        self, name: str, goal: list[float], measured: list[float], **extra: Any
    ) -> dict[str, Any]:
        err = np.degrees(
            np.abs((np.array(measured) - np.array(goal) + np.pi) % (2 * np.pi) - np.pi)
        )
        move = {
            "name": name,
            "controller": self.name,
            "reached": bool(np.all(err < _ANGULAR_ARRIVAL_DEG)),
            "planned_deg": np.round(np.degrees(goal), 2).tolist(),
            "measured_deg": np.round(np.degrees(measured), 2).tolist(),
            "planned_pinch_z": float(forward_kinematics(goal, self.geom)[0][2]),
            "measured_pinch_z": float(forward_kinematics(measured, self.geom)[0][2]),
        } | extra
        self.moves.append(move)
        _logger.info(
            "Move %s (%s): pinch z planned %+.4f measured %+.4f (floor %+.3f); max "
            "joint error %.2f deg.",
            name,
            self.name,
            move["planned_pinch_z"],
            move["measured_pinch_z"],
            self.geom.floor_z,
            float(err.max()),
        )
        return move


class _AngularMover(_Mover):
    """Jimmy Wu's pick_object moves: the onboard controller, torque loop released."""

    name = "angular"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._aborted = False
        self._joints = list(self.arm.get_arm_state())

    def joints(self) -> list[float]:
        # With the torque loop released the arm server's joint reading is frozen at
        # the release (run 10: an IK seeded from it flipped branch); each onboard
        # move reports where the arm stopped, so track that instead.
        return list(self._joints)

    def begin(self) -> None:
        self._joints = list(self.arm.get_arm_state())
        self.arm.release()

    def end(self) -> None:
        if not self._aborted:
            self.arm.resume()

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        self.arm.note_step(name)
        try:
            self.check_stop()
            measured = self.arm.move_angular(goal)
        except ArmMotionAborted:
            self._aborted = True
            raise
        self._joints = list(measured)
        return bool(self._record(name, goal, list(measured))["reached"])

    def gripper(self, position: float) -> float:
        self.arm.note_step(f"gripper to {position:.2f}")
        try:
            self.check_stop()
            return self.arm.set_gripper_angular(
                position, float(self.td.get("gripper_timeout_s", 4.0))
            )
        except ArmMotionAborted:
            self._aborted = True
            raise


class _CompliantMover(_Mover):
    """The compliant torque loop.

    As in Tom Silver's cylinder servo, a move streams its joint target and waits for the
    arm to arrive (or stop): the loop settles a few degrees short wherever its gravity
    and friction models miss (24 mm at the gripper in run 3), and for the hover and the
    servo's nudges that does not matter, because the wrist camera measures the cube from
    wherever the gripper really is. Only the descent, which the camera cannot watch, is
    corrected (`correct=True`): after the arm settles the pinch point is re-aimed past
    the goal by the measured error, never below the grasp height, so an arm held up by
    the floor or the cube is not pushed into it.
    """

    name = "compliant"

    def __init__(
        self,
        arm: ArmInterface,
        td: Any,
        geom: TopDownPickGeometry,
        moves: list[dict[str, Any]],
        arm_cfg: DictConfig,
    ) -> None:
        super().__init__(arm, td, geom, moves)
        self.arm_cfg = arm_cfg
        self.command = list(arm.get_arm_state())
        self.grip = arm.get_gripper_state()

    def move(self, name: str, goal: list[float], correct: bool = False) -> bool:
        self.arm.note_step(name)
        self.check_stop()
        goal_pos, approach, closing = forward_kinematics(goal, self.geom)
        if correct and approach[2] < -0.99:
            return self._move_down(name, goal, goal_pos, closing)
        if correct:
            return self._move_joints(name, goal)
        settled = self._settle(list(goal))
        self._record(name, goal, list(self.arm.get_arm_state()), settled=settled)
        return settled

    def _move_down(
        self,
        name: str,
        goal: list[float],
        goal_pos: NDArray[np.float64],
        closing: NDArray[np.float64],
    ) -> bool:
        """A pointing-down move (hover, grasp, lift, servo nudges), corrected in
        Cartesian space: re-aim the pinch point at goal + (goal - measured), keeping
        the tool pointing down, until within `correction_tol_mm`. Near the floor the
        aim never goes below the goal's own height, so an arm held up by the floor
        or the cube is not pushed into it."""
        tol = float(self.td.get("correction_tol_mm", 2.0)) / 1000
        rounds = int(self.td.get("correction_rounds", 3))
        max_push = float(self.td.get("correction_max_mm", 60.0)) / 1000
        near_floor = goal_pos[2] < self.geom.floor_z + self.geom.grasp_height + 0.03
        yaw = math.atan2(closing[1], closing[0])
        aim = np.array(goal_pos, float)
        command = list(goal)
        used = 0
        for used in range(rounds + 1):
            self._settle(command)
            measured_pos = forward_kinematics(self.arm.get_arm_state(), self.geom)[0]
            err = measured_pos - goal_pos
            if np.linalg.norm(err) < tol or used == rounds:
                break
            aim = aim - err
            push = aim - goal_pos
            if np.linalg.norm(push) > max_push:
                aim = goal_pos + push * (max_push / np.linalg.norm(push))
            if near_floor and aim[2] < goal_pos[2]:
                aim[2] = goal_pos[2]
            command = pointing_down_joints(
                aim, yaw, self.geom, seed=self.arm.get_arm_state(), tilt=self.geom.tilt
            )
        measured = list(self.arm.get_arm_state())
        err = forward_kinematics(measured, self.geom)[0] - goal_pos
        off = float(np.linalg.norm(err))
        # The error in the gripper's frame: along its closing axis, across it, up.
        across = np.cross(closing, np.array([0.0, 0.0, -1.0]))
        tool = (float(err @ closing), float(err @ across), float(err[2]))
        return bool(
            self._record(
                name,
                goal,
                measured,
                reached=self._arrived(tool, off, tol),
                pinch_error_mm=round(off * 1000, 1),
                pinch_error_tool_mm=[round(v * 1000, 1) for v in tool],
                commanded_deg=np.round(np.degrees(command), 2).tolist(),
                correction_rounds=used,
            )["reached"]
        )

    def _arrived(
        self, tool: tuple[float, float, float], off: float, tol: float
    ) -> bool:
        """Whether the descent ended close enough to close the gripper: within each
        axis's limit in ``correction_arrive_mm`` (closing, across, down, up; mm), or
        without them within 1.5 x the correction tolerance."""
        limits = self.td.get("correction_arrive_mm")
        if limits is None:
            return off < 1.5 * tol
        closing, across, up = (v * 1000 for v in tool)
        return (
            abs(closing) <= float(limits.closing)
            and abs(across) <= float(limits.across)
            and -float(limits.down) <= up <= float(limits.up)
        )

    def _move_joints(self, name: str, goal: list[float]) -> bool:
        """Any other move (the carry pose): corrected in joint space."""
        tol = math.radians(float(self.td.get("correction_tol_deg", 0.3)))
        rounds = int(self.td.get("correction_rounds", 3))
        max_push = math.radians(float(self.td.get("correction_max_deg", 8.0)))
        command = np.array(goal, float)
        used = 0
        for used in range(rounds + 1):
            self._settle(list(command))
            err = _wrap(np.array(self.arm.get_arm_state()) - np.array(goal))
            if np.all(np.abs(err) < tol) or used == rounds:
                break
            push = _wrap(command - err - np.array(goal))
            command = np.array(goal) + np.clip(push, -max_push, max_push)
        return bool(
            self._record(
                name,
                goal,
                list(self.arm.get_arm_state()),
                commanded_deg=np.round(np.degrees(command), 2).tolist(),
                correction_rounds=used,
            )["reached"]
        )

    def _settle(self, command: list[float]) -> bool:
        """Stream `command` until the arm is within `arrive_deg` of it, or has moved and
        stopped (it can settle short of the command), or `max_ticks` pass.

        The first quiet ticks after a new command are before the arm starts to move, so
        stillness only counts once it has moved or been streamed for a while.
        """
        self.command = command
        arrive = math.radians(float(self.td.get("arrive_deg", 1.0)))
        start = np.array(self.arm.get_arm_state())
        last = start
        still = 0
        for tick in range(int(self.arm_cfg.max_ticks)):
            self.arm.execute_action(command, self.grip)
            now = np.array(self.arm.get_arm_state())
            if np.all(np.abs(_wrap(now - np.array(command))) < arrive):
                return True
            moving = np.any(np.abs(_wrap(now - last)) > math.radians(0.05))
            still = 0 if moving else still + 1
            last = now
            started = np.any(np.abs(_wrap(now - start)) > math.radians(0.5))
            if still >= 5 and (started or tick >= 15):
                return True
            time.sleep(POLICY_CONTROL_PERIOD)
        return False

    def gripper(self, position: float) -> float:
        self.arm.note_step(f"gripper to {position:.2f}")
        self.check_stop()
        self.grip = position
        _set_gripper(self.arm, position, self.td, hold=self.command)
        self.grip = self.arm.get_gripper_state()
        return self.grip


def _grasp_at(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    cfg: DictConfig,
    cube: dict[str, float],
    label: str,
    base: tuple[float, float, float],
    arm: ArmInterface,
    geom: TopDownPickGeometry,
    step: Callable[[str], None],
    attempt: int,
    record: dict[str, Any],
    wrist: "_WristRecorder",
    servo: Any,
    read: Callable[[float], float],
    clip: str,
    stop: threading.Event | None = None,
    bin_fit: BinFit | None = None,
    in_bin: Any = None,
) -> tuple[bool, str | None]:
    """One grasp from where the base parked at `cube`: (held, stop reason or None).

    ``step`` is told each step (the ceiling video's label); ``read`` maps the gripper's
    reading after closing (fake mode's gripper closes on nothing). The wrist clip
    ``clip`` is left recording when the cube is held. With ``bin_fit`` the cube is
    inside the bin: the arm goes through the waypoint above the rim, and no arm pose
    touching a wall is commanded.
    """
    td = cfg.pick.top_down
    ax, ay = arm_frame_target(base, (cube["x"], cube["y"]), geom)
    yaw_rel = math.radians(cube["yaw_deg"]) - base[2]
    try:
        top_down_joint_targets(ax, ay, yaw_rel, geom, seed=arm.get_arm_state())
    except ValueError as e:
        return False, f"pick_failure: {e}"
    moves: list[dict[str, Any]] = []
    record.setdefault("pick_attempts", []).append(
        {
            "attempt": attempt,
            "cube": cube,
            "target": label,
            "arm_target_xy": [ax, ay],
            "base": list(base),
            "wrist_clip": clip,
            "moves": moves,
        }
    )
    above: list[float] | None = None
    clear: Callable[[Sequence[float]], str | None] | None = None
    if bin_fit is not None and in_bin is not None:
        chosen, wall = _bin_grasp_geometry(
            geom, base, cube, bin_fit, in_bin, arm.get_arm_state()
        )
        if chosen is None:
            record["pick_attempts"][-1].update(
                {"gripper": None, "held": False, "note": f"bin wall: {wall}"}
            )
            _logger.warning("Pick attempt %d not started: bin wall: %s.", attempt, wall)
            return False, None
        geom = chosen
        record["pick_attempts"][-1]["closing_turn_deg"] = math.degrees(
            geom.closing_turn
        )
        record["pick_attempts"][-1]["tilt_deg"] = math.degrees(geom.tilt)
        record["pick_attempts"][-1]["open_position"] = geom.open_position
        record["pick_attempts"][-1]["wrist_yaw_offset_deg"] = math.degrees(
            geom.wrist_yaw_offset
        )
        above = _bin_pick_targets(
            geom, (ax, ay, yaw_rel), bin_fit, in_bin, arm.get_arm_state()
        )
        fit, cfg_in_bin = bin_fit, in_bin

        def clear(joints: Sequence[float]) -> str | None:
            return arm_bin_collision(joints, geom, base, fit, cfg_in_bin)

    step(
        f"Pick ({label}): open, "
        + ("arm above the bin's rim, " if above is not None else "")
        + f"arm to 10 cm above the cube (arm frame {ax:+.3f}, {ay:+.3f}), "
        + ("align with the wrist camera, " if servo is not None else "")
        + "descend, close"
    )
    wrist.start(clip)
    mover: _Mover = (
        _AngularMover(arm, td, geom, moves)
        if td.get("arm_controller", "compliant") == "angular"
        else _CompliantMover(arm, td, geom, moves, cfg.arm_home)
    )
    mover.stop = stop
    held, reading, note, servo_log = _grasp(
        mover,
        geom,
        (ax, ay, yaw_rel),
        read,
        servo,
        clip,
        above,
        clear,
    )
    if servo_log is not None:
        record["pick_attempts"][-1]["lateral_servo"] = servo_log
    record["pick_attempts"][-1].update({"gripper": reading, "held": held})
    if note is not None:
        record["pick_attempts"][-1]["note"] = note
    _logger.info(
        "Pick attempt %d: gripper %.3f -> %s%s.",
        attempt,
        reading,
        "held" if held else "missed",
        f" ({note})" if note else "",
    )
    if not held:
        wrist.stop()
    return held, None


def _wrap(angles: Any) -> Any:
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def _tool_axes(
    geom: TopDownPickGeometry, yaw_rel: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The top-down tool's x and closing (y) axes in the arm frame for a cube yaw
    `yaw_rel` (the closing yaw is that, modulo the cube's 90-degree symmetry)."""
    closing = cube_closing_yaw(yaw_rel, geom)
    c, s = math.cos(closing), math.sin(closing)
    return np.array([-s, c]), np.array([c, s])


def _grasp(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    mover: _Mover,
    geom: TopDownPickGeometry,
    start: tuple[float, float, float],
    read: Any,
    servo: TopDownLateralServo | None,
    name: str,
    above: list[float] | None = None,
    clear: Callable[[Sequence[float]], str | None] | None = None,
) -> tuple[bool, float, str | None, dict[str, Any] | None]:
    """Jimmy Wu's pick_object sequence: open, hover, (align with the wrist camera),
    descend, close, read; then lift and carry, or open and back off.

    With ``above`` (a cube inside the bin) the arm enters and leaves through that
    waypoint over the rim, and ``clear`` (why a pose would touch the bin, or None)
    refuses any servo step or grasp that would.

    Returns (held, gripper reading, note, servo log); the note says why the gripper was
    not closed, if it was not.
    """
    td = mover.td
    ax, ay, yaw = start
    hover, grasp = top_down_joint_targets(
        ax, ay, yaw, geom, seed=above if above is not None else mover.joints()
    )
    servo_log: dict[str, Any] | None = None

    def retreat(to: list[float]) -> None:
        mover.move("retreat", to)
        if above is not None:
            mover.move("above", above)

    mover.begin()
    try:
        mover.gripper(geom.open_position)
        if above is not None and not mover.move("above", above):
            mover.move("above", above)
            return False, mover.arm.get_gripper_state(), "above did not arrive", None
        if not mover.move("hover", hover):
            retreat(hover)
            return False, mover.arm.get_gripper_state(), "hover did not arrive", None
        if servo is not None:

            def move_hover(label: str, x: float, y: float, w: float) -> bool:
                here = mover.joints()
                joints, _ = top_down_joint_targets(x, y, w, geom, seed=here)
                if not _servo_step_is_small(here, joints, td):
                    return False
                if clear is not None and clear(joints) is not None:
                    _logger.warning("Servo step refused: %s.", clear(joints))
                    return False
                return mover.move(f"lateral_{label}", joints)

            def measured_xy() -> tuple[float, float]:
                pos = forward_kinematics(mover.joints(), geom)[0]
                return float(pos[0]), float(pos[1])

            result = servo.align(
                (ax, ay, yaw), move_hover, functools.partial(_tool_axes, geom), name
            )
            servo_log = _servo_log(result)
            ax, ay, yaw = result.ax, result.ay, result.yaw
            if result.aligned and isinstance(mover, _CompliantMover):
                # The servo centred the gripper where it really is; the compliant arm
                # lags its command (run 36: 42 mm), so descend from there.
                ax, ay = (float(v) for v in measured_xy())
                servo_log["descend_from_measured"] = [ax, ay]
            hover, grasp = top_down_joint_targets(
                ax, ay, yaw, geom, seed=mover.joints()
            )
            if not result.aligned:
                retreat(hover)
                return False, mover.arm.get_gripper_state(), result.note, servo_log
        if clear is not None and clear(grasp) is not None:
            retreat(hover)
            return (
                False,
                mover.arm.get_gripper_state(),
                f"grasp refused: {clear(grasp)}",
                servo_log,
            )
        if not mover.move("grasp", grasp, correct=True):
            retreat(hover)
            return (
                False,
                mover.arm.get_gripper_state(),
                "grasp did not arrive",
                servo_log,
            )
        reading = read(mover.gripper(float(td.get("close_position", 1.0))))
        held = grasp_succeeded(reading, geom)
        if held:
            mover.move("lift", hover)
            if above is not None:
                mover.move("above", above)
            mover.move("carry", list(CARRY_ARM_CONF))
        else:
            mover.gripper(0.0)
            retreat(hover)
        return held, reading, None, servo_log
    finally:
        mover.end()


def _servo_step_is_small(
    here: Sequence[float], target: Sequence[float], td: Any
) -> bool:
    """As in the cylinder servo, refuse a nudge whose IK jumps an arm joint by more than
    `max_joint_step` (a branch flip near a singularity); the wrist (joint 7) may turn
    further, up to `max_wrist_step`, to square the fingers to the cube."""
    jump = np.abs(_wrap(np.asarray(target) - np.asarray(here)))
    arm_limit = float(td.get("max_joint_step", 0.2))
    wrist_limit = float(td.get("max_wrist_step", 0.9))
    if jump[:6].max() > arm_limit or jump[6] > wrist_limit:
        _logger.warning(
            "Servo nudge refused: joint change %s rad (limits %.2f arm, %.2f wrist); "
            "likely an IK branch flip.",
            np.round(jump, 3).tolist(),
            arm_limit,
            wrist_limit,
        )
        return False
    return True


def _servo_log(result: TopDownServoResult) -> dict[str, Any]:
    return {
        "aligned": result.aligned,
        "note": result.note,
        "final": [result.ax, result.ay, math.degrees(result.yaw)],
        "looks": [
            {
                "label": t.label,
                "pinch_xy": list(t.pinch_xy),
                "cube_uv": None if t.cube is None else [t.cube.u, t.cube.v],
                "error_px": t.error_px,
                "delta_tool_m": None if t.delta_tool is None else list(t.delta_tool),
            }
            for t in result.trace
        ],
    }


def _build_lateral_servo(
    cfg: DictConfig, source: ImageSource | None, log_dir: Path | None
) -> TopDownLateralServo | None:
    """Tom Silver's align loop from ``pick.top_down.lateral_servo``, or None when it is
    off or there is no wrist camera."""
    td = cfg.pick.top_down
    lateral_cfg = td.get("lateral_servo")
    if lateral_cfg is None or not lateral_cfg.get("enabled", False):
        return None
    if source is None:
        return None
    names = {f.name for f in dataclasses.fields(LateralServoParams)}
    kwargs = {k: lateral_cfg[k] for k in names if k in lateral_cfg}
    for key in ("target_uv", "lateral_gain", "lateral_sign"):
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])
    return TopDownLateralServo(
        _SettledImages(source, float(lateral_cfg.get("settle_s", 0.3))),
        LateralServoParams(**kwargs),
        None if log_dir is None else log_dir / "lateral_servo",
    )


class _SettledImages:
    """Wait `settle_s` after a move before taking the wrist image, so the frame shows
    where the arm stopped, not where it was moving."""

    def __init__(self, source: ImageSource, settle_s: float) -> None:
        self._source = source
        self._settle_s = settle_s

    def get_image(self) -> Any:
        """The newest frame after the settle delay."""
        if self._settle_s > 0:
            time.sleep(self._settle_s)
        return self._source.get_image()


def _fake_reading(value: float, _reading: float) -> float:
    """The fake gripper closes on nothing; report `value` instead."""
    return value


def _set_gripper(
    arm: ArmInterface, position: float, td: Any, hold: list[float] | None = None
) -> None:
    """Command the gripper and wait until it stops moving, holding the arm at `hold`
    (the last command: re-sending the measured joints would let a sagging arm sag
    further) or, without it, where it is."""
    target = list(hold) if hold is not None else arm.get_arm_state()
    arm.execute_action(target, position)
    deadline = time.monotonic() + float(td.get("gripper_timeout_s", 4.0))
    last = arm.get_gripper_state()
    while time.monotonic() < deadline:
        time.sleep(0.15)
        arm.execute_action(target, position)
        now = arm.get_gripper_state()
        if abs(now - last) < 0.005:
            return
        last = now


class _CeilingRecorder:
    """Films the whole run from the ceiling cameras into <log_dir>/ceiling.mp4, stamping
    each frame with the time and the step last asked about.

    Frames are encoded as they arrive (H.264), so a long run costs no memory.
    """

    def __init__(self, cfg: DictConfig, log_dir: Path | None) -> None:
        video_cfg = cfg.get("ceiling_video")
        self.label = "start"
        self._running = False
        if log_dir is None or video_cfg is None or not video_cfg.enabled:
            return

        from prpl_tidybot.marker_detector.ceiling_image_client import (  # pylint: disable=import-outside-toplevel
            CeilingImageClient,
            FakeCeilingImageClient,
        )

        self._client: Any = (
            CeilingImageClient(str(video_cfg.host))
            if cfg.mode == "real"
            else FakeCeilingImageClient()
        )
        self._fps = int(video_cfg.fps)
        self._scale = float(video_cfg.scale)
        self._path = log_dir / "ceiling.mp4"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import imageio_ffmpeg  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        period = 1.0 / self._fps
        while self._running:
            start = time.monotonic()
            try:
                img = self._client.get_image()
            except Exception:  # pylint: disable=broad-except
                break  # the publisher went away; keep what was recorded
            if img is not None:
                frame = cv.resize(img, None, fx=self._scale, fy=self._scale)
                frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
                text = f"{start - self._t0:6.1f}s  {self.label.strip()[:90]}"
                cv.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
                cv.putText(frame, text, (8, 22), 0, 0.6, (255, 255, 255), 1)
                if self._writer is None:
                    self._writer = imageio_ffmpeg.write_frames(
                        str(self._path),
                        (frame.shape[1], frame.shape[0]),
                        fps=self._fps,
                        codec="libx264",
                        pix_fmt_out="yuv420p",
                        output_params=["-movflags", "+faststart"],
                        ffmpeg_log_level="error",
                    )
                    self._writer.send(None)
                self._writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    def close(self) -> None:
        """Stop recording and finish the file."""
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=5)
        if self._writer is not None:
            self._writer.close()
        self._client.close()


class _WristRecorder:
    """Records the wrist camera during each pick attempt (when it has a source)."""

    def __init__(self, source: ImageSource | None, log_dir: Path | None) -> None:
        self._enabled = source is not None and log_dir is not None
        self._log_dir = log_dir
        self._source = source
        self._thread: Any = None
        self._frames: list[Any] = []
        self._name = ""
        self._running = False

    def start(self, name: str) -> None:
        """Begin recording a clip called `name`."""
        if not self._enabled:
            return

        self._frames, self._name, self._running = [], name, True

        def grab() -> None:
            assert self._source is not None
            while self._running:
                img = self._source.get_image()
                if img is not None:
                    self._frames.append(img)
                time.sleep(0.1)

        self._thread = threading.Thread(target=grab, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and write the clip to <log_dir>/<name>_wrist.mp4."""
        if not self._enabled or not self._running:
            return
        self._running = False
        self._thread.join(timeout=2)
        if self._frames and self._log_dir is not None:
            from prpl_tidybot.video import (  # pylint: disable=import-outside-toplevel
                write_mp4,
            )

            write_mp4(self._frames, self._log_dir / f"{self._name}_wrist.mp4", fps=10)

    def close(self) -> None:
        """Stop any clip in progress (the source is closed by its owner)."""
        self.stop()
