"""The cube pick skill's own policy: look, perceive, choose a grasp that fits, grasp.

One policy object runs unchanged against the simulator and the real robot: it reads
only a :class:`~prpl_tidybot.cube_pick.robot.PickRobot` (proprioception and wrist
RGB-D frames) and a believed scene (the robot, the static map and the containers,
with the cube wherever perception put it), and it emits absolute per-tick targets.

The steps (as approved for simulation on 2026-10-15), each failure labelled with the
step it happened in (:data:`STAGES`):

1. ``look`` -- drive the base to a placement facing the expected cube position and
   raise the wrist camera to a viewing pose at least 25 cm from it.
2. ``perceive`` -- estimate the cube's pose from that one RGB-D frame.
3. ``propose`` -- grasp candidates (pose + opening) from the configured generator.
4. ``filter`` -- keep candidates with inverse-kinematics solutions that are
   collision-free at the grasp, along the straight approach and at the standoff, in
   the planning scene; rank near-vertical and well-clear first.
5. ``plan`` -- base and arm paths (KINDER's planners) to the standoff along the
   grasp's approach axis, the gripper pre-opened to the chosen width.
6. ``approach`` -- straight, open-loop, to the grasp computed in step 2; stop early if
   the arm stops short (unexpected contact).
7. ``close`` -- close, and check the fingers stop near the cube's width.
8. ``lift`` -- lift a few centimetres and check the fingers still hold that width.
9. ``retract`` -- clear any container and go to the carry pose.

A failed attempt backs the arm out (open, retreat along the approach, rise) and the
pick retries -- with the next candidate after a planning failure, from a fresh look
after anything that may have disturbed the cube -- up to `max_attempts` times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_pick.camera import RGBDFrame
from prpl_tidybot.cube_pick.grasps import (
    AnalyticGraspGenerator,
    CubeGraspSampler,
    GraspCandidate,
    GraspGenerator,
    nearest_equivalent_yaw,
    upright_cube_frame,
)
from prpl_tidybot.cube_pick.perception import (
    CubeEstimate,
    CubePoseEstimator,
    CubePrior,
    PerceptionFailure,
)
from prpl_tidybot.cube_pick.planning import ContainerSpec, PlanningScene
from prpl_tidybot.cube_pick.robot import (
    GRIPPER_CLOSE,
    TOOL_FROM_CAMERA,
    PickRobot,
    RobotCommand,
)

_logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

STAGES = (
    "look",
    "perceive",
    "propose",
    "filter",
    "plan",
    "approach",
    "close",
    "lift",
    "retract",
)

# KINDER's retract configuration, the pose the old pick lifts to.
HOME_ARM = tuple(np.deg2rad([0, -20, 180, -146, 0, -50, 90]).tolist())


class PickFailed(RuntimeError):
    """The pick gave up; `stage` is one of :data:`STAGES`."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass
class PickConfig:
    """Knobs of the pick policy."""

    cube_size: float = 0.05
    observe_heights: tuple[float, ...] = (0.36, 0.30, 0.42)
    base_distances: tuple[float, ...] = (0.55, 0.50, 0.62)
    base_headings: int = 12
    max_base_poses: int = 4
    standoff: float = 0.12
    obstacle_clearance: float = 0.004
    approach_clearance: float = 0.001
    grasp_pool: int = 8
    comfortable_margin: float = 0.01
    tilt_penalty_per_rad: float = 0.01
    max_attempts: int = 3
    max_looks_per_base: int = 4
    max_filter_bases: int = 10
    lift_check_height: float = 0.05
    rim_clearance: float = 0.04
    close_if_short: float = 0.025
    lift_height: float = 0.25
    min_camera_range: float = 0.20
    closed_gap_fraction: float = 0.5
    base_tolerance: float = 0.02
    base_rot_tolerance: float = 0.03
    arm_waypoint_tolerance: float = 0.04
    arm_final_tolerance: float = 0.01
    fine_tolerance: float = 0.006
    pass_tolerance: float = 0.015
    fine_position_tolerance: float = 0.0015
    fine_rotation_tolerance: float = np.radians(1.5)
    motion_timeout: int = 250
    gripper_ticks: int = 15
    generator: GraspGenerator | None = None

    def grasp_generator(self) -> GraspGenerator:
        """The grasp generator (default: the analytic sampler for this cube size)."""
        return self.generator or AnalyticGraspGenerator(
            CubeGraspSampler(cube_size=self.cube_size)
        )


@dataclass
class PickEvent:
    """One entry in the pick's log."""

    tick: int
    stage: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PickResult:
    """How the pick ended (by the policy's own judgement, not ground truth)."""

    success: bool
    stage: str | None
    detail: str
    ticks: int
    events: list[PickEvent]
    estimates: list[CubeEstimate]
    frames: list[tuple[RGBDFrame, CubeEstimate | PerceptionFailure]]
    grasp: GraspCandidate | None


class CubePickPolicy:
    """Look-perceive-plan-grasp policy for one known-size cube."""

    def __init__(
        self,
        *,
        robot: PickRobot,
        estimator: CubePoseEstimator,
        believed_state: Any,
        cube_name: str,
        prior: CubePrior,
        containers: Sequence[ContainerSpec] = (),
        config: PickConfig | None = None,
        extra_obstacle_names: Sequence[str] = (),
        keep_frames: bool = True,
    ) -> None:
        self.robot = robot
        self.estimator = estimator
        self.config = config or PickConfig(cube_size=prior.size)
        self.prior = prior
        self.scene = PlanningScene(
            believed_state,
            cube_name=cube_name,
            containers=containers,
            extra_obstacle_names=extra_obstacle_names,
        )
        self.scene.set_cube_estimate(upright_cube_frame(prior.position, 0.0))
        self.containers = list(containers)
        self.events: list[PickEvent] = []
        self.estimates: list[CubeEstimate] = []
        self.frames: list[tuple[RGBDFrame, CubeEstimate | PerceptionFailure]] = []
        self.keep_frames = keep_frames
        self.grasp: GraspCandidate | None = None
        self._ticks = 0
        self._last_frame: RGBDFrame | None = None
        self._grasp_yaw = 0.0
        self._standoff_joints: list[float] = []

    # --- driving ---------------------------------------------------------------------------

    def run(self, max_ticks: int = 2000) -> PickResult:
        """Run the pick to completion against the robot."""
        try:
            self._run(max_ticks)
            return self._result(True, None, "holding")
        except PickFailed as failure:
            self._log(failure.stage, f"FAILED: {failure.detail}")
            return self._result(False, failure.stage, failure.detail)
        finally:
            self.scene.close()

    def _result(self, success: bool, stage: str | None, detail: str) -> PickResult:
        return PickResult(
            success=success,
            stage=stage,
            detail=detail,
            ticks=self._ticks,
            events=self.events,
            estimates=self.estimates,
            frames=self.frames,
            grasp=self.grasp,
        )

    def _run(self, max_ticks: int) -> None:
        self._budget = max_ticks
        cfg = self.config
        failure: PickFailed | None = None
        estimate: CubeEstimate | None = None
        excluded: set[str] = set()
        for attempt in range(cfg.max_attempts):
            self._log("look", f"attempt {attempt + 1} of {cfg.max_attempts}")
            try:
                if estimate is None:
                    estimate = self._look_and_choose(excluded)
                else:
                    self._choose_grasp(estimate, excluded)
                assert self.grasp is not None
                excluded.add(self.grasp.label)
                self._move_arm_planned(
                    self._standoff_joints, opening=self.grasp.opening, stage="plan"
                )
                self._approach(estimate)
                self._close_and_check()
                self._lift_and_verify()
                self._retract()
                return
            except PickFailed as f:
                failure = f
                self._log(f.stage, f"attempt {attempt + 1} failed: {f.detail}")
                if self._ticks >= self._budget:
                    break
                self._back_out()
                # Anything that may have touched the cube calls for a fresh look; a
                # plan that merely did not work out moves on to the next candidate.
                if f.stage in (
                    "approach",
                    "close",
                    "lift",
                    "retract",
                    "look",
                    "perceive",
                ):
                    estimate = None
                    excluded = set()
                elif f.stage in ("propose", "filter"):
                    estimate = None
        assert failure is not None
        raise failure

    def _look_and_choose(self, excluded: set[str]) -> CubeEstimate:
        """Steps 1-4.

        Look from the first base placement that gives an estimate; then evaluate grasps
        at candidate base placements *in the planning scene* (no driving), and drive
        only to the first placement where one fits.
        """
        cfg = self.config
        bases = self._base_pose_candidates()
        if not bases:
            raise PickFailed("look", "no collision-free base placement to look from")
        estimate: CubeEstimate | None = None
        last: PickFailed = PickFailed("perceive", "no view gave an estimate")
        for base in bases[: cfg.max_base_poses]:
            try:
                self._go_to_base(base)
                estimate = self._observe_from_above()
                break
            except PickFailed as failure:
                last = failure
                self._log(
                    failure.stage,
                    f"no estimate from base {np.round(base, 3)}: {failure.detail}",
                )
        if estimate is None:
            raise last
        self.prior = CubePrior(
            position=estimate.position,
            size=self.prior.size,
            sigma=0.02,
            rgba=self.prior.rgba,
            support_heights=self.prior.support_heights,
        )
        here = self.robot.proprio()
        options = [here.base] + [
            b for b in bases if not self._base_close(b, here.base, 0.03, 0.05)
        ]
        self._hand_blocked: set[tuple[Any, ...]] = set()
        last = PickFailed("filter", "no grasp fits from any base placement")
        for base in options[: cfg.max_filter_bases]:
            self.scene.set_robot(base, here.arm)
            try:
                self._choose_grasp(estimate, excluded, base_heading=base[2])
            except PickFailed as failure:
                last = failure
                self._log(
                    "filter",
                    f"no grasp from base {np.round(base, 3)}: {failure.detail}",
                )
                if getattr(failure, "hand_blocked", False):
                    break  # the hand itself does not fit: no base placement can help
                continue
            if base is not here.base:
                self.scene.set_robot(here.base, here.arm)
                self._go_to_base(base)
                # Re-check at the placement actually reached.
                self._choose_grasp(estimate, excluded, base_heading=base[2])
            return estimate
        self.scene.set_robot(here.base, here.arm)
        raise last

    # --- stage helpers ---------------------------------------------------------------------

    def _base_pose_candidates(self) -> list[tuple[float, float, float]]:
        cfg = self.config
        proprio = self.robot.proprio()
        cx, cy = self.prior.position[:2]
        bx, by, _ = proprio.base
        facing = np.arctan2(cy - by, cx - bx)
        out = []
        for distance in cfg.base_distances:
            for k in range(cfg.base_headings):
                offset = (
                    (k + 1) // 2 * (1 if k % 2 else -1) * 2 * np.pi / cfg.base_headings
                )
                heading = facing + offset
                pose = (
                    cx - distance * np.cos(heading),
                    cy - distance * np.sin(heading),
                    heading,
                )
                out.append((abs(offset), distance, pose))
        out.sort(key=lambda item: (item[0], abs(item[1] - cfg.base_distances[0])))
        poses = []
        for _, _, pose in out:
            if not self.scene.base_pose_collides(pose):
                poses.append(pose)
        # The base the robot already stands at comes first if it is a candidate.
        return poses

    def _go_to_base(self, target: tuple[float, float, float]) -> None:
        proprio = self.robot.proprio()
        self.scene.set_robot(proprio.base, proprio.arm)
        if (
            np.hypot(target[0] - proprio.base[0], target[1] - proprio.base[1]) < 0.01
            and abs(_wrap_angle(target[2] - proprio.base[2])) < 0.02
        ):
            return
        # Tuck the arm before driving: KINDER's base planner checks only the base's
        # footprint, so an arm left reaching out sweeps whatever the base passes.
        if _joint_error(proprio.arm, HOME_ARM) > self.config.arm_waypoint_tolerance:
            tuck = self.scene.plan_arm(
                proprio.arm, HOME_ARM, opening=0.085, include_cube=True
            )
            if tuck is None:
                self._log("look", "no planned path to tuck the arm; driving as it is")
            else:
                self._follow_arm(
                    tuck,
                    opening=None,
                    tolerance=self.config.arm_waypoint_tolerance,
                    stage="look",
                )
            proprio = self.robot.proprio()
            self.scene.set_robot(proprio.base, proprio.arm)
        path = self.scene.plan_base(target)
        if path is None:
            raise PickFailed("look", f"no base path to {np.round(target, 3)}")
        self._log(
            "plan", f"base path of {len(path)} waypoints to {np.round(target, 3)}"
        )
        waypoints = [(p.x, p.y, p.theta()) for p in path]
        cfg = self.config
        ticks = 0
        index = 0
        while True:
            proprio = self.robot.proprio()
            while index < len(waypoints) - 1 and self._base_close(
                proprio.base, waypoints[index], 0.04
            ):
                index += 1
            if index == len(waypoints) - 1 and self._base_close(
                proprio.base, waypoints[-1], cfg.base_tolerance, cfg.base_rot_tolerance
            ):
                break
            self._tick(RobotCommand(base=waypoints[index], arm=proprio.arm))
            ticks += 1
            if ticks > 4 * cfg.motion_timeout:
                raise PickFailed("look", "base did not reach its target")
        proprio = self.robot.proprio()
        self.scene.set_robot(proprio.base, proprio.arm)

    def _observe_poses(self) -> list[FloatArray]:
        """Tool poses that put the camera above the prior looking at it: straight down
        first, then tilted 25 degrees from four sides (for when the arm cannot reach
        overhead or a container wall is in the way)."""
        cfg = self.config
        base_heading = self.robot.proprio().base[2]
        poses = []
        views = [(0.0, 0.0)] + [
            (np.radians(25), a) for a in (np.pi, 0.0, np.pi / 2, -np.pi / 2)
        ]
        for tilt, azimuth in views:
            for height in cfg.observe_heights:
                for yaw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
                    # Camera position on a sphere about the prior, looking at it.
                    direction = np.array(
                        [
                            np.sin(tilt) * np.cos(base_heading + azimuth),
                            np.sin(tilt) * np.sin(base_heading + azimuth),
                            np.cos(tilt),
                        ]
                    )
                    z_axis = -direction
                    heading = base_heading + yaw
                    x_axis = np.array([np.cos(heading), np.sin(heading), 0.0])
                    x_axis -= (x_axis @ z_axis) * z_axis
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    camera = np.eye(4)
                    camera[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
                    camera[:3, 3] = self.prior.position + direction * height
                    poses.append(camera @ np.linalg.inv(TOOL_FROM_CAMERA))
        return poses

    def _observe_from_above(self) -> CubeEstimate:
        cfg = self.config
        proprio = self.robot.proprio()
        failures = []
        looked = 0
        for tool_pose in self._observe_poses():
            joints = self.scene.ik(tool_pose, seed=proprio.arm)
            if joints is None:
                failures.append("ik")
                continue
            if not self.scene.check(
                joints,
                opening=0.085,
                clearance=cfg.obstacle_clearance,
                include_cube=True,
                cube_clearance=0.01,
            ).ok:
                failures.append("collision")
                continue
            try:
                self._move_arm_planned(
                    joints, opening=0.085, stage="look", include_cube=True
                )
            except PickFailed as failure:
                failures.append(failure.detail)
                continue
            estimate = self._perceive("perceive")
            if isinstance(estimate, CubeEstimate):
                return estimate
            failures.append(estimate.reason)
            looked += 1
            if looked >= cfg.max_looks_per_base:
                break
        raise PickFailed(
            "perceive", f"no estimate from above ({'; '.join(failures[:6])})"
        )

    def _perceive(
        self, stage: str, prior: CubePrior | None = None
    ) -> CubeEstimate | PerceptionFailure:
        # Let the arm settle so the frame is not motion-blurred in the real camera.
        for _ in range(3):
            self._tick(RobotCommand())
        frame = self.robot.wrist_frame()
        result = self.estimator.estimate(frame, prior or self.prior)
        self._last_frame = frame
        if self.keep_frames:
            self.frames.append((frame, result))
        if isinstance(result, CubeEstimate):
            self.estimates.append(result)
            self.scene.set_cube_estimate(_upright(result))
            self._log(
                stage,
                f"estimate {np.round(result.position, 4)} yaw {np.degrees(result.yaw):.1f}",
                {
                    "position": result.position.tolist(),
                    "yaw": result.yaw,
                    "quality": result.quality,
                },
            )
        else:
            self._log(stage, f"perception failed: {result.reason}")
        return result

    def _choose_grasp(
        self,
        estimate: CubeEstimate,
        excluded: set[str] | None = None,
        base_heading: float | None = None,
    ) -> tuple[GraspCandidate, list[float], list[float]]:
        cfg = self.config
        proprio = self.robot.proprio()
        frame = self._grasp_frame(estimate, base_heading)
        blocked: set[tuple[Any, ...]] = getattr(self, "_hand_blocked", set())
        hand_only = True
        counts: dict[str, int] = {}
        last_frame = self._last_frame
        mask = estimate.mask
        candidates = cfg.grasp_generator().candidates(
            cube_frame=frame, frame=last_frame, mask=mask
        )
        if not candidates:
            raise PickFailed(
                "propose", f"{cfg.grasp_generator().name} proposed no grasps"
            )
        feasible: list[tuple[float, int, GraspCandidate, list[float], list[float]]] = []
        for rank, candidate in enumerate(candidates):
            if excluded and candidate.label in excluded:
                continue
            key = _grasp_key(candidate.tool_pose(frame), candidate.opening)
            if key in blocked:
                counts["hand blocked (cached)"] = (
                    counts.get("hand blocked (cached)", 0) + 1
                )
                continue
            verdict = self._check_candidate(candidate, frame, proprio.arm)
            if isinstance(verdict, str):
                counts[verdict] = counts.get(verdict, 0) + 1
                if _is_hand_collision(verdict):
                    blocked.add(key)
                else:
                    hand_only = False
                continue
            hand_only = False
            grasp_joints, standoff_joints, margin = verdict
            feasible.append((margin, rank, candidate, grasp_joints, standoff_joints))
            # Among the first few that fit, prefer near-vertical and well clear: an
            # open-loop approach has only the perception error as its margin.
            if len(feasible) >= cfg.grasp_pool or (
                margin >= cfg.comfortable_margin and candidate.tilt == 0
            ):
                break
        if not feasible:
            failure = PickFailed("filter", f"every candidate rejected: {counts}")
            failure.hand_blocked = hand_only  # type: ignore[attr-defined]
            raise failure
        margin, _, candidate, grasp_joints, standoff_joints = max(
            feasible,
            key=lambda f: (
                min(f[0], cfg.comfortable_margin)
                - cfg.tilt_penalty_per_rad * abs(f[2].tilt),
                -f[1],
            ),
        )
        self._standoff_joints = standoff_joints
        self.grasp = candidate
        self._grasp_yaw = float(np.arctan2(frame[1, 0], frame[0, 0]))
        self._log(
            "filter",
            f"grasp {candidate.label} (margin {margin * 1000:.1f} mm, {len(feasible)} fit)",
            {"rejections": counts},
        )
        return candidate, grasp_joints, standoff_joints

    def _standoff_distance(self, grasp_pose: FloatArray) -> float:
        """How far back along the approach the standoff sits: `standoff`, or far enough
        that the standoff is above the rim of a container around the grasp, so the
        coarse joint-space path never enters the container and everything inside it is
        the straight, finely tracked approach."""
        cfg = self.config
        distance = cfg.standoff
        xy = grasp_pose[:2, 3]
        approach = grasp_pose[:3, 2]
        for spec in self.containers:
            local = np.linalg.inv(spec.pose) @ np.append(grasp_pose[:3, 3], 1.0)
            if (
                abs(local[0]) > spec.length / 2 + 0.15
                or abs(local[1]) > spec.width / 2 + 0.15
            ):
                continue
            rim = float((spec.pose @ np.array([0.0, 0.0, spec.height, 1.0]))[2])
            rise = rim + cfg.rim_clearance - grasp_pose[2, 3]
            down = max(-approach[2], 0.5)
            distance = max(distance, rise / down)
        del xy
        return float(distance)

    def _grasp_frame(
        self, estimate: CubeEstimate, reference_yaw: float | None = None
    ) -> FloatArray:
        """The cube's upright frame, its yaw the symmetric equivalent nearest
        `reference_yaw` (default: the base heading).

        Re-targeting passes the yaw the grasp was chosen in, so a refined estimate that
        crosses the +-45 degree wrap does not rotate the grasp by a quarter turn.
        """
        reference = (
            self.robot.proprio().base[2] if reference_yaw is None else reference_yaw
        )
        yaw = nearest_equivalent_yaw(estimate.yaw, reference)
        return upright_cube_frame(estimate.position, yaw)

    def _check_candidate(
        self,
        candidate: GraspCandidate,
        frame: FloatArray,
        seed: Sequence[float],
        approach_clearance: float | None = None,
    ) -> tuple[list[float], list[float], float] | str:
        """Joints at the grasp and the standoff, and the worst clearance margin along
        the approach (metres beyond the required clearance), or why it does not fit."""
        cfg = self.config
        scene = self.scene
        clearance = (
            cfg.approach_clearance if approach_clearance is None else approach_clearance
        )
        margin = np.inf
        grasp_pose = candidate.tool_pose(frame)
        standoff = self._standoff_distance(grasp_pose)
        standoff_pose = candidate.standoff_pose(frame, standoff)
        # The camera must be able to see the cube from the standoff.
        # The hand alone first, at the grasp and along the approach: cheap, and
        # independent of the base placement and the arm.
        report = scene.hand_check(
            grasp_pose,
            opening=candidate.opening,
            clearance=clearance,
            include_cube=True,
        )
        if not report.ok:
            return f"grasp hits {report.culprit}"
        for t in (0.25, 0.5, 0.75, 1.0):
            pose = grasp_pose.copy()
            pose[:3, 3] = grasp_pose[:3, 3] - grasp_pose[:3, 2] * standoff * t
            report = scene.hand_check(
                pose, opening=candidate.opening, clearance=clearance, include_cube=True
            )
            if not report.ok:
                return f"approach hits {report.culprit}"
        standoff_joints = scene.ik(standoff_pose, seed=seed)
        if standoff_joints is None:
            return "ik standoff"
        grasp_joints = scene.ik(grasp_pose, seed=standoff_joints)
        if grasp_joints is None:
            return "ik grasp"
        report = scene.check(
            grasp_joints,
            opening=candidate.opening,
            clearance=clearance,
            include_cube=True,
            cube_clearance=0.0,
        )
        if not report.ok:
            return f"grasp hits {report.culprit}"
        margin = min(margin, report.closest)
        report = scene.check(
            standoff_joints,
            opening=candidate.opening,
            clearance=cfg.obstacle_clearance,
            include_cube=True,
            cube_clearance=0.005,
        )
        if not report.ok:
            return f"standoff hits {report.culprit}"
        path = scene.plan_cartesian(standoff_joints, grasp_pose, step=0.02)
        if path is None:
            return "approach ik"
        for joints in path[1:-1]:
            report = scene.check(
                joints,
                opening=candidate.opening,
                clearance=clearance,
                include_cube=True,
                cube_clearance=0.0,
            )
            if not report.ok:
                return f"approach hits {report.culprit}"
            margin = min(margin, report.closest)
        return grasp_joints, standoff_joints, float(margin)

    def _approach(self, estimate: CubeEstimate) -> None:
        """Straight, open-loop, from the standoff to the grasp perceived in step 2."""
        assert self.grasp is not None
        target = self.grasp.tool_pose(self._grasp_frame(estimate, self._grasp_yaw))
        proprio = self.robot.proprio()
        path = self.scene.plan_cartesian(list(proprio.arm), target, step=0.01)
        if path is None:
            raise PickFailed("approach", "no straight approach from the standoff")
        try:
            self._follow_arm(
                path,
                opening=self.grasp.opening,
                tolerance=self.config.fine_tolerance,
                stage="approach",
            )
        except PickFailed as failure:
            # Stopped early. If the pads are already around the cube (on the approach
            # axis, a little short), close and let step 7 judge; otherwise give up.
            here = self.scene.tool_pose(self.robot.proprio().arm)
            offset = target[:3, 3] - here[:3, 3]
            along = float(offset @ target[:3, 2])
            lateral = float(np.linalg.norm(offset - along * target[:3, 2]))
            if 0.0 <= along <= self.config.close_if_short and lateral <= 0.004:
                self._log(
                    "approach",
                    f"stopped {along * 1000:.0f} mm short, {lateral * 1000:.1f} mm off axis; closing",
                )
                return
            raise PickFailed(
                "approach",
                f"stopped short (contact?) {along * 1000:.0f} mm before the grasp, {lateral * 1000:.1f} mm off axis: {failure.detail}",
            ) from failure

    def _close_and_check(self) -> None:
        cfg = self.config
        hold = list(self.robot.proprio().arm)
        last = None
        for tick in range(cfg.gripper_ticks * 3):
            self._tick(RobotCommand(arm=hold, gripper=GRIPPER_CLOSE))
            gap = self.robot.proprio().gripper_opening
            if (
                tick >= cfg.gripper_ticks
                and last is not None
                and abs(gap - last) < 2e-4
            ):
                break
            last = gap
        gap = self.robot.proprio().gripper_opening
        if gap < cfg.cube_size * cfg.closed_gap_fraction:
            raise PickFailed(
                "close", f"fingers closed to {gap * 1000:.1f} mm: missed the cube"
            )
        if gap > cfg.cube_size + 0.01:
            raise PickFailed(
                "close",
                f"fingers stopped at {gap * 1000:.1f} mm: blocked before the cube",
            )
        self._log("close", f"closed on the cube (gap {gap * 1000:.1f} mm)")

    def _lift_and_verify(self) -> None:
        cfg = self.config
        before = self.robot.proprio().gripper_opening
        self._move_tool_straight(
            np.array([0.0, 0.0, cfg.lift_check_height]), stage="lift", close=True
        )
        for _ in range(5):
            self._tick(RobotCommand(gripper=GRIPPER_CLOSE))
        gap = self.robot.proprio().gripper_opening
        if gap < cfg.cube_size * cfg.closed_gap_fraction or abs(gap - before) > 0.006:
            raise PickFailed(
                "lift",
                f"hold not verified after the lift: gap {before * 1000:.1f} -> {gap * 1000:.1f} mm",
            )
        self._log("lift", f"hold verified (gap {gap * 1000:.1f} mm)")

    def _retract(self) -> None:
        """Clear any container (straight up), then the carry pose."""
        cfg = self.config
        try:
            self._move_tool_straight(
                np.array([0.0, 0.0, cfg.lift_height - cfg.lift_check_height]),
                stage="retract",
                close=True,
            )
        except PickFailed:
            self._log(
                "retract",
                "full straight lift unavailable; planning to the carry pose from here",
            )
        proprio = self.robot.proprio()
        plan = self.scene.plan_arm(
            proprio.arm, HOME_ARM, opening=cfg.cube_size, include_cube=False
        )
        if plan is not None:
            self._follow_arm(
                plan,
                opening=None,
                tolerance=cfg.arm_waypoint_tolerance,
                stage="retract",
                close=True,
            )
        else:
            self._log(
                "retract", "no path to the carry pose; holding the cube where it is"
            )
        gap = self.robot.proprio().gripper_opening
        if gap < cfg.cube_size * cfg.closed_gap_fraction:
            raise PickFailed(
                "retract", f"cube lost while retracting (gap {gap * 1000:.1f} mm)"
            )

    def _move_tool_straight(
        self, delta: FloatArray, *, stage: str, close: bool = False
    ) -> None:
        proprio = self.robot.proprio()
        here = self.scene.tool_pose(proprio.arm)
        target = here.copy()
        target[:3, 3] += delta
        path = self.scene.plan_cartesian(list(proprio.arm), target, step=0.02)
        if path is None:
            raise PickFailed(stage, f"no straight move by {np.round(delta, 3)}")
        opening = None if close else (self.grasp.opening if self.grasp else None)
        self._follow_arm(
            path,
            opening=opening,
            tolerance=self.config.arm_waypoint_tolerance,
            stage=stage,
            close=close,
        )

    def _back_out(self) -> None:
        """After a failure: open, retreat along the approach axis, rise.

        Never leaves the arm stranded in a container; best effort, never raises.
        """
        cfg = self.config
        try:
            opening = self.grasp.opening if self.grasp is not None else 0.085
            proprio = self.robot.proprio()
            for _ in range(cfg.gripper_ticks):
                self._tick(RobotCommand(arm=proprio.arm, gripper=opening))
            here = self.scene.tool_pose(proprio.arm)
            for distance in (
                self._standoff_distance(here),
                cfg.standoff,
                cfg.standoff / 2,
            ):
                back = here.copy()
                back[:3, 3] -= here[:3, 2] * distance
                path = self.scene.plan_cartesian(list(proprio.arm), back, step=0.02)
                if path is not None:
                    self._follow_arm(
                        path,
                        opening=opening,
                        tolerance=cfg.arm_waypoint_tolerance,
                        stage="retract",
                    )
                    break
            self._move_tool_straight(np.array([0.0, 0.0, 0.10]), stage="retract")
        except PickFailed as failure:
            self._log("retract", f"back-out incomplete: {failure.detail}")

    # --- motion primitives -----------------------------------------------------------------

    def _move_arm_planned(
        self,
        goal: Sequence[float],
        *,
        opening: float,
        stage: str,
        include_cube: bool = True,
    ) -> None:
        proprio = self.robot.proprio()
        self.scene.set_robot(proprio.base, proprio.arm)
        # Open to the grasp width first, so the planned clearance is the one executed.
        for _ in range(self.config.gripper_ticks):
            self._tick(RobotCommand(arm=proprio.arm, gripper=opening))
        plan = self.scene.plan_arm(
            proprio.arm, goal, opening=opening, include_cube=include_cube
        )
        if plan is None:
            raise PickFailed(stage, "no arm path")
        self._follow_arm(
            plan,
            opening=opening,
            tolerance=self.config.arm_waypoint_tolerance,
            stage=stage,
        )

    def _follow_arm(
        self,
        path: Sequence[Sequence[float]],
        *,
        opening: float | None,
        tolerance: float,
        stage: str,
        close: bool = False,
    ) -> None:
        cfg = self.config
        gripper: float | str | None = GRIPPER_CLOSE if close else opening
        index = 0
        ticks = 0
        settle = 0
        still = 0
        previous: tuple[float, ...] | None = None
        while True:
            proprio = self.robot.proprio()
            # Pass intermediate waypoints at a looser tolerance (the PD loop never quite
            # settles on each); hold the final one to `tolerance`.
            passing = max(tolerance, cfg.pass_tolerance)
            while index < len(path) - 1 and self._reached(
                proprio.arm, path[index], passing
            ):
                index += 1
            error = _joint_error(proprio.arm, path[index])
            if index == len(path) - 1 and self._reached(
                proprio.arm, path[-1], min(tolerance, cfg.arm_final_tolerance)
            ):
                settle += 1
                if settle >= 2:
                    break
            if previous is not None and _joint_error(proprio.arm, previous) < 3e-4:
                still += 1
            else:
                still = 0
            previous = proprio.arm
            if still >= 12:
                # The arm has stopped short: steady-state tracking error, or contact.
                if index == len(path) - 1 and (
                    error < 3 * tolerance
                    or self._reached(proprio.arm, path[-1], tolerance, scale=2.0)
                ):
                    break
                raise PickFailed(
                    stage,
                    f"arm stopped at waypoint {index}/{len(path) - 1}, {error:.3f} rad short",
                )
            self._tick(RobotCommand(arm=path[index], gripper=gripper))
            ticks += 1
            if ticks > cfg.motion_timeout + 10 * len(path):
                if index == len(path) - 1 and (
                    error < 3 * tolerance
                    or self._reached(proprio.arm, path[-1], tolerance, scale=2.0)
                ):
                    break
                raise PickFailed(
                    stage,
                    f"arm timed out at waypoint {index}/{len(path) - 1}, {error:.3f} rad short",
                )
        proprio = self.robot.proprio()
        self.scene.set_robot(proprio.base, proprio.arm)

    def _reached(
        self,
        arm: Sequence[float],
        target: Sequence[float],
        tolerance: float,
        scale: float = 1.0,
    ) -> bool:
        """Joint-space closeness, or -- for the redundant 7-DOF arm, whose PD loop
        leaves a residual mostly in the null space -- tool-space closeness: within
        `fine_position_tolerance` and `fine_rotation_tolerance` (times `scale`)."""
        if _joint_error(arm, target) < tolerance:
            return True
        cfg = self.config
        if tolerance > cfg.arm_waypoint_tolerance:
            return False
        here = self.scene.tool_pose(arm)
        there = self.scene.tool_pose(target)
        position = float(np.linalg.norm(here[:3, 3] - there[:3, 3]))
        cos = (np.trace(here[:3, :3].T @ there[:3, :3]) - 1) / 2
        angle = float(np.arccos(np.clip(cos, -1.0, 1.0)))
        return (
            position < cfg.fine_position_tolerance * scale
            and angle < cfg.fine_rotation_tolerance * scale
        )

    def _tick(self, cmd: RobotCommand) -> None:
        if self._ticks >= self._budget:
            raise PickFailed(self._current_stage(), "tick budget exhausted")
        self.robot.command(cmd)
        self._ticks += 1

    def _current_stage(self) -> str:
        return self.events[-1].stage if self.events else "look"

    @staticmethod
    def _base_close(
        base: Sequence[float],
        target: Sequence[float],
        tol: float,
        rot_tol: float | None = None,
    ) -> bool:
        return bool(
            abs(base[0] - target[0]) < tol
            and abs(base[1] - target[1]) < tol
            and abs(_wrap_angle(base[2] - target[2]))
            < (rot_tol if rot_tol is not None else tol)
        )

    def _log(
        self, stage: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        self.events.append(PickEvent(self._ticks, stage, message, data or {}))
        _logger.info("[%s t=%d] %s", stage, self._ticks, message)


_HAND_LINKS = ("robotiq", "finger", "knuckle", "pad", "bracelet")


def _is_hand_collision(verdict: str) -> bool:
    """Whether a rejection is the hand itself colliding at the grasp or on the
    approach -- which no other base placement or arm configuration can change."""
    if not verdict.startswith(("grasp hits", "approach hits")) or "(" not in verdict:
        return False
    link = verdict.rsplit("(", 1)[1]
    return any(name in link for name in _HAND_LINKS)


def _grasp_key(tool_pose: FloatArray, opening: float) -> tuple:
    return (*np.round(tool_pose[:3, :].ravel(), 3).tolist(), round(opening, 4))


def _upright(estimate: CubeEstimate) -> FloatArray:
    return upright_cube_frame(estimate.position, estimate.yaw)


def _joint_error(a: Sequence[float], b: Sequence[float]) -> float:
    diff = np.asarray(a[:7], dtype=float) - np.asarray(b[:7], dtype=float)
    for i in (0, 2, 4, 6):
        diff[i] = (diff[i] + np.pi) % (2 * np.pi) - np.pi
    return float(np.max(np.abs(diff)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
