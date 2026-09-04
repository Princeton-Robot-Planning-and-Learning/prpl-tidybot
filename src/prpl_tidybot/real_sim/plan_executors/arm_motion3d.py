"""Plan executors for kinematic3d arm + gripper trajectories.

An arm-only trajectory is a sequence of (state, action) pairs whose
kinder 11-d action holds the base delta at zero, encodes arm joint
deltas in ``action[3:10]``, and a gripper command in ``action[10]``.
The mapping from kinder gripper command to TidyBotAction gripper_goal
is: ``cmd < -0.5`` → close (``gripper_close_position``, 1.0 by default),
``cmd > 0.5`` → open (0.0), otherwise hold the perceived finger state.
A close position below 1.0 stops the fingers short of fully closed so a
compliant object (a Pringles can) is held rather than crushed.

Currently one concrete subclass is exposed:

* :class:`StreamingArmMotion3DPlanExecutor` — discrete-waypoint
  crossover advance. Each pair's intended absolute joint target
  ``state.joints + action.arm_delta`` is precomputed; per tick, the
  cursor advances through any waypoints already within
  ``advance_radius`` of the perceived joints, and the OTG target is
  set to the waypoint at the new cursor. By re-targeting *before* the
  OTG has decelerated to zero at the current waypoint, Ruckig's
  mid-flight replan keeps the arm in cruise — the planned via-points
  are visited within tolerance without per-waypoint accel/decel
  cycles. See the class docstring for full details.

The abstract base :class:`ArmMotion3DPlanExecutor` owns the trajectory
storage and the arm-only validation (rejects any pair with nontrivial
base motion at :meth:`set_trajectory` time). Mixed base + arm/gripper
trajectories are the dispatcher's concern
(:class:`Kinematic3DPlanExecutor`), which segments mixed trajectories
and feeds each homogeneous segment to the appropriate sub-executor.
"""

from __future__ import annotations

import abc
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from kinder_models.structs import SkillCall
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState
from spatialmath import SE2

from prpl_tidybot.real_sim.plan_executors.failures import ExecutionFailure
from prpl_tidybot.structs import TidyBotAction
from prpl_tidybot.visual_servo.cylinder_edges import detect_cylinder_edges

_logger = logging.getLogger(__name__)

JointPositions = list[float]
# Matches pybullet-helpers' create_joint_distance_fn return type. list (not
# Sequence) in the argument slot so that callable is assignable here without
# contravariance complaints; Sequence-typed callers still match via the
# contravariant direction.
JointDistanceFn = Callable[[JointPositions, JointPositions], float]

_BASE_MOTION_EPS = 1e-4
# Perceived-joint motion (distance_fn metric) below which a tick counts as still.
_STILL_EPS = 5e-3


class ArmMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TidyBotAction, ObjectCentricState], abc.ABC
):
    """Abstract base for kinematic3d arm + gripper plan executors.

    Owns trajectory storage and the arm-only validation. Subclasses
    implement :meth:`_on_set_trajectory`, :meth:`step`, and :meth:`done`
    to provide the tracking strategy.
    """

    def __init__(self, robot_name: str = "robot") -> None:
        self._robot_name = robot_name
        self._pairs: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        for _, action in trajectory:
            _validate_arm_only(action)
        self._pairs = list(trajectory)
        self._on_set_trajectory()

    @abc.abstractmethod
    def _on_set_trajectory(self) -> None:
        """Reset strategy-specific state at the start of a new trajectory."""

    @abc.abstractmethod
    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]: ...

    @abc.abstractmethod
    def done(self, sim_state: ObjectCentricState) -> bool: ...


class StreamingArmMotion3DPlanExecutor(ArmMotion3DPlanExecutor):
    """Discrete-waypoint crossover advance for arm + gripper trajectories.

    Each (state, action) pair's intended absolute joint target is
    ``state.joints + action.arm_delta`` (computed wrap-aware by the
    injected ``distance_fn`` at convergence-check time; the raw sum
    stored here may lie outside ``[-pi, pi]`` for circular joints —
    that's fine, the distance function handles it). On each tick:

    1. Advance the cursor past the current target when the perceived
       joints are within ``advance_radius`` (in the ``distance_fn``
       metric) of it, **or** when their projection onto the incoming
       segment (previous waypoint → current target, or the planned start
       configuration → first target) has passed the target. The
       projection test is what makes tracking robust to a persistent
       offset orthogonal to the path: joints the plan never moves but the
       arm holds slightly off (friction, a nudge during teleoperation)
       add to the distance to every waypoint and can exceed
       ``advance_radius`` on their own, which used to stall the cursor
       for good. Multiple waypoints can be passed in a single tick; the
       cursor jumps straight to the furthest one.
    2. Command the new cursor's target as the arm goal in the resulting
       :class:`TidyBotAction`. Same target is re-issued every tick as a
       heartbeat (the underlying Kinova controller's watchdog freezes
       the OTG after ``2.5 * POLICY_CONTROL_PERIOD`` without a fresh
       command — see ``arm_controller.py:104``).

    Because the cursor advances *before* the OTG has decelerated to
    zero at the current waypoint, Ruckig's mid-flight replan uses the
    current nonzero velocity as initial conditions and the arm rounds
    through the waypoint at speed. The radius rule can skip a waypoint
    that is closer than ``advance_radius`` to its predecessor without
    the arm having visited it; the projection rule only ever advances
    past waypoints the arm has actually passed.

    Stalled arrival: the low-level compliant controller settles with a
    steady-state error of a few hundredths of a radian per joint, which
    over seven joints can exceed ``advance_radius`` and leave the
    projection short of 1 on a short segment, with the arm converged as
    far as it will ever get. So when the perceived joints have not moved
    for ``stall_advance_ticks`` ticks and the path progress is at least
    ``stall_advance_min_progress``, the executor logs a warning and
    advances one waypoint anyway. On a mid-path segment that gives the
    carrot a longer lead and the arm moves on; at the waypoint before a
    gripper command it lets the gripper command issue. A genuine
    obstruction shows up as repeated stall-advance warnings before
    ``max_iter_total`` raises.

    If the cursor has not advanced for ``stall_warning_ticks`` ticks a
    warning is logged once (with the distance to the target and the
    projection parameter) so a stall is visible live.

    If ``max_iter_total`` ticks elapse before the final waypoint is
    reached, :meth:`done` logs a warning with the remaining distances and
    raises :class:`ExecutionFailure` rather than reporting the segment
    complete, so the rest of the plan does not run from a configuration it
    was not refined for.

    The base + gripper components of the commanded ``TidyBotAction``
    hold at the perceived state for arm pairs that don't move them
    (``action[10]`` in ``[-0.5, 0.5]`` means "hold gripper"); explicit
    gripper open/close commands surface as ``gripper_goal=0.0`` /
    ``gripper_goal=gripper_close_position``.

    Why this strategy works only because of the OTG: if the underlying
    controller were a pre-computed-trajectory follower, re-targeting
    mid-flight would abort the in-flight motion and restart from rest,
    defeating the whole point. The Kinova controller's Ruckig-based
    online retargeting is what makes mid-flight cursor advance smooth.
    """

    def __init__(
        self,
        distance_fn: JointDistanceFn,
        robot_name: str = "robot",
        advance_radius: float = 0.2,
        arrival_tolerance: float = 0.1,
        max_iter_total: int = 2000,
        gripper_dwell_ticks: int = 0,
        gripper_close_position: float = 1.0,
        stall_warning_ticks: int = 50,
        stall_advance_ticks: int = 30,
        stall_advance_min_progress: float = 0.5,
        confirm_grasp_closes: bool = False,
        prompt_fn: Callable[[str], str] = input,
        compensate_base_error: bool = False,
        compensation_reach: float = 0.8,
        visual_lateral_mode: str = "off",
        image_source: Any = None,
        visual_lateral_gain: float = 0.0003,
        visual_lateral_sign: float = 1.0,
        visual_max_lateral: float = 0.04,
        visual_debug_dir: str | None = None,
    ) -> None:
        super().__init__(robot_name=robot_name)
        if visual_lateral_mode not in ("off", "log", "on"):
            raise ValueError("visual_lateral_mode must be off, log or on")
        if advance_radius <= 0:
            raise ValueError("advance_radius must be > 0")
        if arrival_tolerance <= 0:
            raise ValueError("arrival_tolerance must be > 0")
        if max_iter_total <= 0:
            raise ValueError("max_iter_total must be > 0")
        if gripper_dwell_ticks < 0:
            raise ValueError("gripper_dwell_ticks must be >= 0")
        if not 0.0 < gripper_close_position <= 1.0:
            raise ValueError("gripper_close_position must be in (0, 1]")
        self._distance_fn = distance_fn
        self._advance_radius = advance_radius
        self._arrival_tolerance = arrival_tolerance
        self._max_iter_total = max_iter_total
        self._gripper_dwell_ticks = gripper_dwell_ticks
        self._gripper_close_position = gripper_close_position
        self._stall_warning_ticks = stall_warning_ticks
        self._stall_advance_ticks = stall_advance_ticks
        self._stall_advance_min_progress = stall_advance_min_progress
        # A staging-calibration gate: with confirm_grasp_closes, the operator
        # is prompted once per gripper CLOSE, with the fingers holding at the
        # grasp pose — e.g. to mark where the object must be staged for the
        # open-loop grasp to land on it.
        self._confirm_grasp_closes = confirm_grasp_closes
        self._prompt_fn = prompt_fn
        # Base-error lateral compensation: at each segment's first tick the
        # perceived base pose is compared with the plan's, and joint 1 of
        # every target is offset so the arm plane swings back onto the
        # map-frame end-effector target ~compensation_reach ahead (the
        # planned joints reproduce the end effector in the BASE frame, so a
        # base staging error maps one-to-one into a map-frame arm error;
        # the lateral component dominates and joint 1 corrects exactly it).
        self._compensate_base_error = compensate_base_error
        self._compensation_reach = compensation_reach
        # Wrist-camera lateral correction for grasp segments (segments that
        # contain a gripper CLOSE start at the pre-grasp pose, thanks to the
        # kinder grasp controller's segment marker). The camera measures the
        # can's true lateral offset, which SUPERSEDES the marker-based base
        # compensation for that segment: when a detection succeeds the
        # marker correction is skipped; on detection failure it is the
        # fallback. "log" measures and logs the would-be correction without
        # applying it — run that first on the robot to verify the sign at
        # the 45-degree view (visual_lateral_sign).
        self._visual_lateral_mode = visual_lateral_mode
        self._image_source = image_source
        self._visual_lateral_gain = visual_lateral_gain
        self._visual_lateral_sign = visual_lateral_sign
        self._visual_max_lateral = visual_max_lateral
        self._visual_debug_dir = visual_debug_dir
        self._visual_frame_count = 0
        # The base hands over as soon as it is within tolerance, while it is
        # still creeping and the marker-pose stream lags, so the error must
        # be computed from a SETTLED pose: the arm holds until the perceived
        # base pose is stable for _SETTLE_STABLE_TICKS consecutive ticks
        # (or the timeout passes), and the correction uses the mean of the
        # stable window.

        self._targets: list[JointPositions] = []
        self._start_joints: JointPositions = []
        self._cursor: int = 0
        self._tick_count: int = 0
        self._ticks_since_advance: int = 0
        self._stall_warned: bool = False
        self._last_perceived: JointPositions | None = None
        self._still_ticks: int = 0
        self._lead_from_segment_start: bool = False
        self._done_latched: bool = False
        self._gripper_cursor: int = -1
        self._gripper_ticks_remaining: int = 0
        self._last_gripper_goal: float | None = None
        self._confirmed_close_cursor: int = -1
        self._compensation_applied: bool = False
        self._settle_history: list[tuple[float, float, float]] = []
        self._visual_attempts: int = 0

    def _on_set_trajectory(self) -> None:
        self._targets = [
            _absolute_target(state, action, self._robot_name)
            for state, action in self._pairs
        ]
        self._start_joints = (
            _perceived_joints(self._pairs[0][0], self._robot_name)
            if self._pairs
            else []
        )
        self._cursor = 0
        self._tick_count = 0
        self._ticks_since_advance = 0
        self._stall_warned = False
        self._last_perceived = None
        self._still_ticks = 0
        self._lead_from_segment_start = False
        self._done_latched = False
        self._gripper_cursor = -1
        self._gripper_ticks_remaining = 0
        self._last_gripper_goal = None
        self._confirmed_close_cursor = -1
        self._compensation_applied = False
        self._settle_history = []
        self._visual_attempts = 0

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TidyBotAction, NDArray[np.floating]]:
        if not self._pairs:
            raise RuntimeError(
                "StreamingArmMotion3DPlanExecutor.step called with no trajectory"
            )
        if (
            self._compensate_base_error or self._visual_applicable()
        ) and not self._compensation_applied:
            settled = self._settled_base_pose(sim_state)
            if settled is None:
                # Hold the arm at its perceived joints until the base pose
                # settles; the base target inside the action holds too.
                perceived = _perceived_joints(sim_state, self._robot_name)
                hold = np.zeros(11)
                action = _build_tidybot_action(
                    sim_state,
                    perceived,
                    hold,
                    self._robot_name,
                    self._last_gripper_goal,
                    self._gripper_close_position,
                )
                self._tick_count += 1
                return action, hold
            if self._visual_applicable():
                outcome = self._attempt_visual_correction()
                if outcome == "retry":
                    perceived = _perceived_joints(sim_state, self._robot_name)
                    hold = np.zeros(11)
                    action = _build_tidybot_action(
                        sim_state,
                        perceived,
                        hold,
                        self._robot_name,
                        self._last_gripper_goal,
                        self._gripper_close_position,
                    )
                    self._tick_count += 1
                    return action, hold
                self._compensation_applied = True
                if outcome == "failed" and self._compensate_base_error:
                    self._apply_base_error_compensation(settled)
            else:
                self._compensation_applied = True
                if self._compensate_base_error:
                    self._apply_base_error_compensation(settled)
        perceived = _perceived_joints(sim_state, self._robot_name)
        self._advance_cursor(perceived)
        target = self._command_target(perceived)
        _, sim_action = self._pairs[self._cursor]
        if (
            self._confirm_grasp_closes
            and _is_gripper_cmd(sim_action)
            and float(sim_action[10]) < -0.5
            and self._cursor != self._confirmed_close_cursor
        ):
            # First tick at a CLOSE waypoint: the fingers hold at the grasp
            # pose while the operator responds; the close is commanded on
            # this same tick afterwards.
            self._confirmed_close_cursor = self._cursor
            self._prompt_fn(
                "Gripper is at the grasp pose and about to CLOSE. Mark the "
                "object spot, then press Enter to close..."
            )
        # Remember the most recent explicit open/close so that subsequent "hold"
        # ticks (e.g. the entire retract phase) re-issue the same gripper goal.
        # The planning sim's finger_state may not reflect the real gripper state
        # (kinder does not update finger_state after close actions), so we cannot
        # rely on the planned state; tracking the last command is authoritative.
        if _is_gripper_cmd(sim_action):
            self._last_gripper_goal = (
                self._gripper_close_position if float(sim_action[10]) < -0.5 else 0.0
            )
        action = _build_tidybot_action(
            sim_state,
            target,
            sim_action,
            self._robot_name,
            self._last_gripper_goal,
            self._gripper_close_position,
        )
        self._tick_count += 1
        # Advance past a gripper pair after gripper_dwell_ticks extra ticks.
        # With gripper_dwell_ticks=0 the cursor advances on the very next tick
        # after the command is issued (original behaviour, correct for sim/fake
        # where FakeInterface stores the target immediately).  In real mode set
        # gripper_dwell_ticks to something like 20 (≈5 s at 0.25 s/tick) so the
        # arm stays at the grasp position while the Kinova gripper physically
        # closes around the object before retract begins.
        if _is_gripper_cmd(sim_action) and self._cursor + 1 < len(self._targets):
            if self._cursor != self._gripper_cursor:
                self._gripper_cursor = self._cursor
                self._gripper_ticks_remaining = self._gripper_dwell_ticks
            if self._gripper_ticks_remaining > 0:
                self._gripper_ticks_remaining -= 1
            else:
                self._cursor += 1
        return action, sim_action

    def _visual_applicable(self) -> bool:
        """Visual lateral correction is configured and this is a grasp segment."""
        return (
            self._visual_lateral_mode != "off"
            and self._image_source is not None
            and any(
                float(action[10]) < -0.5
                for _, action in self._pairs
                if not isinstance(action, SkillCall)
            )
        )

    def _attempt_visual_correction(self) -> str:
        """Try one wrist-camera detection; returns "applied", "retry" or "failed".

        "applied" also covers log-only mode (measured, logged, not applied).
        """
        self._visual_attempts += 1
        image = self._image_source.get_image()
        edges = None if image is None else detect_cylinder_edges(image)
        if self._visual_debug_dir is not None and image is not None:
            self._dump_visual_frame(image, edges)
        if edges is None or edges.clipped_left or edges.clipped_right:
            if self._visual_attempts < _VISUAL_MAX_ATTEMPTS:
                return "retry"
            _logger.warning(
                "Visual lateral correction: no usable detection after %d "
                "attempts; falling back to the marker-based compensation.",
                self._visual_attempts,
            )
            return "failed"
        lateral = (
            self._visual_lateral_sign
            * self._visual_lateral_gain
            * float(edges.servo_error_px)
        )
        clamped = max(-self._visual_max_lateral, min(self._visual_max_lateral, lateral))
        delta = clamped / self._compensation_reach
        _logger.info(
            "Visual lateral correction: %.1f px -> %.1f mm lateral -> joint 1 "
            "offset %+.4f rad%s.",
            edges.servo_error_px,
            1000 * clamped,
            delta,
            " (log only, NOT applied)" if self._visual_lateral_mode == "log" else "",
        )
        if self._visual_lateral_mode == "on":
            for target in self._targets:
                target[0] = float(target[0]) + delta
        return "applied"

    def _dump_visual_frame(self, image: Any, edges: Any) -> None:
        """Save the raw wrist frame (and edge overlay when detected) for offline
        detector tuning."""
        import cv2  # type: ignore[import-untyped]

        from prpl_tidybot.visual_servo.cylinder_edges import render_edge_overlay

        directory = Path(self._visual_debug_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._visual_frame_count += 1
        stem = directory / f"grasp_frame_{self._visual_frame_count:04d}"
        cv2.imwrite(str(stem) + ".png", image[:, :, ::-1])
        overlay = render_edge_overlay(image, edges)
        cv2.imwrite(str(stem) + "_overlay.png", overlay[:, :, ::-1])

    def _settled_base_pose(
        self, sim_state: ObjectCentricState
    ) -> tuple[float, float, float] | None:
        """The mean perceived base pose once it is stable, else None (keep waiting).

        Stable means _SETTLE_STABLE_TICKS consecutive readings within
        _SETTLE_POS_TOL / _SETTLE_ROT_TOL of the newest one. After
        _SETTLE_TIMEOUT_TICKS the newest reading is used anyway (with a
        warning) so a drifting marker stream cannot stall the rollout.
        """
        pose = _perceived_base(sim_state, self._robot_name)
        self._settle_history.append(pose)
        recent = self._settle_history[-_SETTLE_STABLE_TICKS:]
        stable = len(recent) == _SETTLE_STABLE_TICKS and all(
            math.hypot(b[0] - pose[0], b[1] - pose[1]) < _SETTLE_POS_TOL
            and abs(_wrap(b[2] - pose[2])) < _SETTLE_ROT_TOL
            for b in recent
        )
        if not stable:
            if len(self._settle_history) < _SETTLE_TIMEOUT_TICKS:
                return None
            _logger.warning(
                "Base pose did not settle within %d ticks; compensating from "
                "the latest reading.",
                _SETTLE_TIMEOUT_TICKS,
            )
            recent = [pose]
        return (
            sum(b[0] for b in recent) / len(recent),
            sum(b[1] for b in recent) / len(recent),
            math.atan2(
                sum(math.sin(b[2]) for b in recent),
                sum(math.cos(b[2]) for b in recent),
            ),
        )

    def _apply_base_error_compensation(
        self, actual: tuple[float, float, float]
    ) -> None:
        """Offset joint 1 of every target for the perceived base staging error.

        The map-frame end-effector target is reconstructed from the PLANNED
        base pose and the last target's joint 1 (arm plane heading = base
        heading + joint 1, target ~compensation_reach along it); the offset
        is whatever joint 1 must become for the ACTUAL base to point the arm
        plane at that same map point. First order and lateral-only: the
        forward error component (a few millimetres for centimetre-scale base
        errors) is absorbed by the grasp/place standoffs.
        """
        if not self._pairs or not self._targets:
            return
        planned = _perceived_base(self._pairs[0][0], self._robot_name)
        last_j1 = float(self._targets[-1][0])
        # Joint 1 positive rotates the arm plane CLOCKWISE in the map frame
        # (verified against the kinder robot model in
        # test_base_error_compensation_recovers_ee_position), hence the
        # negations: plane bearing = base heading - joint 1.
        heading = planned[2] - last_j1
        target_x = planned[0] + self._compensation_reach * math.cos(heading)
        target_y = planned[1] + self._compensation_reach * math.sin(heading)
        needed_j1 = -_wrap(
            math.atan2(target_y - actual[1], target_x - actual[0]) - actual[2]
        )
        delta = _wrap(needed_j1 - last_j1)
        if abs(delta) > 0.15:
            # A correction this large means the base is nowhere near its
            # staging pose (or perception glitched); trust the plan instead.
            _logger.warning(
                "Base-error compensation of %.3f rad exceeds the 0.15 cap; "
                "skipping (planned base %s, perceived %s).",
                delta,
                np.round(planned, 3),
                np.round(actual, 3),
            )
            return
        if abs(delta) > 1e-4:
            _logger.info(
                "Base-error compensation: joint 1 offset %.4f rad "
                "(~%.1f cm lateral at %.2f m).",
                delta,
                100 * delta * self._compensation_reach,
                self._compensation_reach,
            )
        for target in self._targets:
            target[0] = float(target[0]) + delta

    def done(self, sim_state: ObjectCentricState) -> bool:
        if self._done_latched:
            return True
        if not self._pairs:
            self._done_latched = True
            return True
        if self._tick_count >= self._max_iter_total:
            perceived = _perceived_joints(sim_state, self._robot_name)
            to_cursor = self._distance_fn(perceived, self._targets[self._cursor])
            to_final = self._distance_fn(perceived, self._targets[-1])
            message = (
                f"{type(self).__name__} gave up after {self._tick_count} ticks: "
                f"cursor at waypoint {self._cursor + 1}/{len(self._targets)}, "
                f"distance {to_cursor:.3f} to the cursor target (advance_radius "
                f"{self._advance_radius}), {to_final:.3f} to the final target "
                f"(arrival_tolerance {self._arrival_tolerance})."
            )
            _logger.warning(message)
            raise ExecutionFailure(message)
        # Require the cursor to have reached the last waypoint before declaring
        # done. Without this guard, done() returns True immediately when the
        # final target (the retract/home position) happens to equal the robot's
        # initial perceived position — the merged arm segment (approach +
        # gripper + retract) starts and ends at home, so the distance check
        # fires before a single step() is ever called.
        if self._cursor < len(self._targets) - 1:
            return False
        perceived = _perceived_joints(sim_state, self._robot_name)
        final_target = self._targets[-1]
        if self._distance_fn(perceived, final_target) <= self._arrival_tolerance:
            self._done_latched = True
            return True
        return False

    def _advance_cursor(self, perceived: JointPositions) -> None:
        self._track_stillness(perceived)
        advanced = False
        while self._cursor + 1 < len(self._targets) and not _is_gripper_cmd(
            self._pairs[self._cursor][1]
        ):
            within_radius = (
                self._distance_fn(perceived, self._targets[self._cursor])
                <= self._advance_radius
            )
            if not within_radius and self._progress(perceived) < 1.0:
                break
            self._cursor += 1
            self._lead_from_segment_start = False
            advanced = True
        if not advanced and self._stalled_arrival(perceived):
            _logger.warning(
                "%s stalled at waypoint %d/%d: stationary for %d ticks, distance "
                "%.3f to the target, path progress %.2f; treating it as reached.",
                type(self).__name__,
                self._cursor + 1,
                len(self._targets),
                self._still_ticks,
                self._distance_fn(perceived, self._targets[self._cursor]),
                self._progress(perceived),
            )
            self._cursor += 1
            self._still_ticks = 0
            # The arm now projects behind the new segment's start; measure the
            # carrot lead from that start so each stall-advance lengthens it.
            self._lead_from_segment_start = True
            advanced = True
        if advanced:
            self._ticks_since_advance = 0
            self._stall_warned = False
        else:
            self._ticks_since_advance += 1
            if (
                self._ticks_since_advance >= self._stall_warning_ticks
                and not self._stall_warned
            ):
                self._stall_warned = True
                _logger.warning(
                    "%s cursor has not advanced for %d ticks: waypoint %d/%d, "
                    "distance %.3f to its target (advance_radius %s), path "
                    "progress %.2f.",
                    type(self).__name__,
                    self._ticks_since_advance,
                    self._cursor + 1,
                    len(self._targets),
                    self._distance_fn(perceived, self._targets[self._cursor]),
                    self._advance_radius,
                    self._progress(perceived),
                )

    def _track_stillness(self, perceived: JointPositions) -> None:
        """Count consecutive ticks on which the perceived joints did not move."""
        if (
            self._last_perceived is not None
            and self._distance_fn(perceived, self._last_perceived) < _STILL_EPS
        ):
            self._still_ticks += 1
        else:
            self._still_ticks = 0
        self._last_perceived = list(perceived)

    def _stalled_arrival(self, perceived: JointPositions) -> bool:
        """Whether a stationary stall past half the incoming segment should count as
        reaching the cursor waypoint (see the class docstring)."""
        if self._cursor + 1 >= len(self._targets):
            return False
        if _is_gripper_cmd(self._pairs[self._cursor][1]):
            return False
        return (
            self._still_ticks >= self._stall_advance_ticks
            and self._progress(perceived) >= self._stall_advance_min_progress
        )

    def _incoming_segment(self) -> tuple[JointPositions, JointPositions]:
        """The segment ending at the cursor target: (previous waypoint, target)."""
        previous = (
            self._start_joints if self._cursor == 0 else self._targets[self._cursor - 1]
        )
        return previous, self._targets[self._cursor]

    def _progress(self, perceived: JointPositions) -> float:
        """Projection parameter of `perceived` onto the incoming segment.

        0 at the previous waypoint, 1 at the cursor target; may fall outside
        [0, 1]. Computed in joint space with wrap-aware differences, so a
        constant offset orthogonal to the segment does not change it.
        """
        previous, target = self._incoming_segment()
        return _path_progress(previous, target, perceived)

    def _remaining_to_cursor(self, perceived: JointPositions) -> float:
        """Path length (distance_fn metric) from the projected point to the cursor
        target."""
        previous, target = self._incoming_segment()
        progress = self._progress(perceived)
        if self._lead_from_segment_start:
            progress = max(progress, 0.0)
        remaining_fraction = max(0.0, 1.0 - progress)
        return remaining_fraction * self._distance_fn(previous, target)

    def _command_target(self, perceived: JointPositions) -> JointPositions:
        """The joint target to command this tick; subclasses may override."""
        del perceived
        return self._targets[self._cursor]


class CarrotArmMotion3DPlanExecutor(StreamingArmMotion3DPlanExecutor):
    """Streaming executor that commands a constant-lookahead carrot target.

    The discrete crossover advance commands whole waypoints, so with plans
    whose waypoint spacing is comparable to ``advance_radius`` the commanded
    lead alternates between roughly one and two waypoint spacings tick to
    tick, and the OTG's cruise speed surges at that beat — perceived as arm
    jitter on dense plans (measured ~30% speed ripple on the planar
    cylinder-shelf trajectories). Here the command is instead a point
    interpolated along the plan polyline at a fixed ``lookahead`` (in the
    ``distance_fn`` metric) ahead of the perceived joints, so the lead —
    and hence the commanded speed — stays constant while cruising. The walk
    never crosses a gripper waypoint or the final waypoint, which
    reproduces the parent's hold-and-dwell behavior at those points and the
    natural deceleration at the end of the plan.

    Linear interpolation between adjacent waypoints is safe because
    targets are cumulative sums of small per-step deltas (they are not
    wrapped to [-pi, pi], so consecutive targets never jump across a
    circular-joint seam).
    """

    def __init__(
        self,
        distance_fn: JointDistanceFn,
        robot_name: str = "robot",
        advance_radius: float = 0.2,
        arrival_tolerance: float = 0.1,
        max_iter_total: int = 2000,
        gripper_dwell_ticks: int = 0,
        lookahead: float = 0.25,
        gripper_close_position: float = 1.0,
        stall_warning_ticks: int = 50,
        stall_advance_ticks: int = 30,
        stall_advance_min_progress: float = 0.5,
        confirm_grasp_closes: bool = False,
        prompt_fn: Callable[[str], str] = input,
        compensate_base_error: bool = False,
        compensation_reach: float = 0.8,
        visual_lateral_mode: str = "off",
        image_source: Any = None,
        visual_lateral_gain: float = 0.0003,
        visual_lateral_sign: float = 1.0,
        visual_max_lateral: float = 0.04,
        visual_debug_dir: str | None = None,
    ) -> None:
        super().__init__(
            distance_fn=distance_fn,
            robot_name=robot_name,
            advance_radius=advance_radius,
            arrival_tolerance=arrival_tolerance,
            max_iter_total=max_iter_total,
            gripper_dwell_ticks=gripper_dwell_ticks,
            gripper_close_position=gripper_close_position,
            stall_warning_ticks=stall_warning_ticks,
            stall_advance_ticks=stall_advance_ticks,
            stall_advance_min_progress=stall_advance_min_progress,
            confirm_grasp_closes=confirm_grasp_closes,
            prompt_fn=prompt_fn,
            compensate_base_error=compensate_base_error,
            compensation_reach=compensation_reach,
            visual_lateral_mode=visual_lateral_mode,
            image_source=image_source,
            visual_lateral_gain=visual_lateral_gain,
            visual_lateral_sign=visual_lateral_sign,
            visual_max_lateral=visual_max_lateral,
            visual_debug_dir=visual_debug_dir,
        )
        if lookahead <= 0:
            raise ValueError("lookahead must be > 0")
        self._lookahead = lookahead

    def _command_target(self, perceived: JointPositions) -> JointPositions:
        cursor_pair_action = self._pairs[self._cursor][1]
        cursor_target = self._targets[self._cursor]
        if _is_gripper_cmd(cursor_pair_action):
            # Hold at the gripper waypoint while the dwell runs.
            return cursor_target
        # Measure the lead along the path from the perceived joints' projection
        # rather than by raw distance to the cursor target, so an offset
        # orthogonal to the path does not shrink the lookahead.
        budget = self._lookahead - self._remaining_to_cursor(perceived)
        if budget <= 0:
            # The cursor target is already at least a full lookahead away.
            return cursor_target
        index = self._cursor
        while index + 1 < len(self._targets):
            if _is_gripper_cmd(self._pairs[index][1]):
                # Never command past a gripper waypoint.
                return self._targets[index]
            segment = self._distance_fn(self._targets[index], self._targets[index + 1])
            if segment > budget:
                fraction = budget / segment
                return [
                    a + fraction * (b - a)
                    for a, b in zip(self._targets[index], self._targets[index + 1])
                ]
            budget -= segment
            index += 1
        return self._targets[-1]


# ============================================================================
# Module-level helpers
# ============================================================================


def _is_gripper_cmd(action: NDArray[np.floating]) -> bool:
    """True when action[10] encodes an explicit open or close command (|cmd| > 0.5)."""
    return abs(float(action[10])) > 0.5


def _validate_arm_only(action: NDArray[np.floating]) -> None:
    base_moves = bool(np.any(np.abs(action[0:3]) > _BASE_MOTION_EPS))
    if base_moves:
        raise ValueError(
            "Arm-motion plan executors require arm-only pairs; got "
            f"base_delta={action[0:3]}."
        )


def _wrapped_difference(a: JointPositions, b: JointPositions) -> np.ndarray:
    """Per-joint ``a - b`` wrapped to [-pi, pi], so continuous joints compare across
    the seam."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.arctan2(np.sin(diff), np.cos(diff))


def _path_progress(
    start: JointPositions, end: JointPositions, point: JointPositions
) -> float:
    """Projection parameter of `point` onto the segment start→end (0 at start, 1 at
    end). A degenerate segment counts as passed."""
    direction = _wrapped_difference(end, start)
    length_sq = float(direction @ direction)
    if length_sq < 1e-12:
        return 1.0
    return float(_wrapped_difference(point, start) @ direction / length_sq)


def _perceived_joints(sim_state: ObjectCentricState, robot_name: str) -> JointPositions:
    robot = sim_state.get_object_from_name(robot_name)
    return [sim_state.get(robot, f"joint_{j + 1}") for j in range(7)]


def _perceived_base(
    sim_state: ObjectCentricState, robot_name: str
) -> tuple[float, float, float]:
    robot = sim_state.get_object_from_name(robot_name)
    return (
        float(sim_state.get(robot, "pos_base_x")),
        float(sim_state.get(robot, "pos_base_y")),
        float(sim_state.get(robot, "pos_base_rot")),
    )


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# Base-pose settle detection for the compensation (see compensate_base_error):
# consecutive readings required, their agreement tolerances, and the give-up
# horizon (ticks are ~0.1 s in real mode).
_SETTLE_STABLE_TICKS = 4
_SETTLE_POS_TOL = 0.004
_SETTLE_ROT_TOL = 0.006
_SETTLE_TIMEOUT_TICKS = 50
# Wrist-camera detection attempts (one per tick) before the visual lateral
# correction gives up and falls back to the marker-based compensation.
_VISUAL_MAX_ATTEMPTS = 10


def _absolute_target(
    state: ObjectCentricState,
    sim_action: NDArray[np.floating],
    robot_name: str,
) -> JointPositions:
    """Per-pair absolute joint target = state.joints + arm_delta.

    The raw sum may fall outside ``[-pi, pi]`` for circular joints;
    that's fine — downstream distance/equality checks must be
    wrap-aware (the convergence check goes through ``distance_fn``,
    which is, and the Kinova arm controller's
    :func:`execute_action_angular` path also wraps the target itself
    inside the inner loop, so the OTG receives a wrap-safe value
    either way).
    """
    robot = state.get_object_from_name(robot_name)
    return [
        float(state.get(robot, f"joint_{j + 1}")) + float(sim_action[3 + j])
        for j in range(7)
    ]


def _build_tidybot_action(
    sim_state: ObjectCentricState,
    arm_target: JointPositions,
    sim_action: NDArray[np.floating],
    robot_name: str,
    last_gripper_goal: float | None = None,
    gripper_close_position: float = 1.0,
) -> TidyBotAction:
    """Pack the commanded arm goal + held base pose + gripper into a TidyBotAction.

    For "hold" gripper commands (|action[10]| <= 0.5), uses ``last_gripper_goal``
    as the hold target when provided (the last explicit open/close command issued
    by the executor), falling back to perceived finger_state when no explicit
    command has been issued yet. This ensures the gripper stays closed throughout
    retract after a gripper-close pair instead of reverting to perceived state.
    """
    robot = sim_state.get_object_from_name(robot_name)
    base_goal = SE2(
        x=float(sim_state.get(robot, "pos_base_x")),
        y=float(sim_state.get(robot, "pos_base_y")),
        theta=float(sim_state.get(robot, "pos_base_rot")),
    )
    hold_finger = (
        last_gripper_goal
        if last_gripper_goal is not None
        else float(sim_state.get(robot, "finger_state"))
    )
    gripper_goal = _gripper_target(hold_finger, sim_action, gripper_close_position)
    return TidyBotAction(
        arm_goal=list(arm_target),
        base_pose_target_map=base_goal,
        gripper_goal=gripper_goal,
    )


def _gripper_target(
    current_finger: float,
    sim_action: NDArray[np.floating],
    gripper_close_position: float = 1.0,
) -> float:
    """Convert the kinder bipolar gripper command to a TidyBotAction absolute target."""
    gripper_cmd = float(sim_action[10])
    if gripper_cmd < -0.5:
        return gripper_close_position  # close
    if gripper_cmd > 0.5:
        return 0.0  # open
    return current_finger
