"""Tests for `prpl_tidybot.toss_cycles.run_toss_cycles` (scripts/toss_cycles.py) in fake
mode: pick -> drive to the toss pose -> toss -> retract, repeated, with a confirmation
before every action and a reset (retract, back to the toss pose) whenever the run
stops."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (
    CARRY_ARM_CONF,
    forward_kinematics,
    geometry_from_config,
    pointing_down_joints,
)
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
TOSS_POSE = (0.23, 0.01, 1.5708)


@pytest.fixture(autouse=True)
def _no_control_period_sleep(monkeypatch):
    """The fake arm needs no real-time ticks; a lagging or sagging fake would
    otherwise sleep through every settle."""
    monkeypatch.setattr("prpl_tidybot.toss_cycles.POLICY_CONTROL_PERIOD", 0.0)


def _compose(**extra):
    overrides = ["mode=fake", "seed=0", "interactive=true", "settle_after_toss_s=0"]
    overrides.append("confirm=each")
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="toss_cycles", overrides=overrides)


def _configured_geometry():
    """The pick geometry the run uses (conf/toss_cycles.yaml), not the defaults."""
    return geometry_from_config(_compose().pick.top_down)


class _Operator:
    """Answers every prompt 'y' except those containing a word in `decline`."""

    def __init__(self, decline=()):
        self.prompts: list[str] = []
        self._decline = decline

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "n" if any(word in prompt for word in self._decline) else "y"

    def asked(self, word):
        """How many prompts mentioned `word`."""
        return sum(word in p for p in self.prompts)


@pytest.fixture(name="arm_log")
def _arm_log(monkeypatch):
    """Every FakeArmInterface toss and arm goal, in order."""
    log: list[tuple] = []
    original_toss = FakeArmInterface.toss
    original_execute = FakeArmInterface.execute_action

    def toss(self, **kwargs):
        log.append(("toss", kwargs))
        return original_toss(self, **kwargs)

    def execute_action(self, arm_goal, gripper_goal):
        log.append(("arm", list(arm_goal), gripper_goal))
        original_execute(self, arm_goal, gripper_goal)

    original_angular = FakeArmInterface.move_angular

    def move_angular(self, goal):
        log.append(("angular", list(goal), self.gripper_state))
        return original_angular(self, goal)

    monkeypatch.setattr(FakeArmInterface, "toss", toss)
    monkeypatch.setattr(FakeArmInterface, "execute_action", execute_action)
    monkeypatch.setattr(FakeArmInterface, "move_angular", move_angular)
    return log


def test_two_cycles_pick_home_drive_toss_then_reset(tmp_path, arm_log):
    """Two full cycles, each confirmed step by step, two tosses at 140/600, then the
    arm retracts and the base ends at the toss pose; cycles.json records both."""
    operator = _Operator()
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 2
    assert result.finish_reason == "completed 2/2 cycles"
    tosses = [e[1] for e in arm_log if e[0] == "toss"]
    assert tosses == [{"heading": 0.0, "max_vel": 140.0, "gripper_release_ms": 600}] * 2
    for word in ("pick", "Drive to the toss pose", "Toss?", "retract the arm"):
        assert operator.asked(word) >= 2, word
    assert operator.asked("home") == 0
    assert operator.asked("Reset") == 1
    last_arm = [e for e in arm_log if e[0] in ("arm", "angular")][-1]
    assert last_arm[1] == pytest.approx(list(RETRACT_ARM_CONF))
    x, y, theta = result.final_base_pose
    assert (x, y) == pytest.approx(TOSS_POSE[:2], abs=0.02)
    assert math.cos(theta - TOSS_POSE[2]) > math.cos(0.05)
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert [c["cycle"] for c in log["cycles"]] == [1, 2]
    assert all(c["tossed"] for c in log["cycles"])
    assert log["toss_pose"] == pytest.approx(list(TOSS_POSE))


def test_the_base_drives_to_the_toss_pose_with_the_cube_in_the_carry_pose(
    tmp_path, arm_log
):
    """No separate home step: the pick ends in the carry pose and the arm does not move
    again until the toss."""
    run_toss_cycles(_compose(cycles=1), log_dir=tmp_path, prompt_fn=_Operator())
    toss_at = next(i for i, e in enumerate(arm_log) if e[0] == "toss")
    moves = [e for e in arm_log[:toss_at] if e[0] in ("arm", "angular")]
    assert moves[-1][1] == pytest.approx(list(CARRY_ARM_CONF))


def test_the_arm_retracts_after_each_toss_before_the_next_pick(tmp_path, arm_log):
    """The toss leaves the arm raised in its release pose; it is retracted, under the
    compliant loop, before the base drives off for the next pick (run 7 drove to
    cycle 2 with the arm up)."""
    run_toss_cycles(_compose(cycles=2), log_dir=tmp_path, prompt_fn=_Operator())
    first_toss = next(i for i, e in enumerate(arm_log) if e[0] == "toss")
    after = [e for e in arm_log[first_toss + 1 :] if e[0] in ("arm", "angular")]
    assert after[0][0] == "arm"
    assert after[0][1] == pytest.approx(list(RETRACT_ARM_CONF))


def test_every_retract_runs_under_the_compliant_loop(tmp_path, arm_log):
    """Only the grasp uses the onboard controller; retracts (after a toss, before the
    drive back after a miss, and in the reset) stay compliant."""
    cfg = _compose(
        **{
            "pick.top_down.fake_grasp_reading": 0.996,
            "pick.top_down.max_attempts": 2,
            "pick.top_down.ask_to_move_after_miss": 0,
        }
    )
    run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=_Operator())
    retract = list(RETRACT_ARM_CONF)
    assert not [
        e for e in arm_log if e[0] == "angular" and e[1] == pytest.approx(retract)
    ]
    assert [e for e in arm_log if e[0] == "arm" and e[1] == pytest.approx(retract)]


def test_declining_the_drive_stops_without_tossing_and_resets(tmp_path, arm_log):
    """A declined step ends the run; nothing is thrown; the reset still happens."""
    operator = _Operator(decline=("Drive to the toss pose",))
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 0
    assert "declined" in result.finish_reason
    assert not [e for e in arm_log if e[0] == "toss"]
    assert operator.asked("Reset") == 1
    last_arm = [e for e in arm_log if e[0] == "arm"][-1]
    assert last_arm[1] == pytest.approx(list(RETRACT_ARM_CONF))


def test_declining_the_reset_leaves_the_robot_where_it_is(tmp_path, arm_log):
    """Saying no to the reset prompt means no further motion."""
    operator = _Operator(decline=("Toss?", "Reset"))
    run_toss_cycles(_compose(cycles=1), log_dir=tmp_path, prompt_fn=operator)
    retracts = [
        e
        for e in arm_log
        if e[0] == "arm" and e[1] == pytest.approx(list(RETRACT_ARM_CONF))
    ]
    assert not retracts


def test_no_cube_stops_and_asks_for_it_to_be_moved(tmp_path, arm_log, monkeypatch):
    """When the localizer finds no cube the run stops, says so, and resets."""

    def no_cube(self):
        raise RuntimeError(
            "CeilingCubeTargetSource: no cube found in the drivable area"
        )

    monkeypatch.setattr(ConstantTargetSource, "get_target", no_cube)
    operator = _Operator()
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 0
    assert "no cube found" in result.finish_reason
    assert "move the cube" in result.finish_reason
    assert not [e for e in arm_log if e[0] == "toss"]
    assert operator.asked("Reset") == 1


def test_top_down_pick_hovers_then_descends_pointing_down(tmp_path, arm_log):
    """The default pick drives to the standoff, then streams a hover target and a grasp
    target straight below it with the tool pointing down, opening before and closing
    after."""
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )
    geom = _configured_geometry()
    downward = []
    for entry in arm_log:
        if entry[0] not in ("arm", "angular"):
            continue
        pos, approach, _ = forward_kinematics(entry[1], geom)
        if approach[2] < -0.99 and pos[2] < geom.floor_z + 0.2:
            downward.append((round(pos[2], 3), entry[2]))
    heights = sorted({z for z, _ in downward})
    assert len(heights) >= 2
    assert heights[-1] - heights[0] == pytest.approx(geom.hover_height, abs=0.005)
    assert any(g == 0.0 for _, g in downward) and any(g == 1.0 for _, g in downward)


def test_each_pick_move_logs_planned_and_measured_joints(tmp_path):
    """cycles.json records, for every arm move of a pick attempt, the planned and the
    measured joints and pinch-point heights, so sag and floor contact can be read off.
    """
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = log["cycles"][0]["pick_attempts"][0]["moves"]
    assert [m["name"] for m in moves] == ["hover", "grasp", "lift", "carry"]
    for move in moves:
        assert len(move["planned_deg"]) == len(move["measured_deg"]) == 7
        assert {"planned_pinch_z", "measured_pinch_z", "reached"} <= move.keys()
    hover, grasp = moves[0], moves[1]
    assert hover["planned_pinch_z"] - grasp["planned_pinch_z"] == pytest.approx(
        _configured_geometry().hover_height, abs=0.005
    )


def test_the_pick_runs_under_the_onboard_controller_by_default(tmp_path, arm_log):
    """Hover, grasp, lift and carry are onboard-controller moves by default (the
    compliant loop creeps too much for the wrist servo, run 9), with the torque loop
    released around them."""
    run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "angular"}
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    angular = [e for e in arm_log if e[0] == "angular"]
    assert len(angular) == 4
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = log["cycles"][0]["pick_attempts"][0]["moves"]
    assert all(m["controller"] == "angular" for m in moves)


def test_the_compliant_pick_is_selectable(tmp_path, arm_log):
    """arm_controller=compliant streams every grasp move through the torque loop."""
    run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "compliant"}
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert not [e for e in arm_log if e[0] == "angular"]
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = log["cycles"][0]["pick_attempts"][0]["moves"]
    assert all(m["controller"] == "compliant" for m in moves)


def test_an_angular_move_that_does_not_arrive_is_not_closed_on(
    tmp_path, arm_log, monkeypatch
):
    """If the arm stops more than a degree off the hover or grasp pose the gripper is
    not closed; the attempt counts as a miss and the arm backs off to the hover."""
    logged = FakeArmInterface.move_angular

    def short(self, goal):
        self.angular_offset = 0.05  # about 3 degrees short on every joint
        return logged(self, goal)

    monkeypatch.setattr(FakeArmInterface, "move_angular", short)
    cfg = _compose(
        cycles=1,
        skip_toss="true",
        **{"pick.top_down.max_attempts": 1, "pick.top_down.arm_controller": "angular"},
    )
    result = run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=_Operator())
    assert "pick failed" in result.finish_reason
    log = json.loads((tmp_path / "cycles.json").read_text())
    attempt = log["cycles"][0]["pick_attempts"][0]
    assert attempt["held"] is False
    assert "did not arrive" in attempt["note"]
    assert not any(e[0] == "angular" and e[2] == 1.0 for e in arm_log)


def test_ctrl_c_during_a_pick_move_stops_without_further_motion(tmp_path, monkeypatch):
    """After an aborted onboard move nothing else is commanded: no reset prompt, no
    torque-loop restart; the arm is left halted where it stopped."""
    from prpl_tidybot.interfaces.arm_interface import (  # pylint: disable=import-outside-toplevel
        ArmMotionAborted,
    )

    def aborted(self, goal):
        raise ArmMotionAborted("Ctrl-C")

    resumes = []
    monkeypatch.setattr(FakeArmInterface, "move_angular", aborted)
    monkeypatch.setattr(FakeArmInterface, "resume", lambda self: resumes.append(1))
    operator = _Operator()
    result = run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "angular"}
        ),
        log_dir=tmp_path,
        prompt_fn=operator,
    )
    assert "aborted" in result.finish_reason
    assert operator.asked("Reset") == 0
    assert not resumes


def test_the_whole_run_is_filmed_from_the_ceiling(tmp_path):
    """ceiling.mp4 in the log dir covers the run, from the first prompt to the reset."""
    import cv2 as cv  # pylint: disable=import-outside-toplevel

    run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )
    video = cv.VideoCapture(str(tmp_path / "ceiling.mp4"))
    assert video.isOpened()
    assert video.get(cv.CAP_PROP_FRAME_COUNT) > 0


def test_ceiling_video_can_be_turned_off(tmp_path):
    """ceiling_video.enabled=false records nothing."""
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true", **{"ceiling_video.enabled": "false"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert not (tmp_path / "ceiling.mp4").exists()


def test_the_toss_pose_faces_the_bin_out_of_reach_and_behind_the_tape():
    """The 2026-10-20 opposite-side layout: the base faces +y at the bin, whose near
    wall is beyond the arm's 1.14 m reach from the base centre (so it has to be
    tossed), whose centre is inside the real toss standoffs, and whose base front
    edge stays `barrier.clearance` behind the measured tape (y = 0.349 + 0.0267 x)."""
    cfg = _compose()
    x, y, heading = cfg.toss_pose
    x0, x1, y0, y1 = cfg.bin_stop_box
    assert math.sin(heading) > 0.99  # throwing towards +y
    assert x0 < x < x1
    assert y0 - y > 1.14
    lo, hi = cfg.real_toss_bounds.standoff
    assert lo <= math.hypot((x0 + x1) / 2 - x, (y0 + y1) / 2 - y) <= hi
    assert y + 0.274 + cfg.barrier.clearance <= 0.349 + 0.0267 * x


def test_each_toss_waits_for_the_cube_to_settle_before_it_is_localized(
    tmp_path, monkeypatch
):
    """Localizing right after the swing caught the cube mid-flight and mid-slide (the
    fit fell to 0.11 of the 0.6 needed) and the run stopped; every toss now waits
    settle_after_toss_s before anything looks for the cube."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    events: list[str] = []
    monkeypatch.setattr(
        tc.time, "sleep", lambda s: events.append(f"sleep {s}") if s >= 1 else None
    )
    original = ConstantTargetSource.get_target

    def get_target(self):
        events.append("localize")
        return original(self)

    monkeypatch.setattr(ConstantTargetSource, "get_target", get_target)
    run_toss_cycles(
        _compose(cycles=2, settle_after_toss_s=5.0),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    settles = [i for i, e in enumerate(events) if e == "sleep 5.0"]
    assert len(settles) == 2
    assert all("localize" in events[i + 1 :] for i in settles[:1])


def test_a_compliant_move_that_sags_is_corrected_onto_its_target(tmp_path, monkeypatch):
    """The compliant loop settles a few degrees short under unmodelled torque (24 mm at
    the gripper on the robot); the descent, which the wrist camera cannot watch, is
    re-aimed past its target by the measured error until the pinch point is within
    2 mm."""
    # The joint errors the real arm settled with at the hover in run 3 (42 mm short).
    sag = np.radians([-0.13, 1.29, 1.86, -1.28, -5.09, -3.24, -3.26])

    def sagging(self, arm_goal, gripper_goal):
        self.arm_state = list(np.array(arm_goal) + sag)
        self.gripper_state = gripper_goal

    monkeypatch.setattr(FakeArmInterface, "execute_action", sagging)
    run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "compliant"}
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = {m["name"]: m for m in log["cycles"][0]["pick_attempts"][0]["moves"]}
    assert moves["grasp"]["reached"] and moves["grasp"]["pinch_error_mm"] < 3.0


class _LaggingArm(FakeArmInterface):
    """Moves a fraction of the way to each command per tick, the short way round,
    like a real arm (and like the lagging arm in the cylinder-servo tests)."""

    def execute_action(self, arm_goal, gripper_goal):
        here = np.array(self.arm_state)
        gap = (np.array(arm_goal) - here + np.pi) % (2 * np.pi) - np.pi
        self.arm_state = list(here + 0.3 * gap)
        self.gripper_state = gripper_goal


def test_a_compliant_move_waits_for_a_lagging_arm_to_arrive(tmp_path, monkeypatch):
    """A move is not over until the arm gets there (run 8: the first quiet ticks
    were taken for arrival, the hover was 'corrected' from the start pose and ended
    14 mm off). Hover and lift end within 1 deg; the corrected descent within 2 mm."""
    monkeypatch.setattr(FakeArmInterface, "execute_action", _LaggingArm.execute_action)
    run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "compliant"}
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = {m["name"]: m for m in log["cycles"][0]["pick_attempts"][0]["moves"]}
    for name in ("hover", "lift"):
        err = np.array(moves[name]["measured_deg"]) - moves[name]["planned_deg"]
        assert np.abs((err + 180) % 360 - 180).max() < 1.0, name
    assert moves["grasp"]["reached"] and moves["grasp"]["pinch_error_mm"] < 3.0


def test_the_correction_never_aims_the_grasp_below_its_planned_height(
    tmp_path, monkeypatch
):
    """An arm held 8 mm up (the floor, the cube) at the grasp is not pushed down: no
    command puts the pinch point below the grasp height measured for the 2F-85 pads."""
    geom = _configured_geometry()
    grasp_z = geom.floor_z + geom.grasp_height
    commanded_z: list[float] = []

    def blocked(self, arm_goal, gripper_goal):
        pos, approach, closing = forward_kinematics(arm_goal, geom)
        commanded_z.append(float(pos[2]))
        state = list(arm_goal)
        if approach[2] < -0.99 and pos[2] < grasp_z + 0.008:
            state = pointing_down_joints(
                (pos[0], pos[1], grasp_z + 0.008),
                math.atan2(closing[1], closing[0]),
                geom,
                seed=arm_goal,
            )
        self.arm_state = state
        self.gripper_state = gripper_goal

    monkeypatch.setattr(FakeArmInterface, "execute_action", blocked)
    run_toss_cycles(
        _compose(
            cycles=1, skip_toss="true", **{"pick.top_down.arm_controller": "compliant"}
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert min(commanded_z) > grasp_z - 0.001


def test_a_compliant_grasp_8_mm_off_still_closes(tmp_path, monkeypatch):
    """Run 35 (2026-10-20): the compliant descent ended 8 mm from the grasp after
    every correction round and the gripper was never closed. The 2F-85 opens 85 mm
    round the 5 cm cube, so a pinch point within `correction_arrive_mm` (10 mm) of the
    grasp closes; a farther one still does not."""
    geom = _configured_geometry()
    grasp_z = geom.floor_z + geom.grasp_height

    def held_up(lift):
        def execute_action(self, arm_goal, gripper_goal):
            pos, approach, closing = forward_kinematics(arm_goal, geom)
            state = list(arm_goal)
            if approach[2] < -0.99 and pos[2] < grasp_z + lift:
                state = pointing_down_joints(
                    (pos[0], pos[1], grasp_z + lift),
                    math.atan2(closing[1], closing[0]),
                    geom,
                    seed=arm_goal,
                )
            self.arm_state = state
            self.gripper_state = gripper_goal

        return execute_action

    for lift, closes in ((0.008, True), (0.015, False)):
        monkeypatch.setattr(FakeArmInterface, "execute_action", held_up(lift))
        run_toss_cycles(
            _compose(
                cycles=1,
                skip_toss="true",
                **{
                    "pick.top_down.arm_controller": "compliant",
                    "pick.top_down.max_attempts": 1,
                },
            ),
            log_dir=tmp_path,
            prompt_fn=_Operator(),
        )
        log = json.loads((tmp_path / "cycles.json").read_text())
        attempt = log["cycles"][0]["pick_attempts"][0]
        moves = {m["name"]: m for m in attempt["moves"]}
        assert moves["grasp"]["reached"] is closes, lift
        assert (attempt.get("note") != "grasp did not arrive") is closes, lift


def test_confirm_once_asks_a_single_question_for_the_whole_run(
    tmp_path, arm_log, caplog
):
    """confirm=once: one prompt before the first action; the picks, tosses, retracts
    and the reset then run without asking, each step still printed."""
    operator = _Operator()
    with caplog.at_level("INFO", logger="prpl_tidybot.toss_cycles"):
        result = run_toss_cycles(
            _compose(confirm="once"), log_dir=tmp_path, prompt_fn=operator
        )
    assert result.completed == 2
    assert len(operator.prompts) == 1
    steps = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("Step")
    ]
    assert any("Drive to the toss pose" in m for m in steps)
    assert any("retract the arm after the toss" in m for m in steps)
    assert len([e for e in arm_log if e[0] == "toss"]) == 2


def test_by_default_a_miss_retracts_returns_to_the_toss_pose_and_ends(
    tmp_path, arm_log
):
    """One attempt: a miss is not retried; the arm retracts and the base returns to
    the toss pose."""
    cfg = _compose(
        confirm="once",
        **{"pick.top_down.fake_grasp_reading": 0.996},
    )
    result = run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=_Operator())
    assert "pick failed after 1 attempts" in result.finish_reason
    assert not [e for e in arm_log if e[0] == "toss"]
    last = [e for e in arm_log if e[0] in ("arm", "angular")][-1]
    assert last[1] == pytest.approx(list(RETRACT_ARM_CONF))
    assert result.final_base_pose[:2] == pytest.approx(TOSS_POSE[:2], abs=0.02)


def test_a_missed_grasp_retries_then_stops(tmp_path, arm_log):
    """Readings that look like an empty gripper retry up to max_attempts, then stop."""
    operator = _Operator()
    cfg = _compose(
        **{
            "pick.top_down.fake_grasp_reading": 0.996,
            "pick.top_down.max_attempts": 2,
            "pick.top_down.ask_to_move_after_miss": 0,
        }
    )
    result = run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 0
    assert "pick failed after 2 attempts" in result.finish_reason
    assert operator.asked("Pick attempt") == 2
    assert not [e for e in arm_log if e[0] == "toss"]
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert [a["held"] for a in log["cycles"][0]["pick_attempts"]] == [False, False]


def test_a_missed_grasp_drives_back_to_the_toss_pose_to_relocalize(
    tmp_path, monkeypatch
):
    """Between attempts the arm retracts and the base returns to the toss pose, where
    the ceiling cameras see the cube unoccluded, before driving to a fresh standoff."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    drives: list[tuple[float, ...]] = []
    original = tc._drive_to  # pylint: disable=protected-access

    def drive_to(real_env, pose, *args):
        drives.append(tuple(round(v, 3) for v in pose))
        return original(real_env, pose, *args)

    monkeypatch.setattr(tc, "_drive_to", drive_to)
    operator = _Operator()
    cfg = _compose(
        **{
            "pick.top_down.fake_grasp_reading": 0.996,
            "pick.top_down.max_attempts": 2,
            "pick.top_down.ask_to_move_after_miss": 0,
        }
    )
    run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=operator)
    order = [
        p
        for p in operator.prompts
        if "standoff" in p or "Pick attempt" in p or "re-localize" in p
    ]
    assert ["standoff" in p for p in order] == [True, False, False, True, False]
    assert "re-localize" in order[2]
    toss = tuple(round(v, 3) for v in TOSS_POSE)
    assert drives[0] != toss and drives[1] == toss and drives[2] != toss


def test_skip_toss_stops_holding_the_cube(tmp_path, arm_log):
    """skip_toss ends after a successful pick, with no throw."""
    result = run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )
    assert result.finish_reason == "picked (toss skipped)"
    assert not [e for e in arm_log if e[0] == "toss"]


def test_place_at_sets_the_cube_down_on_the_target_and_lets_go(tmp_path, arm_log):
    """place_at carries the picked cube to a map (x, y) instead of throwing it: drive
    to that point's standoff, lower the closed gripper to the grasp height (the cube's
    bottom on the floor, so it is never dropped), open there, then lift."""
    result = run_toss_cycles(
        _compose(cycles=1, place_at="[0.3,0.0]"),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.finish_reason == "placed at (+0.300, +0.000)"
    assert not [e for e in arm_log if e[0] == "toss"]
    geom = _configured_geometry()
    place = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0]["place"]
    assert [m["name"] for m in place["moves"]] == [
        "place_hover",
        "place_down",
        "place_up",
    ]
    down = place["moves"][1]
    assert down["planned_pinch_z"] == pytest.approx(
        geom.floor_z + geom.grasp_height, abs=0.002
    )
    # The base parked so the target is where the arm reaches for every pick.
    assert place["arm_target_xy"][0] == pytest.approx(geom.reach, abs=0.03)
    assert place["arm_target_xy"][1] == pytest.approx(0.0, abs=0.03)
    # Closed on the way down; opened only at the bottom.
    moves = [e for e in arm_log if e[0] == "angular"][-3:]
    assert moves[1][2] == pytest.approx(1.0) and moves[2][2] == pytest.approx(0.0)


@pytest.mark.usefixtures("arm_log")
def test_the_kinder_pick_is_still_selectable(tmp_path):
    """pick.method=kinder runs the shelf3d PickController as before."""
    result = run_toss_cycles(
        _compose(cycles=1, **{"pick.method": "kinder"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.completed == 1


def test_the_real_cube_source_searches_the_graspable_range():
    """The ceiling cube localizer is built with the graspable search box, not the
    drivable box; other target sources (fake mode) are built unchanged."""
    from prpl_tidybot.toss_cycles import (  # pylint: disable=import-outside-toplevel
        _cube_source_kwargs,
    )

    cfg = _compose()
    real = {"_target_": "prpl_tidybot.cube_localizer.CeilingCubeTargetSource"}
    bounds = _cube_source_kwargs(cfg, real)["floor_bounds"]
    assert bounds[3] > 1.84 and bounds[0] < cfg.preview.floor_bounds[0]
    fake = {
        "_target_": "prpl_tidybot.real_sim.perceivers.target_source.ConstantTargetSource"
    }
    assert not _cube_source_kwargs(cfg, fake)


def test_the_real_cube_source_saves_failed_localizations_in_the_run_log(tmp_path):
    """A missed localization lands beside the run's own logs."""
    from prpl_tidybot.toss_cycles import (  # pylint: disable=import-outside-toplevel
        _cube_source_kwargs,
    )

    cfg = _compose()
    real = {"_target_": "prpl_tidybot.cube_localizer.CeilingCubeTargetSource"}
    kwargs = _cube_source_kwargs(cfg, real, tmp_path)
    assert kwargs["failure_dir"] == str(tmp_path / "localization_failures")


def test_a_missed_grasp_tries_the_other_cameras_estimate(tmp_path, monkeypatch):
    """Josh, 2026-10-15: when both ceiling cameras see the cube, grasp at the more
    confident estimate first and, if that misses, at the other one, before giving up."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    readings = iter([1.0, 0.49])  # fully closed on nothing, then closed on the cube
    monkeypatch.setattr(tc, "_fake_reading", lambda _v, _r: next(readings))
    monkeypatch.setattr(
        ConstantTargetSource,
        "alternates",
        lambda self: [(self.get_target()[0] + 0.03, self.get_target()[1], 0.0)],
        raising=False,
    )
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true", **{"pick.top_down.max_attempts": 1}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    log = json.loads((tmp_path / "cycles.json").read_text())
    tries = log["cycles"][0]["pick_attempts"]
    assert [t["held"] for t in tries] == [False, True]
    assert tries[1]["cube"]["x"] == pytest.approx(tries[0]["cube"]["x"] + 0.03)
    assert tries[1]["target"] == "other camera"


def test_the_top_down_pick_builds_no_planner_models(tmp_path, monkeypatch):
    """The KINDER planner's env models (~480 MB each, never freed) are only needed by
    the kinder pick; building them on every top-down run ran CI out of memory."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    def refuse(*_args, **_kwargs):
        raise AssertionError("the top-down pick built the planner's env models")

    monkeypatch.setattr(tc, "build_planner_env_models", refuse)
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )


def test_each_cycle_throws_the_next_setting_of_the_toss_grid(tmp_path, arm_log):
    """A calibration grid: cycle k throws grid setting k, and each cycle's record
    carries the setting it threw and when the swing started (to align other video)."""
    grid = "[[100,500],[120,675],[140,850]]"
    run_toss_cycles(
        _compose(cycles=3, **{"+toss.grid": grid}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    tosses = [kw for kind, kw, *_ in arm_log if kind == "toss"]
    assert [(t["max_vel"], t["gripper_release_ms"]) for t in tosses] == [
        (100.0, 500),
        (120.0, 675),
        (140.0, 850),
    ]
    log = json.loads((tmp_path / "cycles.json").read_text())
    for cycle, (speed, release) in zip(
        log["cycles"], [(100, 500), (120, 675), (140, 850)]
    ):
        assert cycle["toss"]["max_vel_deg_s"] == pytest.approx(speed)
        assert cycle["toss"]["gripper_release_ms"] == release
        assert cycle["toss"]["swing_start_time"] > 0


def test_every_toss_can_ask_even_when_the_run_confirms_once(tmp_path):
    """Untried throws (the grid's corners) are confirmed one by one, while the rest of
    the run still asks once."""
    operator = _Operator()
    run_toss_cycles(
        _compose(cycles=2, confirm="once", **{"+toss.confirm_each": "true"}),
        log_dir=tmp_path,
        prompt_fn=operator,
    )
    assert operator.asked("Toss?") == 2
    assert operator.asked("without asking again") == 1


def test_a_cube_that_ends_up_in_the_bin_is_put_back_by_the_person(
    tmp_path, monkeypatch
):
    """The pick's hover is below the bin's rim, so a cube in (or against) the bin is
    not picked: the person is asked to put it on the floor, and the record says so."""
    # Cycle 1's pick localizes twice (before and after its drive); the check before
    # cycle 2's pick finds the thrown cube in the bin.
    spots = iter([(0.5, 0.3, 0.025), (0.5, 0.3, 0.025), (0.13, 1.68, 0.025)])

    def get_target(_self):
        return next(spots, (0.5, 0.3, 0.025))

    monkeypatch.setattr(ConstantTargetSource, "get_target", get_target)
    operator = _Operator()
    run_toss_cycles(
        _compose(cycles=2, **{"bin_stop_box": "[-0.13,0.39,1.45,1.91]"}),
        log_dir=tmp_path,
        prompt_fn=operator,
    )
    assert operator.asked("in the bin") == 1
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert log["cycles"][1]["cube_in_bin"] == {"x": 0.13, "y": 1.68}


def test_a_cube_seen_inside_the_bin_box_is_put_back_by_the_person(
    tmp_path, monkeypatch
):
    """A cube inside the bin lies beyond the floor search and in shadow (run 31), so the
    in-bin check asks the source to look inside the bin box first."""
    monkeypatch.setattr(
        ConstantTargetSource,
        "find_in_box",
        lambda _self, box: (0.33, 2.02),
        raising=False,
    )
    operator = _Operator()
    run_toss_cycles(
        _compose(cycles=2, **{"bin_stop_box": "[0.02,0.44,1.85,2.05]"}),
        log_dir=tmp_path,
        prompt_fn=operator,
    )
    assert operator.asked("in the bin") == 1
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert log["cycles"][1]["cube_in_bin"] == {"x": 0.33, "y": 2.02}


def test_the_toss_logs_when_its_swing_started_not_when_it_was_called(tmp_path):
    """The arm server's toss winds up first (seconds), then swings; the record carries
    the swing's own start, so the release is swing start + release ms."""
    run_toss_cycles(
        _compose(cycles=1, **{"+toss.grid": "[[140,675]]"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    toss = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0]["toss"]
    assert toss["swing_start_time"] >= toss["toss_call_time"]
    assert toss["release_time"] == pytest.approx(toss["swing_start_time"] + 0.675)


def test_a_missed_pick_asks_the_person_to_move_the_cube_then_retries(
    tmp_path, monkeypatch
):
    """Josh, 2026-10-16: on a missed pick, ask the person to move the cube (e.g. out
    of the camera overlap strip) and try again, instead of ending the run."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    readings = iter([1.0, 0.49])  # closed on nothing, then on the cube
    monkeypatch.setattr(tc, "_fake_reading", lambda _v, _r: next(readings))
    operator = _Operator()
    summary = run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=operator
    )
    assert operator.asked("move the cube") == 1
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert [a["held"] for a in log["cycles"][0]["pick_attempts"]] == [False, True]
    assert log["cycles"][0]["cube_moved_by_person"] == 1
    assert "picked" in summary.finish_reason


def test_a_cube_the_cameras_cannot_find_asks_the_person_to_move_it(
    tmp_path, monkeypatch
):
    """Josh, 2026-10-16: when the cube can't be found (e.g. beside the robot, inside its
    exclusion zone), ask the person to move it and try again, like a missed pick."""
    original = ConstantTargetSource.get_target
    calls = {"n": 0}

    def get_target(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CeilingCubeTargetSource: no cube found")
        return original(self)

    monkeypatch.setattr(ConstantTargetSource, "get_target", get_target)
    operator = _Operator()
    summary = run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=operator
    )
    assert operator.asked("can't find the cube") == 1
    assert summary.finish_reason == "picked (toss skipped)"
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert log["cycles"][0]["cube_moved_by_person"] == 1
