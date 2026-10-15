"""Tests for `prpl_tidybot.toss_cycles.run_toss_cycles` (scripts/toss_cycles.py) in fake
mode: pick -> arm home -> drive to the toss pose -> toss, repeated, with a confirmation
before every action and a reset (retract, back to the toss pose) whenever the run
stops."""

import json
import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
TOSS_POSE = (1.326, -0.493, -3.132)


def _compose(**extra):
    overrides = ["mode=fake", "seed=0", "interactive=true"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="toss_cycles", overrides=overrides)


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
        original_toss(self, **kwargs)

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
    for word in ("pick", "home", "Drive to the toss pose", "Toss?"):
        assert operator.asked(word) >= 2, word
    assert operator.asked("Reset") == 1
    last_arm = [e for e in arm_log if e[0] == "arm"][-1]
    assert last_arm[1] == pytest.approx(list(RETRACT_ARM_CONF))
    x, y, theta = result.final_base_pose
    assert (x, y) == pytest.approx(TOSS_POSE[:2], abs=0.02)
    assert math.cos(theta - TOSS_POSE[2]) > math.cos(0.05)
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert [c["cycle"] for c in log["cycles"]] == [1, 2]
    assert all(c["tossed"] for c in log["cycles"])
    assert log["toss_pose"] == pytest.approx(list(TOSS_POSE))


def test_arm_goes_home_holding_the_cube_before_the_drive(tmp_path, arm_log):
    """After the pick the arm is sent to its home joints with the gripper still closed,
    and the toss comes after."""
    run_toss_cycles(_compose(cycles=1), log_dir=tmp_path, prompt_fn=_Operator())
    toss_at = next(i for i, e in enumerate(arm_log) if e[0] == "toss")
    holding = [e for e in arm_log[:toss_at] if e[0] == "arm" and e[2] > 0.5]
    assert holding, "the arm should move with the gripper closed before the toss"


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
    geom = TopDownPickGeometry()
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
        0.10, abs=0.005
    )


def test_the_pick_runs_under_the_onboard_controller_by_default(tmp_path, arm_log):
    """Hover, grasp, lift and carry are onboard-controller moves (the compliant loop
    settled 24 mm short on the robot), with the torque loop released around them."""
    run_toss_cycles(
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=_Operator()
    )
    angular = [e for e in arm_log if e[0] == "angular"]
    assert len(angular) == 4
    log = json.loads((tmp_path / "cycles.json").read_text())
    moves = log["cycles"][0]["pick_attempts"][0]["moves"]
    assert all(m["controller"] == "angular" for m in moves)


def test_the_compliant_pick_is_still_selectable(tmp_path, arm_log):
    """pick.top_down.arm_controller=compliant streams the moves as before."""
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
    cfg = _compose(cycles=1, skip_toss="true", **{"pick.top_down.max_attempts": 1})
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
        _compose(cycles=1, skip_toss="true"), log_dir=tmp_path, prompt_fn=operator
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


def test_a_throw_from_the_toss_pose_lands_where_it_can_be_found_again():
    """The first real throw (140 deg/s, 600 ms) came to rest 2.34 m from the base,
    0.31 m beyond the localizer's search box, so the cube was lost. From the toss
    pose that throw must land at least 0.3 m inside the box, and the base (its
    collision box reaches 0.274 m behind its centre) must stop 0.1 m inside the box's
    back edge."""
    cfg = _compose()
    x, _, heading = cfg.toss_pose
    x_min, x_max = cfg.preview.floor_bounds[0], cfg.preview.floor_bounds[1]
    assert math.cos(heading) < -0.99  # throwing towards -x
    assert x - 2.34 >= x_min + 0.3
    assert x + 0.274 == pytest.approx(x_max - 0.1, abs=0.005)


def test_a_missed_grasp_retries_then_stops(tmp_path, arm_log):
    """Readings that look like an empty gripper retry up to max_attempts, then stop."""
    operator = _Operator()
    cfg = _compose(
        **{"pick.top_down.fake_grasp_reading": 0.996, "pick.top_down.max_attempts": 2}
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

    def drive_to(real_env, pose):
        drives.append(tuple(round(v, 3) for v in pose))
        return original(real_env, pose)

    monkeypatch.setattr(tc, "_drive_to", drive_to)
    operator = _Operator()
    cfg = _compose(
        **{"pick.top_down.fake_grasp_reading": 0.996, "pick.top_down.max_attempts": 2}
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


@pytest.mark.usefixtures("arm_log")
def test_the_kinder_pick_is_still_selectable(tmp_path):
    """pick.method=kinder runs the shelf3d PickController as before."""
    result = run_toss_cycles(
        _compose(cycles=1, **{"pick.method": "kinder"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.completed == 1
