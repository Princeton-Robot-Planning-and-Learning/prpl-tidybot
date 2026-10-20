"""Tests for the Tossing3D skill server: the skills on the fake robot in the real room
(conf/skill_server.yaml), the contract's shape, the fake robot planners test against,
and a loopback RPC."""

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from spatialmath import SE2

from prpl_tidybot.aimed_toss import Barrier, Workspace, footprint
from prpl_tidybot.bin_localizer import BinFit, ConstantBinSource
from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_tidybot.robot_skills import InBinPick
from prpl_tidybot.skill_server import (
    BIN_NOT_LOCALIZED,
    FakeTossingRobot,
    SkillBusy,
    SkillStopped,
    TossingRobotClient,
    build_skills,
    serve,
)
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import geometry_from_config
from prpl_tidybot.visual_servo.top_down_cube import TopDownLateralServo

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
CONTRACT_KEYS = {
    "robot",
    "cube",
    "bin",
    "holding",
    "gripper",
    "gripper_command",
    "cube_upright",
    "cube_in_bin",
    "cube_inside_bin",
    "barrier",
    "barrier_x",
    "toss_direction",
    "can_pick",
}
CAN_PICK_KEYS = {"feasible", "note", "refused", "reason"}
HOME = (1.10, -0.78, math.pi)  # the close-side toss stand, the server's home
CLOSE_BIN = (-0.29, -0.77)  # the fake bin: the close spot, long side along y
FAR_BIN = (0.24, 1.40)  # the far spot's measured box x 0.03..0.45, y 1.24..1.56
FLOOR_CUBE = "[0.6,-0.5,0.025]"
IN_BIN = "[-0.27,-0.75,0.025]"
# The measured tape's robot-side face: y = 0.349 + 0.0267 x, the robot on -y.
TAPE = Barrier(
    x=0.0,
    y=0.374,
    length=6.0,
    thickness=0.05,
    yaw=0.026694,
    robot_side=(0.026690, -0.999644),
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No real-time ticks for the fake arm."""
    monkeypatch.setattr("prpl_tidybot.robot_skills.POLICY_CONTROL_PERIOD", 0.0)


@pytest.fixture(autouse=True)
def _home_start(monkeypatch):
    """The fake base starts at the server's home, the close-side toss stand."""
    original = FakeBaseInterface.__init__

    def init(self):
        original(self)
        self.base_state = SE2(*HOME)
        self.map_base_state = SE2(*HOME)

    monkeypatch.setattr(FakeBaseInterface, "__init__", init)


def _cfg(**extra):
    overrides = ["mode=fake", "settle_after_toss_s=0", f"env.cube_pose={FLOOR_CUBE}"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=overrides)


@pytest.fixture(name="skills")
def _skills():
    return build_skills(_cfg())


def _at_far_spot(skills):
    """Move the fake bin to the far spot (across the barrier)."""
    b = skills.cfg.bin
    skills.bin_source = ConstantBinSource(
        FAR_BIN[0], FAR_BIN[1], 0.0, b.length, b.width, b.height, b.wall
    )


def _still(skills):
    return (list(skills.arm.get_arm_state()), skills.observe()["robot"])


def _base(skills):
    """The base's map pose, without a look."""
    return skills._robot()  # pylint: disable=protected-access


class _CountingBins:
    """The skills' bin source, counting its localizations; ``found`` False: no bin."""

    def __init__(self, inner, found=True):
        self.inner = inner
        self.found = found
        self.calls = 0

    def localize(self):
        """The inner source's bin, or the ceiling source's miss."""
        self.calls += 1
        if not self.found:
            raise RuntimeError("CeilingBinSource: no bin found in the search box")
        return self.inner.localize()

    def close(self):
        """Nothing to release."""


def _count_cube_looks(monkeypatch):
    """Count the cube source's floor localizations."""
    looks = []
    original = ConstantTargetSource.get_target

    def get_target(self):
        looks.append(True)
        return original(self)

    monkeypatch.setattr(ConstantTargetSource, "get_target", get_target)
    return looks


def _missing(skills, **cfg):
    """Skills whose gripper closes on nothing (0.996: the empty reading)."""
    del skills
    return build_skills(_cfg(**{"pick.top_down.fake_grasp_reading": 0.996, **cfg}))


# ------------------------------------------------------------------ the real room


def test_the_server_config_is_the_real_room():
    """The measured tape, the measured bin searched for at both spots, the pick out
    of the bin on, the lateral servo on, the compliant arm
    (the toss keeps its own swing), and each spot's fixed toss direction."""
    cfg = _cfg()
    workspace = Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds)
    for x in (-1.5, 0.23, 1.7):
        assert workspace.barrier.depth((x, 0.349 + 0.0267 * x)) == pytest.approx(
            0.0, abs=0.002
        )
    assert workspace.barrier.depth((0.0, 0.0)) > 0  # the robot is on -y
    assert (cfg.bin.length, cfg.bin.width, cfg.bin.height) == (0.39, 0.29, 0.21)
    assert cfg.bin.min_saturation == 130.0
    x0, x1, y0, y1 = cfg.bin.search_bounds
    for spot in (CLOSE_BIN, FAR_BIN):
        assert x0 < spot[0] < x1 and y0 < spot[1] < y1, spot
    td = cfg.pick.top_down
    assert td.arm_controller == "compliant"
    assert td.lateral_servo.enabled and "wrist_servo" not in td
    assert td.in_bin.enabled
    assert cfg.toss.spot_directions.robot_side == pytest.approx(0.0)
    assert cfg.toss.spot_directions.far_side == pytest.approx(-math.pi / 2)
    assert list(cfg.toss_pose) == pytest.approx(list(HOME))


def test_with_a_wrist_camera_the_server_uses_the_lateral_servo():
    """The servo that worked on the robot on 2026-10-20 (runs 37-40)."""
    skills = build_skills(_cfg(), wrist_camera=lambda arm: object())
    assert isinstance(skills.servo, TopDownLateralServo)


# ------------------------------------------------------------------ observe


def test_observe_has_exactly_the_contracts_fields_as_plain_values(skills):
    """Every field, JSON-serialisable; the barrier is the measured tape."""
    obs = skills.observe()
    assert set(obs) == CONTRACT_KEYS
    json.dumps(obs)
    assert obs["robot"] == pytest.approx(list(HOME))
    assert obs["cube"][:2] == pytest.approx([0.6, -0.5])
    assert set(obs["bin"]) == {"x", "y", "yaw", "length", "width", "height"}
    assert obs["barrier"] == pytest.approx(TAPE.as_dict())
    assert obs["barrier_x"] == 0.0
    assert obs["holding"] is False
    assert obs["gripper_command"] == 0.0
    assert obs["cube_upright"] is True
    assert obs["cube_in_bin"] is False and obs["cube_inside_bin"] is False
    assert obs["toss_direction"] == pytest.approx(0.0)


def test_observe_looks_for_the_cube_inside_the_bin_first(skills, monkeypatch):
    """Inside the bin the cube is dim and beyond the floor search (run 31): the bin's
    box is searched first; the cube found there is inside the bin and, 2 cm from its
    centre, in its scoring region too."""
    looked = []

    def find_pose_in_box(_self, box):
        looked.append(box)
        return (CLOSE_BIN[0] + 0.02, CLOSE_BIN[1], 0.3)

    monkeypatch.setattr(
        ConstantTargetSource, "find_pose_in_box", find_pose_in_box, raising=False
    )
    obs = skills.observe()
    assert looked and looked[0][0] < CLOSE_BIN[0] < looked[0][1]
    assert obs["cube"][:2] == pytest.approx([CLOSE_BIN[0] + 0.02, CLOSE_BIN[1]])
    assert obs["cube_inside_bin"] is True and obs["cube_in_bin"] is True


def test_a_cube_inside_the_bin_but_against_its_wall_does_not_score(skills):
    """cube_in_bin is Tossing3D's scoring region (5 cm inside the walls);
    cube_inside_bin is anywhere inside them."""
    skills.cube_source = ConstantTargetSource(CLOSE_BIN[0], CLOSE_BIN[1] + 0.17, 0.025)
    obs = skills.observe()
    assert obs["cube_inside_bin"] is True and obs["cube_in_bin"] is False


def test_the_far_spot_tosses_from_minus_y(skills):
    """A bin across the barrier is the far spot, thrown at from -y."""
    _at_far_spot(skills)
    assert skills.observe()["toss_direction"] == pytest.approx(-math.pi / 2)


# ------------------------------------------------------------------ can_pick, pick


def test_can_pick_is_a_dry_run(skills):
    """Feasible for a floor cube on the robot's side; nothing moves."""
    before = _still(skills)
    reply = skills.can_pick()
    assert reply["feasible"] is True and reply["note"] is None
    assert reply["refused"] is False and reply["reason"] is None
    assert _still(skills) == before


def test_can_pick_refuses_a_cube_across_the_barrier():
    """The robot never crosses the barrier, so that cube is unpickable."""
    skills = build_skills(_cfg(**{"env.cube_pose": "[0.3,0.9,0.025]"}))
    reply = skills.can_pick()
    assert reply["feasible"] is False and reply["note"] == "cube past the barrier"


def test_a_refused_pick_moves_nothing_and_says_why():
    """A pick that may not start (past the barrier) returns refused with the reason,
    and nothing moved: distinguishable from a miss."""
    skills = build_skills(_cfg(**{"env.cube_pose": "[0.3,0.9,0.025]"}))
    before = _still(skills)
    reply = skills.pick()
    assert reply["held"] is False and reply["refused"] is True
    assert reply["reason"] == "cube past the barrier" == reply["note"]
    assert _still(skills) == before


def test_a_cube_in_the_far_bin_is_refused_for_the_person(skills, monkeypatch):
    """The far spot is across the barrier: a cube thrown into it is the person's to
    put back (a human reset), never picked."""
    _at_far_spot(skills)
    monkeypatch.setattr(
        ConstantTargetSource,
        "find_pose_in_box",
        lambda _self, box: (FAR_BIN[0], FAR_BIN[1], 0.0),
        raising=False,
    )
    before = _still(skills)
    reply = skills.pick()
    assert reply["refused"] is True and "barrier" in reply["reason"]
    assert reply["cube"]["in_bin"] is True
    assert _still(skills) == before


def test_a_cube_against_the_bin_wall_is_refused_before_moving():
    """4 cm from the inner wall no grasp clears it: refused, nothing moved."""
    cube = f"[{CLOSE_BIN[0] + 0.10},{CLOSE_BIN[1]},0.025]"
    skills = build_skills(_cfg(**{"env.cube_pose": cube}))
    before = _still(skills)
    reply = skills.pick()
    assert reply["refused"] is True and "bin wall" in reply["reason"]
    assert _still(skills) == before


def test_no_cube_is_a_refusal(skills, monkeypatch):
    """Nothing to pick: refused, and the localizer's reason is kept."""

    def no_cube(_self):
        raise RuntimeError("CeilingCubeTargetSource: no cube found")

    monkeypatch.setattr(ConstantTargetSource, "get_target", no_cube)
    reply = skills.pick()
    assert reply["refused"] is True and "no cube found" in reply["reason"]


def test_a_miss_is_not_a_refusal():
    """The gripper closing on nothing is a real miss: the robot moved (it drove to the
    standoff and back), refused is False, and the note says why."""
    skills = build_skills(_cfg(**{"pick.top_down.fake_grasp_reading": 0.996}))
    reply = skills.pick()
    assert reply["held"] is False and reply["refused"] is False
    assert reply["reason"] is None and reply["note"]
    assert reply["stand"] != pytest.approx(list(HOME))


# ------------------------------------------------------- one look, and no bin


def test_observe_says_whether_the_pick_could_start_from_the_same_look(
    skills, monkeypatch
):
    """Observe() localizes the bin and the cube once and answers can_pick from that
    look, exactly as can_pick() decides it; nothing moves."""
    skills.bin_source = _CountingBins(skills.bin_source)
    looks = _count_cube_looks(monkeypatch)
    before = (
        list(skills.arm.get_arm_state()),
        _base(skills),
    )
    obs = skills.observe()
    assert skills.bin_source.calls == 1 and len(looks) == 1
    assert set(obs["can_pick"]) == CAN_PICK_KEYS
    assert obs["can_pick"] == skills.can_pick()
    assert obs["can_pick"]["feasible"] is True
    json.dumps(obs)
    assert (
        list(skills.arm.get_arm_state()),
        _base(skills),
    ) == before


@pytest.mark.parametrize(
    "cube, reason",
    [("[0.3,0.9,0.025]", "cube past the barrier"), (None, "bin wall")],
)
def test_observes_can_pick_refuses_what_can_pick_refuses(cube, reason):
    """Past the barrier, against the bin wall: the same refusal, the same reason."""
    cube = cube or f"[{CLOSE_BIN[0] + 0.10},{CLOSE_BIN[1]},0.025]"
    skills = build_skills(_cfg(**{"env.cube_pose": cube}))
    answer = skills.observe()["can_pick"]
    assert answer["refused"] is True and reason in answer["reason"]
    assert answer == skills.can_pick()


def test_observe_while_holding_says_the_pick_cannot_start(skills):
    """Holding: can_pick's own first check."""
    assert skills.pick()["held"]
    assert skills.observe()["can_pick"]["reason"] == "already holding the cube"


def test_a_pick_after_observe_plans_from_its_look(skills, monkeypatch):
    """The pick reuses observe's bin and cube: its only new look is the cube's, at the
    standoff."""
    skills.bin_source = _CountingBins(skills.bin_source)
    looks = _count_cube_looks(monkeypatch)
    skills.observe()
    reply = skills.pick()
    assert reply["held"] is True
    assert skills.bin_source.calls == 1
    assert len(looks) == 2


@pytest.mark.parametrize(
    "motion", ["retract", "go_home", "open_gripper", "stop", "toss"]
)
def test_any_motion_skill_invalidates_the_look(skills, monkeypatch, motion):
    """A look taken before a motion is never reused: the pick localizes afresh."""
    skills.observe()
    if motion == "toss":
        assert skills.pick()["held"]
        skills.observe()
        skills.toss(1.2, 120.0, 700.0, 0.0)
    else:
        getattr(skills, motion)()
    skills.bin_source = _CountingBins(skills.bin_source)
    looks = _count_cube_looks(monkeypatch)
    assert skills.pick()["held"] is True
    assert skills.bin_source.calls == 1 and len(looks) == 2


def test_without_a_bin_can_pick_and_pick_refuse_before_moving(skills):
    """A bin the cameras miss is never planned around as no bin (that skips the arm's
    check against its walls): observe, can_pick and pick all refuse, nothing moves."""
    skills.bin_source = _CountingBins(skills.bin_source, found=False)
    before = (
        list(skills.arm.get_arm_state()),
        _base(skills),
    )
    obs = skills.observe()
    assert obs["bin"] is None
    assert obs["can_pick"]["refused"] is True
    assert obs["can_pick"]["reason"] == BIN_NOT_LOCALIZED == "bin not localized"
    assert skills.can_pick()["reason"] == BIN_NOT_LOCALIZED
    reply = skills.pick()
    assert reply["refused"] is True and reply["held"] is False
    assert reply["reason"] == BIN_NOT_LOCALIZED
    assert (
        list(skills.arm.get_arm_state()),
        _base(skills),
    ) == before


def test_with_no_bin_configured_the_pick_still_runs(skills):
    """No bin source (no bin in the config): nothing to localize, so nothing to refuse
    for."""
    skills.bin_source = None
    assert skills.observe()["can_pick"]["feasible"] is True
    assert skills.pick()["held"] is True


def test_the_real_bin_source_saves_missed_localizations_in_the_run_log(tmp_path):
    """build_skills hands the ceiling bin source the run's localization_failures."""
    from prpl_tidybot.skill_server import (  # pylint: disable=import-outside-toplevel
        _bin_source_kwargs,
    )

    real = {"_target_": "prpl_tidybot.bin_localizer.CeilingBinSource"}
    kwargs = _bin_source_kwargs(real, tmp_path)
    assert kwargs["failure_dir"] == str(tmp_path / "localization_failures")
    assert not _bin_source_kwargs(real, None)
    assert not _bin_source_kwargs(
        {"_target_": "prpl_tidybot.bin_localizer.ConstantBinSource"}, tmp_path
    )


# ------------------------------------------------------- a missed pick retries


def test_a_missed_pick_retreats_relocalizes_and_retries(skills, monkeypatch):
    """After a miss the robot retracts and drives back to its home stand (where the
    ceiling cameras see the cube without the robot over it), localizes the bin and
    the cube afresh and tries again: pick.max_attempts (2) in all. Missing both, it
    ends retreated, not over the cube."""
    skills = _missing(skills)
    drives = []
    import prpl_tidybot.skill_server as ss  # pylint: disable=import-outside-toplevel

    original = ss._drive_to  # pylint: disable=protected-access

    def drive_to(real_env, pose, *args):
        drives.append([round(float(v), 3) for v in pose])
        return original(real_env, pose, *args)

    monkeypatch.setattr(ss, "_drive_to", drive_to)
    skills.bin_source = _CountingBins(skills.bin_source)
    reply = skills.pick()
    assert reply["held"] is False and reply["refused"] is False and reply["note"]
    assert [a["attempt"] for a in reply["attempts"]] == [1, 2]
    assert [a["held"] for a in reply["attempts"]] == [False, False]
    home = [round(v, 3) for v in HOME]
    assert drives[0] != home and drives[1] == home
    assert drives[2] != home and drives[3] == home and len(drives) == 4
    assert skills.bin_source.calls == 2  # the plan, then again after the retreat
    assert _base(skills) == pytest.approx(HOME, abs=0.02)
    assert list(skills.arm.get_arm_state()) == pytest.approx(
        list(RETRACT_ARM_CONF), abs=0.06
    )
    assert skills.gripper_command == 0.0
    json.dumps(reply)


def test_a_retry_that_holds_is_a_held_pick(skills):
    """Miss, retreat, re-localize, hold: the reply is held, with both attempts."""
    readings = iter([0.996, 0.49])
    skills._read = lambda _reading: next(readings)  # pylint: disable=protected-access
    reply = skills.pick()
    assert reply["held"] is True and reply["note"] is None
    assert [a["held"] for a in reply["attempts"]] == [False, True]
    assert skills.gripper_command == 1.0


def test_the_number_of_attempts_is_configurable(skills):
    """pick.max_attempts in the config; an in-process caller (toss_cycles, which asks
    its operator between attempts) may pass its own."""
    three = _missing(skills, **{"pick.max_attempts": 3})
    assert len(three.pick()["attempts"]) == 3
    once = _missing(skills)
    reply = once.pick(max_attempts=1)
    assert len(reply["attempts"]) == 1
    assert _base(once) == pytest.approx(HOME, abs=0.02)


def test_a_retry_the_new_look_refuses_ends_retreated(skills):
    """After the retreat the bin is not found: the retry does not start, the robot stays
    retreated, and the reply is a miss (it moved), saying why."""
    skills = _missing(skills)
    inner = skills.bin_source

    class _FirstOnly(_CountingBins):
        def localize(self):
            self.found = self.calls == 0
            return super().localize()

    skills.bin_source = _FirstOnly(inner)
    reply = skills.pick()
    assert reply["held"] is False and reply["refused"] is False
    assert BIN_NOT_LOCALIZED in reply["note"]
    assert [a["attempt"] for a in reply["attempts"]] == [1, 2]
    assert reply["attempts"][1]["held"] is False
    assert _base(skills) == pytest.approx(HOME, abs=0.02)


def test_pick_off_the_floor_is_compliant_and_bin_aware(skills, monkeypatch):
    """The floor pick runs every arm move under the compliant loop, and the standoff
    it drives to keeps the base clear of the bin and the barrier."""
    angular = []
    monkeypatch.setattr(
        FakeArmInterface, "move_angular", lambda self, goal: angular.append(goal)
    )
    reply = skills.pick()
    assert reply["held"] is True and reply["refused"] is False
    assert not angular
    moves = reply["attempts"][-1]["moves"]
    assert {m["controller"] for m in moves} == {"compliant"}
    robot = skills.observe()["robot"]
    fit = BinFit(*CLOSE_BIN, math.pi / 2, 0.39, 0.29, 0.21, 0.005)
    assert skills.workspace.pose_problem(robot, fit) is None


def test_pick_out_of_the_bin_goes_through_the_waypoint_above_the_rim():
    """A cube inside the bin is picked: above the rim, hover, grasp, lift, above the
    rim, carry; one of the cube's two pairs of faces is chosen."""
    skills = build_skills(_cfg(**{"env.cube_pose": IN_BIN}))
    reply = skills.pick()
    assert reply["held"] is True, reply["note"]
    assert reply["cube"]["in_bin"] is True
    attempt = reply["attempts"][-1]
    names = [m["name"] for m in attempt["moves"]]
    assert names[0] == "above" and names[-2:] == ["above", "carry"]
    assert "closing_turn_deg" in attempt
    json.dumps(reply)


def test_pick_then_toss_into_the_bin(skills):
    """Pick holds the cube; the toss drives to the aimed pose, throws with the given
    speed and release, retracts, and reports where the cube landed."""
    assert skills.pick()["held"] is True
    assert skills.gripper_command == 1.0
    tosses = []
    original = FakeArmInterface.toss

    def toss(self, **kwargs):
        tosses.append(kwargs)
        return original(self, **kwargs)

    FakeArmInterface.toss = toss  # restored below
    try:
        out = skills.toss(1.2, 120.0, 700.0, 0.0)
    finally:
        FakeArmInterface.toss = original
    assert out["tossed"] is True
    assert tosses == [{"heading": 0.0, "max_vel": 120.0, "gripper_release_ms": 700}]
    assert skills.gripper_command == 0.0
    robot = skills.observe()["robot"]
    assert robot[:2] == pytest.approx([CLOSE_BIN[0] + 1.2, CLOSE_BIN[1]], abs=0.02)
    assert min(TAPE.depth(c) for c in footprint(robot)) >= 0.05 - 1e-6
    assert {"tossed", "landed", "in_bin", "note"} <= set(out)
    assert out["pose"][:2] == pytest.approx(robot[:2], abs=0.02)
    assert out["direction"] == pytest.approx(0.0)
    json.dumps(out)


@pytest.mark.parametrize(
    "spot, stand",
    [(CLOSE_BIN, (CLOSE_BIN[0] + 1.39, CLOSE_BIN[1], math.pi)), (FAR_BIN, None)],
)
def test_without_a_direction_the_toss_stands_on_the_spots_fixed_side(
    skills, spot, stand
):
    """The real robot throws at the close spot from +x and at the far spot from -y:
    toss(direction=None) stands 1.39 m from the bin on that side, facing it."""
    if spot == FAR_BIN:
        _at_far_spot(skills)
        stand = (FAR_BIN[0], FAR_BIN[1] - 1.39, math.pi / 2)
    assert skills.pick()["held"]
    out = skills.toss(1.39, 120.0, 675.0, None)
    assert out["tossed"] is True
    assert out["pose"][:2] == pytest.approx(stand[:2], abs=1e-6)
    assert math.cos(out["pose"][2] - stand[2]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "args",
    [
        (1.2, 150.0, 700.0, 0.0),
        (1.2, 120.0, 400.0, 0.0),
        (2.0, 120.0, 700.0, 0.0),
        (1.2, 120.0, 700.0, math.pi / 2),  # across the barrier
    ],
)
def test_toss_raises_and_moves_nothing_outside_the_bounds_or_across_the_barrier(
    skills, args
):
    """ValueError before any motion, holding or not."""
    skills.pick()
    before = _still(skills)
    with pytest.raises(ValueError):
        skills.toss(*args)
    assert _still(skills) == before


def test_toss_without_the_cube_does_not_move(skills):
    """Nothing to throw: reported, nothing moves."""
    out = skills.toss(1.2, 120.0, 700.0, 0.0)
    assert out["tossed"] is False and out["landed"] is None
    assert out["in_bin"] is False and out["note"] == "not holding the cube"


def test_open_gripper_retract_and_go_home(skills):
    """The small skills: gripper open (commanded 0), retract, home pose."""
    skills.pick()
    skills.open_gripper()
    assert skills.gripper_command == 0.0
    skills.retract()
    skills.go_home()
    assert skills.observe()["robot"] == pytest.approx(list(HOME), abs=0.02)


def test_stop_halts_the_arm_and_holds_the_base(skills):
    """stop() calls the arm's Stop and sends the base its own pose."""
    halted = []
    skills._halt_arm = lambda: halted.append(True)  # pylint: disable=protected-access
    skills.stop()
    assert halted == [True]
    assert skills.stop_event.is_set()


def test_a_stopped_drive_ends_the_skill_with_an_error(skills, monkeypatch):
    """A stop during the drive halts it: the skill raises SkillStopped (never a
    KeyboardInterrupt, which would kill the RPC thread)."""
    skills.pick()  # away from home, so go_home drives
    original = skills._moving  # pylint: disable=protected-access

    class _KeepStop:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            skills.stop_event.set()

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(skills, "_moving", lambda: _KeepStop(original()))
    with pytest.raises(SkillStopped):
        skills.go_home()


def test_one_skill_moves_at_a_time(skills):
    """A second motion skill while one runs is refused."""
    with skills._moving():  # pylint: disable=protected-access
        with pytest.raises(SkillBusy):
            skills.retract()


# ------------------------------------------------------------------ recording


def test_every_call_is_logged_with_its_arguments_reply_and_timing(tmp_path):
    """skill_calls.jsonl: one line per call, in order, with the arguments, the reply
    (or the error) and how long it took."""
    skills = build_skills(_cfg(), log_dir=tmp_path)
    skills.observe()
    skills.pick()
    with pytest.raises(ValueError):
        skills.toss(2.0, 120.0, 700.0, 0.0)
    skills.toss(1.2, 120.0, 700.0, None)
    skills.close()
    lines = [
        json.loads(line)
        for line in (tmp_path / "skill_calls.jsonl").read_text().splitlines()
    ]
    assert [line["call"] for line in lines] == ["observe", "pick", "toss", "toss"]
    assert [line["i"] for line in lines] == [1, 2, 3, 4]
    assert lines[2]["args"] == {
        "standoff": 2.0,
        "speed": 120.0,
        "release_ms": 700.0,
        "direction": 0.0,
    }
    assert "ValueError" in lines[2]["error"] and "reply" not in lines[2]
    assert lines[1]["reply"]["held"] is True
    assert lines[3]["args"]["direction"] is None and lines[3]["reply"]["tossed"]
    for line in lines:
        assert line["end"] >= line["start"] and line["duration_s"] >= 0


def test_each_pick_films_the_wrist_and_the_session_films_the_ceiling(tmp_path):
    """A wrist clip per pick, named by the pick; one ceiling video for the session,
    labelled by the call in progress."""
    import cv2 as cv  # pylint: disable=import-outside-toplevel

    frame = np.zeros((48, 64, 3), np.uint8)

    class _Camera:
        def get_image(self):
            """A blank wrist frame."""
            return frame

    skills = build_skills(
        _cfg(**{"pick.top_down.lateral_servo.enabled": "false"}),
        log_dir=tmp_path,
        wrist_camera=lambda arm: _Camera(),
    )
    skills.pick()
    skills.close()
    assert sorted(p.name for p in tmp_path.glob("*_wrist.mp4")) == [
        "pick_001_wrist.mp4"
    ]
    video = cv.VideoCapture(str(tmp_path / "ceiling.mp4"))
    assert video.isOpened() and video.get(cv.CAP_PROP_FRAME_COUNT) > 0


# ------------------------------------------------------------------ fake robot


def test_the_fake_robot_has_the_clients_interface():
    """Same public methods as the client."""

    def public(cls):
        return {n for n in vars(cls) if not n.startswith("_")}

    assert public(TossingRobotClient) <= public(FakeTossingRobot)


def test_the_fake_robots_defaults_are_the_real_room():
    """FakeTossingRobot's room is conf/skill_server.yaml's: barrier, bin, drivable box,
    pick geometry, bin-pick clearances, toss bounds, spot directions and home."""
    cfg = _cfg()
    robot = FakeTossingRobot()
    assert robot.workspace == Workspace.from_config(
        cfg.barrier, cfg.preview.floor_bounds
    )
    b = cfg.bin
    assert robot.bin_fit == BinFit(
        *CLOSE_BIN, math.pi / 2, b.length, b.width, b.height, b.wall
    )
    assert robot.geom == geometry_from_config(cfg.pick.top_down)
    assert robot.in_bin == InBinPick(
        **{k: v for k, v in cfg.pick.top_down.in_bin.items() if k != "enabled"}
    )
    assert robot.in_bin_margin == cfg.bin.in_bin_margin
    assert robot.bounds.standoff == tuple(cfg.real_toss_bounds.standoff)
    assert robot.spot_directions == dict(cfg.toss.spot_directions)
    assert robot.home == pytest.approx(list(cfg.toss_pose))
    assert robot.robot == pytest.approx(list(cfg.toss_pose))


def test_the_fake_robot_runs_a_pick_and_toss_episode():
    """Deterministic: pick holds, a toss whose reach matches the standoff scores, and
    the cube in the close-side bin is picked out of it next."""
    robot = FakeTossingRobot()
    obs = robot.observe()
    assert set(obs) == CONTRACT_KEYS
    json.dumps(obs)
    assert robot.can_pick()["feasible"]
    assert robot.pick()["held"]
    assert robot.observe()["holding"] and robot.observe()["cube"] is None
    # Default landing: reach 0.9 + 0.55 * mean position in the box; at the middle of
    # both ranges the cube lands 1.175 m ahead.
    out = robot.toss(1.175, 120.0, 675.0, 0.0)
    assert out["tossed"] and out["in_bin"], out
    assert robot.observe()["cube_in_bin"] and robot.observe()["cube_inside_bin"]
    assert robot.can_pick()["feasible"]  # out of the close-side bin
    reply = robot.pick()
    assert reply["held"] and reply["cube"]["in_bin"]


def test_the_fake_robot_refuses_the_far_bin_and_tosses_from_minus_y():
    """At the far spot the toss stands on -y by default, and a cube thrown into that
    bin is past the barrier: refused, nothing moves. (From -y the base clears the tape
    only at standoffs of about 1.37 m and more.)"""
    far = BinFit(*FAR_BIN, 0.0, 0.39, 0.29, 0.21, 0.005)
    robot = FakeTossingRobot(bin_fit=far, landing=lambda pose, s, r: FAR_BIN)
    assert robot.observe()["toss_direction"] == pytest.approx(-math.pi / 2)
    robot.pick()
    with pytest.raises(ValueError, match="barrier"):
        robot.toss(1.3, 120.0, 675.0, None)
    out = robot.toss(1.39, 120.0, 675.0, None)
    assert out["pose"][:2] == pytest.approx([FAR_BIN[0], FAR_BIN[1] - 1.39])
    assert out["in_bin"]
    before = list(robot.robot)
    reply = robot.pick()
    assert reply["refused"] and "barrier" in reply["reason"] and not reply["held"]
    assert robot.robot == before


def test_the_fake_robot_rejects_what_the_real_one_rejects():
    """Same bounds and barrier checks."""
    robot = FakeTossingRobot()
    robot.pick()
    with pytest.raises(ValueError, match="speed"):
        robot.toss(1.2, 141.0, 600.0, 0.0)
    with pytest.raises(ValueError, match="barrier"):
        robot.toss(1.2, 120.0, 600.0, math.pi / 2)
    assert robot.observe()["holding"]


def test_the_fake_robots_pick_outcomes_are_configurable():
    """Outcomes cycle; a miss leaves the cube on the floor and is not a refusal."""
    robot = FakeTossingRobot(pick_outcomes=(False, True), max_attempts=1)
    miss = robot.pick()
    assert not miss["held"] and not miss["refused"]
    assert robot.observe()["cube"] is not None
    assert robot.pick()["held"]


def test_the_fake_robot_lands_short_and_misses():
    """A toss that falls short of the bin is reported out of it."""
    robot = FakeTossingRobot()
    robot.pick()
    out = robot.toss(1.4, 100.0, 500.0, 0.0)
    assert out["tossed"] and not out["in_bin"]


# ------------------------------------------------------------------- loopback


def test_client_and_server_over_localhost(skills):
    """The client reaches the skills through the real RPC, values come back plain,
    a ValueError raised on the server reaches the client, and only the contract's
    calls are served."""
    port = 50093
    thread = threading.Thread(
        target=serve, args=(skills, "127.0.0.1", port, b"test"), daemon=True
    )
    thread.start()
    client = None
    for _ in range(50):
        try:
            client = TossingRobotClient("127.0.0.1", port, b"test")
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.1)
    assert client is not None
    obs = client.observe()
    assert set(obs) == CONTRACT_KEYS
    json.dumps(obs)
    assert client.can_pick()["feasible"]
    assert client.pick()["held"]
    with pytest.raises(ValueError, match="speed"):
        client.toss(1.2, 160.0, 700.0, 0.0)
    out = client.toss(standoff=1.2, speed=120.0, release_ms=700.0)
    assert out["tossed"] and out["direction"] == pytest.approx(0.0)
    client.open_gripper()
    client.retract()
    client.go_home()
    client.stop()
    with pytest.raises(AttributeError):
        client._skills.close()  # pylint: disable=protected-access


def test_busy_and_stopped_cross_the_rpc_as_builtin_errors(skills):
    """A client without prpl_tidybot (hitl-pmp) must unpickle every error the server
    raises: SkillBusy and SkillStopped arrive as RuntimeError naming them."""
    port = 50095
    thread = threading.Thread(
        target=serve, args=(skills, "127.0.0.1", port, b"test"), daemon=True
    )
    thread.start()
    client = None
    for _ in range(50):
        try:
            client = TossingRobotClient("127.0.0.1", port, b"test")
            break
        except ConnectionRefusedError:
            threading.Event().wait(0.1)
    assert client is not None
    with skills._moving():  # pylint: disable=protected-access
        with pytest.raises(RuntimeError) as busy:
            client.retract()
    assert type(busy.value).__module__ == "builtins"
    assert str(busy.value).startswith("SkillBusy: another skill is moving the robot")

    def stopped():
        raise SkillStopped("stop() was called")

    skills.retract = stopped
    with pytest.raises(RuntimeError) as halted:
        client.retract()
    assert type(halted.value).__module__ == "builtins"
    assert str(halted.value) == "SkillStopped: stop() was called"


def test_the_fake_robot_answers_can_pick_in_observe_and_retries_a_miss():
    """As the real skills: observe carries can_pick; a miss retreats home and tries
    again (max_attempts); a bin it cannot see refuses."""
    robot = FakeTossingRobot(pick_outcomes=(False, True))
    assert robot.observe()["can_pick"] == robot.can_pick()
    reply = robot.pick()
    assert reply["held"] and [a["held"] for a in reply["attempts"]] == [False, True]
    once = FakeTossingRobot(pick_outcomes=(False,), max_attempts=1)
    assert not once.pick()["held"] and once.robot == pytest.approx(list(HOME))
    blind = FakeTossingRobot(bin_localized=False)
    assert blind.observe()["bin"] is None
    assert blind.observe()["can_pick"]["reason"] == BIN_NOT_LOCALIZED
    assert blind.pick()["refused"]
