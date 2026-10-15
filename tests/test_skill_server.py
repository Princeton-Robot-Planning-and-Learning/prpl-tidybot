"""Tests for the Tossing3D skill server: the skills on the fake robot, the contract's
shape, the fake robot planners test against, and a loopback RPC."""

import json
import math
import threading
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.aimed_toss import Barrier, footprint
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.skill_server import (
    FakeTossingRobot,
    SkillBusy,
    SkillStopped,
    TossingRobotClient,
    build_skills,
    serve,
)

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
    "barrier",
    "barrier_x",
}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No real-time ticks for the fake arm."""
    monkeypatch.setattr("prpl_tidybot.toss_cycles.POLICY_CONTROL_PERIOD", 0.0)


@pytest.fixture(autouse=True)
def _robot_side_start(monkeypatch):
    """The fake base starts at the lab's home pose, on the robot's side."""
    from spatialmath import SE2  # pylint: disable=import-outside-toplevel

    from prpl_tidybot.interfaces.base_interface import (  # pylint: disable=import-outside-toplevel
        FakeBaseInterface,
    )

    original = FakeBaseInterface.__init__

    def init(self):
        original(self)
        self.base_state = SE2(1.326, -0.493, -3.132)
        self.map_base_state = SE2(1.326, -0.493, -3.132)

    monkeypatch.setattr(FakeBaseInterface, "__init__", init)


def _cfg(**extra):
    overrides = ["mode=fake", "settle_after_toss_s=0", "env.cube_pose=[0.8,0.8,0.025]"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=overrides)


@pytest.fixture(name="skills")
def _skills():
    return build_skills(_cfg())


def test_observe_has_exactly_the_contracts_fields_as_plain_values(skills):
    """Every field, JSON-serialisable, the barrier on by default in the server."""
    obs = skills.observe()
    assert set(obs) == CONTRACT_KEYS
    json.dumps(obs)
    assert obs["robot"] == pytest.approx([1.326, -0.493, -3.132])
    assert obs["cube"][:2] == pytest.approx([0.8, 0.8])
    assert set(obs["bin"]) == {"x", "y", "yaw", "length", "width", "height"}
    assert obs["barrier"] == Barrier().as_dict()
    assert obs["barrier_x"] == 0.0
    assert obs["holding"] is False
    assert obs["gripper_command"] == 0.0
    assert obs["cube_upright"] is True
    assert obs["cube_in_bin"] is False


def test_can_pick_is_a_dry_run(skills):
    """Feasible for a cube on the robot's side; nothing moves."""
    arm = skills.arm
    before = (list(arm.get_arm_state()), skills.observe()["robot"])
    assert skills.can_pick() == {"feasible": True, "note": None}
    assert (list(arm.get_arm_state()), skills.observe()["robot"]) == before


def test_can_pick_refuses_a_cube_across_the_barrier():
    """The robot never crosses the barrier, so that cube is unpickable."""
    skills = build_skills(_cfg(**{"env.cube_pose": "[-0.8,0.8,0.025]"}))
    assert skills.can_pick() == {"feasible": False, "note": "cube across the barrier"}


def test_pick_then_toss_into_the_bin(skills):
    """Pick holds the cube (gripper commanded closed, reading in the band, cube off
    the floor); the toss drives to the aimed pose, throws with the given speed and
    release, retracts, and reports where the cube landed."""
    result = skills.pick()
    assert result["held"] is True and result["note"] is None
    obs_cube = skills.cube_source  # the fake cube stays where the config put it
    del obs_cube
    assert skills.gripper_command == 1.0
    tosses = []
    original = FakeArmInterface.toss

    def toss(self, **kwargs):
        tosses.append(kwargs)
        original(self, **kwargs)

    FakeArmInterface.toss = toss  # restored below
    try:
        out = skills.toss(1.2, 120.0, 700.0, 0.0)
    finally:
        FakeArmInterface.toss = original
    assert out["tossed"] is True
    assert tosses == [{"heading": 0.0, "max_vel": 120.0, "gripper_release_ms": 700}]
    assert skills.gripper_command == 0.0
    robot = skills.observe()["robot"]
    assert robot[:2] == pytest.approx([-0.45 + 1.2, 0.4], abs=0.02)
    assert footprint(robot)[:, 0].min() >= Barrier().robot_side_x + 0.05 - 1e-6
    assert set(out) == {"tossed", "landed", "in_bin", "note"}


@pytest.mark.parametrize(
    "args",
    [
        (1.2, 150.0, 700.0, 0.0),
        (1.2, 120.0, 400.0, 0.0),
        (2.0, 120.0, 700.0, 0.0),
        (0.9, 120.0, 700.0, math.radians(150)),  # behind the barrier
    ],
)
def test_toss_raises_and_moves_nothing_outside_the_bounds_or_across_the_barrier(
    skills, args
):
    """ValueError before any motion, holding or not."""
    skills.pick()
    before = (list(skills.arm.get_arm_state()), skills.observe()["robot"])
    with pytest.raises(ValueError):
        skills.toss(*args)
    assert (list(skills.arm.get_arm_state()), skills.observe()["robot"]) == before


def test_toss_without_the_cube_does_not_move(skills):
    """Nothing to throw: reported, nothing moves."""
    out = skills.toss(1.2, 120.0, 700.0, 0.0)
    assert out == {
        "tossed": False,
        "landed": None,
        "in_bin": False,
        "note": "not holding the cube",
    }


def test_open_gripper_retract_and_go_home(skills):
    """The small skills: gripper open (commanded 0), retract, home pose."""
    skills.pick()
    skills.open_gripper()
    assert skills.gripper_command == 0.0
    skills.retract()
    skills.go_home()
    assert skills.observe()["robot"] == pytest.approx([1.326, -0.493, -3.132], abs=0.02)


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
    skills.stop_event.set()
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


# ------------------------------------------------------------------ fake robot


def test_the_fake_robot_has_the_clients_interface():
    """Same public methods as the client."""

    def public(cls):
        return {n for n in vars(cls) if not n.startswith("_")}

    assert public(TossingRobotClient) <= public(FakeTossingRobot)


def test_the_fake_robot_runs_a_pick_and_toss_episode():
    """Deterministic: pick holds, a toss whose reach matches the standoff scores."""
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
    assert robot.observe()["cube_in_bin"]
    assert not robot.can_pick()["feasible"]  # the cube is across the barrier now


def test_the_fake_robot_rejects_what_the_real_one_rejects():
    """Same bounds and barrier checks."""
    robot = FakeTossingRobot()
    robot.pick()
    with pytest.raises(ValueError, match="speed"):
        robot.toss(1.2, 141.0, 600.0, 0.0)
    with pytest.raises(ValueError, match="barrier"):
        robot.toss(0.9, 120.0, 600.0, math.radians(150))
    assert robot.observe()["holding"]


def test_the_fake_robots_pick_outcomes_are_configurable():
    """Outcomes cycle; a miss leaves the cube on the floor."""
    robot = FakeTossingRobot(pick_outcomes=(False, True))
    assert not robot.pick()["held"]
    assert robot.observe()["cube"] is not None
    assert robot.pick()["held"]


def test_the_fake_robot_lands_short_and_misses():
    """A toss that falls short of the bin is reported out of it."""
    robot = FakeTossingRobot(bin_fit=BinFit(-0.45, 0.4, 0.0, 0.40, 0.32, 0.19))
    robot.pick()
    out = robot.toss(1.4, 100.0, 500.0, 0.0)
    assert out["tossed"] and not out["in_bin"]


# ------------------------------------------------------------------- loopback


def test_client_and_server_over_localhost(skills):
    """The client reaches the skills through the real RPC, values come back plain,
    and a ValueError raised on the server reaches the client."""
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
    out = client.toss(1.2, 120.0, 700.0, 0.0)
    assert out["tossed"]
    client.open_gripper()
    client.retract()
    client.go_home()
    client.stop()
