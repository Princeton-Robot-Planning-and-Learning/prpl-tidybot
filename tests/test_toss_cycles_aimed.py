"""Fake-mode tests for `run_toss_cycles` tossing at the localized bin (``toss.aimed``)
with the real parameter bounds, and for the barrier the robot never crosses."""

# Fixtures that only patch the fakes are arguments for their side effect.
# pylint: disable=unused-argument

import json
import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.aimed_toss import Barrier, footprint
from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
FAKE_BIN = (-0.45, 0.4)


@pytest.fixture(name="base_log")
def _base_log(monkeypatch):
    """Every map-frame pose the fake base is sent to."""
    log = []
    original = FakeBaseInterface.execute_action

    def execute_action(self, action):
        log.append((float(action.x), float(action.y), float(action.theta())))
        original(self, action)

    monkeypatch.setattr(FakeBaseInterface, "execute_action", execute_action)
    return log


@pytest.fixture(autouse=True)
def _no_control_period_sleep(monkeypatch):
    """The fake arm needs no real-time ticks."""
    monkeypatch.setattr("prpl_tidybot.toss_cycles.POLICY_CONTROL_PERIOD", 0.0)


class _Operator:
    """Answers every prompt 'y'."""

    def __call__(self, prompt: str) -> str:
        return "y"


@pytest.fixture(name="arm_log")
def _arm_log(monkeypatch):
    """Every FakeArmInterface toss and onboard-controller move, in order."""
    log: list[tuple] = []
    original_toss = FakeArmInterface.toss
    original_angular = FakeArmInterface.move_angular

    def toss(self, **kwargs):
        log.append(("toss", kwargs))
        original_toss(self, **kwargs)

    def move_angular(self, goal):
        log.append(("angular", list(goal)))
        return original_angular(self, goal)

    monkeypatch.setattr(FakeArmInterface, "toss", toss)
    monkeypatch.setattr(FakeArmInterface, "move_angular", move_angular)
    return log


def _compose(**extra):
    overrides = ["mode=fake", "seed=0", "interactive=true", "settle_after_toss_s=0"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="toss_cycles", overrides=overrides)


@pytest.fixture(name="robot_side_start")
def _robot_side_start(monkeypatch):
    """The fake base starts at the map origin, on the barrier; the real robot starts on
    its side.

    Start the fake at the lab's toss pose instead.
    """
    from spatialmath import SE2  # pylint: disable=import-outside-toplevel

    original = FakeBaseInterface.__init__

    def init(self):
        original(self)
        self.base_state = SE2(1.326, -0.493, -3.132)
        self.map_base_state = SE2(1.326, -0.493, -3.132)

    monkeypatch.setattr(FakeBaseInterface, "__init__", init)


def _aimed(**extra):
    overrides = {
        "toss.aimed.enabled": "true",
        "toss.aimed.standoff": 1.2,
        "toss.aimed.speed": 120.0,
        "toss.aimed.release_ms": 700.0,
        "toss.aimed.direction": 0.0,
        "cycles": 1,
        # The pick on the robot's side, so the drive to the toss is clear of the bin.
        "env.cube_pose": "[0.8,0.8,0.025]",
    }
    overrides.update(extra)
    return _compose(**overrides)


def test_an_aimed_toss_drives_to_face_the_bin_and_throws_with_its_parameters(
    tmp_path, arm_log, robot_side_start
):
    """The base parks `standoff` from the localized bin facing it; the throw uses the
    aimed speed and release, the arm straight ahead."""
    result = run_toss_cycles(_aimed(), log_dir=tmp_path, prompt_fn=_Operator())
    assert result.completed == 1, result.finish_reason
    tosses = [e[1] for e in arm_log if e[0] == "toss"]
    assert tosses == [{"heading": 0.0, "max_vel": 120.0, "gripper_release_ms": 700}]
    record = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0]
    x, y, theta = record["aimed"]["pose"]
    assert (x, y) == pytest.approx((FAKE_BIN[0] + 1.2, FAKE_BIN[1]))
    assert math.cos(theta - math.pi) == pytest.approx(1.0)
    assert record["aimed"]["bin"]["x"] == pytest.approx(FAKE_BIN[0])


def test_a_toss_grid_and_an_aimed_toss_together_fail_before_anything_moves(
    tmp_path, arm_log, base_log, robot_side_start
):
    """The grid is calibration thrown from the fixed toss pose, and the aim needs that
    calibration first: both enabled is a config error, raised before the first prompt
    or any arm or base command."""
    prompts: list[str] = []

    def operator(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    cfg = _aimed(**{"+toss.grid": "[[100,500],[120,675]]"})
    with pytest.raises(ValueError, match="toss.grid .* and toss.aimed cannot both"):
        run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=operator)
    assert not prompts
    assert not arm_log
    assert not base_log


@pytest.mark.parametrize(
    "param, value",
    [
        ("toss.aimed.speed", 150.0),
        ("toss.aimed.speed", 80.0),
        ("toss.aimed.release_ms", 900.0),
        ("toss.aimed.standoff", 2.0),
    ],
)
def test_parameters_outside_the_real_bounds_are_refused_before_moving(
    tmp_path, arm_log, base_log, robot_side_start, param, value
):
    """Out-of-bounds parameters end the run before the base drives to toss or the arm
    throws."""
    result = run_toss_cycles(
        _aimed(**{param: value}), log_dir=tmp_path, prompt_fn=_Operator()
    )
    assert result.finish_reason.startswith("refused"), result.finish_reason
    assert not [e for e in arm_log if e[0] == "toss"]
    record = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0]
    assert "aimed" not in record


def test_a_toss_pose_across_the_barrier_is_refused(tmp_path, arm_log, robot_side_start):
    """With the barrier on, a standoff too short to keep the base on its side is
    refused (0.9 m from a bin at x = -0.6 would put the base's front at x = 0.026)."""
    result = run_toss_cycles(
        _aimed(
            **{
                "barrier.enabled": "true",
                "env.cube_pose": "[0.8,0.8,0.025]",
                "toss.aimed.standoff": 0.9,
                "bin.pipelines.fake.x": -0.6,
            }
        ),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.finish_reason.startswith("refused"), result.finish_reason
    assert "barrier" in result.finish_reason
    assert not [e for e in arm_log if e[0] == "toss"]


def test_with_the_barrier_every_base_pose_stays_on_the_robots_side(
    tmp_path, arm_log, base_log, robot_side_start
):
    """A full aimed cycle with the barrier on: pick a cube on the robot's side, toss at
    a bin across it, reset; the base never puts its footprint over the line."""
    result = run_toss_cycles(
        _aimed(**{"barrier.enabled": "true", "env.cube_pose": "[0.8,0.8,0.025]"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.completed == 1, result.finish_reason
    assert base_log
    for pose in base_log:
        assert min(Barrier().depth(c) for c in footprint(pose)) >= 0.05 - 1e-6, pose


def test_a_cube_across_the_barrier_is_not_picked(
    tmp_path, arm_log, base_log, robot_side_start
):
    """The robot never crosses: a cube on the far side ends the run unpicked."""
    result = run_toss_cycles(
        _aimed(**{"barrier.enabled": "true", "env.cube_pose": "[-0.8,0.8,0.025]"}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert "cube past the barrier" in result.finish_reason
    assert not [e for e in arm_log if e[0] == "angular"]
