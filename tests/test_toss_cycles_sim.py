"""`run_toss_cycles` in mode=sim: the same pick, drive, toss and retract commands as
the real robot, executed on the kinematic TidyBot of `prpl_tidybot.sim_robot`, with a
video of the robot's paths."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.sim_robot import SimArmInterface
from prpl_tidybot.sim_video import gl_available
from prpl_tidybot.third_party import arm_server
from prpl_tidybot.toss import plan_toss_swing, toss_windup_pose
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
GRID = [[100, 500], [120, 675], [140, 850]]
TOSS_POSE = "[0.13,-1.13,1.5708]"


@pytest.fixture(autouse=True)
def _no_wall_clock_waits(monkeypatch):
    """The sim keeps its own clock; skip the real-time waits around it."""
    monkeypatch.setattr("prpl_tidybot.toss_cycles.POLICY_CONTROL_PERIOD", 0.0)
    monkeypatch.setattr(arm_server, "time", SimpleNamespace(sleep=lambda _s: None))


def _compose(*overrides):
    base = [
        "mode=sim",
        "seed=0",
        f"toss_pose={TOSS_POSE}",
        "pick.top_down.wrist_servo.enabled=false",
        "sim.video.enabled=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="toss_cycles", overrides=base + list(overrides))


@pytest.fixture(name="tosses")
def _tosses(monkeypatch):
    """Every toss the sim arm is asked for, and the swing it then executed."""
    log: list[dict] = []
    original = SimArmInterface.toss

    def toss(self, **kwargs):
        original(self, **kwargs)
        log.append(kwargs | {"swing": self.world.swings[-1]})

    monkeypatch.setattr(SimArmInterface, "toss", toss)
    return log


def test_sim_mode_throws_every_grid_setting_without_asking(tmp_path, tosses):
    """Josh's calibration run in sim: no operator, one grid setting per cycle, every
    cycle picks the (placed) cube again and the run ends retracted at the toss pose."""
    asked = []
    result = run_toss_cycles(
        _compose(
            "cycles=3",
            f"+toss.grid={json.dumps(GRID)}",
            "+toss.confirm_each=true",
        ),
        log_dir=tmp_path,
        prompt_fn=asked.append,
    )
    assert result.finish_reason == "completed 3/3 cycles"
    assert not asked
    assert [(t["max_vel"], t["gripper_release_ms"]) for t in tosses] == [
        (100.0, 500),
        (120.0, 675),
        (140.0, 850),
    ]
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert all(c["picked"] and c["tossed"] for c in log["cycles"])
    assert all(c["pick_attempts"][-1]["held"] for c in log["cycles"])
    x, y, _ = result.final_base_pose
    assert (x, y) == pytest.approx((0.13, -1.13), abs=0.02)


def test_the_swing_in_sim_is_the_one_the_real_arm_would_stream(tmp_path, tosses):
    """Each executed swing equals the arm server's plan for its grid setting."""
    run_toss_cycles(
        _compose("cycles=3", f"+toss.grid={json.dumps(GRID)}"), log_dir=tmp_path
    )
    for toss, (speed, release) in zip(tosses, GRID):
        planned = plan_toss_swing(
            toss_windup_pose(0.0), 0.0, max_vel=speed, gripper_release_ms=release
        )
        executed = toss["swing"]["executed_deg"]
        assert executed.shape == planned.shape
        assert np.abs((executed - planned + 180) % 360 - 180).max() < 1e-3
        assert toss["swing"]["release_index"] == release


def test_a_cube_thrown_into_the_bin_is_put_back_by_the_scripted_operator(tmp_path):
    """A landing inside bin_stop_box stops the next pick for the person; in sim the
    scripted operator puts the cube back and the run carries on."""
    result = run_toss_cycles(
        _compose(
            "cycles=2",
            "+bin_stop_box=[-0.13,0.39,1.45,1.91]",
            "sim.landing_offset=[2.7,0.0]",
        ),
        log_dir=tmp_path,
    )
    assert result.finish_reason == "completed 2/2 cycles"
    log = json.loads((tmp_path / "cycles.json").read_text())
    assert log["cycles"][1]["cube_in_bin"]["y"] == pytest.approx(1.57, abs=0.05)


def test_sim_mode_records_the_path_safety_summary(tmp_path):
    """Each step's clearances, contacts and joint-limit margins: sim_safety.json."""
    run_toss_cycles(_compose("cycles=1"), log_dir=tmp_path)
    summary = json.loads((tmp_path / "sim_safety.json").read_text())
    steps = [s["step"] for s in summary["steps"]]
    assert any("toss" in s.lower() for s in steps)
    for step in summary["steps"]:
        assert {"min_floor_clearance_m", "contacts", "min_joint_limit_margin_deg"} <= (
            set(step)
        )


@pytest.mark.skipif(not gl_available(), reason="no OpenGL context for MuJoCo")
def test_sim_mode_writes_a_video(tmp_path):
    """The run is filmed, overhead and oblique side by side."""
    import imageio.v3 as iio  # pylint: disable=import-outside-toplevel

    run_toss_cycles(
        _compose("cycles=1", "sim.video.enabled=true", "settle_after_toss_s=0"),
        log_dir=tmp_path,
    )
    path = tmp_path / "sim.mp4"
    assert path.exists()
    frame = next(iio.imiter(path))
    assert frame.shape[1] == 2 * frame.shape[0] * 4 // 3
