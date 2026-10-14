"""Tests for the repeated pick -> toss -> re-perceive loop (`cycles=N`) of
`prpl_tidybot.pick_and_toss.run_pick_and_toss`, in fake mode.

In fake mode the cube is perceived at `env.cube_pose` and the fake ceiling reports
it landing at `env.cube_pose` too, so by default every cycle finds the cube where
it was picked from; the tests move the reported landing to trigger each stop.
"""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from kinder_bilevel_planning.agent import AgentFailure

from prpl_tidybot import pick_and_toss
from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.pick_and_toss import run_pick_and_toss

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


_ENV_MODELS_CACHE: dict = {}


@pytest.fixture(autouse=True)
def _reuse_env_models(monkeypatch):
    """Build the planner's env models once per module rather than per run: each build
    holds a pybullet sim that is never torn down, and the suite's peak memory adds up.
    """
    original = pick_and_toss.build_planner_env_models

    def cached(env_name, make_kwargs, env_model_kwargs=None):
        if env_name not in _ENV_MODELS_CACHE:
            _ENV_MODELS_CACHE[env_name] = original(
                env_name, make_kwargs, env_model_kwargs
            )
        return _ENV_MODELS_CACHE[env_name]

    monkeypatch.setattr(pick_and_toss, "build_planner_env_models", cached)


def _compose(cycles: int = 3, mode: str = "fake", **extra):
    overrides = [
        f"mode={mode}",
        "seed=0",
        f"cycles={cycles}",
        "capture.settle_s=0",
        "capture.post_toss_record_s=0",
    ]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="pick_and_toss", overrides=overrides)


def _cycles_log(log_dir: Path) -> dict:
    return json.loads((log_dir / "cycles.json").read_text())


def test_loop_runs_every_cycle_and_logs_each(tmp_path):
    """Three cycles: three picks, three tosses, one JSON entry per cycle with the
    perceived cube pose (including its yaw) and the landing."""
    result = run_pick_and_toss(_compose(cycles=3), log_dir=tmp_path)
    assert result.finish_reason == "completed 3/3 cycles"
    log = _cycles_log(tmp_path)
    assert log["stop_reason"] is None
    assert [c["cycle"] for c in log["cycles"]] == [1, 2, 3]
    for entry in log["cycles"]:
        assert entry["picked"] is True
        assert entry["tossed"] is True
        assert entry["landing"]["marker_id"] == 23
        assert set(entry["cube_perceived"]) == {"x", "y", "z", "yaw"}
        assert (tmp_path / f"cycle_{entry['cycle']:02d}" / "toss_outcome.json").exists()
    assert result.cycles == log["cycles"]


def test_loop_uses_the_perceived_cube_yaw(tmp_path, monkeypatch):
    """The cube's rotation comes from the perceiver; a perceiver that reports a
    rotated cube gets that yaw into the log (and into the PickController's state)."""
    original = pick_and_toss.cube_pose_from_state

    def rotated(state):
        pose = original(state)
        return {**pose, "yaw": 0.25}

    monkeypatch.setattr(pick_and_toss, "cube_pose_from_state", rotated)
    run_pick_and_toss(_compose(cycles=1), log_dir=tmp_path)
    assert _cycles_log(tmp_path)["cycles"][0]["cube_perceived"]["yaw"] == 0.25


def test_loop_stops_when_the_cube_is_not_visible(tmp_path):
    """No cube marker after a toss: stop with that reason after logging the cycle."""
    result = run_pick_and_toss(
        _compose(cycles=3, **{"capture.pipelines.fake.marker_client.targets": "[]"}),
        log_dir=tmp_path,
    )
    log = _cycles_log(tmp_path)
    assert len(log["cycles"]) == 1
    assert "not visible" in log["stop_reason"]
    assert result.finish_reason == f"stopped after 1/3 cycles: {log['stop_reason']}"


def test_loop_stops_when_the_cube_is_outside_the_base_pose_bounds(tmp_path):
    """A landing outside preview.floor_bounds is not pickable from the floor box."""
    run_pick_and_toss(
        _compose(
            cycles=2,
            **{"capture.pipelines.fake.marker_client.targets": "[[23,3.0,0.0]]"},
        ),
        log_dir=tmp_path,
    )
    log = _cycles_log(tmp_path)
    assert len(log["cycles"]) == 1
    assert "base pose bounds" in log["stop_reason"]


def test_loop_stops_when_the_cube_is_outside_the_pickable_region(tmp_path):
    """loop.pickable_region narrows where the next pick may be attempted."""
    run_pick_and_toss(
        _compose(cycles=2, **{"loop.pickable_region": "[-1.0,0.0,-1.0,1.0]"}),
        log_dir=tmp_path,
    )
    assert "pickable region" in _cycles_log(tmp_path)["stop_reason"]


def test_loop_stops_when_perception_disagrees_with_the_landing(tmp_path):
    """The perceiver's cube must be where the ceiling saw it land; anything else
    (e.g. a stale cached detection) stops the loop rather than picking thin air."""
    run_pick_and_toss(
        _compose(
            cycles=2,
            **{"capture.pipelines.fake.marker_client.targets": "[[23,1.0,0.0]]"},
        ),
        log_dir=tmp_path,
    )
    log = _cycles_log(tmp_path)
    assert len(log["cycles"]) == 2
    assert "perceived" in log["stop_reason"]
    assert log["cycles"][1]["picked"] is False


def test_loop_stops_when_a_pick_fails(tmp_path, monkeypatch):
    """No pick trajectory on the second cycle: stop, cycle 2 logged as not picked."""
    # pylint: disable-next=protected-access
    original = pick_and_toss.PickSkillAgent._run_planning
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise AgentFailure("no pick")
        return original(self)

    monkeypatch.setattr(pick_and_toss.PickSkillAgent, "_run_planning", flaky)
    run_pick_and_toss(_compose(cycles=3), log_dir=tmp_path)
    log = _cycles_log(tmp_path)
    assert [c["picked"] for c in log["cycles"]] == [True, False]
    assert "pick failed" in log["stop_reason"]


def test_loop_stops_when_the_arm_faults(tmp_path, monkeypatch):
    """An exception out of the toss (e.g. the wind-up pose not reached) stops the
    loop with the arm fault as the reason."""
    calls = {"n": 0}
    original = FakeArmInterface.toss

    def faulty(self, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Arm stopped 3.00 deg from the toss wind-up pose")
        original(self, **kwargs)

    monkeypatch.setattr(FakeArmInterface, "toss", faulty)
    run_pick_and_toss(_compose(cycles=3), log_dir=tmp_path)
    log = _cycles_log(tmp_path)
    assert len(log["cycles"]) == 2
    assert log["cycles"][1]["tossed"] is False
    assert "arm fault" in log["stop_reason"] and "wind-up" in log["stop_reason"]


@pytest.mark.parametrize("first_only,expected", [("false", 3), ("true", 1)])
def test_loop_confirms_every_cycle_unless_told_otherwise(
    tmp_path, first_only, expected
):
    """By default every toss is confirmed; loop.confirm_first_cycle_only=true asks
    only on the first cycle."""
    prompts: list[str] = []

    def prompt_fn(text):
        prompts.append(text)
        return "y"

    cfg = _compose(
        cycles=3, interactive="true", **{"loop.confirm_first_cycle_only": first_only}
    )
    result = run_pick_and_toss(cfg, log_dir=tmp_path, prompt_fn=prompt_fn)
    assert result.finish_reason == "completed 3/3 cycles"
    assert sum("Toss?" in p for p in prompts) == expected


def test_loop_rejected_toss_stops(tmp_path):
    """Declining a toss ends the loop cleanly."""
    answers = iter(["y", "n"])
    run_pick_and_toss(
        _compose(cycles=3, interactive="true"),
        log_dir=tmp_path,
        prompt_fn=lambda _: next(answers),
    )
    log = _cycles_log(tmp_path)
    assert len(log["cycles"]) == 2
    assert "rejected" in log["stop_reason"]


def test_preview_gate_follows_the_confirm_setting():
    """In real mode the plan preview prompts on every cycle by default and only on
    the first with confirm_first_cycle_only; the floor check runs regardless."""
    every = _compose(cycles=3, mode="real")
    first = _compose(cycles=3, mode="real", **{"loop.confirm_first_cycle_only": "true"})
    assert [pick_and_toss.preview_prompts(every, c) for c in (1, 2, 3)] == [True] * 3
    assert [pick_and_toss.preview_prompts(first, c) for c in (1, 2, 3)] == [
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "extra",
    [
        {"skip_pick": "true"},
        {"skip_toss": "true"},
        {"capture.enabled": "false"},
    ],
)
def test_loop_needs_both_steps_and_the_capture(extra):
    """More than one cycle needs the pick, the toss, and the landing readout."""
    with pytest.raises(ValueError):
        run_pick_and_toss(_compose(cycles=2, **extra))


def test_loop_is_not_available_in_sim():
    """The kinematic sim has no toss, so there is nothing to repeat."""
    with pytest.raises(ValueError):
        run_pick_and_toss(_compose(cycles=2, mode="sim"))


def test_fake_toss_leaves_the_arm_at_the_release_pose():
    """Like the real arm, the fake ends a toss at the release pose (radians wrapped to
    [-pi, pi) as `Arm.get_joint_angles` reports them),
    so the next cycle's pick plans from there."""
    arm = FakeArmInterface()
    arm.toss(heading=10.0)
    assert arm.get_arm_state() == pytest.approx(
        [0.17453293, 0.34906585, -3.14159265, -0.61086524, 0.0, 0.43633231, 1.57079633]
    )
