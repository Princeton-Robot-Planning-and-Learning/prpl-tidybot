"""End-to-end tests for `prpl_tidybot.pick_and_toss.run_pick_and_toss` (the logic
behind scripts/pick_and_toss.py), composed from conf/pick_and_toss.yaml like
`tests/test_pipeline.py` composes conf/config.yaml."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.pick_and_toss import run_pick_and_toss
from prpl_tidybot.real_sim.perceivers import kinematic3d

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def _compose(mode: str = "fake", **extra):
    overrides = [
        f"mode={mode}",
        "seed=0",
        "capture.settle_s=0",
        "capture.post_toss_record_s=0",
    ]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="pick_and_toss", overrides=overrides)


@pytest.fixture(name="arm_log")
def _arm_log(monkeypatch):
    """Every FakeArmInterface call the run makes, in order."""
    log: list[tuple] = []
    original_toss = FakeArmInterface.toss
    original_execute = FakeArmInterface.execute_action

    def toss(self, **kwargs):
        log.append(("toss", kwargs))
        original_toss(self, **kwargs)

    def execute_action(self, arm_goal, gripper_goal):
        log.append(("execute_action", gripper_goal))
        original_execute(self, arm_goal, gripper_goal)

    monkeypatch.setattr(FakeArmInterface, "toss", toss)
    monkeypatch.setattr(FakeArmInterface, "execute_action", execute_action)
    return log


def test_pick_then_toss_in_fake_mode(tmp_path, arm_log):
    """The PickController's trajectory is executed, then the arm tosses at the CLI's
    heading / speed / release, and the landing is logged from the (fake) ceiling."""
    cfg = _compose(**{"toss.heading": 10.0})
    result = run_pick_and_toss(cfg, log_dir=tmp_path)
    assert result.picked is True
    assert result.pick_steps > 0
    assert result.tossed is True
    tosses = [entry for entry in arm_log if entry[0] == "toss"]
    assert tosses == [
        ("toss", {"heading": 10.0, "max_vel": 60.0, "gripper_release_ms": 600})
    ]
    # The pick closed the gripper before the toss.
    assert ("execute_action", 1.0) in arm_log
    assert arm_log.index(("execute_action", 1.0)) < arm_log.index(tosses[0])
    outcome = json.loads((tmp_path / "toss_outcome.json").read_text())
    assert outcome["status"] == "landed"
    assert outcome["landing"]["marker_id"] == 23
    assert outcome["toss"]["max_vel_deg_s"] == 60.0
    assert (tmp_path / "toss.mp4").exists()
    assert result.toss_outcome == outcome


def test_pick_only_stops_holding_the_cube(tmp_path, arm_log):
    """skip_toss: the pick runs and nothing is thrown or filmed."""
    cfg = _compose(skip_toss="true")
    result = run_pick_and_toss(cfg, log_dir=tmp_path)
    assert result.picked is True
    assert result.tossed is False
    assert not [entry for entry in arm_log if entry[0] == "toss"]
    assert ("execute_action", 1.0) in arm_log
    assert not (tmp_path / "toss_outcome.json").exists()


def test_toss_only_hands_the_object_in_without_perception(
    tmp_path, arm_log, monkeypatch
):
    """skip_pick: open, wait for the operator, close, confirm, toss. Neither the
    perceiver nor the planner's models are built."""

    def _no_perceiver(*args, **kwargs):
        raise AssertionError("toss-only must not build a perceiver")

    monkeypatch.setattr(kinematic3d.Shelf3DPerceiver, "__init__", _no_perceiver)
    monkeypatch.setattr(
        "prpl_tidybot.pick_and_toss.build_planner_env_models", _no_perceiver
    )
    prompts: list[str] = []
    answers = iter(["", "y"])

    def prompt_fn(text: str) -> str:
        prompts.append(text)
        arm_log.append(("prompt", len(prompts)))
        return next(answers)

    cfg = _compose(skip_pick="true", interactive="true")
    result = run_pick_and_toss(cfg, log_dir=tmp_path, prompt_fn=prompt_fn)
    assert result.picked is None
    assert result.tossed is True
    assert [entry[0] for entry in arm_log] == [
        "execute_action",
        "prompt",
        "execute_action",
        "prompt",
        "toss",
    ]
    assert arm_log[0] == ("execute_action", 0.0)  # open for the hand-in
    assert arm_log[2] == ("execute_action", 1.0)  # close on it
    assert "hand" in prompts[0].lower()
    assert "toss" in prompts[1].lower() and "60" in prompts[1]
    assert (tmp_path / "toss_outcome.json").exists()


def test_rejected_toss_does_not_throw(tmp_path, arm_log):
    """Anything but y at the toss confirmation leaves the object in the gripper."""
    answers = iter(["", "n"])
    cfg = _compose(skip_pick="true", interactive="true")
    result = run_pick_and_toss(cfg, log_dir=tmp_path, prompt_fn=lambda _: next(answers))
    assert result.tossed is False
    assert "rejected" in result.finish_reason
    assert not [entry for entry in arm_log if entry[0] == "toss"]


def test_toss_without_capture(tmp_path, arm_log):
    """capture.enabled=false tosses without touching the ceiling cameras."""
    cfg = _compose(skip_pick="true", **{"capture.enabled": "false"})
    result = run_pick_and_toss(cfg, log_dir=tmp_path)
    assert result.tossed is True
    assert result.toss_outcome is None
    assert [entry[0] for entry in arm_log].count("toss") == 1
    assert not (tmp_path / "toss_outcome.json").exists()


def test_skipping_both_steps_is_an_error():
    """Nothing to do is a configuration mistake."""
    with pytest.raises(ValueError):
        run_pick_and_toss(_compose(skip_pick="true", skip_toss="true"))


def test_toss_only_is_not_available_in_sim():
    """The kinematic sim has no toss, so toss-only has nothing to run in sim."""
    with pytest.raises(ValueError):
        run_pick_and_toss(_compose(mode="sim", skip_pick="true"))


def test_pick_in_sim_mode_skips_the_toss(tmp_path):
    """In sim the pick runs in the kinder sim (the cube ends up held); the toss is
    not simulated and is reported as skipped."""
    cfg = _compose(mode="sim")
    result = run_pick_and_toss(cfg, log_dir=tmp_path)
    assert result.picked is True
    assert result.tossed is False
    assert "sim" in result.finish_reason
    assert result.final_state.grasped_object == "cube0"
