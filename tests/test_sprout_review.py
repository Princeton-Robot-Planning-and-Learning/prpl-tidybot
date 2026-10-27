"""Terminal approval must not bypass scene, preview or readiness checks."""

import json
import time

import pytest

from prpl_tidybot.sprout.review import approve_and_execute, digest


@pytest.fixture
def review(tmp_path):
    for name in ("preview.mp4", "trajectory.json", "scene.json"):
        (tmp_path / name).write_bytes(b"synthetic test artifact")
    observed = {
        "grasp_xyz": [0.5, 0, 0.1],
        "base_xyyaw": [0, 0, 0],
        "arm_joints_rad": [0] * 7,
    }
    payload = {
        "execution_ready": True,
        "completed": True,
        "source": "live_camera",
        "artifact_sha256": {
            name: digest(tmp_path / name)
            for name in ("preview.mp4", "trajectory.json", "scene.json")
        },
        "observed": observed,
        "maximum_observation_age_s": 1,
        "revalidation_tolerances": {
            "grasp_xyz": 0.005,
            "base_xyyaw": 0.005,
            "arm_joints_rad": 0.01,
        },
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload))
    events = []

    def observe():
        events.append("observe")
        return {**observed, "captured_at_unix": time.time(), "ready": True}

    def execute(report):
        events.append("execute")
        return "done"

    return path, payload, observe, execute, events


@pytest.mark.parametrize("answer", ["", "yes", "YES"])
def test_only_after_terminal_approval_and_fresh_read(review, answer):
    path, _, observe, execute, events = review

    def prompt(_):
        assert events == []
        events.append("approved")
        return answer

    assert (
        approve_and_execute(path, observe, execute, prompt=prompt, interactive=True)
        == "done"
    )
    assert events == ["approved", "observe", "execute"]


@pytest.mark.parametrize(
    "field,value",
    [("execution_ready", False), ("completed", False), ("source", "synthetic_demo")],
)
def test_unready_preview_never_prompts_or_executes(review, field, value):
    path, payload, observe, execute, events = review
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        approve_and_execute(
            path,
            observe,
            execute,
            prompt=lambda _: pytest.fail("Unexpected prompt"),
            interactive=True,
        )
    assert events == []


def test_piped_enter_cannot_authorize(review):
    path, _, observe, execute, events = review
    with pytest.raises(RuntimeError, match="interactive"):
        approve_and_execute(
            path, observe, execute, prompt=lambda _: "", interactive=False
        )
    assert events == []


def test_video_changed_while_waiting_requires_new_preview(review):
    path, _, observe, execute, events = review

    def prompt(_):
        (path.parent / "preview.mp4").write_bytes(b"changed")
        return "yes"

    with pytest.raises(RuntimeError, match="artifact changed"):
        approve_and_execute(path, observe, execute, prompt=prompt, interactive=True)
    assert events == []


@pytest.mark.parametrize(
    "key", ["grasp_xyz", "base_xyyaw", "arm_joints_rad", "captured_at_unix"]
)
def test_scene_change_or_staleness_blocks_motion(review, key):
    path, _, observe, execute, events = review

    def changed():
        fresh = observe()
        if key == "captured_at_unix":
            fresh[key] -= 2
        else:
            fresh[key] = [value + 0.1 for value in fresh[key]]
        return fresh

    with pytest.raises(RuntimeError):
        approve_and_execute(
            path, changed, execute, prompt=lambda _: "yes", interactive=True
        )
    assert events == ["observe"]


@pytest.mark.parametrize("answer", ["no", "y", "execute"])
def test_other_answers_cancel(review, answer):
    path, _, observe, execute, events = review
    with pytest.raises(RuntimeError, match="declined"):
        approve_and_execute(
            path, observe, execute, prompt=lambda _: answer, interactive=True
        )
    assert events == []


def test_eof_cancels(review):
    path, _, observe, execute, events = review

    def eof(_):
        raise EOFError

    with pytest.raises(RuntimeError, match="cancelled"):
        approve_and_execute(path, observe, execute, prompt=eof, interactive=True)
    assert events == []
