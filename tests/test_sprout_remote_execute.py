"""SSH approval transport: EOF, wrong artifact and heartbeat loss latch stop."""

import io
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from prpl_tidybot.sprout.remote_execute import Session
from prpl_tidybot.sprout.review import digest


def test_exact_review_approval_only():
    session = Session("expected")
    session.receive({"kind": "approve", "review_sha256": "wrong"})
    assert session.stopped.is_set()
    session.receive({"kind": "approve", "review_sha256": "expected"})
    assert not session.approved.is_set()


def test_approved_then_stop_cannot_resume():
    session = Session("expected")
    session.receive({"kind": "approve", "review_sha256": "expected"})
    assert session.prompt("unused") == "yes"
    session.receive({"kind": "stop"})
    session.receive({"kind": "heartbeat"})
    with pytest.raises(RuntimeError, match="stop latched"):
        session.prompt("unused")


@pytest.mark.parametrize("stream", ["", "not-json\n", "[]\n"])
def test_connection_eof_or_malformed_data_stops_runtime(stream):
    events = []
    session = Session("expected")
    session.bind(SimpleNamespace(stop=lambda: events.append("stop")))
    session.read(io.StringIO(stream))
    assert events == ["stop"]
    assert session.stopped.is_set()


def test_stop_before_runtime_is_bound_latches():
    events = []
    session = Session("expected")
    session.stop()
    session.bind(SimpleNamespace(stop=lambda: events.append("stop")))
    assert events == ["stop"]


def test_heartbeat_loss_calls_stop():
    stopped = threading.Event()
    session = Session("expected", timeout_s=0.01)
    session.bind(SimpleNamespace(stop=stopped.set))
    watcher = threading.Thread(target=session.watch)
    watcher.start()
    try:
        assert stopped.wait(1)
        assert session.stopped.is_set()
    finally:
        session.finished.set()
        watcher.join(1)


def test_operator_confirmation_is_bound_to_live_phase_and_nonce():
    from prpl_tidybot.sprout.live_runtime import Confirmations

    confirmations = Confirmations()
    session = Session("expected")
    session.bind(SimpleNamespace(confirm=confirmations.confirm, stop=lambda: None))
    session.receive({"kind": "approve", "review_sha256": "expected"})
    confirmations.request("grasp")
    phase, token = confirmations.pending
    session.receive({"kind": "confirm", "phase": phase, "token": token})
    assert confirmations.accepted.is_set() and not session.stopped.is_set()
    confirmations.request("support")
    session.receive({"kind": "confirm", "phase": phase, "token": token})
    assert session.stopped.is_set() and not confirmations.accepted.is_set()


def test_real_process_approval_transport_with_fake_runtime(tmp_path):
    """Exercise stdin/threads/import/revalidation/execute; no hardware imports."""
    observed = {
        "grasp_xyz": [1, 2, 0.1],
        "base_xyyaw": [0, 0, 0],
        "arm_joints_rad": [0] * 7,
    }
    artifacts = ("preview.mp4", "trajectory.json", "scene.json")
    for name in artifacts:
        (tmp_path / name).write_text("fixture")
    report = {
        "execution_ready": True,
        "completed": True,
        "source": "live_camera",
        "observed": observed,
        "maximum_observation_age_s": 1,
        "revalidation_tolerances": {key: 0.01 for key in observed},
        "artifact_sha256": {name: digest(tmp_path / name) for name in artifacts},
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(report))
    (tmp_path / "fixture_runtime.py").write_text(
        "import time\n"
        "class Runtime:\n"
        f"    def observe(self): return {{**{observed!r}, 'ready': True, 'captured_at_unix': time.time()}}\n"
        "    def execute(self, review): return {'fake_cycle_completed': True}\n"
        "    def stop(self): pass\n"
        "def factory(review_path): return Runtime()\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "prpl_tidybot.sprout.remote_execute",
            "--review",
            str(review_path),
            "--runtime",
            "fixture_runtime:factory",
        ],
        env=env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.stdin.write(
            json.dumps({"kind": "approve", "review_sha256": digest(review_path)}) + "\n"
        )
        process.stdin.flush()
        assert process.wait(timeout=5) == 0, process.stderr.read()
        assert '"fake_cycle_completed": true' in process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
