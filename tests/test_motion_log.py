"""Tests for the skill server's continuous motion log (motion.jsonl)."""

import json
import threading
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from spatialmath import SE2

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.motion_log import MotionLog
from prpl_tidybot.skill_server import build_skills

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
HOME = (1.10, -0.78, 3.141592653589793)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No real-time ticks for the fake arm."""
    monkeypatch.setattr("prpl_tidybot.robot_skills.POLICY_CONTROL_PERIOD", 0.0)


@pytest.fixture(autouse=True)
def _home_start(monkeypatch):
    """The fake base starts at the server's home."""
    original = FakeBaseInterface.__init__

    def init(self):
        original(self)
        self.base_state = SE2(*HOME)
        self.map_base_state = SE2(*HOME)

    monkeypatch.setattr(FakeBaseInterface, "__init__", init)


def _cfg(*extra):
    overrides = ["mode=fake", "settle_after_toss_s=0", *extra]
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="skill_server", overrides=overrides)


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


class _FakeClock:
    """Time that passes only when the sampler waits, and stops at ``limit`` s after the
    start (the sampler then idles in real time)."""

    def __init__(self, start=1000.0, limit=1e9):
        self.t = start
        self.end = start + limit
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.t

    def wait(self, dt):
        """Advance by dt (the sampler's wait until its next slot)."""
        with self._lock:
            if self.t + dt <= self.end + 1e-9:
                self.t += dt
                stopped = False
            else:
                stopped = True
        time.sleep(0.001 if stopped else 0)


def _counter():
    n = [0]

    def read():
        n[0] += 1
        return [float(n[0])] * 7

    return read


def test_samples_at_the_configured_rate_on_a_fake_clock(tmp_path):
    """20 Hz: one sample every 50 ms of (fake) time, flushed about every second."""
    clock = _FakeClock(limit=3.0)
    log = MotionLog(
        tmp_path / "motion.jsonl",
        {"q": _counter()},
        hz=20.0,
        clock=clock,
        wait=clock.wait,
    ).start()
    _until(lambda: len(_lines(log.path)) >= 20)  # flushed while running
    _until(lambda: log.samples >= 61)
    time.sleep(0.05)
    log.close()
    lines = _lines(log.path)
    assert len(lines) == log.written == log.samples == 61  # 3 s at 20 Hz
    gaps = [b["t"] - a["t"] for a, b in zip(lines, lines[1:])]
    assert all(gap == pytest.approx(0.05, abs=1e-3) for gap in gaps)
    assert log.skipped == 0


def test_a_late_sample_skips_its_missed_slots(tmp_path):
    """A read that takes 3 slots costs 2 skipped samples, not a burst after it."""
    clock = _FakeClock(limit=1.0)
    count = _counter()

    def slow():
        value = count()
        if value[0] == 5:
            clock.wait(0.15)
        return value

    log = MotionLog(
        tmp_path / "motion.jsonl", {"q": slow}, hz=20.0, clock=clock, wait=clock.wait
    ).start()
    _until(lambda: log.samples >= 10)
    log.close()
    lines = _lines(log.path)
    gaps = [round(b["t"] - a["t"], 3) for a, b in zip(lines, lines[1:])]
    assert log.skipped == 2
    assert max(gaps) == pytest.approx(0.15) and min(gaps) == pytest.approx(0.05)


def test_an_idle_robot_writes_a_heartbeat_only(tmp_path):
    """Unchanged samples are not written, but one per heartbeat second is."""
    clock = _FakeClock(limit=5.0)
    log = MotionLog(
        tmp_path / "motion.jsonl",
        {"q": lambda: [0.1] * 7, "i": lambda: ()},
        hz=20.0,
        clock=clock,
        wait=clock.wait,
    ).start()
    _until(lambda: log.samples >= 101)  # 5 s
    log.close()
    lines = _lines(log.path)
    assert log.samples == 101
    assert len(lines) == 6  # t = 0, 1, ..., 5 s
    assert lines[0]["q"] == [0.1] * 7 and lines[0]["i"] == []


def test_a_failing_reader_gives_null_and_sampling_goes_on(tmp_path):
    """A reader that raises leaves its field null; the others are still logged."""
    clock = _FakeClock()

    def broken():
        raise ConnectionError("arm server gone")

    log = MotionLog(
        tmp_path / "motion.jsonl",
        {"q": _counter(), "odom": broken},
        hz=20.0,
        clock=clock,
        wait=clock.wait,
    ).start()
    _until(lambda: log.samples >= 5)
    log.close()
    line = _lines(log.path)[0]
    assert line["odom"] is None and line["q"] is not None


def test_the_skill_server_logs_the_call_in_progress(tmp_path):
    """On the fake robot: while open_gripper() runs, the samples carry its index from
    skill_calls.jsonl; before and after, none."""
    skills = build_skills(_cfg(), log_dir=tmp_path)
    assert skills.motion_log is not None
    entered, release = threading.Event(), threading.Event()
    execute = skills.arm.execute_action

    def held(arm_goal, gripper_goal):
        entered.set()
        release.wait(5.0)
        execute(arm_goal, gripper_goal)

    skills.arm.execute_action = held  # type: ignore[method-assign]
    skills.observe()
    worker = threading.Thread(target=skills.open_gripper)
    worker.start()
    assert entered.wait(5.0)
    time.sleep(0.3)  # ~6 samples during the call
    release.set()
    worker.join(10.0)
    assert not worker.is_alive()
    time.sleep(0.2)
    skills.close()
    calls = _lines(tmp_path / "skill_calls.jsonl")
    index = next(c["i"] for c in calls if c["call"] == "open_gripper")
    call = next(c for c in calls if c["call"] == "open_gripper")
    lines = _lines(tmp_path / "motion.jsonl")
    during = [ln for ln in lines if ln["i"] == [index]]
    # The held arm does not move: the call's start is one line, then heartbeats.
    assert during
    assert all(call["start"] - 0.01 <= ln["t"] <= call["end"] + 0.01 for ln in during)
    assert not [ln for ln in lines if ln["t"] < call["start"] - 0.06 and ln["i"]]
    assert lines[-1]["i"] == []
    assert during[0]["q"] is not None and len(during[0]["q"]) == 7
    assert during[0]["odom"] == pytest.approx(list(HOME), abs=1e-4)
    assert during[0]["map"] == pytest.approx(list(HOME), abs=1e-4)
    assert any("step" in ln for ln in lines)


def test_a_hung_read_never_blocks_a_skill_or_the_shutdown(tmp_path):
    """A reader stuck in its RPC holds nothing a skill needs: skills run at full speed,
    and close() returns anyway."""
    skills = build_skills(_cfg(), log_dir=None)
    stuck = threading.Event()
    log = MotionLog(
        tmp_path / "motion.jsonl",
        {"q": skills.arm.get_arm_state, "odom": lambda: stuck.wait(30.0)},
        hz=200.0,
    ).start()
    skills.motion_log = log
    t0 = time.monotonic()
    skills.retract()
    skills.open_gripper()
    skills.observe()
    assert time.monotonic() - t0 < 10.0
    t1 = time.monotonic()
    skills.close()
    assert time.monotonic() - t1 < 4.0
    stuck.set()


def test_sampling_alongside_skills_reads_the_robot_safely(tmp_path):
    """Fast sampling of the live fake robot while skills move it: every line is whole,
    and the skills' replies are unchanged."""
    skills = build_skills(_cfg("recording.motion_hz=200"), log_dir=tmp_path)
    for _ in range(3):
        skills.retract()
        skills.open_gripper()
        assert skills.observe()["robot"] is not None
    skills.close()
    lines = _lines(tmp_path / "motion.jsonl")
    assert lines and all(len(ln["q"]) == 7 for ln in lines)


def test_close_stops_the_sampler_and_flushes(tmp_path):
    """skills.close() ends the thread and writes the last lines; nothing after."""
    skills = build_skills(_cfg(), log_dir=tmp_path)
    log = skills.motion_log
    assert log is not None
    time.sleep(0.2)
    skills.close()
    assert not log._thread.is_alive()  # pylint: disable=protected-access
    path = tmp_path / "motion.jsonl"
    size = path.stat().st_size
    assert size > 0
    time.sleep(0.2)
    assert path.stat().st_size == size


def test_the_motion_log_is_the_skill_servers_only():
    """conf/skill_server.yaml samples at 20 Hz; toss_cycles (no recording block) does
    not start one."""
    assert _cfg().recording.motion_hz == pytest.approx(20.0)
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="toss_cycles", overrides=["mode=fake"])
    assert cfg.get("recording") is None


def test_readings_that_are_not_live_are_logged_as_null(tmp_path):
    """While the arm's readings are not live, live is false and q, g are null."""
    clock = _FakeClock()
    live = [True]
    log = MotionLog(
        tmp_path / "motion.jsonl",
        {"live": lambda: live[0], "q": _counter(), "g": lambda: 0.5},
        hz=20.0,
        clock=clock,
        wait=clock.wait,
    )
    first = log.sample(clock())
    live[0] = False
    second = log.sample(clock() + 0.05)
    assert first is not None and second is not None
    assert first["live"] is True and len(first["q"]) == 7 and first["g"] == 0.5
    assert second["live"] is False and second["q"] is None and second["g"] is None


class _FakeArmProxy:
    """The arm server's Arm, recording whether the interface thought its readings live
    during each call."""

    def __init__(self, iface):
        self.iface = iface
        self.seen = []

    def release(self):
        """Stop the cyclic loop."""
        self.seen.append(("release", self.iface.readings_are_live()))

    def resume(self):
        """Restart it."""
        self.seen.append(("resume", self.iface.readings_are_live()))

    def toss(self, **_):
        """The toss releases and resumes on the server; it may fail."""
        self.seen.append(("toss", self.iface.readings_are_live()))
        raise RuntimeError("wind-up missed")


def test_the_real_arm_interface_knows_when_its_readings_are_live():
    """Live while the cyclic loop runs: not from release() until resume() returns, nor
    during the toss (the server resumes after it, even when it fails)."""
    from prpl_tidybot.interfaces.real_arm_interface import (  # pylint: disable=import-outside-toplevel
        RealArmInterface,
    )

    iface = RealArmInterface.__new__(RealArmInterface)
    iface._cyclic = True  # pylint: disable=protected-access
    iface.arm = _FakeArmProxy(iface)
    assert iface.readings_are_live()
    iface.release()
    assert not iface.readings_are_live()
    iface.resume()
    assert iface.readings_are_live()
    with pytest.raises(RuntimeError):
        iface.toss()
    assert iface.readings_are_live()
    assert iface.arm.seen == [("release", False), ("resume", False), ("toss", False)]


def test_an_angular_pick_is_logged_as_not_live(tmp_path, monkeypatch):
    """The pilot's pick runs under the onboard controller (arm released): its samples
    say so, under the pick's call index, and the arm is live again after."""
    move = FakeArmInterface.move_angular

    def slow(self, goal):
        time.sleep(0.03)  # an onboard move takes time
        return move(self, goal)

    monkeypatch.setattr(FakeArmInterface, "move_angular", slow)
    skills = build_skills(
        _cfg("recording.motion_hz=200", "env.cube_pose=[0.6,-0.5,0.025]"),
        log_dir=tmp_path,
    )
    assert skills.cfg.pick.top_down.arm_controller == "angular"
    assert skills.pick()["held"] is True
    time.sleep(0.05)
    skills.close()
    index = next(
        c["i"] for c in _lines(tmp_path / "skill_calls.jsonl") if c["call"] == "pick"
    )
    lines = _lines(tmp_path / "motion.jsonl")
    released = [ln for ln in lines if ln["live"] is False]
    assert released and all(ln["q"] is None and ln["g"] is None for ln in released)
    assert all(ln["i"] == [index] for ln in released)
    assert lines[-1]["live"] is True and len(lines[-1]["q"]) == 7
