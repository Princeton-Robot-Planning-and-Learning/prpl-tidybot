# Test doubles use pytest fixture names and controller signatures.
# pylint: disable=unused-argument,redefined-outer-name
"""Verified retract completion, failure handling and explicit connection choice."""

from types import SimpleNamespace

import numpy as np
import pytest

from prpl_tidybot.sprout import shutdown_arm


@pytest.mark.parametrize("fault", [None, "stalled", "invalid", "interrupted", "resume"])
def test_server_retract_waits_for_stable_feedback_and_always_closes(monkeypatch, fault):
    """Do not infer success from sending a fixed number of commands."""
    clock = SimpleNamespace(now=0.0)
    calls = []
    target = np.ones(7) * 0.2

    class Arm:
        """Fake arm with configurable feedback faults, no hardware access."""

        def resume(self):
            """Record controller startup, optionally fail before any motion."""
            calls.append("resume")
            if fault == "resume":
                raise RuntimeError("resume failed")

        def get_gripper_position(self):
            """Return an unchanged gripper position to preserve during retract."""
            return 0.37

        def get_joint_angles(self):
            """Return a target-equivalent wrapped pose or simulated failure."""
            calls.append("feedback")
            if fault == "interrupted":
                raise KeyboardInterrupt
            if fault == "invalid":
                return np.full(7, np.nan)
            return np.zeros(7) if fault == "stalled" else target + 2 * np.pi

        def execute_action_angular(self, qpos, gripper_pos):
            """Check target and that no command opens or closes the fingers."""
            assert np.allclose(qpos, target) and gripper_pos == 0.37
            calls.append("command")

        def close(self):
            """Record controller shutdown on every outcome."""
            calls.append("close")

    def sleep(period):
        """Advance the synthetic clock without waiting."""
        clock.now += period

    monkeypatch.setattr(shutdown_arm.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(shutdown_arm.time, "sleep", sleep)
    if fault:
        with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
            shutdown_arm.retract_via_server(Arm(), target, timeout=0.8)
    else:
        shutdown_arm.retract_via_server(Arm(), target, timeout=0.8)
        assert calls.count("feedback") >= 6
    assert calls[-1] == "close"
    if fault in ("resume", "invalid", "interrupted"):
        assert "command" not in calls


@pytest.mark.parametrize("error", [ConnectionRefusedError, TimeoutError, OSError])
def test_only_connection_refused_selects_direct_control(monkeypatch, error):
    """An unresponsive RPC server must not trigger a competing direct owner."""

    def connect(*args, **kwargs):
        """Simulate network errors without opening sockets."""
        raise error("synthetic failure")

    monkeypatch.setattr(shutdown_arm.socket, "create_connection", connect)
    if error is ConnectionRefusedError:
        assert not shutdown_arm.server_available()
    else:
        with pytest.raises(error):
            shutdown_arm.server_available()
