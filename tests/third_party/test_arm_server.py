"""Tests for third_party/arm_server.py.

`Arm.toss` is exercised against a recording stand-in for `TorqueControlledArm`, so the
order of mode switches and the trajectory handed to the 1 kHz loop are checked without
the Kortex API. The RPC tests run the real `ArmManager` server in-process against a fake
Kortex session, to check that every client shares one Arm and that at most one torque
loop ever runs.
"""

import threading
from unittest import mock

import numpy as np
import pytest

from prpl_tidybot.third_party.arm_server import Arm
from prpl_tidybot.toss import plan_toss_swing, toss_windup_pose


class _RecordingArm:
    """Stands in for `TorqueControlledArm`; logs every call in order."""

    def __init__(self, reached_offset_deg: float = 0.0) -> None:
        self.calls: list[tuple] = []
        self.cyclic_running = True
        self._reached_offset = reached_offset_deg
        self._target: list[float] | None = None

    def stop_cyclic(self):
        """Stop the torque loop."""
        self.calls.append(("stop_cyclic",))
        self.cyclic_running = False

    def init_cyclic(self, callback):
        """Start the torque loop."""
        del callback
        self.calls.append(("init_cyclic",))
        self.cyclic_running = True

    def move_angular(self, joint_angles):
        """High-level joint move; lands at the target plus the offset."""
        self.calls.append(("move_angular", list(joint_angles)))
        self._target = list(joint_angles)

    def get_measured_joint_angles(self):
        """Where the high-level move ended."""
        assert self._target is not None
        return [q + self._reached_offset for q in self._target]

    def get_feedback_joint_positions(self):
        """Same as the measured angles."""
        return self.get_measured_joint_angles()

    def run_position_trajectory(self, trajectory, gripper_release_step):
        """Record the swing."""
        self.calls.append(("run_position_trajectory", trajectory, gripper_release_step))

    def gripper_to(self, value, timeout):
        """High-level gripper move; reports where it stopped."""
        self.calls.append(("gripper_to", value, timeout))
        return 0.5


def _arm(recording: _RecordingArm) -> Arm:
    arm = Arm.__new__(Arm)
    arm.arm = recording
    arm.command_queue = mock.Mock()
    arm.command_queue.empty.return_value = True
    arm.controller = None
    return arm


def test_toss_releases_winds_up_swings_and_resumes(monkeypatch):
    """Torque loop off, high-level move to the wind-up pose, Jimmy's position-servo
    swing from the measured pose, then the torque loop back on."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    recording = _RecordingArm()
    _arm(recording).toss(heading=20.0, max_vel=60.0, gripper_release_ms=900)
    names = [call[0] for call in recording.calls]
    assert names == [
        "stop_cyclic",
        "move_angular",
        "run_position_trajectory",
        "init_cyclic",
    ]
    assert recording.calls[1][1] == toss_windup_pose(20.0)
    _, trajectory, release_step = recording.calls[2]
    expected = plan_toss_swing(toss_windup_pose(20.0), 20.0, max_vel=60.0)
    assert np.allclose(trajectory, expected)
    assert release_step == 900


def test_toss_does_not_swing_when_the_windup_pose_is_not_reached(monkeypatch):
    """More than 1 deg off the wind-up pose: no swing, the arm is re-acquired, and the
    failure is raised to the caller."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    recording = _RecordingArm(reached_offset_deg=2.0)
    with pytest.raises(RuntimeError, match="wind-up"):
        _arm(recording).toss()
    names = [call[0] for call in recording.calls]
    assert "run_position_trajectory" not in names
    assert names[-1] == "init_cyclic"


def test_toss_rejects_bad_parameters_before_moving(monkeypatch):
    """A speed the swing planner rejects fails before the torque loop is touched."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    recording = _RecordingArm()
    with pytest.raises(ValueError):
        _arm(recording).toss(max_vel=500.0)
    assert not recording.calls


def test_move_joints_angular_takes_radians_and_reports_the_measured_pose():
    """Radians in, Kortex degrees (0-360) to the onboard controller, wrapped radians
    of the measured pose back."""
    recording = _RecordingArm(reached_offset_deg=0.5)
    recording.cyclic_running = False
    goal = [0.0, 1.0, -3.0, -1.5, 0.2, -0.4, 1.6]
    measured = _arm(recording).move_joints_angular(goal)
    sent = recording.calls[0][1]
    assert all(0.0 <= v < 360.0 for v in sent)
    assert np.allclose(np.radians(sent) % (2 * np.pi), np.array(goal) % (2 * np.pi))
    err = (np.array(measured) - goal + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(err, np.radians(0.5))


def test_move_joints_angular_refuses_while_the_torque_loop_runs():
    """The onboard controller cannot move the arm under low-level servoing."""
    with pytest.raises(RuntimeError, match="release"):
        _arm(_RecordingArm()).move_joints_angular([0.0] * 7)


def test_gripper_angular_moves_the_gripper_under_the_onboard_controller():
    """Returns where the gripper stopped (closing on a cube never reaches 1.0)."""
    recording = _RecordingArm()
    recording.cyclic_running = False
    assert _arm(recording).gripper_angular(1.0, timeout=3.0) == 0.5
    assert recording.calls == [("gripper_to", 1.0, 3.0)]


class _FakeKortexArm:
    """Stands in for `TorqueControlledArm` (one Kortex session) and tracks whether its 1
    kHz torque loop runs, so a test can count the loops commanding the arm."""

    sessions: list["_FakeKortexArm"] = []

    def __init__(self) -> None:
        self.cyclic_running = False
        self.connected = True
        self.q = np.zeros(7)
        self.gripper_pos = 0.0
        _FakeKortexArm.sessions.append(self)

    def set_joint_limits(self, speed_limits, acceleration_limits):
        """No-op."""
        del speed_limits, acceleration_limits

    def clear_faults(self):
        """No-op."""

    def open_gripper(self):
        """No-op."""

    def retract(self):
        """No-op."""

    def init_cyclic(self, control_callback):
        """Start the torque loop; asserts as the real one does."""
        del control_callback
        assert not self.cyclic_running, "Cyclic thread is already running"
        self.cyclic_running = True

    def stop_cyclic(self):
        """Stop the torque loop."""
        self.cyclic_running = False

    def disconnect(self):
        """End the Kortex session."""
        self.connected = False


def _live_loops() -> int:
    return sum(session.cyclic_running for session in _FakeKortexArm.sessions)


@pytest.fixture(name="kortex")
def _kortex(monkeypatch):
    """The arm server with every Kortex session faked and no shared Arm yet."""
    # pylint: disable=import-outside-toplevel
    import time as real_time
    from types import SimpleNamespace

    from prpl_tidybot.third_party import arm_server

    monkeypatch.setattr(_FakeKortexArm, "sessions", [])
    monkeypatch.setattr(arm_server, "TorqueControlledArm", _FakeKortexArm)
    monkeypatch.setattr(arm_server, "IKSolver", lambda ee_offset: None)
    monkeypatch.setattr(
        arm_server, "time", SimpleNamespace(sleep=lambda _: None, time=real_time.time)
    )
    monkeypatch.setattr(arm_server, "_shared_arm", None, raising=False)
    return arm_server


_SERVERS: list = []  # keeps each test's in-process RPC server alive


@pytest.fixture(name="arm_rpc")
def _arm_rpc(kortex):
    """An in-process arm RPC server (as on the NUC); returns a client factory, each call
    a separate client process's `manager.Arm()`."""
    manager = kortex.ArmManager(address=("127.0.0.1", 0), authkey=b"test")
    server = manager.get_server()
    _SERVERS.append(server)
    # Only the accepter (and the stop event serve_forever would make): serve_forever
    # itself resets sys.stdout and sys.exit()s, which breaks pytest's capture.
    server.stop_event = threading.Event()
    threading.Thread(target=server.accepter, daemon=True).start()

    def client():
        connection = kortex.ArmManager(address=server.address, authkey=b"test")
        connection.connect()
        return connection.Arm()  # pylint: disable=no-member

    return client


def test_every_client_drives_the_same_arm_with_one_torque_loop(arm_rpc):
    """Two clients (the second after the first died without close(), runs 21c/21d):

    one Kortex session and one torque loop, not one of each per client.
    """
    first = arm_rpc()
    first.reset(reset_arm=False)
    second = arm_rpc()
    second.reset(reset_arm=False)
    assert len(_FakeKortexArm.sessions) == 1
    assert _live_loops() == 1


def test_resume_while_the_loop_runs_restarts_it_rather_than_adding_one(kortex):
    """`resume()` on a running loop stops it first; it never asserts or doubles."""
    arm = kortex.Arm()
    arm.resume()
    arm.resume()
    assert _live_loops() == 1


def test_close_stops_the_loop_and_the_next_client_reconnects(arm_rpc):
    """Close() ends the loop and the Kortex session; the next client's reset() opens a
    fresh session on the same shared Arm, as each client got before."""
    first = arm_rpc()
    first.reset(reset_arm=False)
    first.close()
    assert _live_loops() == 0
    assert not _FakeKortexArm.sessions[0].connected
    second = arm_rpc()
    second.reset(reset_arm=False)
    assert [s.connected for s in _FakeKortexArm.sessions] == [False, True]
    assert _live_loops() == 1


def test_the_server_reports_whether_a_torque_loop_is_running(arm_rpc):
    """A client can see a loop a previous client left running."""
    arm_rpc().reset(reset_arm=False)
    assert arm_rpc().is_cyclic_running()
    arm_rpc().release()
    assert not arm_rpc().is_cyclic_running()
