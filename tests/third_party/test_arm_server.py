"""Tests for the toss sequence in third_party/arm_server.py.

`Arm.toss` is exercised against a recording stand-in for `TorqueControlledArm`, so
the order of mode switches and the trajectory handed to the 1 kHz loop are checked
without the Kortex API.
"""

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
