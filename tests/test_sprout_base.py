# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Base path, body-envelope and shutdown failure checks without hardware."""

import copy
import itertools
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from prpl_tidybot.sprout import base_motion
from prpl_tidybot.sprout.base_motion import validate_navigation
from prpl_tidybot.sprout.base_planning import (
    free,
    route,
    segment_free,
    validate_base_plan,
)
from prpl_tidybot.sprout.body_obstacle import check_contained, detect_body
from prpl_tidybot.sprout.shutdown_arm import retract

BOX = {"center": [0, 0, 0.2], "half_size": [0.3, 0.4, 0.2]}
BOUNDS = [[-2, -2], [2, 2]]


def test_route_avoids_box_along_entire_transit():
    """Verify route avoids box along entire transit."""
    path = route([-1, 0], [1, 0], [BOX], 0.2, BOUNDS)
    assert len(path) > 2
    for a, b in itertools.pairwise(path):
        assert segment_free(a, b, [BOX], 0.2, BOUNDS)
    assert not segment_free([-1, 0], [1, 0], [BOX], 0.2, BOUNDS)


def test_unreachable_and_boundary_paths_rejected():
    """Verify unreachable and boundary paths rejected."""
    assert not free([1.9, 0], [BOX], 0.2, BOUNDS)
    with pytest.raises(ValueError):
        route([0, 0], [1, 1], [BOX], 0.2, BOUNDS)
    wall = {"center": [0, 0, 0.1], "half_size": [0.1, 2, 0.1]}
    with pytest.raises(ValueError, match="No collision-free"):
        route([-1, 0], [1, 0], [wall], 0.2, [[-1.5, -1], [1.5, 1]])


def test_swept_check_catches_thin_obstacle_between_samples():
    """Verify swept check catches thin obstacle between samples."""
    tiny = {"center": [0.005, 0, 0.1], "half_size": [0.001, 0.001, 0.1]}
    assert not segment_free([0, 0], [0.02, 0], [tiny], 0.001, BOUNDS)


def test_validation_rejects_corrupt_or_incomplete_plans():
    """Verify validation rejects corrupt or incomplete plans."""
    plan = {
        "required": True,
        "path_xyyaw": [[-1, 1, 0], [1, 1, 0]],
        "radius_m": 0.2,
        "bounds_xy": BOUNDS,
        "obstacles": [BOX],
    }
    validate_base_plan(plan)
    for key, value in [
        ("radius_m", float("nan")),
        ("bounds_xy", [[0, 0], [0, 0]]),
        ("path_xyyaw", [[0, 0, 0]]),
        ("obstacles", []),
    ]:
        bad = copy.deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_base_plan(bad)


def test_body_cannot_shrink_to_hide_a_moved_limb():
    """Verify body cannot shrink to hide a moved limb."""
    check_contained(BOX, BOX)
    moved = copy.deepcopy(BOX)
    moved["center"][0] += 0.1
    with pytest.raises(RuntimeError):
        check_contained(moved, BOX)


def test_body_requires_two_camera_views():
    """Verify body requires two camera views."""
    with pytest.raises(ValueError, match="both calibrated"):
        detect_body(
            np.zeros((100, 100, 3), np.uint8),
            np.zeros((100, 100, 3), np.uint8),
            {"maximum_height_m": 0.3, "margin_m": 0.05, "views": []},
        )


def test_navigation_requires_explicit_physical_verification():
    """Verify navigation requires explicit physical verification."""
    with pytest.raises(ValueError, match="verification"):
        validate_navigation({})
    with pytest.raises(ValueError, match="speed_m_s"):
        validate_navigation({"verified": True})


class FakeBase:
    """Test double for FakeBase."""

    def __init__(self, event):
        self.event = event
        self.calls = []
        self.callback = None

    def SetServoingMode(self, mode):
        """Provide the SetServoingMode test fixture or callback."""
        self.calls.append("mode")

    def ReadAllActions(self, query):
        """Provide the ReadAllActions test fixture or callback."""
        return SimpleNamespace(action_list=[SimpleNamespace(name="Retract", handle=3)])

    def OnNotificationActionTopic(self, callback, options):
        """Provide the OnNotificationActionTopic test fixture or callback."""
        self.callback = callback
        return 1

    def ExecuteActionFromReference(self, handle):
        """Provide the ExecuteActionFromReference test fixture or callback."""
        self.calls.append("retract")
        if self.event is not None:
            self.callback(SimpleNamespace(action_event=self.event))

    def Stop(self):
        """Provide the Stop test fixture or callback."""
        self.calls.append("stop")

    def Unsubscribe(self, handle):
        """Provide the Unsubscribe test fixture or callback."""
        self.calls.append("unsubscribe")


@pytest.mark.parametrize("event,success", [(1, True), (2, False), (None, False)])
def test_shutdown_requires_action_end(event, success):
    """Verify shutdown requires action end."""
    base = FakeBase(event)
    messages = SimpleNamespace(
        ServoingModeInformation=SimpleNamespace,
        SINGLE_LEVEL_SERVOING=1,
        RequestedActionType=SimpleNamespace,
        REACH_JOINT_ANGLES=2,
        ACTION_END=1,
        ACTION_ABORT=2,
        NotificationOptions=SimpleNamespace,
    )
    if success:
        retract(SimpleNamespace(base=base), messages, timeout=0.001)
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            retract(SimpleNamespace(base=base), messages, timeout=0.001)
    assert ("stop" in base.calls) is not success
    assert base.calls[-1] == "unsubscribe"


@pytest.mark.parametrize("fault", [None, "body_lost", "arm_moved", "stop"])
def test_base_driver_stops_on_arrival_and_faults(monkeypatch, fault):
    """Verify base driver stops on arrival and faults."""

    clock = SimpleNamespace(now=100.0)
    latch = threading.Event()

    class Vehicle:
        """Test double for Vehicle."""

        def __init__(self, **kwargs):
            self.control_loop_running = False
            self.dx = np.zeros(3)
            self.pose = np.array([0.0, 1.0, 0.0])
            self.stopped = False

        def start_control(self):
            """Provide the start control test fixture or callback."""
            self.control_loop_running = True

        def stop_control(self):
            """Provide the stop control test fixture or callback."""
            self.control_loop_running = False
            self.stopped = True

        def set_target_velocity(self, v, frame):
            """Provide the set target velocity test fixture or callback."""
            self.dx = np.asarray(v)

    vehicle = Vehicle()

    class Perception:
        """Test double for Perception."""

        def base(self):
            """Provide the base test fixture or callback."""
            return vehicle.pose.copy(), base_motion.time.time()

        def body(self):
            """Provide the body test fixture or callback."""
            if fault == "body_lost":
                raise RuntimeError("body lost")
            if fault == "stop":
                latch.set()
            return {"captured_at_unix": base_motion.time.time()}

    def reader():
        """Provide the reader test fixture or callback."""
        return {"joints_rad": [0.1 if fault == "arm_moved" else 0.0] * 7}

    config = {
        "verified": True,
        "speed_m_s": 0.04,
        "yaw_speed_rad_s": 0.1,
        "acceleration_m_s2": 0.08,
        "yaw_acceleration_rad_s2": 0.15,
        "position_tolerance_m": 0.005,
        "yaw_tolerance_rad": 0.01,
        "path_tolerance_m": 0.015,
        "timeout_s": 10.0,
        "period_s": 0.02,
        "arm_tolerance_rad": 0.01,
        "camera_max_age_s": 0.25,
    }

    def sleep(dt):
        """Provide the sleep test fixture or callback."""
        clock.now += dt
        vehicle.pose += vehicle.dx * dt

    monkeypatch.setattr(base_motion.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(base_motion.time, "sleep", sleep)
    plan = {
        "required": True,
        "path_xyyaw": [[0, 1, 0], [0.02, 1, 0]],
        "radius_m": 0.2,
        "bounds_xy": BOUNDS,
        "obstacles": [BOX],
    }
    motion = base_motion.BaseMotion(
        config, Perception(), reader, latch, vehicle_factory=lambda **kwargs: vehicle
    )
    if fault:
        with pytest.raises(RuntimeError):
            motion.execute(plan, np.zeros(7))
    else:
        motion.execute(plan, np.zeros(7))
    assert vehicle.stopped and not vehicle.control_loop_running
