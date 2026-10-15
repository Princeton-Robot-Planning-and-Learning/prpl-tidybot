"""The wrist-camera servo inside scripts/toss_cycles.py's top-down pick (fake mode,
with a synthetic wrist camera that renders the cube where it really is relative to
the gripper)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
TARGET = (318.0, 280.0)


@pytest.fixture(autouse=True)
def _no_control_period_sleep(monkeypatch):
    """The fake arm needs no real-time ticks; a lagging or sagging fake would
    otherwise sleep through every settle."""
    monkeypatch.setattr("prpl_tidybot.toss_cycles.POLICY_CONTROL_PERIOD", 0.0)


def _compose(**extra):
    overrides = ["mode=fake", "seed=0", "interactive=true", "settle_after_toss_s=0"]
    overrides += ["cycles=1", "skip_toss=true", "pick.top_down.wrist_servo.settle_s=0"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="toss_cycles", overrides=overrides)


class _WristCamera:
    """Renders the cube at TARGET + K r, r its offset from the pinch point in the tool
    frame's horizontal axes. The true cube is `error` (arm frame) away from where the
    first hover aimed, i.e. where the ceiling cameras put it."""

    def __init__(self, arm, error, visible=True):
        self.arm = arm
        self.error = np.array(error)
        self.visible = visible
        self.k = np.array([[300.0, -1500.0], [-1450.0, -250.0]])
        self.cube = None
        self.cube_yaw = 0.0
        self.geom = TopDownPickGeometry()

    def _pinch(self):
        pos, approach, closing = forward_kinematics(self.arm.get_arm_state(), self.geom)
        x_axis = np.cross(closing, approach)[:2]
        return (
            pos[:2],
            x_axis / np.linalg.norm(x_axis),
            closing[:2] / np.linalg.norm(closing[:2]),
            approach[2] < -0.99,
        )

    def get_image(self):
        """The wrist view from where the fake arm is now."""
        import cv2 as cv  # pylint: disable=import-outside-toplevel

        image = np.full((480, 640, 3), 200, np.uint8)
        pinch, x_axis, y_axis, down = self._pinch()
        if not down:  # looking at the room, not the floor
            return image
        if self.cube is None:  # the first look down is from the planned hover
            self.cube = pinch + self.error
            # ... and the cube is turned 8 degrees from the planned closing yaw.
            self.cube_yaw = math.atan2(y_axis[1], y_axis[0]) + math.radians(8)
        if not self.visible:
            return image
        r = self.cube - pinch
        uv = np.array(TARGET) + self.k @ np.array([r @ x_axis, r @ y_axis])
        edge = np.array([math.cos(self.cube_yaw), math.sin(self.cube_yaw)])
        img_edge = self.k @ np.array([edge @ x_axis, edge @ y_axis])
        angle = math.degrees(math.atan2(img_edge[1], img_edge[0]))
        box = cv.boxPoints(((uv[0], uv[1]), (60, 60), angle))
        cv.fillPoly(image, [box.astype(np.int32)], (40, 90, 230))
        return image


def _run(tmp_path, monkeypatch, error, visible=True, **extra):
    cameras = []

    def factory(arm):
        cameras.append(_WristCamera(arm, error, visible))
        return cameras[-1]

    log = []
    original = FakeArmInterface.execute_action

    def execute_action(self, arm_goal, gripper_goal):
        log.append((list(arm_goal), gripper_goal))
        original(self, arm_goal, gripper_goal)

    monkeypatch.setattr(FakeArmInterface, "execute_action", execute_action)
    result = run_toss_cycles(
        _compose(**extra),
        log_dir=tmp_path,
        prompt_fn=lambda _: "y",
        wrist_camera=factory,
    )
    return result, cameras, log


@pytest.mark.parametrize("controller", ["compliant", "angular"])
def test_the_servo_moves_the_grasp_onto_a_cube_the_ceiling_misplaced(
    tmp_path, monkeypatch, controller
):
    """With the cube 2.2 cm from where the ceiling cameras put it, the gripper closes
    with its pinch point within 3 mm of the cube."""
    result, cameras, _ = _run(
        tmp_path,
        monkeypatch,
        (0.018, -0.012),
        **{"pick.top_down.arm_controller": controller},
    )
    assert result.finish_reason == "picked (toss skipped)"
    log = json.loads((tmp_path / "cycles.json").read_text())
    attempt = log["cycles"][0]["pick_attempts"][0]
    assert attempt["servo"]["aligned"], attempt["servo"]
    grasp = next(m for m in attempt["moves"] if m["name"] == "grasp")
    pinch, _, _ = forward_kinematics(
        np.radians(grasp["measured_deg"]), TopDownPickGeometry()
    )
    assert math.dist(pinch[:2], cameras[0].cube) < 0.003
    _, _, closing = forward_kinematics(
        np.radians(grasp["measured_deg"]), TopDownPickGeometry()
    )
    turn = math.atan2(closing[1], closing[0]) - cameras[0].cube_yaw
    assert abs((turn + math.pi / 4) % (math.pi / 2) - math.pi / 4) < math.radians(4)
    assert list((tmp_path / "wrist_servo").glob("*_raw.png"))


def test_a_nudge_whose_ik_flips_branch_is_refused(tmp_path, monkeypatch):
    """As in the cylinder servo, a nudge whose IK would swing an arm joint by more
    than max_joint_step is not commanded; the servo stops and nothing descends."""
    import prpl_tidybot.toss_cycles as tc  # pylint: disable=import-outside-toplevel

    real = tc.top_down_joint_targets
    calls = {"n": 0}

    def flipping(ax, ay, yaw, geom, seed=None):
        calls["n"] += 1
        hover, grasp = real(ax, ay, yaw, geom, seed=seed)
        if calls["n"] >= 3:  # the servo's probes and nudges
            hover = list(hover)
            hover[3] += 1.0
        return hover, grasp

    monkeypatch.setattr(tc, "top_down_joint_targets", flipping)
    result, _, log = _run(
        tmp_path, monkeypatch, (0.01, 0.0), **{"pick.top_down.max_attempts": 1}
    )
    assert "pick failed" in result.finish_reason
    attempt = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0][
        "pick_attempts"
    ][0]
    assert "did not arrive" in attempt["note"]
    assert not [g for _, g in log if g == 1.0]


def test_no_cube_in_the_wrist_camera_means_no_descent_and_no_close(
    tmp_path, monkeypatch
):
    """The cube must be seen before the gripper goes down: otherwise the attempt is a
    miss without descending or closing."""
    result, _, log = _run(
        tmp_path,
        monkeypatch,
        (0.0, 0.0),
        visible=False,
        **{"pick.top_down.max_attempts": 1},
    )
    assert "pick failed" in result.finish_reason
    attempt = json.loads((tmp_path / "cycles.json").read_text())["cycles"][0][
        "pick_attempts"
    ][0]
    assert "not seen" in attempt["note"]
    assert [m["name"] for m in attempt["moves"]] == ["hover", "retreat"]
    assert not [g for _, g in log if g == 1.0]
