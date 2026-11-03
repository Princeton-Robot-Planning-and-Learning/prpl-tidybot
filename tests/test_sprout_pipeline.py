# Test doubles keep production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Exercise Sprout through the actual main Runner/env/executors without hardware."""

import json
import threading

import numpy as np
import pytest

from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.real_sim.plan_executors.arm_motion3d import (
    CarrotArmMotion3DPlanExecutor,
)
from prpl_tidybot.real_sim.plan_executors.base_motion3d import (
    PurePursuitBaseMotion3DPlanExecutor,
)
from prpl_tidybot.sprout.pipeline import (
    SproutPipeline,
    arm_pairs,
    base_pairs,
    fake_interface,
)
from prpl_tidybot.sprout.pipeline_runtime import PipelineRuntime, real_interface
from prpl_tidybot.sprout.sequence import GraspPose, Stopped
from prpl_tidybot.sprout.urdf_collision import PRPLCollisionScene


@pytest.fixture(name="controls")
def fixture_controls():
    """All inputs are synthetic test settings, never a hardware calibration."""
    return {
        "navigation": {"position_tolerance_m": 0.005, "yaw_tolerance_rad": 0.01},
        "motion": {"close_position": 0.7},
        "servo": {
            "startup_timeout_s": 1,
            "max_updates": 20,
            "tolerance_m": 0.005,
            "max_step_m": 0.005,
            "max_travel_m": 0.03,
        },
    }


def test_only_supplied_obstacles_block_the_robot():
    """Unregistered cabinets are absent, but a perceived Sprout box still blocks."""
    q = [0, 0.2487, np.pi, -2.29235, -0.0247, 0.87245, np.pi / 2]
    with PRPLCollisionScene() as world:
        world.set([-0.5, 1.5, 0], q)
        assert not world.collision(set())
        sprout = world.box([-0.5, 1.5, 0.15], [0.3, 0.3, 0.15])
        assert world.collision({sprout}, base_only=True)


@pytest.mark.parametrize("model_matches", [False, True])
def test_asset_mismatch_rejected_before_connecting_rpc(monkeypatch, model_matches):
    """A reviewed model mismatch must fail before hardware imports/connections."""
    monkeypatch.setenv("SPROUT_TIDYBOT_MODEL", "/unused-public-model.xml")
    monkeypatch.setattr(
        "prpl_tidybot.sprout.pipeline_runtime.model_digest",
        lambda _: "expected" if model_matches else "changed",
    )
    monkeypatch.setattr(
        "prpl_tidybot.sprout.pipeline_runtime.urdf_digest", lambda: "changed"
    )
    with pytest.raises(ValueError, match="assets differ"):
        real_interface(
            {"model_sha256": "expected", "collision_model_sha256": "expected"}
        )


def test_main_pipeline_base_arm_close_open(controls):
    """Base and all arm/gripper actions really pass through upstream executors."""
    commands = []
    interface = fake_interface({"arm_joints_rad": [0] * 7, "base_xyyaw": [0, 0, 3.13]})
    pipeline = SproutPipeline(
        interface,
        controls,
        threading.Event(),
        control_period=0,
        on_step=lambda *args: commands.append(args[2]),
    )
    assert isinstance(pipeline.env, RealTidyBotEnv)
    assert isinstance(
        pipeline.executor._base_executor, PurePursuitBaseMotion3DPlanExecutor
    )
    assert isinstance(pipeline.executor._arm_executor, CarrotArmMotion3DPlanExecutor)
    try:
        pipeline.run(
            "base", base_pairs(pipeline.state, [[0, 0, 3.13], [0.4, 0.2, -3.13]])
        )
        assert interface.get_map_base_state().x == pytest.approx(0.4, abs=0.005)
        for phase, q, grip in (
            ("pregrasp", 0.2, "open"),
            ("close", 0.2, "close"),
            ("lift", 0.3, "close"),
            ("lower", 0.2, "close"),
            ("open", 0.2, "open"),
            ("retreat", 0.3, "open"),
        ):
            pipeline.run(
                phase,
                arm_pairs(
                    pipeline.state, [{"arm_joints_rad": [q] * 7, "gripper": grip}]
                ),
            )
        assert interface.get_gripper_state() == 0
        assert interface.get_arm_state() == pytest.approx([0.3] * 7)
        assert any(command.gripper_goal == 0.7 for command in commands)
    finally:
        pipeline.close()


def test_stop_interrupts_original_runner_before_next_tick(controls):
    """The transport stop must interrupt the inherited Runner inner loop."""
    stop, ticks = threading.Event(), []

    def stop_after_one(*_):
        ticks.append(True)
        stop.set()

    pipeline = SproutPipeline(
        fake_interface({"arm_joints_rad": [0] * 7, "base_xyyaw": [0, 0, 0]}),
        controls,
        stop,
        control_period=0,
        on_step=stop_after_one,
    )
    with pytest.raises(Stopped):
        pipeline.run("base", base_pairs(pipeline.state, [[0, 0, 0], [1, 0, 0]]))
    assert len(ticks) == 1
    pipeline.close()


def test_runtime_uses_main_stack_and_preserves_servo_correction(
    tmp_path, controls, monkeypatch
):
    """Read-only construction, main execution, camera age off, corrected descent."""
    monkeypatch.setenv("SPROUT_TIDYBOT_MODEL", "/unused-public-model.xml")
    report = {"observed": {"grasp_xyz": [0.3, 0.4, 0.1]}, "plan": {"grasp": {"yaw": 0}}}
    (tmp_path / "review.json").write_text(json.dumps(report))
    (tmp_path / "live.json").write_text(json.dumps(controls))
    trajectory = []
    for phase, z, grip in (
        ("pregrasp", 0.2, "open"),
        ("descend", 0.1, "open"),
        ("close", 0.1, "close"),
        ("lift", 0.25, "close"),
        ("lower", 0.1, "close"),
        ("open", 0.1, "open"),
        ("retreat", 0.2, "open"),
    ):
        trajectory.append(
            {
                "phase": phase,
                "arm_joints_rad": [0.3, 0.4, z, 0, 0, 0, 0],
                "pinch_xyz": [0.3, 0.4, z],
                "gripper": grip,
            }
        )
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
    camera_options, activated = [], []

    class Perception:
        """Ceiling feedback with deliberately old acquisition metadata."""

        def __init__(self, *_, **kwargs):
            camera_options.append(kwargs)

        def target(self):
            """Return the observed shifted forearm."""
            return GraspPose(0.32, 0.4, 0.1, 0), -1000

        def base(self):
            """Return the measured stationary map pose."""
            return np.zeros(3), -1000

        def close(self):
            """Release fake camera resources."""

    class Kinematics:
        """Cartesian-as-joint fixture for checking retained servo corrections."""

        def __init__(self, *_):
            pass

        def full_pose(self, q):
            """Read the fixture pose."""
            return np.asarray(q[:3]), np.eye(3)

        def solve(self, xyz, _rotation, seed):
            """Solve the fixture Cartesian target."""
            return [*xyz, *seed[3:]]

        def check_path(self, *_, **__):
            """Fixture paths contain no obstacles."""

    interface = fake_interface(
        {"arm_joints_rad": [0.3, 0.4, 0.2, 0, 0, 0, 0], "base_xyyaw": [0, 0, 0]}
    )

    def connect(_):
        activated.append(True)
        return interface

    runtime = PipelineRuntime(
        tmp_path / "review.json",
        interface_factory=connect,
        perception_factory=Perception,
        kinematics_factory=Kinematics,
        arm_reader=lambda _: {"joints_rad": interface.get_arm_state()},
    )
    assert not activated
    assert runtime.observe()["ready"] and not activated
    # Keep the real Runner/env/executors but skip wall-clock sleeps in the test.
    monkeypatch.setattr("prpl_tidybot.real_env.time.sleep", lambda _: None)
    result = runtime.execute(report)
    assert result["execution_backend"] == "prpl_main_runner"
    assert camera_options == [{"check_acquisition_age": False}]
    assert interface.get_arm_state()[:3] == pytest.approx([0.32, 0.4, 0.2])
    assert interface.get_gripper_state() == 0
    runtime.close()
