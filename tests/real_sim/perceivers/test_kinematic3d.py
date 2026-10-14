"""Tests for real_sim/perceivers/kinematic3d.py."""

import math

import numpy as np
import pytest
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from spatialmath import SE2

from prpl_tidybot.camera_constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.real_sim.perceivers.kinematic3d import (
    BaseMotion3DPerceiver,
    CylinderShelf3DPerceiver,
    PrplLab3DPerceiver,
    Shelf3DPerceiver,
)
from prpl_tidybot.real_sim.perceivers.target_source import (
    ConstantCylinderTargets,
    ConstantTargetSource,
)
from prpl_tidybot.structs import TidyBotObservation


def _make_obs() -> TidyBotObservation:
    return TidyBotObservation(
        arm_conf=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        base_pose=SE2(x=0.0, y=0.0, theta=0.0),
        map_base_pose=SE2(x=2.0, y=1.0, theta=-0.4),
        gripper=0.6,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )


def test_step_populates_robot_features():
    """The perceiver lifts obs.{map_base_pose, arm_conf, gripper} into the kinematic3d
    robot feature dict; grasp_active is 0 and grasp_tf is identity."""
    perceiver = PrplLab3DPerceiver()
    state = perceiver.step(_make_obs(), {})
    robot = state.get_object_from_name("robot")
    assert state.get(robot, "pos_base_x") == 2.0
    assert state.get(robot, "pos_base_y") == 1.0
    assert state.get(robot, "pos_base_rot") == pytest.approx(-0.4)
    assert state.get(robot, "joint_1") == 0.1
    assert state.get(robot, "joint_7") == 0.7
    assert state.get(robot, "finger_state") == 0.6
    assert state.get(robot, "grasp_active") == 0.0
    assert state.get(robot, "grasp_tf_qw") == 1.0
    for f in ("grasp_tf_x", "grasp_tf_y", "grasp_tf_z", "grasp_tf_qx"):
        assert state.get(robot, f) == 0.0


def test_reset_matches_step():
    """Reset() and step() are equivalent for this stateless perceiver."""
    perceiver = PrplLab3DPerceiver()
    obs = _make_obs()
    assert perceiver.reset(obs, {}) == perceiver.step(obs, {})


def test_base_motion3d_perceiver_emits_target_from_source():
    """`BaseMotion3DPerceiver` puts the TargetSource's (x, y, z) into the target
    object's features verbatim."""
    perceiver = BaseMotion3DPerceiver(
        target_source=ConstantTargetSource(1.5, -0.5, 0.2)
    )
    state = perceiver.step(_make_obs(), {})
    target = state.get_object_from_name("target")
    assert state.get(target, "x") == pytest.approx(1.5)
    assert state.get(target, "y") == pytest.approx(-0.5)
    assert state.get(target, "z") == pytest.approx(0.2)


def test_cylinder_shelf3d_perceiver_emits_every_cylinder_and_the_shelf():
    """`CylinderShelf3DPerceiver` emits one upright cylinder per spec, in spec order,
    each with its own radius and height at its targets' position, plus the shelf fixture
    at the threaded-in pose."""
    cylinders = [
        {"radius": 0.039, "height": 0.233, "fake_xy": [0.5, 0.0]},
        {"radius": 0.040, "height": 0.210, "fake_xy": [0.3, -0.6]},
    ]
    perceiver = CylinderShelf3DPerceiver(
        cylinders=cylinders,
        targets=ConstantCylinderTargets(cylinders),
        shelf_pose=(1.5, 1.5, 0.02),
    )
    state = perceiver.step(_make_obs(), {})
    first = state.get_object_from_name("cylinder0")
    assert state.get(first, "pose_x") == pytest.approx(0.5)
    assert state.get(first, "pose_y") == pytest.approx(0.0)
    assert state.get(first, "pose_z") == pytest.approx(0.1165)
    assert state.get(first, "half_extent_x") == pytest.approx(0.039)
    assert state.get(first, "half_extent_z") == pytest.approx(0.1165)
    second = state.get_object_from_name("cylinder1")
    assert state.get(second, "pose_x") == pytest.approx(0.3)
    assert state.get(second, "pose_y") == pytest.approx(-0.6)
    assert state.get(second, "pose_z") == pytest.approx(0.105)
    assert state.get(second, "half_extent_y") == pytest.approx(0.040)
    assert state.get(second, "half_extent_z") == pytest.approx(0.105)
    assert len(state.get_objects(Kinematic3DCuboidType)) == 2
    shelf = state.get_object_from_name("shelf")
    assert state.get(shelf, "pose_x") == pytest.approx(1.5)
    assert state.get(shelf, "pose_y") == pytest.approx(1.5)
    assert state.get(shelf, "pose_z") == pytest.approx(0.02)


def test_cylinder_shelf3d_perceiver_caches_targets_at_reset():
    """Targets are read once at reset; a source that changes afterwards is ignored until
    the next reset."""

    class _Moving:
        def __init__(self):
            self.calls = 0

        def get_targets(self):
            """A cylinder that drifts 1 m per call."""
            self.calls += 1
            return [(float(self.calls), 0.0, 0.1)]

    perceiver = CylinderShelf3DPerceiver(
        cylinders=[{"radius": 0.03, "height": 0.2}],
        targets=_Moving(),  # type: ignore[arg-type]
        shelf_pose=(1.5, 1.5, 0.02),
    )
    state = perceiver.reset(_make_obs(), {})
    cylinder = state.get_object_from_name("cylinder0")
    assert state.get(cylinder, "pose_x") == pytest.approx(1.0)
    state = perceiver.step(_make_obs(), {})
    assert state.get(cylinder, "pose_x") == pytest.approx(1.0)
    state = perceiver.reset(_make_obs(), {})
    assert state.get(cylinder, "pose_x") == pytest.approx(2.0)


def test_cylinder_shelf3d_perceiver_rejects_target_count_mismatch():
    """A targets source that reports a different number of cylinders than the specs is
    an error, not a silent truncation."""

    class _TooFew:
        def get_targets(self):
            """One position for two specs."""
            return [(0.0, 0.0, 0.1)]

    perceiver = CylinderShelf3DPerceiver(
        cylinders=[{"radius": 0.03, "height": 0.2}, {"radius": 0.03, "height": 0.2}],
        targets=_TooFew(),  # type: ignore[arg-type]
        shelf_pose=(1.5, 1.5, 0.02),
    )
    with pytest.raises(RuntimeError, match="2 cylinder spec"):
        perceiver.reset(_make_obs(), {})


def _cube_yaw(state):
    cube = state.get_object_from_name("cube0")
    assert state.get(cube, "pose_qx") == pytest.approx(0.0)
    assert state.get(cube, "pose_qy") == pytest.approx(0.0)
    return 2 * math.atan2(state.get(cube, "pose_qz"), state.get(cube, "pose_qw"))


def _angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize(
    "cube_xy,source_yaw",
    [((3.0, 1.0), 0.0), ((0.5, 0.0), 0.0), ((2.0, 3.0), 0.3), ((0.4, 2.2), -0.6)],
)
def test_shelf3d_perceiver_turns_the_cube_yaw_to_face_the_robot(cube_xy, source_yaw):
    """A 90-degree-symmetric cube reports the equivalent yaw (source + k * 90 degrees)
    nearest the robot-to-cube bearing, inside [-3pi/4, 3pi/4] (the shelf3d pick asserts
    on angles outside that once its own offset is added)."""
    half = 0.025
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(*cube_xy, half, yaw=source_yaw),
        shelf_pose=(1.5, 1.5, 0.02),
        cube_half_extents=(half, half, half),
    )
    yaw = _cube_yaw(perceiver.reset(_make_obs(), {}))  # robot at (2.0, 1.0)
    period = math.pi / 2
    k = (yaw - source_yaw) / period
    assert k == pytest.approx(round(k), abs=1e-6)
    assert -3 * math.pi / 4 - 1e-9 <= yaw <= 3 * math.pi / 4 + 1e-9
    bearing = math.atan2(cube_xy[1] - 1.0, cube_xy[0] - 2.0)
    for j in range(-4, 5):
        other = source_yaw + j * period
        if -3 * math.pi / 4 <= other <= 3 * math.pi / 4:
            assert (
                abs(_angle_diff(yaw, bearing))
                <= abs(_angle_diff(other, bearing)) + 1e-9
            )


def test_shelf3d_perceiver_keeps_a_block_symmetric_only_every_180_degrees():
    """The default 10 x 5 x 5 cm block only looks the same every 180 degrees."""
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(0.5, 0.0, 0.025, yaw=0.2),
        shelf_pose=(1.5, 1.5, 0.02),
    )
    yaw = _cube_yaw(perceiver.reset(_make_obs(), {}))
    k = (yaw - 0.2) / math.pi
    assert k == pytest.approx(round(k), abs=1e-6)


def test_shelf3d_perceiver_without_yaw_is_unchanged():
    """A position-only source still gives a cube lying square (identity orientation)
    when the robot is on its -x side."""
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(3.0, 1.0, 0.025), shelf_pose=(1.5, 1.5, 0.02)
    )
    assert _cube_yaw(perceiver.reset(_make_obs(), {})) == pytest.approx(0.0)
