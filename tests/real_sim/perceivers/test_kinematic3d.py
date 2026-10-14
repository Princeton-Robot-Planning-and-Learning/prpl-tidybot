"""Tests for real_sim/perceivers/kinematic3d.py."""

import dataclasses

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
    TargetSource,
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


def _obs_at(x: float, y: float) -> TidyBotObservation:
    return dataclasses.replace(_make_obs(), map_base_pose=SE2(x=x, y=y, theta=0.0))


def _cube_yaw(state) -> float:
    cube = state.get_object_from_name("cube0")
    assert state.get(cube, "pose_qx") == 0.0 and state.get(cube, "pose_qy") == 0.0
    return 2.0 * np.arctan2(state.get(cube, "pose_qz"), state.get(cube, "pose_qw"))


def _angle_diff(a: float, b: float) -> float:
    return float(np.angle(np.exp(1j * (a - b))))


_CUBE = (0.025, 0.025, 0.025)
_BLOCK = (0.05, 0.025, 0.025)


def test_shelf3d_perceiver_uses_the_sources_yaw_as_rotation_about_z():
    """The cube's position comes from the source and its yaw becomes a z rotation."""
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(0.5, 0.0, 0.025, yaw=0.3),
        shelf_pose=(1.5, 1.5, 0.02),
        cube_half_extents=_CUBE,
    )
    state = perceiver.reset(_obs_at(0.0, 0.0), {})
    cube = state.get_object_from_name("cube0")
    assert state.get(cube, "pose_x") == pytest.approx(0.5)
    assert state.get(cube, "pose_z") == pytest.approx(0.025)
    assert _cube_yaw(state) == pytest.approx(0.3)


def test_shelf3d_perceiver_without_a_yaw_source_keeps_identity():
    """A position-only `TargetSource` still yields a cube lying flat at yaw 0."""

    class _PositionOnly(TargetSource):
        def get_target(self):
            return (0.5, 0.0, 0.025)

    perceiver = Shelf3DPerceiver(
        target_source=_PositionOnly(), shelf_pose=(1.5, 1.5, 0.02)
    )
    state = perceiver.reset(_obs_at(0.0, 0.0), {})
    cube = state.get_object_from_name("cube0")
    assert state.get(cube, "pose_qw") == pytest.approx(1.0)
    assert state.get(cube, "pose_qz") == pytest.approx(0.0)


@pytest.mark.parametrize(
    "half_extents, raw_yaw, robot_xy, expected_yaw",
    [
        # A cube is symmetric under 90 degrees: any face pair is the same grasp.
        (_CUBE, 0.3 + np.pi / 2, (0.0, 0.0), 0.3),
        (_CUBE, 0.3 - np.pi, (0.0, 0.0), 0.3),
        # The yaw is the representative nearest the robot -> cube bearing, so the
        # pick approaches from the robot's side...
        (_CUBE, 0.3, (0.5, 1.0), 0.3 - np.pi / 2),
        (_CUBE, 0.3, (0.5, -1.0), 0.3 + np.pi / 2),
        # ...among those within [-3pi/4, 3pi/4]: here the bearing is pi, and
        # 0.3 + pi lies outside that range.
        (_CUBE, 0.3, (1.0, 0.0), 0.3 + np.pi / 2),
        # A 10 x 5 cm block is only symmetric under 180 degrees.
        (_BLOCK, 0.3 + np.pi / 2, (0.0, 0.0), 0.3 - np.pi / 2),
        (_BLOCK, 0.3 + np.pi, (0.0, 0.0), 0.3),
        (_BLOCK, 0.3, (1.0, 0.0), 0.3),
    ],
)
def test_shelf3d_perceiver_wraps_yaw_by_symmetry_toward_the_robot(
    half_extents, raw_yaw, robot_xy, expected_yaw
):
    """The reported yaw is the symmetry-equivalent one closest to the bearing from
    the robot to the cube, with the symmetry period set by the half-extents."""
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(0.5, 0.0, 0.025, yaw=raw_yaw),
        shelf_pose=(1.5, 1.5, 0.02),
        cube_half_extents=half_extents,
    )
    state = perceiver.reset(_obs_at(*robot_xy), {})
    assert _angle_diff(_cube_yaw(state), expected_yaw) == pytest.approx(0.0, abs=1e-9)


def test_shelf3d_perceiver_caches_the_cube_yaw_at_reset():
    """The yaw chosen at reset stays fixed while the robot moves during the episode."""
    perceiver = Shelf3DPerceiver(
        target_source=ConstantTargetSource(0.5, 0.0, 0.025, yaw=0.3),
        shelf_pose=(1.5, 1.5, 0.02),
        cube_half_extents=_CUBE,
    )
    perceiver.reset(_obs_at(0.0, 0.0), {})
    state = perceiver.step(_obs_at(1.0, 0.0), {})
    assert _cube_yaw(state) == pytest.approx(0.3)


@pytest.mark.parametrize("half_extents", [_CUBE, _BLOCK])
def test_shelf3d_perceiver_yaw_stays_where_the_pick_controller_can_use_it(
    half_extents,
):
    """The shelf3d pick faces the base along `cube yaw + rot`, rot in [-pi/4, pi/4],
    and builds an `SE2Pose` from that angle unwrapped, which asserts it lies in
    [-pi, pi]; so every reported yaw must lie in [-3pi/4, 3pi/4] (and still be
    equivalent to the measured one under the cube's symmetry)."""
    period = np.pi / 2 if half_extents[0] == half_extents[1] else np.pi
    for raw_yaw in np.linspace(-np.pi, np.pi, 37):
        for robot_angle in np.linspace(-np.pi, np.pi, 13):
            robot_xy = (0.5 + np.cos(robot_angle), np.sin(robot_angle))
            perceiver = Shelf3DPerceiver(
                target_source=ConstantTargetSource(0.5, 0.0, 0.025, yaw=raw_yaw),
                shelf_pose=(1.5, 1.5, 0.02),
                cube_half_extents=half_extents,
            )
            yaw = _cube_yaw(perceiver.reset(_obs_at(*robot_xy), {}))
            assert abs(yaw) <= 3 * np.pi / 4 + 1e-9
            offset = np.remainder(yaw - raw_yaw + period / 2, period)
            assert offset == pytest.approx(period / 2, abs=1e-6)
