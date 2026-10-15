"""The simulator side of the pick's robot port: KINDER's MuJoCo TidyBot.

:class:`KinderWristCamera` renders the robot's wrist camera the way the real vision
module sees: colour at 1280x720 with the colour sensor's field of view; depth at
480x270 with the depth sensor's field of view, from the depth sensor's own position
(offset from the colour sensor); depth converted from MuJoCo's z-buffer to metres,
blacked out inside the sensor's 18 cm minimum range and passed through
:class:`~prpl_tidybot.cube_pick.camera.DepthNoiseModel`; then registered onto the colour
grid with the same function the real robot uses. The camera's pose comes from the
model's kinematics, as the real one comes from forward kinematics.

:class:`KinderPickRobot` implements :class:`~prpl_tidybot.cube_pick.robot.PickRobot`
over a KINDER dynamic3d env: proprioception from the robot's joints, commands turned
into KINDER's delta actions with the same proportional tracking its own controllers
use. Nothing here reads an object's pose; the harness that scores a run does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_pick.camera import (  # noqa: F401
    KINOVA_WRIST_COLOR,
    KINOVA_WRIST_COLOR_FISHEYE,
    KINOVA_WRIST_DEPTH,
    DepthNoiseModel,
    FisheyeIntrinsics,
    PinholeIntrinsics,
    RGBDFrame,
    WristCalibration,
    assemble_wrist_frame,
)
from prpl_tidybot.cube_pick.planning import driver_to_opening, opening_to_command
from prpl_tidybot.cube_pick.robot import GRIPPER_CLOSE, Proprio, RobotCommand

FloatArray = NDArray[np.float64]

# MuJoCo cameras look down their -z with y up; OpenCV's look down +z with y down.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class KinderWristCamera:
    """Render the wrist RGB-D pair from a live KINDER MuJoCo sim."""

    sim: Any
    robot_name: str = "robot"
    fisheye: FisheyeIntrinsics | None = KINOVA_WRIST_COLOR_FISHEYE
    color: PinholeIntrinsics = KINOVA_WRIST_COLOR
    depth: PinholeIntrinsics = KINOVA_WRIST_DEPTH
    # With a fisheye model, the pinhole render it is resampled from (wide enough to
    # cover the fisheye's corners, at twice its resolution); without one, the colour
    # camera itself.
    render: PinholeIntrinsics | None = None
    # The depth module sits beside the colour sensor on the Kinova vision module; the
    # vendor's factory extrinsics give the exact offset. 2.5 cm along the colour
    # camera's x axis is a stand-in until those are read off the robot.
    depth_offset: tuple[float, float, float] = (0.025, 0.0, 0.0)
    noise: DepthNoiseModel = field(default_factory=DepthNoiseModel)

    def __post_init__(self) -> None:
        if self.render is None:
            self.render = (
                self.color
                if self.fisheye is None
                else PinholeIntrinsics(
                    width=1280, height=960, fx=600.0, fy=600.0, cx=639.5, cy=479.5
                )
            )
        self._model = self.sim.model.mj_model
        self._data = self.sim.data.mj_data
        self._cam = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self.robot_name}_wrist"
        )
        if self._cam < 0:
            raise ValueError(f"no camera {self.robot_name}_wrist in the scene")
        self._frames = 0

    def world_from_camera(self) -> FloatArray:
        """The colour camera's pose (OpenCV axes) from the model's kinematics."""
        out = np.eye(4)
        out[:3, :3] = self._data.cam_xmat[self._cam].reshape(3, 3) @ _MJ_TO_CV
        out[:3, 3] = self._data.cam_xpos[self._cam]
        return out

    def capture(self) -> RGBDFrame:
        """One registered RGB-D frame, assembled exactly as on the robot."""
        model, data, cam = self._model, self._data, self._cam
        fovy0 = float(model.cam_fovy[cam])
        pos0 = model.cam_pos[cam].copy()
        world_from_color = self.world_from_camera()
        # Sites are annotations (goal regions drawn as translucent boxes), not things
        # a real camera would see.
        render = self.render
        assert render is not None  # set in __post_init__
        vopt = (
            self.sim._render_context_offscreen.vopt
        )  # pylint: disable=protected-access
        sitegroup0 = np.array(vopt.sitegroup).copy()
        vopt.sitegroup[:] = 0
        try:
            model.cam_fovy[cam] = render.vertical_fov_deg
            rgb = self.sim.render(
                width=render.width,
                height=render.height,
                camera_name=f"{self.robot_name}_wrist",
            )
            # Move the camera to the depth sensor (its offset is given in the colour
            # camera's OpenCV frame; cam_pos is in the parent body's frame).
            offset_cv = np.asarray(self.depth_offset)
            body_from_cam = model.cam_quat[cam]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, body_from_cam)
            model.cam_pos[cam] = pos0 + rot.reshape(3, 3) @ (_MJ_TO_CV @ offset_cv)
            model.cam_fovy[cam] = self.depth.vertical_fov_deg
            mujoco.mj_camlight(model, data)
            _, zbuffer = self.sim.render(
                width=self.depth.width,
                height=self.depth.height,
                camera_name=f"{self.robot_name}_wrist",
                depth=True,
            )
        finally:
            vopt.sitegroup[:] = sitegroup0
            model.cam_fovy[cam] = fovy0
            model.cam_pos[cam] = pos0
            mujoco.mj_camlight(model, data)
        extent = model.stat.extent
        near, far = model.vis.map.znear * extent, model.vis.map.zfar * extent
        depth_m = near / (1.0 - zbuffer.astype(np.float64) * (1.0 - near / far))
        self.noise.reseed(self.noise.seed * 100003 + self._frames)
        self._frames += 1
        depth_noisy = self.noise.apply(depth_m)
        color_from_depth = np.eye(4)
        color_from_depth[:3, 3] = self.depth_offset
        assert self.render is not None
        fisheye_rgb = (
            np.ascontiguousarray(rgb)
            if self.fisheye is None
            else self.fisheye.distort_from_pinhole(
                np.ascontiguousarray(rgb), self.render
            )
        )
        frame = assemble_wrist_frame(
            color_fisheye_image=fisheye_rgb,
            depth_image=depth_noisy,
            calibration=WristCalibration(
                color_fisheye=self.fisheye,
                depth=self.depth,
                color_from_depth=color_from_depth,
                output=self.color,
            ),
            world_from_color=world_from_color,
        )
        return frame


class KinderPickRobot:
    """:class:`PickRobot` over a KINDER dynamic3d TidyBot env.

    `env` is the gym env (`kinder.make(...)` with `allow_state_access=True`). Each
    :meth:`command` is one `env.step`. `on_step` is called after every step with the
    devectorized state (the harness uses it to record video and score the run).
    """

    ARM_GAIN = 2.0  # KINDER's controllers' proportional gain on joint error
    MAX_BASE_STEP = 0.1  # KINDER's base action bound per step

    def __init__(
        self,
        env: Any,
        *,
        camera_noise_seed: int = 0,
        on_step: Callable[[Any], None] | None = None,
    ) -> None:
        self._env = env
        self._oc = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        self.sim = self._oc._robot_env.sim  # pylint: disable=protected-access
        model = self.sim.model.mj_model
        robot = next(iter(self.state().get_objects(_robot_type())))
        self.robot_name = robot.name
        self._robot = robot
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.robot_name}_right_driver_joint"
        )
        self._driver_qpos = int(model.jnt_qposadr[joint])
        self.camera = KinderWristCamera(
            self.sim,
            robot_name=self.robot_name,
            noise=DepthNoiseModel(seed=camera_noise_seed),
        )
        self._on_step = on_step
        self._last_gripper_command = 0.0
        self.ticks = 0
        self.frames_captured = 0

    def state(self) -> Any:
        """The live object-centric state (for the harness and the believed scene)."""
        return self._oc._get_current_state()  # pylint: disable=protected-access

    def proprio(self) -> Proprio:
        state = self.state()
        r = self._robot
        driver = float(self.sim.data.mj_data.qpos[self._driver_qpos])
        return Proprio(
            base=(
                state.get(r, "pos_base_x"),
                state.get(r, "pos_base_y"),
                state.get(r, "pos_base_rot"),
            ),
            arm=tuple(state.get(r, f"pos_arm_joint{i + 1}") for i in range(7)),
            gripper_opening=driver_to_opening(driver),
        )

    def wrist_frame(self) -> RGBDFrame:
        self.frames_captured += 1
        return self.camera.capture()

    def command(self, cmd: RobotCommand) -> None:
        current = self.proprio()
        action = np.zeros(11, dtype=np.float32)
        if cmd.base is not None:
            dx = cmd.base[0] - current.base[0]
            dy = cmd.base[1] - current.base[1]
            drot = (cmd.base[2] - current.base[2] + np.pi) % (2 * np.pi) - np.pi
            action[0:3] = np.clip(
                [dx, dy, drot], -self.MAX_BASE_STEP, self.MAX_BASE_STEP
            )
        if cmd.arm is not None:
            error = np.asarray(cmd.arm[:7], dtype=float) - np.asarray(current.arm)
            for i in (0, 2, 4, 6):
                error[i] = (error[i] + np.pi) % (2 * np.pi) - np.pi
            action[3:10] = self.ARM_GAIN * error
        if cmd.gripper is not None:
            self._last_gripper_command = (
                1.0
                if cmd.gripper == GRIPPER_CLOSE
                else opening_to_command(float(cmd.gripper))
            )
        action[10] = self._last_gripper_command
        self._env.step(action)
        self.ticks += 1
        if self._on_step is not None:
            self._on_step(self.state())


def _robot_type() -> Any:
    from kinder.envs.dynamic3d.object_types import (  # pylint: disable=import-outside-toplevel
        MujocoTidyBotRobotObjectType,
    )

    return MujocoTidyBotRobotObjectType
