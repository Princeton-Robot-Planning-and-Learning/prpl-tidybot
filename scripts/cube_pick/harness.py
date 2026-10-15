"""Shared simulation harness for the cube-pick evaluations (Tossing3D scene).

Owns one live KINDER `Tossing3D-o1` env, restores recorded states into it, runs the old
KINDER pick or the new pipeline, scores the result on ground truth, and optionally
records a lab2 video with the wrist view beside it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import kinder  # noqa: E402
import kinder.envs.dynamic3d.envs  # noqa: E402,F401  (the module, so mujoco imports)

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2 as cv  # noqa: E402
import mujoco  # type: ignore[import-untyped]  # noqa: E402

from prpl_tidybot.cube_pick.camera import RGBDFrame  # noqa: E402
from prpl_tidybot.cube_pick.kinder_sim import KinderPickRobot  # noqa: E402
from prpl_tidybot.cube_pick.perception import (  # noqa: E402
    ColourSegmenter,
    CubeEstimate,
    CubePrior,
    DepthFitEstimator,
    FoundationPoseEstimator,
    PerceptionClient,
    PerceptionFailure,
    Sam2Segmenter,
)
from prpl_tidybot.cube_pick.planning import ContainerSpec  # noqa: E402
from prpl_tidybot.cube_pick.policy import CubePickPolicy, PickConfig  # noqa: E402

CUBE = "cube_0"
# Failures before anything is executed: the record and a still, no video.
PLAN_TIME_STAGES = (
    "look",
    "perceive",
    "propose",
    "filter",
    "plan",
    "no_grasp_candidate",
)
BIN = "bin_0"
BARRIER = "cuboid_barrier"


def parse_rows(spec: str, n: int) -> list[int]:
    """'' = all; 'a:b' a slice; '1,5,9' a list."""
    if not spec:
        return list(range(n))
    if ":" in spec and "," not in spec:
        a, b = spec.split(":")
        return list(range(int(a or 0), int(b or n)))
    return [int(v) for v in spec.split(",") if v]


class VideoRecorder:
    """Scene (free camera following the gripper) | wrist view, written with libx264."""

    def __init__(
        self, sim: Any, out_dir: Path, robot_name: str = "robot", every: int = 2
    ) -> None:
        self.sim = sim
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[np.ndarray] = []  # unused: frames stream to the writer
        self._writer: Any = None
        self.count_written = 0
        self.every = every
        self.count = 0
        model = sim.model.mj_model
        self._site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{robot_name}_pinch_site"
        )
        self._wrist = f"{robot_name}_wrist"
        self.overlay: dict[str, Any] = {}
        self._lookat: np.ndarray | None = None

    def scene_view(self) -> np.ndarray:
        ctx = self.sim._render_context_offscreen  # pylint: disable=protected-access
        data = self.sim.data.mj_data
        target = np.array(data.site_xpos[self._site])
        target[2] = max(0.05, target[2] * 0.5)
        self._lookat = (
            target if self._lookat is None else 0.8 * self._lookat + 0.2 * target
        )
        ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        ctx.cam.lookat[:] = self._lookat
        ctx.cam.distance = 1.6
        ctx.cam.azimuth = 135.0
        ctx.cam.elevation = -35.0
        rgb = self.sim.render(width=640, height=480, camera_name=None)
        ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        return rgb

    def wrist_view(self, robot: KinderPickRobot) -> np.ndarray:
        cam = robot.camera
        model = self.sim.model.mj_model
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._wrist)
        fovy0 = float(model.cam_fovy[cid])
        model.cam_fovy[cid] = cam.color.vertical_fov_deg  # undistorted view, for video
        vopt = (
            self.sim._render_context_offscreen.vopt
        )  # pylint: disable=protected-access
        sitegroup0 = np.array(vopt.sitegroup).copy()
        vopt.sitegroup[:] = 0
        try:
            rgb = self.sim.render(width=640, height=480, camera_name=self._wrist)
        finally:
            vopt.sitegroup[:] = sitegroup0
            model.cam_fovy[cid] = fovy0
        intr = cam.color
        frame = RGBDFrame(
            rgb.copy(), np.zeros((480, 640), np.float32), intr, cam.world_from_camera()
        )
        return draw_overlay(
            frame,
            self.overlay.get("estimate"),
            self.overlay.get("mask_frame"),
            self.overlay.get("text", ""),
        )

    def __call__(self, robot: KinderPickRobot) -> None:
        self.count += 1
        if self.count % self.every:
            return
        scene = self.scene_view()
        wrist = self.wrist_view(robot)
        frame = np.concatenate([scene, wrist], axis=1)
        text = self.overlay.get("caption", "")
        if text:
            cv.putText(
                frame,
                text,
                (10, 470),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
        if self._writer is None:
            import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel

            self._writer = imageio.get_writer(
                self.out_dir / "video.mp4",
                fps=20,
                codec="libx264",
                quality=7,
                macro_block_size=8,
            )
        self._writer.append_data(frame)
        self.count_written += 1
        self.frames = [
            frame
        ]  # keep only the latest, so `if recorder.frames` still works

    def write(self, name: str = "video", fps: int = 20) -> Path:
        """Close the stream (frames were written as they came); keep the last frame as
        still.png."""
        del name, fps
        if self.frames:
            cv.imwrite(
                str(self.out_dir / "still.png"),
                cv.cvtColor(self.frames[-1], cv.COLOR_RGB2BGR),
            )
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.out_dir / "video.mp4"


def draw_overlay(
    frame: RGBDFrame,
    estimate: CubeEstimate | None,
    mask_frame: tuple[RGBDFrame, np.ndarray] | None,
    text: str,
) -> np.ndarray:
    """The wrist colour image with the mask outline and the estimated cube's edges."""
    img = np.ascontiguousarray(frame.rgb.copy())
    if mask_frame is not None:
        mframe, mask = mask_frame
        if (
            mask is not None
            and mask.shape == img.shape[:2]
            and np.allclose(
                mframe.world_from_camera, frame.world_from_camera, atol=1e-4
            )
        ):
            contours, _ = cv.findContours(
                mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            cv.drawContours(img, contours, -1, (255, 0, 255), 2)
    if estimate is not None:
        s = estimate.size / 2
        corners = np.array(
            [[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)]
        )
        world = (
            corners @ estimate.world_from_cube[:3, :3].T
            + estimate.world_from_cube[:3, 3]
        )
        uv, z = frame.project_world(world)
        if np.all(z > 0.02):
            for i in range(8):
                for j in range(i + 1, 8):
                    if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                        p0 = tuple(int(v) for v in uv[i])
                        p1 = tuple(int(v) for v in uv[j])
                        cv.line(img, p0, p1, (0, 255, 255), 2, cv.LINE_AA)
    if text:
        cv.putText(
            img,
            text,
            (8, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    return img


class TossingSim:
    """One live Tossing3D env reused across replays."""

    OLD_STEP_LIMIT = 400  # hitl-pmp's KinderBackend.pick_step_limit

    def __init__(
        self,
        env_id: str = "kinder/Tossing3D-o1-v0",
        perception_port: int = 50111,
        grasp_port: int = 50112,
    ) -> None:
        self.perception_port = perception_port
        self.grasp_port = grasp_port
        # KINDER internals (private object-centric env, devectorize) are untyped here.
        self.env: Any = kinder.make(
            env_id, render_mode="rgb_array", scene_bg=True, allow_state_access=True
        )
        unwrapped: Any = self.env.unwrapped
        self.oc = unwrapped._object_centric_env  # pylint: disable=protected-access
        from kinder_models.dynamic3d.tossing.state_abstractions import (  # pylint: disable=import-outside-toplevel
            Tossing3DStateAbstractor,
        )

        # Built before the seeded reset: its constructor resets the sim unseeded.
        self.abstractor = Tossing3DStateAbstractor(self.oc)
        obs, _ = self.env.reset(seed=0)
        space: Any = self.env.observation_space
        self.template = space.devectorize(obs)
        self.robot_obj = next(
            o for o in self.template if o.type.name == "mujoco_tidybot_robot"
        )
        bin_object = self.oc.get_object(BIN)
        self.bin_dims = (
            bin_object.length,
            bin_object.width,
            bin_object.height,
            bin_object.wall_thickness,
        )
        self.sim = self.oc._robot_env.sim  # pylint: disable=protected-access
        self._clients: dict[str, PerceptionClient] = {}

    # --- state --------------------------------------------------------------------------

    def restore(self, plain: dict[str, list[float]]) -> Any:
        """Put the scene into a recorded state (float64, colliders from the live
        model)."""
        template = self.template
        by_name = {o.name: o for o in template}
        data = {
            o: template[o].copy() for o in template if o.name.startswith("collider:")
        }
        for name, values in plain.items():
            if name.startswith("collider:") or name not in by_name:
                continue
            data[by_name[name]] = np.array(values, dtype=np.float64)
        state = type(template)(data, template.type_features)
        self.oc.set_state(state)
        return self.oc._get_current_state()  # pylint: disable=protected-access

    def live(self) -> Any:
        return self.oc._get_current_state()  # pylint: disable=protected-access

    def holding(self) -> dict[str, Any]:
        state = self.live()
        robot = state.get_object_from_name(self.robot_obj.name)
        cube = state.get_object_from_name(CUBE)
        evidence = self.abstractor.holding_evidence(state, robot, cube)
        out = evidence.as_dict()
        out.pop("contacts", None)
        return out

    def support_heights(self, state: Any) -> tuple[Any, ...]:
        """The floor and, when the bin stands upright, its inner floor (the map)."""
        from prpl_tidybot.cube_pick.cube_fit import (  # pylint: disable=import-outside-toplevel
            SupportSurface,
        )

        obj = state.get_object_from_name(BIN)
        qw, qx, qy, qz = (state.get(obj, k) for k in ("qw", "qx", "qy", "qz"))
        up = 1 - 2 * (qx**2 + qy**2)
        heights: list[Any] = [0.0]
        if up > 0.99:
            length, width, _, wall = self.bin_dims
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
            heights.append(
                SupportSurface(
                    z=float(state.get(obj, "z")) + wall,
                    centre=(float(state.get(obj, "x")), float(state.get(obj, "y"))),
                    half_extents=(length / 2 - wall, width / 2 - wall),
                    yaw=yaw,
                )
            )
        return tuple(heights)

    def containers(self, state: Any) -> list[ContainerSpec]:
        obj = state.get_object_from_name(BIN)
        from scipy.spatial.transform import (  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel
            Rotation,
        )

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(
            [state.get(obj, k) for k in ("qx", "qy", "qz", "qw")]
        ).as_matrix()
        pose[:3, 3] = [state.get(obj, k) for k in ("x", "y", "z")]
        length, width, height, wall = self.bin_dims
        return [ContainerSpec(BIN, pose, length, width, height, wall)]

    # --- arms ---------------------------------------------------------------------------

    def run_old(
        self, plain: dict[str, list[float]], video: Path | None = None
    ) -> dict[str, Any]:
        """KINDER's pick, dispatched as hitl-pmp's KinderBackend.run_pick_cube does."""
        from kinder_models.dynamic3d.tossing.parameterized_skills import (  # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
        from kinder_models.dynamic3d.utils import (  # pylint: disable=import-outside-toplevel
            PyBulletSim,
        )
        from pybullet_helpers.geometry import (  # pylint: disable=import-outside-toplevel
            Pose,
        )

        state = self.restore(plain)
        obs_state = self._observed()
        pybullet_sim = PyBulletSim(obs_state)
        obj = obs_state.get_object_from_name(BIN)
        length, width, height, wall = self.bin_dims
        pybullet_sim.add_bin(
            name=BIN,
            pose=Pose(
                tuple(obs_state.get(obj, f) for f in ("x", "y", "z")),
                tuple(obs_state.get(obj, f) for f in ("qx", "qy", "qz", "qw")),
            ),
            length=length,
            width=width,
            height=height,
            wall_thickness=wall,
        )
        recorder = VideoRecorder(self.sim, video) if video else None
        robot_view = KinderPickRobot(self.env) if recorder else None
        try:
            lifted = create_lifted_controllers(
                self.env.action_space,
                pybullet_sim=pybullet_sim,
                init_constant_state=obs_state,
            )
            objects = tuple(
                obs_state.get_object_from_name(n)
                for n in (self.robot_obj.name, CUBE, BARRIER)
            )
            controller: Any = lifted["pick_cube"].ground(objects)
            try:
                controller.reset(obs_state, None)
            except (
                BaseException
            ) as exc:  # noqa: BLE001  (TrajectorySamplingFailure is not an Exception)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                stage: str | None = "filter" if "grasp" in str(exc).lower() else "plan"
                if recorder is not None:
                    # A refusal moves nothing: two seconds of the state it refused.
                    recorder.overlay["caption"] = (
                        f"old KINDER pick: refused ({type(exc).__name__}: {exc})"[:110]
                    )
                    for _ in range(20):
                        assert robot_view is not None
                        recorder(robot_view)
                return {
                    "success": False,
                    "stage": stage,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "ticks": 0,
                }
            ticks = 0
            error = None
            for ticks in range(1, self.OLD_STEP_LIMIT + 1):
                try:
                    action = controller.step()
                    obs, _, _, _, _ = self.env.step(action)
                except BaseException as exc:  # noqa: BLE001
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    break
                if recorder is not None:
                    recorder.overlay["caption"] = (
                        f"old KINDER pick  phase {controller.current_phase.name}"
                    )
                    assert robot_view is not None
                    recorder(robot_view)
                controller.observe(self.env.observation_space.devectorize(obs))
                if controller.terminated():
                    break
            holding = self.holding()
            phase = controller.current_phase.name
            success = bool(holding["contact_holding"])
            if success:
                stage = None
            elif error is not None:
                stage = "plan" if "grasp" not in error.lower() else "no_grasp_candidate"
            elif phase in (
                "OPEN_GRIPPER",
                "BASE_MOTION",
                "MOVE_ARM_TO_HOVER_OVER_CUBE",
                "MOVE_ARM_DOWN_AROUND_CUBE",
            ):
                stage = "plan"
            elif (
                phase == "CLOSE_GRIPPER_TO_GRASP_CUBE" or not holding["finger_contact"]
            ):
                stage = "closure"
            else:
                stage = "lift"
            return {
                "success": success,
                "stage": stage,
                "detail": error or f"ended in {phase}",
                "ticks": ticks,
                "holding": holding,
            }
        finally:
            pybullet_sim.close()
            if recorder is not None and recorder.frames:
                recorder.write("video")

    def _observed(self) -> Any:
        """The float32 observation the old controller is handed in hitl-pmp."""
        obs = self.oc.get_obs() if hasattr(self.oc, "get_obs") else None
        if obs is None:
            return self.live()
        return obs

    def estimator(self, name: str) -> Any:
        if name == "colour":
            return DepthFitEstimator(ColourSegmenter())
        sam2 = Sam2Segmenter(self._client("perception", self.perception_port))
        if name == "sam2":
            return DepthFitEstimator(sam2)
        if name == "foundationpose":
            return FoundationPoseEstimator(
                sam2, self._client("perception", self.perception_port)
            )
        raise ValueError(name)

    def _client(self, key: str, port: int) -> PerceptionClient:
        if key not in self._clients:
            self._clients[key] = PerceptionClient(port=port)
        return self._clients[key]

    def run_new(
        self,
        plain: dict[str, list[float]],
        *,
        estimator: str,
        generator: str,
        prior_offset: np.ndarray,
        max_ticks: int,
        video: Path | None = None,
        config: PickConfig | None = None,
    ) -> dict[str, Any]:
        """The new pipeline from the recorded state, on rendered wrist RGB-D only."""
        state = self.restore(plain)
        cube = state.get_object_from_name(CUBE)
        true_position = np.array([state.get(cube, k) for k in ("x", "y", "z")])
        size = float(state.get(cube, "bb_x"))
        prior_position = true_position.copy()
        prior_position[:2] += prior_offset
        recorder = VideoRecorder(self.sim, video) if video else None
        robot = KinderPickRobot(
            self.env,
            camera_noise_seed=int(abs(hash(tuple(np.round(true_position, 4)))) % 2**31),
            on_step=(lambda _s: recorder(robot)) if recorder else None,  # type: ignore[has-type]
        )
        believed = state.copy()
        config = config or PickConfig(cube_size=size)
        if generator == "analytic45":
            from prpl_tidybot.cube_pick.grasps import (  # pylint: disable=import-outside-toplevel
                AnalyticGraspGenerator,
                CubeGraspSampler,
            )

            config.generator = AnalyticGraspGenerator(
                CubeGraspSampler(
                    cube_size=size, tilts_deg=(0.0, 10.0, 20.0, 30.0, 40.0, 45.0)
                ),
                name="analytic45",
            )
        elif generator != "analytic":
            config.generator = self.generator(generator, size)
        prior = CubePrior(
            position=prior_position,
            size=size,
            sigma=0.05,
            support_heights=self.support_heights(state),
        )
        est = self.estimator(estimator)
        if recorder is not None:
            est = _RecordingEstimator(est, recorder)
        policy = CubePickPolicy(
            robot=robot,
            estimator=est,
            believed_state=believed,
            cube_name=CUBE,
            prior=prior,
            containers=self.containers(state),
            config=config,
            keep_frames=False,
        )
        if recorder is not None:
            recorder.overlay["caption"] = f"new pipeline ({estimator} + {generator})"
        result = policy.run(max_ticks=max_ticks)
        # Hold still briefly so a slipping cube shows up in the ground truth.
        for _ in range(5):
            robot.command(_hold(robot))
        holding = self.holding()
        live = self.live()
        cube_live = live.get_object_from_name(CUBE)
        errors = []
        for e in result.estimates:
            errors.append(float(np.linalg.norm(e.position - true_position)))
        success = bool(holding["contact_holding"])
        stage = (
            None
            if success
            else (result.stage or ("lift" if holding["finger_contact"] else "closure"))
        )
        if recorder is not None and recorder.frames:
            recorder.write("video")
        return {
            "success": success,
            "policy_success": result.success,
            "stage": stage,
            "detail": result.detail,
            "ticks": result.ticks,
            "frames": robot.frames_captured,
            "grasp": result.grasp.label if result.grasp else None,
            "estimate_errors": errors,
            "events": [(e.tick, e.stage, e.message) for e in result.events],
            "holding": holding,
            "cube_final_z": float(live.get(cube_live, "z")),
        }

    def generator(self, name: str, size: float) -> Any:
        from prpl_tidybot.cube_pick import (  # pylint: disable=import-outside-toplevel
            learned_grasps,
        )

        return learned_grasps.LearnedGraspGenerator(
            client=self._client("grasps", self.grasp_port), model=name, cube_size=size
        )


def _hold(robot: KinderPickRobot) -> Any:
    from prpl_tidybot.cube_pick.robot import (  # pylint: disable=import-outside-toplevel
        RobotCommand,
    )

    return RobotCommand(arm=robot.proprio().arm)


class _RecordingEstimator:
    """Pass-through estimator that hands each result to the video overlay."""

    def __init__(self, inner: Any, recorder: VideoRecorder) -> None:
        self.inner = inner
        self.recorder = recorder
        self.name = inner.name

    def estimate(self, frame: RGBDFrame, prior: CubePrior) -> Any:
        result = self.inner.estimate(frame, prior)
        self.count = getattr(self, "count", 0) + 1
        est = result if isinstance(result, CubeEstimate) else None
        img = draw_overlay(
            frame,
            est,
            (frame, result.mask),
            getattr(result, "backend", getattr(result, "reason", ""))[:70],
        )
        depth = frame.depth.copy()
        vis = np.zeros_like(img)
        valid = depth > 0
        if valid.any():
            lo, hi = np.percentile(depth[valid], [2, 98])
            vis[valid] = cv.applyColorMap(
                np.clip(255 * (depth[valid] - lo) / max(hi - lo, 1e-3), 0, 255).astype(
                    np.uint8
                )[:, None],
                cv.COLORMAP_TURBO,
            )[:, 0, ::-1]
        cv.imwrite(
            str(self.recorder.out_dir / f"perception_{self.count:02d}.png"),
            cv.cvtColor(np.concatenate([img, vis], axis=1), cv.COLOR_RGB2BGR),
        )
        if isinstance(result, CubeEstimate):
            self.recorder.overlay["estimate"] = result
            self.recorder.overlay["mask_frame"] = (frame, result.mask)
            self.recorder.overlay["text"] = (
                f"{result.backend}: {np.round(result.position, 3)}"
            )
        elif isinstance(result, PerceptionFailure):
            self.recorder.overlay["mask_frame"] = (frame, result.mask)
            self.recorder.overlay["text"] = f"perception failed: {result.reason[:60]}"
        return result
