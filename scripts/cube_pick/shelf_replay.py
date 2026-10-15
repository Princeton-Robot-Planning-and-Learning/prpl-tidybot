"""Replay SweepOffShelf3D benchmark states and pick a cube with the old and new pick.

States come from hitl-pmp's shelf skill benchmark (`benchmarks/sweepoffshelf3d/
states.jsonl`): whole MuJoCo qpos/qvel/ctrl, restored exactly. The cases evaluated
are the cube picks: `ReturnCubeToShelfFromBox` (cube in the box) and
`PickCubeFromFloor` / `ReturnCubeToShelfFromFloor` (same states, cube on the floor);
only the pick is run and scored here, not the shelf place that follows a return.

* ``old`` -- KINDER's `PickCubeController` with objects (robot, cube) and the box added
  to its PyBullet scene, as hitl-pmp's `KinderSkills.return_cube_to_shelf` builds it
  (bin held-object threshold for a cube in the box).
* ``new`` -- the cube_pick policy on rendered wrist RGB-D. The target cube enters its
  planning scene only through perception; the other cubes, the cupboard and the box
  are in it at their true poses (the map, and -- a stated simplification -- the other
  cubes as the old controller also sees them).

Success: the cube lifted 5 cm or more above where it started and touching both pads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402  pylint: disable=wrong-import-position
import mujoco  # type: ignore[import-untyped]  # noqa: E402  pylint: disable=wrong-import-position

BOX = "box_0"
BOX_DIMENSIONS = {"length": 0.25, "width": 0.5, "height": 0.2, "wall_thickness": 0.01}
PICK_SKILLS = ("ReturnCubeToShelfFromBox", "PickCubeFromFloor")


class ShelfSim:
    """One live SweepOffShelf3D-o5 env reused across replays."""

    OLD_STEP_LIMIT = 600

    def __init__(self, perception_port: int, grasp_port: int) -> None:
        self.env: Any = harness.kinder.make(
            "kinder/SweepOffShelf3D-o5-v0",
            render_mode="rgb_array",
            scene_bg=True,
            allow_state_access=True,
        )
        unwrapped: Any = self.env.unwrapped
        self.oc = unwrapped._object_centric_env  # pylint: disable=protected-access
        self.env.reset(seed=0)
        self.sim = self.oc._robot_env.sim  # pylint: disable=protected-access
        self.model = self.sim.model.mj_model
        self.data = self.sim.data.mj_data
        self.tossing = harness.TossingSim.__new__(harness.TossingSim)
        self.tossing.perception_port = perception_port
        self.tossing.grasp_port = grasp_port
        self.tossing._clients = {}  # pylint: disable=protected-access
        self._pads = {
            # pylint: disable-next=no-member  # mujoco is a C extension
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in (
                "robot_left_pad1",
                "robot_left_pad2",
                "robot_right_pad1",
                "robot_right_pad2",
            )
        }

    def restore(self, row: dict[str, Any]) -> Any:
        """Restore a recorded MuJoCo state exactly and return the live state."""
        assert len(row["qpos"]) == self.model.nq, (len(row["qpos"]), self.model.nq)
        self.data.qpos[:] = row["qpos"]
        self.data.qvel[:] = row["qvel"]
        self.data.ctrl[:] = row["ctrl"]
        # pylint: disable-next=no-member  # mujoco is a C extension
        mujoco.mj_forward(self.model, self.data)
        # The object-centric env caches its state at each step; refresh it from the
        # restored MuJoCo data.
        # pylint: disable-next=protected-access  # KINDER exposes no public setter
        self.oc._current_state = self.oc._get_object_centric_state()
        return self.live()

    def live(self) -> Any:
        """The live object-centric state."""
        return self.oc._get_current_state()  # pylint: disable=protected-access

    def cube_geoms(self, cube: str) -> set[int]:
        """The geom ids of a cube's body."""
        # pylint: disable-next=no-member  # mujoco is a C extension
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube)
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body}

    def holding(self, cube: str, start_z: float) -> dict[str, Any]:
        """Whether both pads touch the cube and it is lifted 5 cm."""
        geoms = self.cube_geoms(cube)
        left = right = False
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if pair & geoms and pair & self._pads:
                # pylint: disable=no-member  # mujoco is a C extension
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, (pair & self._pads).pop()
                )
                left |= "left" in name
                right |= "right" in name
        state = self.live()
        z = float(state.get(state.get_object_from_name(cube), "z"))
        return {
            "both_pads": bool(left and right),
            "cube_z": z,
            "lifted": bool(z > start_z + 0.05),
            "contact_holding": bool(left and right and z > start_z + 0.05),
        }

    def box_spec(self, state: Any) -> Any:
        """The box as a container spec, and its pose."""
        from scipy.spatial.transform import (  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel
            Rotation,
        )

        obj = state.get_object_from_name(BOX)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(
            [state.get(obj, k) for k in ("qx", "qy", "qz", "qw")]
        ).as_matrix()
        pose[:3, 3] = [state.get(obj, k) for k in ("x", "y", "z")]
        d = BOX_DIMENSIONS
        return (
            harness.ContainerSpec(
                BOX, pose, d["length"], d["width"], d["height"], d["wall_thickness"]
            ),
            pose,
        )

    def support_heights(self, state: Any) -> tuple[Any, ...]:
        """The floor and the box's inner floor (the map)."""
        from prpl_tidybot.cube_pick.cube_fit import (  # pylint: disable=import-outside-toplevel
            SupportSurface,
        )

        spec, pose = self.box_spec(state)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        return (
            0.0,
            SupportSurface(
                z=float(pose[2, 3]) + spec.wall_thickness,
                centre=(float(pose[0, 3]), float(pose[1, 3])),
                half_extents=(
                    spec.length / 2 - spec.wall_thickness,
                    spec.width / 2 - spec.wall_thickness,
                ),
                yaw=yaw,
            ),
        )

    def run_old(self, row: dict[str, Any], cube: str) -> dict[str, Any]:
        """Run KINDER's pick on one case and score it."""
        from kinder_models.dynamic3d.tossing.parameterized_skills import (  # pylint: disable=import-outside-toplevel
            PickCubeController,
        )
        from kinder_models.dynamic3d.utils import (  # pylint: disable=import-outside-toplevel
            PyBulletSim,
        )
        from pybullet_helpers.geometry import (  # pylint: disable=import-outside-toplevel
            Pose,
        )

        state = self.restore(row)
        start_z = float(state.get(state.get_object_from_name(cube), "z"))
        sim = PyBulletSim(state)
        box = state.get_object_from_name(BOX)
        sim.add_bin(
            name=BOX,
            pose=Pose(
                tuple(state.get(box, k) for k in ("x", "y", "z")),
                tuple(state.get(box, k) for k in ("qx", "qy", "qz", "qw")),
            ),
            **BOX_DIMENSIONS,
        )
        robot = next(o for o in state if o.type.name == "mujoco_tidybot_robot")
        try:
            pick = PickCubeController(
                objects=[robot, state.get_object_from_name(cube)], pybullet_sim=sim
            )
            pick.HELD_OBJECT_COLLISION_THRESHOLD = (
                pick.BIN_HELD_OBJECT_COLLISION_THRESHOLD
            )
            try:
                pick.reset(state, np.zeros(0))
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                # TrajectorySamplingFailure is not an Exception subclass.
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                stage: str | None = "filter" if "grasp" in str(exc).lower() else "plan"
                return {
                    "success": False,
                    "stage": stage,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "ticks": 0,
                }
            error = None
            ticks = 0
            for ticks in range(1, self.OLD_STEP_LIMIT + 1):
                try:
                    obs, _, _, _, _ = self.env.step(pick.step())
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    # Any failure ends the run; it is recorded and scored, not raised.
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    break
                pick.observe(self.env.observation_space.devectorize(obs))
                if pick.terminated():
                    break
            holding = self.holding(cube, start_z)
            phase = pick.current_phase.name
            success = holding["contact_holding"]
            if success:
                stage = None
            elif error:
                stage = "plan"
            elif phase in (
                "OPEN_GRIPPER",
                "BASE_MOTION",
                "MOVE_ARM_TO_HOVER_OVER_CUBE",
            ):
                stage = "plan"
            elif phase == "MOVE_ARM_DOWN_AROUND_CUBE":
                stage = "approach"
            elif not holding["both_pads"]:
                stage = "close"
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
            sim.close()

    def run_new(
        self,
        row: dict[str, Any],
        cube: str,
        *,
        estimator: str,
        generator: str,
        prior_offset: np.ndarray,
        max_ticks: int,
        video: Path | None,
    ) -> dict[str, Any]:
        """Run the cube_pick policy on one case and score it."""
        state = self.restore(row)
        obj = state.get_object_from_name(cube)
        true_position = np.array([state.get(obj, k) for k in ("x", "y", "z")])
        start_z = float(true_position[2])
        size = float(state.get(obj, "bb_x"))
        prior_position = true_position.copy()
        prior_position[:2] += prior_offset
        recorder = harness.VideoRecorder(self.sim, video) if video else None
        robot = harness.KinderPickRobot(
            self.env,
            camera_noise_seed=zlib.crc32(row["state_id"].encode()) % 2**31,
            on_step=(lambda _s: recorder(robot)) if recorder else None,  # type: ignore[has-type]
        )
        config = harness.PickConfig(cube_size=size)
        if generator != "analytic":
            config.generator = self.tossing.generator(generator, size)
        est = self.tossing.estimator(estimator)
        if recorder is not None:
            est = harness._RecordingEstimator(  # pylint: disable=protected-access
                est, recorder
            )
        spec, _ = self.box_spec(state)
        r, g, b, alpha = (
            float(v) for v in self.model.geom_rgba[min(self.cube_geoms(cube))]
        )
        cube_rgba = (r, g, b, alpha)
        prior = harness.CubePrior(
            position=prior_position,
            size=size,
            sigma=0.05,
            rgba=cube_rgba,
            support_heights=self.support_heights(state),
        )
        policy = harness.CubePickPolicy(
            robot=robot,
            estimator=est,
            believed_state=state.copy(),
            cube_name=cube,
            prior=prior,
            containers=[spec],
            config=config,
            keep_frames=False,
        )
        result = policy.run(max_ticks=max_ticks)
        for _ in range(5):
            robot.command(harness._hold(robot))  # pylint: disable=protected-access
        holding = self.holding(cube, start_z)
        success = holding["contact_holding"]
        if recorder is not None and recorder.frames:
            recorder.write("video")
        return {
            "success": success,
            "policy_success": result.success,
            "stage": (
                None
                if success
                else (result.stage or ("lift" if holding["both_pads"] else "close"))
            ),
            "detail": result.detail,
            "ticks": result.ticks,
            "frames": robot.frames_captured,
            "grasp": result.grasp.label if result.grasp else None,
            "estimate_errors": [
                float(np.linalg.norm(e.position - true_position))
                for e in result.estimates
            ],
            "events": [(e.tick, e.stage, e.message) for e in result.events],
            "holding": holding,
        }


def main() -> None:
    """Replay every pick case for each arm, appending one record per run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="old,new")
    parser.add_argument("--estimator", default="sam2")
    parser.add_argument("--generator", default="analytic")
    parser.add_argument("--prior-noise", type=float, default=0.02)
    parser.add_argument("--max-ticks", type=int, default=4000)
    parser.add_argument("--perception-port", type=int, default=50111)
    parser.add_argument("--grasp-port", type=int, default=50112)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--video-ids", default="")
    args = parser.parse_args()
    cases = []
    seen = set()
    for line in open(args.states, encoding="utf-8"):
        row = json.loads(line)
        for case in row["cases"]:
            if (
                case["skill"] in PICK_SKILLS
                and (row["state_id"], case["cube"]) not in seen
            ):
                seen.add((row["state_id"], case["cube"]))
                cases.append((row, case))
    k, n = (int(v) for v in args.shard.split("/"))
    cases = cases[k::n]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["id"], rec["arm"]))
    sim = ShelfSim(args.perception_port, args.grasp_port)
    video_ids = set(args.video_ids.split(",")) if args.video_ids else set()
    with open(args.out, "a", encoding="utf-8") as out:
        for row, case in cases:
            cid = f"{row['state_id']}:{case['cube']}"
            for arm in args.arms.split(","):
                if (cid, arm) in done:
                    continue
                start = time.time()
                video = (
                    Path(args.video_dir) / f"{row['state_id']}_{arm}"
                    if cid in video_ids and args.video_dir
                    else None
                )
                try:
                    if arm == "old":
                        record = sim.run_old(row, case["cube"])
                    else:
                        rng = np.random.default_rng(zlib.crc32(cid.encode()))
                        record = sim.run_new(
                            row,
                            case["cube"],
                            estimator=args.estimator,
                            generator=args.generator,
                            prior_offset=rng.normal(scale=args.prior_noise, size=2),
                            max_ticks=args.max_ticks,
                            video=video,
                        )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # One case's failure is recorded; the sweep goes on.
                    record = {
                        "success": False,
                        "stage": "harness_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                    }
                record.update(
                    id=cid,
                    state_id=row["state_id"],
                    skill=case["skill"],
                    family=row["family"],
                    arm=arm,
                    estimator=args.estimator if arm != "old" else "ground_truth",
                    generator=args.generator if arm != "old" else "kinder_topdown",
                    seconds=round(time.time() - start, 2),
                )
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"{cid} {arm}: {'OK ' if record['success'] else 'FAIL'} "
                    f"{record.get('stage')} {str(record.get('detail'))[:120]} "
                    f"({record['seconds']} s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
