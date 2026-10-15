"""Replay recorded Tossing3D states and pick the cube with the old and the new pick.

Each input row carries a full KINDER state (`{object_name: [features]}`, as
hitl-pmp's `tossing3d_state_log.jsonl` records it). For every row and every arm the
scene is restored to exactly that state and one pick is run in MuJoCo:

* ``old`` -- KINDER's `PickCubeController`, dispatched exactly as hitl-pmp does
  (`create_lifted_controllers(..., pybullet_sim=<bin-aware sim>,
  init_constant_state=state)["pick_cube"]`, objects (robot, cube, barrier), 400-step
  limit). It plans on the simulator's ground-truth cube pose.
* ``new`` -- :class:`prpl_tidybot.cube_pick.policy.CubePickPolicy` on rendered wrist
  RGB-D only. Its planning scene gets the robot, the room, the barrier and the bin;
  the cube enters only through perception. The prior it starts from is the true
  cube position plus seeded Gaussian noise (sigma `--prior-noise` per horizontal
  axis), standing in for the task-level estimate the real robot would have.

Success is ground truth for both arms: KINDER's own `contact_holding` evidence
(gripper closed, cube off the ground, finger contact) after the pick ends.

Writes one JSON line per (row, arm). Optionally records lab2 `task_view` video and
wrist frames with the perception overlay.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="old,new")
    parser.add_argument("--rows", default="", help="comma list of row indices or a:b")
    parser.add_argument("--estimator", default="colour")
    parser.add_argument("--generator", default="analytic")
    parser.add_argument("--prior-noise", type=float, default=0.02)
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--video-rows", default="")
    parser.add_argument(
        "--video-failures",
        action="store_true",
        help="record every run and keep the video of each failure (and of --video-rows)",
    )
    parser.add_argument("--max-ticks", type=int, default=4000)
    parser.add_argument("--perception-port", type=int, default=50111)
    parser.add_argument("--grasp-port", type=int, default=50112)
    args = parser.parse_args()

    rows = json.load(open(args.states))
    selected = harness.parse_rows(args.rows, len(rows))
    video_rows = (
        set(harness.parse_rows(args.video_rows, len(rows)))
        if args.video_rows
        else set()
    )
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            rec = json.loads(line)
            done.add((rec["row"], rec["arm"]))
    sim = harness.TossingSim(
        perception_port=args.perception_port, grasp_port=args.grasp_port
    )
    with open(args.out, "a") as out:
        for index in selected:
            row = rows[index]
            for arm in args.arms.split(","):
                if (index, arm) in done:
                    continue
                generator = args.generator if arm != "old" else "kinder_topdown"
                video = None
                if args.video_dir and (index in video_rows or args.video_failures):
                    safe = row["id"].replace("/", "_").replace("#", "-")
                    video = Path(args.video_dir) / f"{safe}__{arm}__{generator}"
                start = time.time()
                try:
                    if arm == "old":
                        record = sim.run_old(row["state"], video=video)
                    else:
                        rng = np.random.default_rng(zlib.crc32(row["id"].encode()))
                        record = sim.run_new(
                            row["state"],
                            estimator=args.estimator,
                            generator=args.generator,
                            prior_offset=rng.normal(scale=args.prior_noise, size=2),
                            max_ticks=args.max_ticks,
                            video=video,
                        )
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "success": False,
                        "stage": "harness_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                    }
                record.update(
                    row=index,
                    id=row["id"],
                    d4=row["d4"],
                    arm=arm,
                    estimator=args.estimator if arm != "old" else "ground_truth",
                    generator=args.generator if arm != "old" else "kinder_topdown",
                    seconds=round(time.time() - start, 2),
                )
                if video is not None:
                    import shutil  # pylint: disable=import-outside-toplevel

                    if record["success"] and index not in video_rows:
                        shutil.rmtree(video, ignore_errors=True)
                    elif (
                        not record["success"]
                        and record.get("stage") in harness.PLAN_TIME_STAGES
                        and index not in video_rows
                    ):
                        # Nothing worth filming: keep the status record and a still.
                        (video / "video.mp4").unlink(missing_ok=True)
                        record["still"] = str(video / "still.png")
                    else:
                        record["video"] = str(video / "video.mp4")
                        record["still"] = str(video / "still.png")
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"row {index} {arm}: {'OK ' if record['success'] else 'FAIL'} "
                    f"{record.get('stage')} {str(record.get('detail'))[:120]} ({record['seconds']} s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
