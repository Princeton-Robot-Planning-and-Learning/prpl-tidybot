"""Hardware calibration: measure the wrist servo's target pixel by placing the cube.

Run on the robot (the arm server must already be up), the base parked with clear
floor in front of it. The arm goes to the top-down grasp pose at (--x, --y) in the
arm frame (the pick's own heights from conf/toss_cycles.yaml), you put the cube on
the floor between the open fingers, and it is then grasped, centred, set down at the
grasp height (never dropped), looked at from the pick's hover, and regrasped,
--samples times (prpl_tidybot.wrist_target_calibration). Every move and gripper
action asks y/N first; Ctrl-C stops the arm.

    python hardware_tests/calibrate_wrist_target.py [--samples 5] [--max-offset 0.01]

Prints the mean target pixel, its per-axis spread and the line to paste into
``pick.top_down.wrist_servo.target_uv``; the samples and every look (raw and
annotated) go to --out (default hardware_tests/artifacts/wrist_target_<time>/).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hydra import compose, initialize_config_dir

from prpl_tidybot.interfaces.real_arm_interface import RealArmInterface
from prpl_tidybot.robot_skills import _SettledImages  # pylint: disable=protected-access
from prpl_tidybot.top_down_pick import geometry_from_config
from prpl_tidybot.visual_servo.image_sources import KinovaWristCameraSource
from prpl_tidybot.wrist_target_calibration import (
    CalibrationParams,
    calibrate_wrist_target,
)

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> int:
    """Interactive place-and-look calibration with a confirmation before every step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.55, help="arm-frame x (m)")
    parser.add_argument("--y", type=float, default=0.0, help="arm-frame y (m)")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-offset", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--higher-look", type=float, default=0.05)
    parser.add_argument("--open-steps", type=int, default=4)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides of conf/toss_cycles.yaml"
    )
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(
            config_name="toss_cycles", overrides=["mode=real", *args.overrides]
        )
    td = cfg.pick.top_down
    geom = geometry_from_config(td)
    out = args.out or ARTIFACT_DIR / time.strftime("wrist_target_%Y%m%d_%H%M%S")
    params = CalibrationParams(
        arm_xy=(args.x, args.y),
        samples=args.samples,
        max_offset=args.max_offset,
        seed=args.seed,
        center=not args.no_center,
        higher_look=args.higher_look,
        open_steps=args.open_steps,
        gripper_timeout_s=float(td.get("gripper_timeout_s", 4.0)),
    )
    print(
        f"Grasp height {geom.floor_z + geom.grasp_height:+.4f} m, hover "
        f"{geom.floor_z + geom.grasp_height + geom.hover_height:+.4f} m (arm frame); "
        f"writing to {out}."
    )
    camera = KinovaWristCameraSource()
    print("Connecting to the real arm interface (no reset)...")
    arm = RealArmInterface(reset_arm=False)
    try:
        result = calibrate_wrist_target(
            arm,
            _SettledImages(camera, args.settle_s),
            geom,
            params,
            input,
            out,
            float(td.get("close_position", 1.0)),
        )
    finally:
        arm.close()
        camera.close()
    data = json.loads((out / "calibration.json").read_text())
    print(f"\nStopped early: {result.note}" if result.note else "\nDone.")
    print(f"Good samples {result.n_good}/{len(result.samples)}.")
    for sample in result.samples:
        hover = sample.hover
        where = "no cube" if hover is None else f"({hover['u']:.1f}, {hover['v']:.1f})"
        print(f"  sample {sample.index}: {where} {'' if sample.good else 'BAD'}")
    if result.n_good:
        print(
            f"target_uv mean ({result.target_uv[0]:.1f}, {result.target_uv[1]:.1f}) "
            f"px, std ({result.std_uv[0]:.1f}, {result.std_uv[1]:.1f}) px"
        )
        if result.gain:
            print(f"gain from the higher look: {result.gain}")
        print(f"Paste into pick.top_down.wrist_servo:\n  {data['yaml']}")
    return 0 if result.note is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
