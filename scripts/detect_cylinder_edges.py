"""Run the cylinder edge detector on an image and write the debug overlay.

Three sources of wrist images, for tuning the detector without the robot in
the loop:

    # A saved frame (any image file):
    python scripts/detect_cylinder_edges.py --image test_images/wrist_image.jpg

    # The kinder simulator's wrist camera at a recorded state (e.g. the
    # pre-grasp tick of a trajectory), with the cylinder-shelf env:
    python scripts/detect_cylinder_edges.py --state logs/.../trajectory/000125/state.pkl

    # The live Kinova wrist camera (on the NUC), one frame or a loop:
    python scripts/detect_cylinder_edges.py --camera [--loop]

Detector parameters can be overridden with --param name=value (see
EdgeDetectorParams). The overlay goes to --out (default: next to the input,
or ./cylinder_edges.png).
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.visual_servo.cylinder_edges import (
    EdgeDetectorParams,
    detect_cylinder_edges,
    render_edge_overlay,
)


def _parse_params(overrides: list[str]) -> EdgeDetectorParams:
    values: dict[str, Any] = {}
    fields = {f.name: f.type for f in dataclasses.fields(EdgeDetectorParams)}
    for item in overrides:
        name, _, raw = item.partition("=")
        if name not in fields:
            raise SystemExit(
                f"Unknown detector parameter {name!r}; known: {sorted(fields)}"
            )
        values[name] = int(raw) if "int" in str(fields[name]) else float(raw)
    return EdgeDetectorParams(**values)


def _load_image_from_state(state_path: Path) -> np.ndarray:
    from kinder.envs.kinematic3d.cylinder_shelf3d import (  # pylint: disable=import-outside-toplevel
        ObjectCentricCylinderShelf3DEnv,
    )

    with open(state_path, "rb") as f:
        state = pickle.load(f)
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    env.reset(seed=0)
    env.set_state(state)
    try:
        return np.asarray(env.render_ee_camera(), dtype=np.uint8)
    finally:
        env.close()


def _report(
    image: np.ndarray, params: EdgeDetectorParams, out: Path, label: str
) -> int:
    edges = detect_cylinder_edges(image, params)
    overlay = render_edge_overlay(image, edges, params, label=label)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out), cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
    if edges is None:
        print(f"{label}: no cylinder edges found; overlay at {out}")
        return 1
    print(
        f"{label}: left={edges.left_x:.1f} right={edges.right_x:.1f} "
        f"center={edges.center_x:.1f} width={edges.width_px:.1f}px "
        f"error={edges.lateral_error_px:+.1f}px "
        f"(responses {edges.left_response:.2f}/{edges.right_response:.2f}); "
        f"overlay at {out}"
    )
    return 0


def main() -> int:
    """Parse arguments, fetch one (or a stream of) image(s), detect, and report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="an RGB image file")
    source.add_argument("--state", type=Path, help="a recorded state.pkl to render")
    source.add_argument("--camera", action="store_true", help="the live wrist camera")
    parser.add_argument("--loop", action="store_true", help="with --camera: keep going")
    parser.add_argument("--out", type=Path, default=None, help="overlay output path")
    parser.add_argument("--param", action="append", default=[], help="name=value")
    args = parser.parse_args()
    params = _parse_params(args.param)

    if args.camera:
        from prpl_tidybot.visual_servo.image_sources import (  # pylint: disable=import-outside-toplevel
            KinovaWristCameraSource,
        )

        camera = KinovaWristCameraSource()
        out = args.out or Path("cylinder_edges.png")
        try:
            index = 0
            while True:
                frame = camera.get_image()
                if frame is None:
                    time.sleep(0.05)
                    continue
                image = np.asarray(frame, dtype=np.uint8)
                target = (
                    out
                    if not args.loop
                    else out.with_name(f"{out.stem}_{index:04d}{out.suffix}")
                )
                _report(image, params, target, f"frame {index}")
                index += 1
                if not args.loop:
                    return 0
                time.sleep(0.2)
        finally:
            camera.close()

    if args.image is not None:
        bgr = cv.imread(str(args.image))
        if bgr is None:
            raise SystemExit(f"Could not read {args.image}")
        file_image = np.asarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), dtype=np.uint8)
        out = args.out or args.image.with_name(f"{args.image.stem}_edges.png")
        return _report(file_image, params, out, args.image.name)

    state_image = _load_image_from_state(args.state)
    out = args.out or args.state.with_name("ee_camera_edges.png")
    return _report(state_image, params, out, args.state.parent.name)


if __name__ == "__main__":
    sys.exit(main())
