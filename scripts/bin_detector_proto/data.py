"""Replay dataset for the bin-detector prototypes: saved ceiling frame sets and video
windows from 2026-10-20, each labelled with where the bin was (close / far spot, or
'wall' when it stood against the far wall, partly out of view)."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from prpl_tidybot.cube_localizer import load_ceiling_cameras

RAW = Path("/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/raw/2026-10-20")
VID_ROOT = Path("/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/analysis/2026-10-20-pilot/binsat_nuc")
SERIALS = ["AD8A29AE", "5A5C29AE"]
SPOTS = {"close": (-0.299, -0.774), "far": (0.226, 1.374)}
TOL = 0.12
SEARCH = (-1.8, 2.0, -1.8, 1.8)
# Bin spot per run, for the video windows (the previous agent's labels, final.py).
VID_RUN = {
    "10-37-51": "far", "16-24-41": "far", "17-09-09": "far", "17-24-13": "far",
    "10-15-57": "far", "17-19-19": "far", "17-20-02": "far", "17-22-30": "far",
    "10-59-22": "close", "11-51-28": "close", "11-24-44": "close", "11-38-12": "close",
    "11-43-15": "close", "11-56-48": "close", "12-34-53": "close",
}
# Video windows where the bin was never near its spot (occluded/moved): final.py.
VID_EXCLUDED = {"10-59-22@998", "11-56-48@673", "12-34-53@157", "12-34-53@471"}


def png_label(run: str, idx: int) -> str:
    """Spot per saved frame set, read off the contact sheets (one per set)."""
    if run in ("09-23-05", "10-03-03"):
        return "wall"
    if run in ("10-59-22", "11-51-28"):
        return "close"
    if run == "17-27-17":
        return "close" if idx == 9 else "far"
    if run == "18-10-36":
        return "far" if idx in (0, 1, 21) else "close"
    return "far"


def cameras():
    return load_ceiling_cameras(SERIALS)


def png_items():
    out = []
    for d in sorted(glob.glob(str(RAW / "[0-9]*" / "localization_failures" / "*"))):
        run = d.split("/")[-3]
        name = os.path.basename(d)
        idx = int(name.split("_")[0])
        info = json.loads(Path(d, "localization.json").read_text())
        out.append(dict(kind="png", label=f"png:{run}/{name}", run=run, path=d,
                        spot=png_label(run, idx), exclude=info.get("exclusions", []),
                        shelf=run < "18-0"))
    return out


def vid_items():
    out = []
    for run, spot in VID_RUN.items():
        path = VID_ROOT / run / "ceiling.mp4"
        n = int(cv.VideoCapture(str(path)).get(cv.CAP_PROP_FRAME_COUNT))
        if n <= 0:  # 16-24-41: truncated mp4 (no moov atom)
            continue
        for k in range(6):
            start = int(k * max(n - 10, 0) / 5)
            key = f"{run}@{start}"
            if key in VID_EXCLUDED:
                continue
            out.append(dict(kind="vid", label=f"vid:{key}", run=run, path=str(path),
                            start=start, spot=spot, exclude=[], shelf=True))
    return out


def load(item, n=10):
    if item["kind"] == "png":
        fs = sorted(glob.glob(f"{item['path']}/frame_*.png"))[-n:]
        return [cv.imread(f) for f in fs]
    v = cv.VideoCapture(item["path"])
    v.set(cv.CAP_PROP_POS_FRAMES, item["start"])
    frames = []
    for _ in range(n):
        ok, f = v.read()
        if not ok:
            break
        frames.append(cv.resize(f, (1280, 1440), interpolation=cv.INTER_LINEAR))
    return frames


def split(frame, num=2):
    rows = frame.shape[0] // num
    return [frame[i * rows:(i + 1) * rows] for i in range(num)]


def plane_homography(cam, z):
    """3x3 H mapping map-frame (x, y, 1) on the plane at height z to pixels."""
    rot = cv.Rodrigues(cam.rvec)[0]
    m = np.c_[rot[:, 0], rot[:, 1], rot[:, 2] * z + cam.tvec.reshape(3)]
    return cam.camera_matrix @ m


def rectify(img, cam, z, bounds, res):
    """Top-down view of the plane at height z: pixel (r, c) <-> map
    (x0 + c*res, y1 - r*res), so +y is up the image."""
    x0, x1, y0, y1 = bounds
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    grid_to_map = np.array([[res, 0, x0], [0, -res, y1], [0, 0, 1]], np.float64)
    hm = plane_homography(cam, z) @ grid_to_map
    out = cv.warpPerspective(img, hm, (w, h), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                             borderMode=cv.BORDER_CONSTANT, borderValue=0)
    valid = cv.warpPerspective(np.full(img.shape[:2], 255, np.uint8), hm, (w, h),
                               flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP)
    return out, valid > 0
