"""Replay every labelled frame set / video window through one detector, gates off, and
write the raw top candidate and its gate statistics (thresholds are applied later by
analyze.py, so they can be chosen on the dev runs and frozen for the test runs).

Each positive item is also replayed with the bin's own footprint excluded from the
search ("bin hidden"): anything returned then is a wrong bin, by construction.
"""
import json
import math
import pickle
import sys
from multiprocessing import Pool

import data
import detectors as D

import os

import numpy as np

METHOD = sys.argv[1] if __name__ == "__main__" else None
OUT = sys.argv[2] if __name__ == "__main__" else None
MODEL = sys.argv[3] if __name__ == "__main__" and len(sys.argv) > 3 else None
PERTURB = os.environ.get("PERTURB", "none")
_det = None


class Baseline:
    """The deployed CeilingBinSource algorithm (bin_localizer.localize_bin)."""

    def __init__(self, sat):
        from prpl_tidybot import bin_localizer as bl
        self.bl, self.sat = bl, sat
        self.cams = data.cameras()
        self.model = bl.BinModel(0.39, 0.29, 0.21, 0.005)

    def localize(self, frames, exclude):
        import time
        t0 = time.time()
        fit = self.bl.localize_bin(frames, self.cams, self.model, data.SEARCH, exclude, 0.7,
                                   data.SERIALS, 0.85, self.sat)
        out = dict(baseline=True, found=fit is not None, t=time.time() - t0)
        if fit is not None:
            out.update(x=fit.x, y=fit.y, yaw=fit.yaw, score=fit.iou)
        return out


def perturb(frame):
    """Photometric stress: the same scene under other lighting / a paler bin."""
    f = frame.astype(np.float32)
    if PERTURB == "dim":  # 40 % less light
        f = f * 0.6
    elif PERTURB == "bright":  # 35 % more light, clipped (washes highlights out)
        f = f * 1.35
    elif PERTURB == "desat":  # half the colour: a paler, washed-out bin (and floor)
        g = f.mean(axis=2, keepdims=True)
        f = g + 0.5 * (f - g)
    elif PERTURB == "warm":  # warmer white balance
        f = f * np.array([0.8, 1.0, 1.15], np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def init():
    global _det
    if METHOD.startswith("sat"):
        _det = Baseline(float(METHOD[3:]))
        return
    model = pickle.load(open(MODEL, "rb")) if MODEL else None
    _det = D.Detector(METHOD, data.cameras(), model,
                      dict(min_score=-9, unique_ratio=1e9, size_ratio=1e9))


def run(item):
    frames = [perturb(f) for f in data.load(item)] if PERTURB != "none" else data.load(item)
    rows = []
    variants = [("normal", list(item["exclude"]))]
    if item["spot"] in data.SPOTS:
        sx, sy = data.SPOTS[item["spot"]]
        variants.append(("hidden", list(item["exclude"]) + [[sx, sy, 0.40]]))
    for name, exclude in variants:
        r = _det.localize(frames, exclude)
        r.update(perturb=PERTURB, label=item["label"], spot=item["spot"], variant=name, kind=item["kind"],
                 run=item["run"], shelf=item["shelf"])
        if "x" in r and item["spot"] in data.SPOTS:
            sx, sy = data.SPOTS[item["spot"]]
            r["d"] = math.hypot(r["x"] - sx, r["y"] - sy)
        rows.append(r)
    return rows


if __name__ == "__main__":
    items = data.png_items() + data.vid_items()
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 10, initializer=init) as pool, open(OUT, "w") as f:
        for rows in pool.imap_unordered(run, items):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
