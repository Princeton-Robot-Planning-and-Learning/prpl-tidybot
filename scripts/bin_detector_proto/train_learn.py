"""Train candidate B's per-pixel cardboard classifier on a few dev frame sets.

Positives: pixels inside the bin's rim (projected at rim height) on the dev sets,
using the fitted pose from the edge detector on those same sets (checked by eye in
overlay crops). Negatives: every pixel outside the bin's silhouette grown by 6 cm
(the outer walls and the shadow band are left out). Everything else in view - floor,
shelf, robot, chair, tape, people - is a negative.
"""
import json
import math
import pickle
import sys

import cv2 as cv
import numpy as np

import data
import detectors as D

TRAIN = ["png:10-37-51/00_10-37-58", "png:10-59-22/00_10-59-30",
         "vid:10-15-57@0", "vid:10-15-57@3849", "vid:10-59-22@249", "vid:10-59-22@748"]


def rect_poly(cam, x, y, yaw, l, w, z):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [[x + c * a * l / 2 - s * b * w / 2, y + s * a * l / 2 + c * b * w / 2, z]
           for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return cam.project(np.array(pts))


def main(fits_path, out_path):
    fits = {}
    for line in open(fits_path):
        r = json.loads(line)
        if r["variant"] == "normal" and "x" in r:
            fits[r["label"]] = r
    cams = data.cameras()
    items = {it["label"]: it for it in data.png_items() + data.vid_items()}
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for lab in TRAIN:
        it, f = items[lab], fits[lab]
        frame = data.load(it)[-1]
        i = f["cam"]
        img = data.split(frame)[i]
        small = cv.resize(img, (640, 360), interpolation=cv.INTER_AREA)
        feats = D.learn_features(small)
        scale = 0.5
        pos = np.zeros(small.shape[:2], np.uint8)
        cv.fillConvexPoly(pos, np.int32(rect_poly(cams[i], f["x"], f["y"], f["yaw"], D.L - 0.02, D.W - 0.02, D.H) * scale), 1)
        grown = np.zeros_like(pos)
        hull = cv.convexHull(np.float32(np.r_[rect_poly(cams[i], f["x"], f["y"], f["yaw"], D.L + 0.12, D.W + 0.12, D.H),
                                               rect_poly(cams[i], f["x"], f["y"], f["yaw"], D.L + 0.12, D.W + 0.12, 0.0)] * scale))
        cv.fillConvexPoly(grown, np.int32(hull.reshape(-1, 2)), 1)
        p_idx = np.argwhere(pos > 0)
        n_idx = np.argwhere(grown == 0)
        n_idx = n_idx[rng.choice(len(n_idx), size=min(len(n_idx), 20 * len(p_idx)), replace=False)]
        for idx, lbl in ((p_idx, 1.0), (n_idx, 0.0)):
            xs.append(feats[idx[:, 0], idx[:, 1]])
            ys.append(np.full(len(idx), lbl))
        print(lab, "pos", len(p_idx), "neg", len(n_idx))
    x, y = np.vstack(xs), np.concatenate(ys)
    model = D.Logistic()
    model.fit(x, y)
    p = model.predict(x)
    print("train acc pos", float((p[y > 0.5] > 0.5).mean()), "neg", float((p[y < 0.5] < 0.5).mean()))
    pickle.dump(model, open(out_path, "wb"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
