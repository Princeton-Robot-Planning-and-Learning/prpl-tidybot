"""Image-space pose refinement with the full 3D box: the bin's silhouette in a camera
is the convex hull of its 8 corners (rim on the sides facing away from the camera,
floor contact on the sides facing it). Maximise the mean log-intensity gradient
across that hull's edges, sampled every ~1.5 px in the camera image, by coordinate
descent from the rim-plane fit."""
import math

import cv2 as cv
import numpy as np

import detectors as D

_CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)])


def hull(cam, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = _CORNERS * [D.L, D.W, D.H]
    w = np.c_[x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1], loc[:, 2]]
    return cv.convexHull(cam.project(w).astype(np.float32)).reshape(-1, 2).astype(np.float64)


def sample(img, pts):
    return cv.remap(img, pts[:, 0].astype(np.float32).reshape(1, -1),
                    pts[:, 1].astype(np.float32).reshape(1, -1), cv.INTER_LINEAR).ravel()


def edge_score(gx, gy, poly, inset=0.12):
    tot, n = 0.0, 0
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        seg = b - a
        ln = float(np.hypot(*seg))
        if ln < 4:
            continue
        t = np.linspace(inset, 1 - inset, max(int(ln / 1.5), 2))
        pts = a + t[:, None] * seg
        nrm = np.array([-seg[1], seg[0]]) / ln
        g = np.abs(sample(gx, pts) * nrm[0] + sample(gy, pts) * nrm[1])
        tot += float(g.sum()); n += len(g)
    return tot / max(n, 1)


def refine(bgr_cam, cam, x, y, yaw):
    lg = cv.GaussianBlur(D.log_gray(bgr_cam), (0, 0), 1.0)
    gx = cv.Sobel(lg, cv.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv.Sobel(lg, cv.CV_32F, 0, 1, ksize=3) / 8.0
    pose = np.array([x, y, yaw])
    best = edge_score(gx, gy, hull(cam, *pose))
    step = np.array([0.008, 0.008, math.radians(2.0)])
    while step[0] > 4e-4:
        improved = False
        for k in range(3):
            for sgn in (-1, 1):
                cand = pose.copy(); cand[k] += sgn * step[k]
                sc = edge_score(gx, gy, hull(cam, *cand))
                if sc > best:
                    best, pose, improved = sc, cand, True
        if not improved:
            step = step / 2
    return float(pose[0]), float(pose[1]), float(pose[2]), best
