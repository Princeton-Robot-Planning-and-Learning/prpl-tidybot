"""Three prototype bin detectors sharing one search.

Shared geometry: each camera's image is warped onto the bin's *rim plane* (z = 0.21 m)
through the calibrated homography, so the rim - whatever the bin's position - becomes
an exact 0.39 x 0.29 m rectangle in a metric top-down view. For each yaw the view is
rendered rotated (straight from the camera image, one warp per yaw), so the rectangle
is axis-aligned and every score is a handful of box filters. Candidates are the best
pose per camera; the safety checks (uniqueness, size specificity, robot exclusion) are
the same for all three. What differs is the evidence:

* ``edge``  - log-intensity gradient across each of the four rim sides (min over sides).
* ``learn`` - a per-pixel cardboard probability from a classifier trained on a few
  labelled frame sets, on illumination-normalized colour; inside-minus-ring.
* ``contrast`` - Lab distance from the *local* floor (large median), normalized by the
  floor's own noise; inside-minus-ring. No colour model at all.
"""
from __future__ import annotations

import math
import time
from typing import Any

import cv2 as cv
import numpy as np

import data

L, W, H = 0.39, 0.29, 0.21
RES = 0.01  # coarse grid (m/px)
FINE_RES = 0.004
YAWS = np.deg2rad(np.arange(0.0, 180.0, 3.0))
SEAM_PX = 8
ROBOT_HALF = 0.45


def grid_homography(cam, centre, yaw, res, size, z=H):
    """H: rotated-grid pixel (u, v) -> camera pixel, for a size x size grid centred on
    ``centre`` whose +u axis is the map direction ``yaw`` (and +v is yaw - 90 deg)."""
    c, s = math.cos(yaw), math.sin(yaw)
    h = (size - 1) / 2.0
    # map = centre + res*(u-h)*[c, s] + res*(v-h)*[s, -c]
    g = np.array([
        [res * c, res * s, centre[0] - res * h * (c + s)],
        [res * s, -res * c, centre[1] - res * h * (s - c)],
        [0, 0, 1.0],
    ])
    return data.plane_homography(cam, z) @ g, g


def warp(img, hm, size, interp=cv.INTER_LINEAR):
    return cv.warpPerspective(img, hm, (size, size), flags=interp | cv.WARP_INVERSE_MAP,
                              borderMode=cv.BORDER_CONSTANT, borderValue=0)


def box(img, hu, hv):
    """Mean over the (2hu+1) x (2hv+1) box centred on each pixel (u half-width hu)."""
    return cv.boxFilter(img, cv.CV_32F, (2 * hu + 1, 2 * hv + 1), normalize=True,
                        borderType=cv.BORDER_CONSTANT)


def shift(img, du, dv):
    """out[v, u] = img[v + dv, u + du] (zero outside)."""
    m = np.float32([[1, 0, -du], [0, 1, -dv]])
    return cv.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST,
                         borderMode=cv.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- pixel features
def log_gray(bgr):
    g = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    return np.log(g + 4.0)


def lab_f(bgr):
    return cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)


def local_floor(img, k_px):
    """Large median per channel (computed at 1/8 res), the local floor."""
    h, w = img.shape[:2]
    small = cv.resize(img, (w // 8, h // 8), interpolation=cv.INTER_AREA)
    chans = small.reshape(small.shape[0], small.shape[1], -1)
    out = []
    for ch in range(chans.shape[2]):
        a = np.clip(chans[:, :, ch], 0, 255).astype(np.uint8)
        out.append(cv.resize(cv.medianBlur(a, k_px), (w, h), interpolation=cv.INTER_LINEAR))
    return np.dstack(out).astype(np.float32)


def contrast_map(bgr):
    """Lab distance from the local floor, in units of the floor's own robust spread,
    squashed to [0, 1): 0.5 at 4 spreads."""
    lab = lab_f(bgr)
    floor = local_floor(lab, 41)
    d = np.sqrt(((lab - floor) ** 2).sum(axis=2))
    sigma = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), 1.0)
    z = d / sigma
    return (z / (z + 4.0)).astype(np.float32)


def learn_features(bgr):
    """Illumination-normalized colour: chromaticity, Lab a*/b*, and L and S relative to
    the local floor's."""
    f = bgr.astype(np.float32) + 1.0
    s = f.sum(axis=2, keepdims=True)
    chrom = f / s
    lab = lab_f(bgr)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    floor = local_floor(np.dstack([lab[:, :, 0], hsv[:, :, 1], lab[:, :, 1], lab[:, :, 2]]), 41)
    l_rel = np.log((lab[:, :, 0] + 4) / (floor[:, :, 0] + 4))
    s_rel = (hsv[:, :, 1] - floor[:, :, 1]) / 64.0
    da = (lab[:, :, 1] - floor[:, :, 2]) / 16.0
    db = (lab[:, :, 2] - floor[:, :, 3]) / 16.0
    return np.dstack([chrom[:, :, 2] * 3 - 1, chrom[:, :, 1] * 3 - 1, l_rel, s_rel, da, db,
                      (lab[:, :, 1] - 128) / 16.0, (lab[:, :, 2] - 128) / 16.0])


class Logistic:
    """Logistic regression on quadratic features of ``learn_features`` (numpy)."""

    def __init__(self):
        self.w = None
        self.mu = None
        self.sd = None

    @staticmethod
    def expand(x):
        n = x.shape[1]
        cols = [x] + [x[:, i:i + 1] * x[:, j:j + 1] for i in range(n) for j in range(i, n)]
        return np.hstack(cols)

    def fit(self, x, y, iters=400, lam=1e-3):
        z = self.expand(x)
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        z = np.c_[(z - self.mu) / self.sd, np.ones(len(z))]
        w = np.zeros(z.shape[1])
        wpos = 0.5 / max(y.mean(), 1e-6)
        wneg = 0.5 / max(1 - y.mean(), 1e-6)
        sw = np.where(y > 0.5, wpos, wneg)
        lr = 0.5
        for _ in range(iters):  # full-batch gradient descent
            p = 1 / (1 + np.exp(-z @ w))
            g = z.T @ (sw * (p - y)) / len(y) + lam * w
            w -= lr * g
        self.w = w

    def predict(self, x):
        shp = x.shape[:-1]
        z = self.expand(x.reshape(-1, x.shape[-1]))
        z = np.c_[(z - self.mu) / self.sd, np.ones(len(z))]
        return (1 / (1 + np.exp(-z @ self.w))).reshape(shp).astype(np.float32)


# ---------------------------------------------------------------- scoring
def side_segments(gu, gv, hl, hw, t=1, inset=2, nseg=3):  # t, inset in px
    """Straight-edge evidence along each rim side, in ``nseg`` segments per side: the
    mean gradient *across* the side minus the mean gradient *along* it (a straight
    edge has the first, texture has both). Returns a list of 4*nseg maps."""
    out = []
    for normal, tangent, half_len, off, along_u in (
        (gv, gu, hl - inset, hw, True), (gu, gv, hw - inset, hl, False)
    ):
        e = normal - tangent
        seg = max(int(half_len / nseg), 1)
        k = box(e, seg, t) if along_u else box(e, t, seg)
        for sign in (-1, 1):
            for j in range(nseg):
                c = int(round(-half_len + (2 * j + 1) * half_len / nseg))
                out.append(shift(k, c, sign * off) if along_u else shift(k, sign * off, c))
    return out


def edge_score(gu, gv, hl, hw, res, mode="min"):
    """``min`` over the 12 side segments detects (every side must be a straight
    edge); the ``mean`` localizes (a thin strip, all of the rim's evidence)."""
    if mode == "min":
        s = side_segments(gu, gv, hl, hw, t=max(1, int(round(0.01 / res))),
                          inset=max(2, int(round(0.02 / res))))
        return np.minimum.reduce(s), s
    s = side_segments(gu, gv, hl, hw, t=1, inset=max(2, int(round(0.02 / res))), nseg=1)
    return np.mean(s, axis=0), s


def region_score(p, valid, hl, hw, res):
    """Mean p over the rectangle shrunk by 2 cm minus mean p over a ring 4-12 cm
    outside it (each normalized by the visible area)."""
    shr = int(round(0.02 / res))
    r0, r1 = int(round(0.04 / res)), int(round(0.12 / res))
    pv = p * valid

    def mean_box(a, b):
        num = box(pv, a, b)
        den = box(valid, a, b)
        return num, den

    n_in, d_in = mean_box(hl - shr, hw - shr)
    n_o0, d_o0 = mean_box(hl + r0, hw + r0)
    n_o1, d_o1 = mean_box(hl + r1, hw + r1)
    a_o0 = (2 * (hl + r0) + 1) * (2 * (hw + r0) + 1)
    a_o1 = (2 * (hl + r1) + 1) * (2 * (hw + r1) + 1)
    ring_num = n_o1 * a_o1 - n_o0 * a_o0
    ring_den = d_o1 * a_o1 - d_o0 * a_o0
    inside = n_in / np.maximum(d_in, 1e-6)
    ring = ring_num / np.maximum(ring_den, 1e-6)
    ok = (d_in > 0.9) & (ring_den / (a_o1 - a_o0) > 0.6)
    return np.where(ok, inside - ring, -1.0).astype(np.float32), inside, ring


class Detector:
    """``method`` in {'edge', 'learn', 'contrast'}."""

    def __init__(self, method, cams, model: Logistic | None = None, params=None):
        self.method = method
        self.cams = cams
        self.model = model
        p = dict(min_score={"edge": 0.10, "learn": 0.45, "contrast": 0.30}[method],
                 unique_ratio=0.75, size_ratio=0.85, n_frames=3, clutter_w=0.0, refine_mode="mean")
        p.update(params or {})
        self.p = p

    # pixel evidence in the camera image (full res), before warping
    def evidence(self, bgr):
        if self.method == "edge":
            return cv.GaussianBlur(log_gray(bgr), (0, 0), 1.2)
        if self.method == "contrast":
            return cv.GaussianBlur(contrast_map(bgr), (0, 0), 1.2)
        prob = self.model.predict(learn_features(cv.resize(bgr, (640, 360), interpolation=cv.INTER_AREA)))
        return cv.GaussianBlur(cv.resize(prob, (bgr.shape[1], bgr.shape[0])), (0, 0), 1.2)

    def score_maps(self, ev, valid_img, cam, centre, yaw, res, size, dims=(L, W), mode="min"):
        hm, g = grid_homography(cam, centre, yaw, res, size)
        e = warp(ev, hm, size)
        valid = (warp(valid_img, hm, size, cv.INTER_NEAREST) > 0).astype(np.float32)
        hl = int(round(dims[0] / 2 / res))
        hw = int(round(dims[1] / 2 / res))
        if self.method == "edge":
            gu = cv.Sobel(e, cv.CV_32F, 1, 0, ksize=3) / 8.0
            gv = cv.Sobel(e, cv.CV_32F, 0, 1, ksize=3) / 8.0
            vmask = cv.erode(valid, np.ones((5, 5), np.uint8))
            gu, gv = np.abs(gu) * vmask, np.abs(gv) * vmask
            # per cm of rim, independent of grid resolution
            k = 0.01 / res
            sc, sides = edge_score(gu * k, gv * k, hl, hw, res, mode)
            # clutter: the mean gradient in a strip 4-12 cm outside each rim side
            # (floor around a bin is smooth; a chair, the shelf or a person is not);
            # the median over the four sides, so one cluttered side (a person
            # standing by the bin) does not veto it
            mag = np.maximum(gu, gv) * k
            r0, r1 = int(round(0.04 / res)), int(round(0.12 / res))
            mid, half = (r0 + r1) // 2, max((r1 - r0) // 2, 1)
            num_l = box(mag, hl, half); den_l = box(vmask, hl, half)
            num_s = box(mag, half, hw); den_s = box(vmask, half, hw)
            rings = []
            for nm, dn, du, dv in ((num_l, den_l, 0, -(hw + mid)), (num_l, den_l, 0, hw + mid),
                                   (num_s, den_s, -(hl + mid), 0), (num_s, den_s, hl + mid, 0)):
                n_, d_ = shift(nm, du, dv), shift(dn, du, dv)
                rings.append(np.where(d_ > 0.5, n_ / np.maximum(d_, 1e-6), 1.0))
            rs = np.sort(np.stack(rings), axis=0)
            ring_only = 0.5 * (rs[1] + rs[2])
            cover = box(valid, hl + 2, hw + 2)
            raw = sc
            sc = sc - self.p["clutter_w"] * ring_only
            sc = np.where(cover > 0.97, sc, -1.0)
            return sc, g, dict(raw=raw, ring=ring_only, sides=sides, valid=valid, e=e, gu=gu * k, gv=gv * k, hl=hl, hw=hw)
        sc, inside, ring = region_score(e, valid, hl, hw, res)
        return sc, g, dict(inside=inside, ring=ring, valid=valid, e=e, hl=hl, hw=hw)

    def camera_valid(self, shape, i, n):
        v = np.full(shape[:2], 255, np.uint8)
        if i > 0:
            v[:SEAM_PX] = 0
        if i < n - 1:
            v[-SEAM_PX:] = 0
        return v

    def best_per_camera(self, ev, vimg, cam, exclude, bounds):
        x0, x1, y0, y1 = bounds
        centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        size = int(math.ceil(math.hypot(x1 - x0, y1 - y0) / RES)) + 8
        cands = []
        for yaw in YAWS:
            sc, g, _ = self.score_maps(ev, vimg, cam, centre, yaw, RES, size)
            # map coordinates of every grid centre
            vv, uu = np.mgrid[0:size, 0:size]
            mx = g[0, 0] * uu + g[0, 1] * vv + g[0, 2]
            my = g[1, 0] * uu + g[1, 1] * vv + g[1, 2]
            ok = (mx >= x0) & (mx <= x1) & (my >= y0) & (my <= y1)
            for ex, ey, half in exclude:
                ok &= ~((np.abs(mx - ex) < half) & (np.abs(my - ey) < half))
            sc = np.where(ok, sc, -1.0)
            # local maxima, top few per yaw
            dil = cv.dilate(sc, np.ones((9, 9), np.uint8))
            peaks = np.argwhere((sc >= dil) & (sc > 0))
            if len(peaks) == 0:
                continue
            vals = sc[peaks[:, 0], peaks[:, 1]]
            for k in np.argsort(-vals)[:5]:
                r, c = peaks[k]
                cands.append((float(vals[k]), float(mx[r, c]), float(my[r, c]), float(yaw)))
        cands.sort(key=lambda t: -t[0])
        # non-maximum suppression in pose space (position only)
        kept = []
        for cand in cands:
            if all(math.hypot(cand[1] - k[1], cand[2] - k[2]) > 0.25 for k in kept):
                kept.append(cand)
            if len(kept) >= 6:
                break
        return kept

    def refine(self, ev, vimg, cam, x, y, yaw):
        """Fine grid around a coarse pose; returns (score, x, y, yaw, aux)."""
        size = int(round((L + 0.36) / FINE_RES)) | 1
        mid, reach = size // 2, int(round(0.03 / FINE_RES))
        best = (-9.0, x, y, yaw, None)
        for dyaw in np.deg2rad(np.arange(-4.0, 4.01, 0.5)):
            sc, g, aux = self.score_maps(ev, vimg, cam, (x, y), yaw + dyaw, FINE_RES, size, mode=self.p['refine_mode'])
            win = np.full(sc.shape, -9.0, np.float32)
            win[mid - reach:mid + reach + 1, mid - reach:mid + reach + 1] = sc[mid - reach:mid + reach + 1, mid - reach:mid + reach + 1]
            r, c = np.unravel_index(int(np.argmax(win)), sc.shape)
            if sc[r, c] > best[0]:
                mx = g[0, 0] * c + g[0, 1] * r + g[0, 2]
                my = g[1, 0] * c + g[1, 1] * r + g[1, 2]
                best = (float(sc[r, c]), float(mx), float(my), float(yaw + dyaw), aux)
        return best

    def score_at(self, ev, vimg, cam, x, y, yaw, dims):
        size = int(round((dims[0] + 0.36) / RES)) | 1
        sc, _, _ = self.score_maps(ev, vimg, cam, (x, y), yaw, RES, size, dims)
        return float(sc[size // 2, size // 2])

    def localize(self, frames, exclude=(), bounds=data.SEARCH, diag=None):
        t0 = time.time()
        frames = frames[-self.p["n_frames"]:]
        mean = np.mean([f.astype(np.float32) for f in frames], axis=0).astype(np.uint8)
        per_cam = []
        for i, (cam, img) in enumerate(zip(self.cams, data.split(mean, len(self.cams)))):
            ev = self.evidence(img)
            vimg = self.camera_valid(img.shape, i, len(self.cams))
            foot = np.array([cam.to_floor(u, v, H) for u, v in
                             ((0, 0), (img.shape[1], 0), (0, img.shape[0]), (img.shape[1], img.shape[0]))])
            cb = (max(bounds[0], foot[:, 0].min()), min(bounds[1], foot[:, 0].max()),
                  max(bounds[2], foot[:, 1].min()), min(bounds[3], foot[:, 1].max()))
            kept = self.best_per_camera(ev, vimg, cam, exclude, cb) if cb[0] < cb[1] and cb[2] < cb[3] else []
            per_cam.append((i, ev, vimg, kept))
        # pool all candidates across cameras; a candidate seen by both cameras (the
        # overlap strip) is one candidate
        pool = [(k[0], i, k) for i, _, _, kept in per_cam for k in kept]
        pool.sort(key=lambda t: -t[0])
        out: dict[str, Any] = dict(found=False, reason="no candidate")
        if diag is not None:
            diag["pool"] = [(s, i, k[1], k[2], k[3]) for s, i, k in pool[:8]]
        if not pool:
            out["t"] = time.time() - t0
            return out
        s0, i0, k0 = pool[0]
        _, ev0, vimg0, _ = per_cam[i0]
        fs, fx, fy, fyaw, aux = self.refine(ev0, vimg0, self.cams[i0], k0[1], k0[2], k0[3])
        rivals = [s for s, i, k in pool[1:] if math.hypot(k[1] - fx, k[2] - fy) > 0.25]
        second = max(rivals, default=0.0)
        det = self.score_at(ev0, vimg0, self.cams[i0], fx, fy, fyaw, (L, W))
        fs_fine, fs = fs, max(det, s0)
        big = self.score_at(ev0, vimg0, self.cams[i0], fx, fy, fyaw, (L + 0.08, W + 0.08))
        small = self.score_at(ev0, vimg0, self.cams[i0], fx, fy, fyaw, (L - 0.08, W - 0.08))
        # whole bin in view with a 10 cm margin, and its inside darker than the floor
        # ring around it (log gray, inside shrunk 3 cm vs a ring 4-12 cm out)
        lg = cv.GaussianBlur(log_gray(data.split(mean, len(self.cams))[i0]), (0, 0), 1.2)
        size = int(round((L + 0.5) / RES)) | 1
        hm, _ = grid_homography(self.cams[i0], (fx, fy), fyaw, RES, size)
        e = warp(lg, hm, size)
        val = (warp(vimg0, hm, size, cv.INTER_NEAREST) > 0).astype(np.float32)
        hl, hw = int(round(L / 2 / RES)), int(round(W / 2 / RES))
        m10 = int(round(0.10 / RES))
        c0 = size // 2
        margin_ok = bool(val[c0 - hw - m10:c0 + hw + m10 + 1, c0 - hl - m10:c0 + hl + m10 + 1].min() > 0)
        _, ins, ring = region_score(e, val, hl - 1, hw - 1, RES)
        dark = float(ins[c0, c0] - ring[c0, c0])
        out.update(margin_ok=margin_ok, dark=dark)
        out.update(x=fx, y=fy, yaw=(fyaw + math.pi / 2) % math.pi - math.pi / 2, score=fs, fine=fs_fine,
                   coarse=s0, second=second, big=big, small=small, cam=i0)
        p = self.p
        if fs < p["min_score"]:
            out["reason"] = f"score {fs:.3f} < {p['min_score']}"
        elif second > p["unique_ratio"] * fs:
            out["reason"] = f"ambiguous: rival {second:.3f} vs {fs:.3f}"
        elif max(big, small) > p["size_ratio"] * fs:
            out["reason"] = f"not size-specific: +8cm {big:.3f} -8cm {small:.3f} vs {fs:.3f}"
        else:
            out["found"] = True
            out["reason"] = "ok"
        out["t"] = time.time() - t0
        return out
