"""Comparison figure: outcomes per spot, photometric stress, score margin, hard frames."""
import json
import math
import sys

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import gates as G
import overlay

R = "/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/analysis/2026-10-20-pilot/bin_detector/results"
OUT = sys.argv[1]
METHODS = [("sat130", "deployed: saturation >130", "#999999", "//"),
           ("sat150", "saturation >150", "#555555", "xx"),
           ("edge", "A: rim edges", "#0072B2", None),
           ("learn", "B: learned colour", "#E69F00", None),
           ("contrast", "C: floor contrast", "#CC79A7", None)]
PERTS = [("none", "as recorded"), ("dim", "40% darker"), ("bright", "35% brighter\n(clipped)"),
         ("desat", "half the\nsaturation"), ("warm", "warmer white\nbalance")]
WRONG = "#D55E00"
MISS = "#E4E4E0"


def rows(m, p):
    return [json.loads(l) for l in open(f"{R}/{m}_{p}.jsonl")]


def outcomes(m, p):
    rs = rows(m, p)
    tau, _, _ = G.choose(rows(m, "none"))
    res = {}
    for key in ("close", "far", "hidden", "wall"):
        sel = [r for r in rs if (r["variant"] == "hidden") == (key == "hidden") and (key == "hidden" or r["spot"] == key)]
        cs = [G.classify(r, tau) for r in sel]
        res[key] = {c: cs.count(c) for c in ("correct", "wrong", "miss", "fp", "reject")} | {"n": len(cs)}
        res[key]["err"] = [r["d"] for r, c in zip(sel, cs) if c == "correct"]
    return res, tau


def main():
    fig = plt.figure(figsize=(16, 15.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.15], hspace=0.42, wspace=0.35)
    clean = {m: outcomes(m, "none") for m, *_ in METHODS}

    # row 1: outcomes per spot (clean)
    for col, spot in enumerate(("close", "far")):
        ax = fig.add_subplot(gs[0, 3 * col:3 * col + 3])
        for k, (m, name, colr, hatch) in enumerate(METHODS):
            o = clean[m][0][spot]
            y = len(METHODS) - 1 - k
            left = 0
            for part, c, h in (("correct", colr, hatch), ("wrong", WRONG, "\\\\"), ("miss", MISS, None)):
                v = o[part]
                if v:
                    ax.barh(y, v, left=left, color=c, hatch=h, edgecolor="white", linewidth=2, height=0.62)
                left += v
            n = o["n"]
            e = o["err"]
            txt = f"{o['correct']}/{n} correct · {o['wrong']}/{n} wrong · {o['miss']}/{n} miss"
            if e:
                txt += f"\nerror vs spot: mean {100 * np.mean(e):.1f} cm, max {100 * max(e):.1f} cm"
            ax.text(n + 1, y, txt, va="center", fontsize=8.5, color="#222")
        ax.set_yticks(range(len(METHODS)), [name for _, name, *_ in METHODS][::-1] if col == 0 else [], fontsize=9)
        ax.set_xlim(0, n * 2.05)
        ax.set_xticks([0, n // 2, n])
        ax.set_xlabel("frame sets")
        lab = "close spot (-0.30, -0.78), bottom camera" if spot == "close" else "far spot (0.23, 1.37), top camera"
        ax.set_title(f"{lab}: n={n} frame sets", fontsize=10.5, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.02, 0.975, "Bin localization from the ceiling cameras, 2026-10-20 replays (Tossing3D real-robot pilot)",
             fontsize=13, weight="bold")
    fig.text(0.02, 0.958, "Saved frame sets + video windows; correct = within 12 cm of the labelled spot. "
             "Orange-red = a confident WRONG bin. Thresholds for A/B/C frozen on the dev runs (before 11:00).",
             fontsize=9.5, color="#444")

    # row 2 left: stress test
    ax = fig.add_subplot(gs[1, 0:4])
    w = 0.16
    for k, (m, name, colr, hatch) in enumerate(METHODS):
        for j, (p, plab) in enumerate(PERTS):
            o, _ = outcomes(m, p)
            n = o["close"]["n"] + o["far"]["n"]
            ok = o["close"]["correct"] + o["far"]["correct"]
            bad = o["close"]["wrong"] + o["far"]["wrong"] + o["hidden"]["fp"] + o["wall"]["fp"]
            x = j + (k - 2) * w
            ax.bar(x, ok, width=w - 0.02, color=colr, hatch=hatch, edgecolor="white", linewidth=1)
            if bad:
                ax.text(x, ok + 3, f"✕{bad}", ha="center", fontsize=7.5, color=WRONG, weight="bold", rotation=90)
    ax.set_xticks(range(len(PERTS)), [p[1] for p in PERTS], fontsize=8.5)
    ax.set_ylabel("correct frame sets")
    ax.set_ylim(0, 200)
    ax.set_title(f"Photometric stress (same frames re-lit / desaturated): correct of n={n} positives\n"
                 "✕N = wrong bins returned (wrong pose + bin-hidden + wall); A returns none in any condition", fontsize=9.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=h) for _, _, c, h in METHODS]
    ax.legend(handles, [nm for _, nm, *_ in METHODS], fontsize=8, ncol=5, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # row 2 right: margin
    ax = fig.add_subplot(gs[1, 4:6])
    for k, (m, name, colr, _) in enumerate(METHODS[2:]):
        rs = rows(m, "none")
        tau, _, _ = G.choose(rs)
        t = [r["score"] / tau for r in rs if G.is_true(r) and G.common(r)]
        f = [max(r["score"], 1e-3 * tau) / tau for r in rs if not G.is_true(r) and G.common(r)]
        rng = np.random.default_rng(k)
        ax.scatter(t, k + 0.15 + rng.uniform(-0.08, 0.08, len(t)), s=9, color=colr, alpha=0.6, linewidths=0)
        ax.scatter(f, k - 0.15 + rng.uniform(-0.08, 0.08, len(f)), s=12, marker="x", color="#666", linewidths=0.8)
        ratio = min(t) / max(f) if f else float("inf")
        ax.text(0.06, k + 0.38, f"weakest bin / strongest non-bin = {ratio:.2f}x", fontsize=7.5, va="center")
    ax.axvline(1, color="#999", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 60)
    ax.set_yticks(range(3), [nm for _, nm, *_ in METHODS[2:]], fontsize=8.5)
    ax.set_ylim(-0.5, 2.6)
    ax.set_xlabel("score / frozen threshold (log)")
    ax.set_title("Score margin (clean frames): dots = true bins,\nx = best non-bin per bin-hidden / wall set", fontsize=9.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # row 3: hard frames
    cams = data.cameras()
    items = {it["label"]: it for it in data.png_items() + data.vid_items()}
    cases = [("png:10-59-22/00_10-59-30", "close spot, shelf in view"),
             ("png:17-24-13/00_17-24-39_bin", "far spot, dim corner + shelf"),
             ("png:18-06-59/00_18-07-16_bin", "far spot, a person beside it"),
             ("vid:12-34-53@628", "close spot, low-res video"),
             ("png:18-10-36/21_18-30-26", "far spot, shelf gone"),
             ("shelf", "where saturation >150 puts the bin")]
    fits = {m: {(r["label"], r.get("perturb", "none")): r for r in rows(m, "none") if r["variant"] == "normal"}
            for m in ("edge", "sat150")}
    for j, (lab, title) in enumerate(cases):
        ax = fig.add_subplot(gs[2, j])
        if lab == "shelf":
            bad = [r for r in fits["sat150"].values() if r.get("found") and r["spot"] == "close" and r.get("d", 0) > 0.12]
            r = bad[0]
            it = items[r["label"]]
            fr = data.load(it)[-1]
            ci = 0 if r["y"] > 0 else 1
            img = overlay.crop(fr, cams, ci, r["x"], r["y"], r["yaw"], half=0.4, color=(0, 0, 255))
            title = f"{title}\n({r['label'][4:]}: x={r['x']:.2f}, y={r['y']:.2f})"
        else:
            it = items[lab]
            fr = data.load(it)[-1]
            r = fits["edge"][(lab, "none")]
            sx, sy = data.SPOTS[it["spot"]]
            img = overlay.crop(fr, cams, r["cam"], r["x"], r["y"], r["yaw"], color=(178, 114, 0), cx=sx, cy=sy)
            title = f"{title}\nA: d={100 * r['d']:.1f} cm"
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    fig.text(0.02, 0.33, "Hard frames, rim-plane view (2 mm/px): blue = A's fit; red (last) = the saturation detector's "
             "confident fit on the wooden shelf", fontsize=9.5, color="#444")
    fig.savefig(OUT, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
