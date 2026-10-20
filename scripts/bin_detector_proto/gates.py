"""Freeze each method's score threshold on the dev runs, apply it to the test runs.

Common physical gates for every method: the whole bin lies >= 10 cm inside the chosen
camera's view, and its inside is darker than the floor ring around it (log ratio
< -0.3; the darkest true bin read -0.58). Optional: uniqueness (no rival candidate
elsewhere scores >= ``unique`` x the best). The method's own score threshold is the
geometric mean of the dev runs' weakest true bin and strongest non-bin (or 1.1x the
strongest non-bin when they overlap: a miss is acceptable, a wrong bin is not).
"""
import json
import math
import sys
from collections import defaultdict

from analyze import DEV_RUNS, TOL

DARK = -0.3


def common(r, unique=None):
    if r.get("baseline"):
        return r["found"]
    if "x" not in r or not r["margin_ok"] or r["dark"] >= DARK:
        return False
    if unique is not None and r["second"] >= unique * r["score"]:
        return False
    return True


def is_true(r):
    return r["variant"] == "normal" and r["spot"] in ("close", "far") and r.get("d", 9) < TOL


def choose(rows, unique=None):
    dev = [r for r in rows if r["run"] in DEV_RUNS]
    t = [r["score"] for r in dev if is_true(r) and common(r, unique)]
    f = [r["score"] for r in dev if not is_true(r) and common(r, unique)]
    lo, hi = max(f, default=0.0), min(t, default=float("inf"))
    tau = math.sqrt(lo * hi) if lo < hi and lo > 0 else lo * 1.1 if lo > 0 else hi * 0.5
    return tau, lo, hi


def classify(r, tau, unique=None):
    found = common(r, unique) and (r.get("baseline") or r["score"] >= tau)
    if r["variant"] == "hidden" or r["spot"] == "wall":
        return "fp" if found else "reject"
    if not found:
        return "miss"
    return "correct" if r["d"] < TOL else "wrong"


def summarize(rows, tau, unique=None):
    res = {}
    for split in ("dev", "test", "all"):
        out = defaultdict(lambda: defaultdict(int))
        errs = defaultdict(list)
        wrongs = []
        for r in rows:
            if split == "dev" and r["run"] not in DEV_RUNS:
                continue
            if split == "test" and r["run"] in DEV_RUNS:
                continue
            key = "hidden" if r["variant"] == "hidden" else r["spot"]
            c = classify(r, tau, unique)
            out[key][c] += 1
            if c == "correct":
                errs[key].append(r["d"])
            if c in ("wrong", "fp"):
                wrongs.append((r["label"], r["variant"], round(r["x"], 3), round(r["y"], 3), round(r["score"], 4)))
        res[split] = dict(counts={k: dict(v) for k, v in out.items()},
                          err={k: v for k, v in errs.items()}, wrongs=wrongs)
    return res


def safe_interval(rows, unique=None):
    """Thresholds giving zero wrong/fp AND zero misses on all data (None if none)."""
    t = [r["score"] for r in rows if is_true(r) and common(r, unique)]
    n_true_all = sum(1 for r in rows if r["variant"] == "normal" and r["spot"] in ("close", "far"))
    f = [r["score"] for r in rows if not is_true(r) and common(r, unique)]
    lo, hi = max(f, default=0.0), min(t, default=0.0)
    return dict(false_max=lo, true_min=hi, true_passing_common=len(t), n_true=n_true_all,
                ratio=(hi / lo if lo > 0 else float("inf")))


if __name__ == "__main__":
    args = sys.argv[1:]
    tau_from = None
    if args[0] == "--tau-from":
        tau_from, args = args[1], args[2:]
    for path in args:
        rows = [json.loads(l) for l in open(path)]
        ungated = sum(is_true(r) for r in rows if r["variant"] == "normal")
        npos = sum(1 for r in rows if r["variant"] == "normal" and r["spot"] in ("close", "far"))
        print(f"== {path.split('/')[-1]}: ungated top candidate correct {ungated}/{npos}")
        for unique in (None, 0.8):
            tau, lo, hi = choose([json.loads(l) for l in open(tau_from)] if tau_from else rows, unique)
            s = summarize(rows, tau, unique)
            print(f" unique={unique} tau={tau:.4f} (dev false max {lo:.4f}, dev true min {hi:.4f}); all-data {safe_interval(rows, unique)}")
            for split in ("dev", "test", "all"):
                c = s[split]["counts"]
                parts = []
                for k in ("close", "far"):
                    o = c.get(k, {}); n = sum(o.values())
                    e = s[split]["err"].get(k, [])
                    parts.append(f"{k} {o.get('correct',0)}/{n} ok {o.get('wrong',0)}/{n} wrong {o.get('miss',0)}/{n} miss" + (f" (err {100*sum(e)/len(e):.1f}/{100*max(e):.1f} cm)" if e else ""))
                for k in ("hidden", "wall"):
                    o = c.get(k, {}); n = sum(o.values())
                    if n:
                        parts.append(f"{k} {o.get('fp',0)}/{n} returned")
                print(f"   {split:4s} " + " | ".join(parts))
            if s["all"]["wrongs"]:
                print("   WRONG:", s["all"]["wrongs"][:10])
