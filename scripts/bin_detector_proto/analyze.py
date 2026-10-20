"""Apply gates to evaluate.py's raw rows and tabulate correct / wrong / miss."""
import json
import sys
from collections import defaultdict

DEV_RUNS = {"09-23-05", "10-03-03", "10-15-57", "10-37-51", "10-59-22"}
TOL = 0.12


def load(path):
    return [json.loads(l) for l in open(path)]


def decide(r, g):
    if "x" not in r or r.get("score", -9) < g["min_score"]:
        return False
    if r["second"] > g["unique_ratio"] * r["score"]:
        return False
    if max(r["big"], r["small"]) > g["size_ratio"] * r["score"]:
        return False
    return True


def classify(r, g):
    found = decide(r, g)
    if r["variant"] == "hidden" or r["spot"] == "wall":
        return "fp" if found else "reject"
    if not found:
        return "miss"
    return "correct" if r["d"] < TOL else "wrong"


def table(rows, g, split=None):
    out = defaultdict(lambda: defaultdict(int))
    errs = defaultdict(list)
    for r in rows:
        if split == "dev" and r["run"] not in DEV_RUNS:
            continue
        if split == "test" and r["run"] in DEV_RUNS:
            continue
        key = ("hidden" if r["variant"] == "hidden" else r["spot"])
        c = classify(r, g)
        out[key][c] += 1
        if c == "correct":
            errs[key].append(r["d"])
    return out, errs


def fmt(out, errs):
    lines = []
    for key in ("close", "far", "wall", "hidden"):
        o = out.get(key, {})
        n = sum(o.values())
        if not n:
            continue
        if key in ("close", "far"):
            e = sorted(errs[key])
            es = f"err mean {100*sum(e)/len(e):.1f} cm, max {100*e[-1]:.1f} cm" if e else ""
            lines.append(f"  {key:6s} correct {o['correct']}/{n}  wrong {o['wrong']}/{n}  miss {o['miss']}/{n}  {es}")
        else:
            lines.append(f"  {key:6s} returned-a-bin {o['fp']}/{n}  rejected {o['reject']}/{n}")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = load(sys.argv[1])
    g = json.loads(sys.argv[2])
    for split in ("dev", "test", None):
        out, errs = table(rows, g, split)
        print(split or "all")
        print(fmt(out, errs))
