"""Paired before/after analysis of replayed picks: success/total per state class, exact
McNemar tests between arms on the shared states, failures by step, and figures.

usage: analyze.py --tossing <jsonl...> --shelf <jsonl...> --out-dir <dir> [--arms a,b,...]

An "arm" is (estimator, generator) for the new pipeline, or "old". Rows are paired by
state id; a test between two arms uses only the states both ran.
"""

from __future__ import annotations

import argparse
import collections
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOSSING_CLASSES = {
    "in bin near wall": "in bin",
    "in bin, clear of walls": "in bin",
    "beside bin (outside)": "floor",
    "open floor": "floor",
    "near room wall": "floor",
    "beside barrier": "floor",
    "tipped bin": "tipped bin",
}
COLOURS = {"old": "#999999"}
PALETTE = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def arm_name(rec: dict) -> str:
    return "old" if rec["arm"] == "old" else f"{rec['estimator']}+{rec['generator']}"


def load(paths: list[str], domain: str) -> list[dict]:
    out = []
    for path in paths:
        for line in open(path):
            rec = json.loads(line)
            rec["domain"] = domain
            if domain == "tossing":
                rec["key"] = rec["id"]
                rec["cls"] = rec["d4"]
                rec["group"] = TOSSING_CLASSES.get(rec["d4"], "other")
            else:
                rec["key"] = rec["id"]
                rec["cls"] = "box" if rec["skill"] == "ReturnCubeToShelfFromBox" else "floor"
                rec["group"] = rec["cls"]
            rec["armname"] = arm_name(rec)
            out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tossing", nargs="*", default=[])
    parser.add_argument("--shelf", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="cube-pick")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = load(args.tossing, "tossing") + load(args.shelf, "shelf")
    by = collections.defaultdict(dict)  # (domain, key) -> armname -> rec
    for r in recs:
        by[(r["domain"], r["key"])][r["armname"]] = r
    arms = sorted({r["armname"] for r in recs}, key=lambda a: (a != "old", a))
    report: dict = {"arms": arms, "tables": {}, "tests": [], "stages": {}}
    lines = []
    for domain in ("tossing", "shelf"):
        keys = [k for k in by if k[0] == domain]
        if not keys:
            continue
        # Per class success/total per arm.
        table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for key in keys:
            for arm, rec in by[key].items():
                for cls in (rec["cls"], f"[{rec['group']}]", "[all]"):
                    cell = table[cls][arm]
                    cell[0] += int(bool(rec["success"]))
                    cell[1] += 1
        report["tables"][domain] = {c: {a: v for a, v in t.items()} for c, t in table.items()}
        lines.append(f"\n## {domain}\n")
        lines.append("| class | " + " | ".join(arms) + " |")
        lines.append("|---|" + "---|" * len(arms))
        for cls in sorted(table, key=lambda c: (c.startswith("["), c)):
            cells = []
            for arm in arms:
                s, n = table[cls].get(arm, [0, 0])
                cells.append(f"{s}/{n}" if n else "-")
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")
        # Paired tests.
        for group in ("[all]", *sorted({by[k][a]["group"] for k in keys for a in by[k]})):
            for i, a in enumerate(arms):
                for b_arm in arms[i + 1 :]:
                    both = [
                        k
                        for k in keys
                        if a in by[k] and b_arm in by[k] and (group == "[all]" or by[k][a]["group"] == group)
                    ]
                    if not both:
                        continue
                    sa = sum(by[k][a]["success"] for k in both)
                    sb = sum(by[k][b_arm]["success"] for k in both)
                    b = sum(by[k][a]["success"] and not by[k][b_arm]["success"] for k in both)
                    c = sum(by[k][b_arm]["success"] and not by[k][a]["success"] for k in both)
                    p = mcnemar_exact(b, c)
                    report["tests"].append(
                        {"domain": domain, "group": group, "a": a, "b": b_arm, "n": len(both), "a_success": sa, "b_success": sb, "a_only": b, "b_only": c, "p": p}
                    )
                    lines.append(
                        f"- {domain} {group}: {a} {sa}/{len(both)} vs {b_arm} {sb}/{len(both)}; discordant {b} vs {c}; exact McNemar p = {p:.3g}"
                    )
        # Failures by step.
        for arm in arms:
            stages = collections.Counter(
                by[k][arm]["stage"] for k in keys if arm in by[k] and not by[k][arm]["success"]
            )
            report["stages"][f"{domain}/{arm}"] = dict(stages)
            lines.append(f"- {domain} {arm} failures by step: {dict(stages)}")
    (out_dir / f"{args.prefix}-summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / f"{args.prefix}-summary.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))
    for domain, title in (("tossing", "Tossing3D: replayed pick-controller refusals"), ("shelf", "SweepOffShelf3D: replayed cube picks")):
        table = report["tables"].get(domain)
        if not table:
            continue
        classes = [c for c in sorted(table, key=lambda c: (c.startswith("["), c)) if not c.startswith("[") or c == "[all]"]
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(classes) * max(1, len(arms)) / 2), 4.2))
        width = 0.8 / len(arms)
        for j, arm in enumerate(arms):
            colour = COLOURS.get(arm, PALETTE[(j - 1) % len(PALETTE)])
            xs, ys, labels = [], [], []
            for i, cls in enumerate(classes):
                s, n = table[cls].get(arm, [0, 0])
                xs.append(i + (j - (len(arms) - 1) / 2) * width)
                ys.append(s / n if n else 0)
                labels.append(f"{s}/{n}" if n else "")
            ax.bar(xs, ys, width=width * 0.95, color=colour, label=arm)
            for x, y, lab in zip(xs, ys, labels):
                ax.text(x, y + 0.01, lab, ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.strip("[]") for c in classes], fontsize=8)
        ax.set_ylabel("picks holding the cube (fraction of states)")
        ax.set_ylim(0, 1.25)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.prefix}-{domain}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
