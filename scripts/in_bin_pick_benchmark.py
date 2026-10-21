"""Benchmark the pick out of the bin in sim against every real in-bin cube rest.

    python scripts/in_bin_pick_benchmark.py --out-dir <dir>
    python scripts/in_bin_pick_benchmark.py --out-dir <dir> \
        --controller current \
        --controller "margin5=pick.top_down.in_bin.wall_margin=0.005"

Each ``--controller`` is ``name`` or ``name=override[,override...]`` (Hydra overrides
on conf/skill_server.yaml; ``current`` is the config as it stands). Writes
``<name>.json`` (every case's sim result and the summary) and ``<name>.md`` (the
table) per controller, and prints the summary. See
:mod:`prpl_tidybot.in_bin_pick_benchmark`.
"""

import argparse
import json
import logging
from pathlib import Path

from prpl_tidybot.in_bin_pick_benchmark import (
    CURRENT,
    DEFAULT_FIXTURE,
    Controller,
    format_table,
    load_cases,
    run_benchmark,
    summarize,
)


def parse_controller(text: str) -> Controller:
    """``name`` or ``name=override[,override...]``."""
    if text == CURRENT.name:
        return CURRENT
    name, _, rest = text.partition("=")
    return Controller(name=name, overrides=tuple(o for o in rest.split(",") if o))


def main() -> None:
    """Run every controller on every case and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--controller", action="append", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cases = load_cases(args.fixture)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for text in args.controller or [CURRENT.name]:
        controller = parse_controller(text)
        results = run_benchmark(cases, controller)
        summary = summarize(results, cases)
        out = {
            "controller": {
                "name": controller.name,
                "overrides": list(controller.overrides),
            },
            "fixture": str(args.fixture),
            "summary": summary,
            "results": results,
        }
        (args.out_dir / f"{controller.name}.json").write_text(
            json.dumps(out, indent=1) + "\n", encoding="utf-8"
        )
        table = format_table(results, cases)
        (args.out_dir / f"{controller.name}.md").write_text(
            table + "\n", encoding="utf-8"
        )
        print(f"## {controller.name} {list(controller.overrides)}")
        print(table)
        for key, value in summary.items():
            if key != "disagreements":
                print(f"{key}: {value[0]}/{value[1]}")
        for d in summary["disagreements"]:
            print(f"disagreement {d['id']}: real {d['real']}, sim {d['sim']}")


if __name__ == "__main__":
    main()
