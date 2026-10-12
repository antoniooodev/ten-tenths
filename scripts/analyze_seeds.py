"""Summarise several training seeds per configuration on one protocol, against the teacher.

    python scripts/analyze_seeds.py --teacher outputs/evaluation/teacher_monza_R0_reduced_U.jsonl \
        --group S1_label outputs/evaluation/S1_label_s0_monza_R0_reduced_U.jsonl ... \
        --group S2_label ... --output outputs/analysis/phase2_monza_R0_reduced.json

Each --group takes a configuration name followed by the result files of its seeds (one file per seed, same
episodes). Writes the summaries (evaluation/analysis.py: seed_summary) and prints one row per configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import load_records, seed_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar="NAME FILE",
                        help="configuration name followed by its result files")
    parser.add_argument("--teacher", type=Path, help="teacher result file on the same protocol")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    teacher = load_records(args.teacher) if args.teacher else None
    summaries = {}
    for name, *files in args.group:
        if not files:
            raise SystemExit(f"group {name} has no result files")
        summaries[name] = seed_summary({Path(f).stem: load_records(f) for f in files}, teacher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=1))
    logging.info("%-14s %5s %8s %6s %13s %7s %7s %10s", "configuration", "seeds", "robust", "sd", "pooled",
                 "every", "none", "lap ratio")
    for name, s in summaries.items():
        pooled = s["robust_pooled"]
        ratio = f"{s['lap_time_ratio_mean']:.4f}" if s["lap_time_ratio_mean"] is not None else "-"
        logging.info("%-14s %5d %8.3f %6.3f %6d/%-6d %7d %7d %10s", name, s["n_seeds"], s["robust_rate_mean"],
                     s["robust_rate_sd"], pooled["k"], pooled["n"], s["robust_in_every_seed"],
                     s["robust_in_no_seed"], ratio)
    if teacher:
        t = summaries[next(iter(summaries))]["teacher"]
        logging.info("teacher: robust %d/%d", t["robust"]["k"], t["n"])
    logging.info("-> %s", args.output)


if __name__ == "__main__":
    main()
