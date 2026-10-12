"""Measure 0b (docs/REDESIGN.md §25.5): the constant-scale reference c* of an executor, on one protocol.

    python scripts/analyze_plan_reference.py \
        --teacher outputs/evaluation/phase2/teacher_monza_R0_reduced_U.jsonl \
        --base outputs/evaluation/multi/S2_td3bc_mu_all4_s0_muhat_all4_monza_R0_reduced_U.jsonl \
        --scaled 1.04=outputs/evaluation/plan_coach/S2_td3bc_mu_all4_s0_muhat_plan1.04_monza_R0_reduced_U.jsonl \
        --scaled 1.06=... --output outputs/analysis/plan_reference_muhat_monza_R0_reduced.json

``--base`` is the executor at scale 1 and ``--scaled C=FILE`` the same executor at the constant scale C; the rule of
c* is in the docstring of ``evaluation/analysis.py`` (``plan_reference``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import load_records, plan_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="executor at scale 1")
    parser.add_argument("--scaled", action="append", required=True, metavar="SCALE=FILE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scaled = {float(scale): load_records(path) for scale, path in (item.split("=", 1) for item in args.scaled)}
    result = plan_reference(load_records(args.base), scaled, load_records(args.teacher))
    out = {"teacher": str(args.teacher), "base": str(args.base), "scaled": dict(item.split("=", 1)
                                                                               for item in args.scaled), **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logging.info("c* %.2f; per scale: %s", result["c_star"],
                 {scale: f"{row['robust']}/{row['n']} at {row['teacher_ratio']:.3f}" if row["teacher_ratio"] else
                  f"{row['robust']}/{row['n']}" for scale, row in result["scales"].items()})


if __name__ == "__main__":
    main()
