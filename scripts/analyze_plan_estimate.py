"""Measure 0a (docs/REDESIGN.md §25): the cost of the friction estimate under a plan coach, on one protocol.

    python scripts/analyze_plan_estimate.py \
        --teacher outputs/evaluation/phase2/teacher_monza_R0_reduced_U.jsonl \
        --executor outputs/evaluation/multi/S2_td3bc_mu_all4_s0_muhat_all4_monza_R0_reduced_U.jsonl \
        --coach s0=outputs/evaluation/plan_coach/P_fail_monza_s0_monza_R0_reduced_U.jsonl,\
outputs/evaluation/plan_coach/P_monza_s0_muhat_monza_R0_reduced_U.jsonl \
        --coach s1=... --output outputs/analysis/plan_estimate_monza_R0_reduced.json

``--coach NAME=TRUE_MU,ESTIMATE`` gives the results of one coach with its executor on the true mu and on the
estimate (scripts/evaluate.py with --mu-estimator); ``--executor`` the executor on the estimate at the teacher's
plan. The outcome rules are in the docstring of ``evaluation/analysis.py`` (``estimate_cost``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import estimate_cost, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="executor on the estimate, plan scale 1")
    parser.add_argument("--coach", action="append", required=True, metavar="NAME=TRUE_MU,ESTIMATE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    teacher, executor = load_records(args.teacher), load_records(args.executor)
    out = {"teacher": str(args.teacher), "executor": str(args.executor), "coaches": {}}
    for item in args.coach:
        name, files = item.split("=", 1)
        true_path, estimate_path = (Path(p) for p in files.split(","))
        result = estimate_cost(load_records(estimate_path), load_records(true_path), executor, teacher)
        out["coaches"][name] = {"true_mu": str(true_path), "estimate": str(estimate_path), **result}
        logging.info("%s: outcome %s; robust %d (true mu %d, executor %d); estimate / true mu %s; extra failures %d "
                     "(%d on an overestimate); student robust floor %d", name, result["outcome"],
                     result["on_estimate"]["robust"], result["true_mu"]["robust"], result["executor"]["robust"],
                     "n/a" if result["estimate_over_true_mu"] is None
                     else f"{result['estimate_over_true_mu']['ratio_mean']:.4f}",
                     len(result["extra_failures"]), result["extra_failures_on_overestimate"],
                     result["student_robust_floor"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
