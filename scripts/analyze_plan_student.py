"""Criterion of a sensor plan student over several seeds (docs/REDESIGN.md §26.3), against the coach and c* on the
estimate.

    python scripts/analyze_plan_student.py --teacher outputs/evaluation/phase2/teacher_monza_R0_reduced_U.jsonl \
        --coach outputs/evaluation/plan_coach/P_monza_s1_muhat_monza_R0_reduced_U.jsonl \
        --c-star outputs/evaluation/plan_coach/S2_td3bc_mu_all4_s0_muhat_plan1.08_monza_R0_reduced_U.jsonl \
        --student s0=outputs/evaluation/plan_student/PS_monza_s0_monza_R0_reduced_U.jsonl --student s1=... \
        --output outputs/analysis/plan_student_monza_R0_reduced.json

The rule (non-inferior robustness, faster than c*, two-level bootstrap intervals over seeds and episodes) is in the
docstring of ``evaluation/analysis.py`` (``plan_student_criterion``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import (
    BOOTSTRAP_RESAMPLES,
    STUDENT_ROBUST_MARGIN,
    load_records,
    plan_student_criterion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--coach", type=Path, required=True, help="the plan coach on the estimate")
    parser.add_argument("--c-star", type=Path, required=True, help="the executor at the scale c* on the estimate")
    parser.add_argument("--student", action="append", required=True, metavar="NAME=FILE", help="one per seed")
    parser.add_argument("--margin", type=float, default=STUDENT_ROBUST_MARGIN, help="non-inferiority margin")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = dict(item.split("=", 1) for item in args.student)
    result = plan_student_criterion({name: load_records(path) for name, path in files.items()},
                                    load_records(args.coach), load_records(args.c_star), load_records(args.teacher),
                                    args.margin, args.resamples, args.bootstrap_seed)
    out = {"teacher": str(args.teacher), "coach": str(args.coach), "c_star": str(args.c_star), "students": files,
           **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    d, star, coach = result["robust_difference"], result["versus_c_star"], result["versus_coach"]
    logging.info("passed %s %s; robust %d/%d; failure difference %.3f %s; student / c* %.4f %s; student / coach "
                 "%.4f %s", result["passed"], result["checks"], result["pooled"]["robust"], result["pooled"]["n"],
                 d["point"], [round(x, 3) for x in d["ci95"]], star["point"], [round(x, 4) for x in star["ci95"]],
                 coach["point"], [round(x, 4) for x in coach["ci95"]])


if __name__ == "__main__":
    main()
