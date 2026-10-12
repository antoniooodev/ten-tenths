"""Train a sensor plan student: the plan scale chosen from onboard sensors only (learning/plan_student.py).

    python scripts/train_plan_student.py --executor outputs/models/multi/S2_td3bc_mu_all4_s0 \
        --mu-estimator outputs/estimators/friction_all4_cl2_s0 --guide outputs/models/plan_coach/P_monza_s1 \
        --racing-line data/tracks/monza/racing_line_R0.npz --output outputs/models/plan_student/PS_monza_s0 --seed 0

The executor is a student policy of level S1/S2 with the mu entry, reading the estimate of ``--mu-estimator``; the
guide is a privileged plan coach that chooses the scale in a share of the training episodes (exploration only).
The student is evaluated like a plan coach, with its own estimator:

    python scripts/evaluate.py --protocol data/protocols/monza_R0_reduced.json \
        --model outputs/models/plan_student/PS_monza_s0 --condition U --workers 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.plan_student import PlanStudentConfig, train_plan_student

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executor", type=Path, required=True, help="student policy that drives (S1/S2 with mu)")
    parser.add_argument("--mu-estimator", type=Path, required=True, help="friction estimator the executor reads")
    parser.add_argument("--guide", type=Path, help="privileged plan coach of the guided episodes")
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PlanStudentConfig.seed)
    parser.add_argument("--env-steps", type=int, default=PlanStudentConfig.env_steps, help="training budget")
    parser.add_argument("--guide-share", type=float, default=PlanStudentConfig.guide_share)
    parser.add_argument("--validate-every", type=int, default=PlanStudentConfig.validate_every)
    parser.add_argument("--threads", type=int, default=PlanStudentConfig.threads)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = PlanStudentConfig(seed=args.seed, env_steps=args.env_steps, guide_share=args.guide_share,
                               validate_every=args.validate_every, threads=args.threads)
    report = train_plan_student(args.executor, args.mu_estimator, args.guide, args.racing_line, args.output, config)
    last = report["validations"][-1]
    logging.info("plan student: %d/%d validation laps (teacher %d), lap time / teacher %s, mean scale %.4f -> %s",
                 last["completed"], len(last["laps"]), last["teacher_completed"], last["lap_time_ratio"],
                 last["plan_scale"], args.output)


if __name__ == "__main__":
    main()
