"""Train a plan coach: a privileged policy that rewrites the speed plan a student drives (learning/plan_coach.py).

    python scripts/train_plan_coach.py --executor outputs/models/multi/S2_td3bc_mu_all4_s0 \
        --racing-line data/tracks/monza/racing_line_R0.npz --output outputs/models/plan_coach/P_monza_s0 --seed 0

The executor is a student policy of level S1/S2 that reads the v_ref preview; every second the coach chooses the
scale of that preview, and the executor drives. The teacher is the reference of the validation laps. The coach is
evaluated with its executor like a policy, and a constant scale of the teacher's plan is its baseline:

    python scripts/evaluate.py --protocol data/protocols/monza_R0_reduced.json \
        --model outputs/models/plan_coach/P_monza_s0 --condition U --workers 8
    python scripts/evaluate.py --protocol data/protocols/monza_R0_reduced.json \
        --model outputs/models/multi/S2_td3bc_mu_all4_s0 --plan-scale 1.08 --condition U --workers 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.plan_coach import PlanCoachConfig, train_plan_coach

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executor", type=Path, required=True, help="student policy that drives (level S1/S2)")
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PlanCoachConfig.seed)
    parser.add_argument("--env-steps", type=int, default=PlanCoachConfig.env_steps, help="training budget")
    parser.add_argument("--validate-every", type=int, default=PlanCoachConfig.validate_every)
    parser.add_argument("--threads", type=int, default=PlanCoachConfig.threads)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = PlanCoachConfig(seed=args.seed, env_steps=args.env_steps, validate_every=args.validate_every,
                             threads=args.threads)
    report = train_plan_coach(args.executor, args.racing_line, args.output, config)
    last = report["validations"][-1]
    logging.info("plan coach: %d/%d validation laps (teacher %d), lap time / teacher %s, mean scale %.4f -> %s",
                 last["completed"], len(last["laps"]), last["teacher_completed"], last["lap_time_ratio"],
                 last["plan_scale"], args.output)


if __name__ == "__main__":
    main()
