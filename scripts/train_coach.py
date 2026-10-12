"""Train a coach: a privileged policy that learns its speed online, without the teacher's plan (learning/coach.py).

    python scripts/train_offline_rl.py --dataset datasets/recipe_monza_mdp.npz --algorithm td3bc --level C --oracle \
        --output outputs/models/planless/C_td3bc_s0 --seed 0
    python scripts/train_coach.py --dataset datasets/recipe_monza_mdp.npz \
        --racing-line data/tracks/monza/racing_line_R0.npz --init-policy outputs/models/planless/C_td3bc_s0 \
        --output outputs/models/coach/C_monza_s0 --seed 0

The dataset is the prior data of the replay (an offline MDP dataset, scripts/build_dataset.py --mdp) and holds the
labels of the BC term; --init-policy is the policy of level C trained offline that the coach starts from (without it
the coach starts from scratch, which fails: docs/REDESIGN.md, section 24.4). The teacher is the reference of the
validation laps. The coach is evaluated like a student:

    python scripts/evaluate.py --protocol data/protocols/monza_R0_reduced.json \
        --model outputs/models/coach/C_monza_s0 --condition U --workers 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.coach import CoachConfig, train_coach

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="prior data (offline MDP dataset)")
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-policy", type=Path, help="policy of level C trained offline to start from")
    parser.add_argument("--seed", type=int, default=CoachConfig.seed)
    parser.add_argument("--env-steps", type=int, default=CoachConfig.env_steps, help="training budget")
    parser.add_argument("--validate-every", type=int, default=CoachConfig.validate_every)
    parser.add_argument("--threads", type=int, default=CoachConfig.threads)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = CoachConfig(seed=args.seed, env_steps=args.env_steps, validate_every=args.validate_every,
                         threads=args.threads, init_policy=None if args.init_policy is None else str(args.init_policy))
    report = train_coach(args.dataset, args.racing_line, args.output, config)
    last = report["validations"][-1]
    held = last["held"]  # the coach decides at the decision rate
    logging.info("coach: %d/%d validation laps (teacher %d), lap time / teacher %s -> %s", held["completed"],
                 last["laps"], last["teacher_completed"], held["lap_time_ratio"], args.output)


if __name__ == "__main__":
    main()
