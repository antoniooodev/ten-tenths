"""Train a student policy by offline reinforcement learning (IQL or TD3+BC) with an asymmetric critic.

    python scripts/train_offline_rl.py --dataset datasets/treatment_a_mdp.npz --algorithm iql \
        --output outputs/models/phase3/S2_iql_mu_s0 --seed 0
    python scripts/train_offline_rl.py --dataset datasets/treatment_a_mdp.npz --algorithm td3bc \
        --output outputs/models/phase3/S2_td3bc_mu_s0 --seed 0

The critic reads the privileged observation, the actor the student observation of --level (default S2 with the mu
oracle entry). IQL learns from the executed actions only; TD3+BC adds a behaviour-cloning term on the teacher
labels (learning/offline_rl.py). The policy is evaluated like a behaviour-cloned student (scripts/evaluate.py).
--no-failures (an ablation) bootstraps the failed laps like truncated ones, without the crash penalty.
--cost-limit RHO (TD3+BC) adds a critic of the traction cost and a Lagrange multiplier that keeps the policy's
cost within the fraction RHO of the teacher labels' cost on the same states:

    python scripts/train_offline_rl.py --dataset datasets/recipe_monza_mdp.npz --algorithm td3bc --cost-limit 0.5 \
        --output outputs/models/traction/S2_td3bc_mu_c050_s0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.offline_rl import ALGORITHMS, OfflineRLConfig, train_offline_rl
from contextual_bicycle_pacejka.spec.layout import STUDENT_LEVELS, STUDENT_ORACLES

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--level", choices=STUDENT_LEVELS[1:], default=OfflineRLConfig.level)
    parser.add_argument("--oracle", nargs="*", choices=STUDENT_ORACLES, default=list(OfflineRLConfig.oracle))
    parser.add_argument("--seed", type=int, default=OfflineRLConfig.seed)
    parser.add_argument("--steps", type=int, default=OfflineRLConfig.steps)
    parser.add_argument("--gamma", type=float, default=OfflineRLConfig.gamma)
    parser.add_argument("--beta", type=float, default=OfflineRLConfig.beta, help="IQL temperature")
    parser.add_argument("--alpha", type=float, default=OfflineRLConfig.alpha, help="TD3+BC Q weight")
    parser.add_argument("--threads", type=int, default=OfflineRLConfig.threads)
    parser.add_argument("--no-failures", action="store_true", help="failures carry no signal (ablation)")
    parser.add_argument("--cost-limit", type=float, metavar="RHO",
                        help="TD3+BC: traction-cost budget, a fraction of the labels' cost (default: no constraint)")
    parser.add_argument("--cost-lr", type=float, default=OfflineRLConfig.cost_lr, help="step of the multiplier")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = OfflineRLConfig(algorithm=args.algorithm, seed=args.seed, level=args.level, oracle=tuple(args.oracle),
                             steps=args.steps, gamma=args.gamma, beta=args.beta, alpha=args.alpha,
                             threads=args.threads, failures=not args.no_failures, cost_limit=args.cost_limit,
                             cost_lr=args.cost_lr)
    report = train_offline_rl(args.dataset, args.output, args.racing_line, config)
    logging.info("last: %s", json.dumps(report["curve"][-1] if report["curve"] else {}))
    logging.info("critic: %s", json.dumps(report["diagnostics"]))
    if "cost_diagnostics" in report:
        logging.info("cost: %s", json.dumps(report["cost_diagnostics"]))


if __name__ == "__main__":
    main()
