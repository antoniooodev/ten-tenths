"""Drive closed-loop laps of a student with the mu oracle entry and log them for the friction estimator.

    python scripts/collect_closed_loop.py --policy outputs/models/phase2/S2_label_mu_s0 \
        --estimator outputs/estimators/friction_s0 --output datasets/friction_closed_loop_monza.npz \
        --variants muhat mu-0.10 mu+0.10 mu+0.20 --workers 8

One lap per demonstration setup (20) and variant, from a start drawn per lap (learning/closed_loop.py), on the racing
line given (Monza R0 by default; the laps record its circuit, the folder of the racing line):

    python scripts/collect_closed_loop.py --policy outputs/models/multi/S2_td3bc_mu_all4_s0 \
        --estimator outputs/estimators/friction_all4_s0 --racing-line data/tracks/spa/racing_line_R1.npz \
        --output datasets/friction_all4_closed_loop_spa.npz --workers 8

Variants: muhat (the policy reads the estimate of --estimator) and mu<+/-b> (the true mu plus b). The labels are the
episode frictions. Train an estimator on the demonstrations plus these laps with scripts/train_friction.py. A later
DAgger iteration drives with the estimator of the previous one, from new starts (--seed) and with its own --label:

    python scripts/collect_closed_loop.py --policy outputs/models/phase2/S2_label_mu_s0 \
        --estimator outputs/estimators/friction_cl_s0 --output datasets/friction_closed_loop2_monza.npz \
        --seed 1 --label closed_loop2 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.closed_loop import DEFAULT_LABEL, MAX_STEPS, collect
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True, help="student policy with the mu oracle entry")
    parser.add_argument("--estimator", type=Path, help="friction estimator of the muhat variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--variants", nargs="+", default=["muhat", "mu-0.10", "mu+0.10", "mu+0.20"])
    parser.add_argument("--seed", type=int, default=0, help="seed of the lap starts")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="prefix of the lap kinds (one per DAgger iteration)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = collect(args.racing_line, args.policy, episode_specs(args.racing_line), args.variants, args.output,
                      estimator_dir=args.estimator, seed=args.seed, workers=args.workers, max_steps=args.max_steps,
                      label=args.label)
    logging.info("closed-loop laps -> %s: %s", args.output, json.dumps(summary))


if __name__ == "__main__":
    main()
