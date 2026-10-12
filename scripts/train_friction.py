"""Train, calibrate and evaluate a friction estimator on an offline (MDP) dataset.

    python scripts/build_dataset.py --name treatment_a --mdp --kinds speed lateral none none_k1.00 none_k1.10 \
        --out datasets/friction_monza_mdp.npz
    python scripts/train_friction.py --dataset datasets/friction_monza_mdp.npz --output outputs/estimators/friction_s0
    python scripts/collect_closed_loop.py --policy outputs/models/phase2/S2_label_mu_s0 \
        --estimator outputs/estimators/friction_s0 --output datasets/friction_closed_loop_monza.npz
    python scripts/train_friction.py --dataset datasets/friction_monza_mdp.npz datasets/friction_closed_loop_monza.npz \
        --output outputs/estimators/friction_cl_s0
    python scripts/train_friction.py --dataset datasets/friction_monza_mdp.npz datasets/friction_closed_loop_monza.npz \
        --recalibrate outputs/estimators/friction_cl_s0 --bias-side over --output outputs/estimators/friction_cl_over_s0

The estimator (learning/friction.py) reads the tyre channels only. Several datasets are concatenated in order.
Held-out setups (one per friction value) give the calibration of the filter and of the lower bound and the report in
estimator.json, overall, per kind of lap and per circuit (datasets of several circuits: the same setups are held out on
every circuit, and the worst circuit sets the margin). A student with the mu oracle entry drives on its estimate with
scripts/evaluate.py --mu-estimator. --recalibrate DIR copies the trained estimator in DIR and calibrates it again
(--bias-side over: b from the overestimates only) on the held-out laps of --dataset, which should be the datasets it
was trained on; --patience at least --epochs disables early stopping (the best validation state is kept).
--overestimate-limit and --margin-floor set the bound on mu_hat / mu - 1 at the held-out steps and the smallest margin
k of the lower bound, in training and in recalibration. A recalibration keeps the settings of DIR that are not given
(--bias-side, --overestimate-limit, --margin-floor):

    python scripts/train_friction.py --dataset ... --recalibrate outputs/estimators/friction_cl2_long_over_s0 \
        --overestimate-limit 0.06 --margin-floor 0 --output outputs/estimators/friction_cl2_long_over_l06_s0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.friction import FrictionConfig, recalibrate, train_friction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FrictionConfig.seed)
    parser.add_argument("--frames", type=int, default=FrictionConfig.n_frames, help="frames of the input window")
    parser.add_argument("--stride", type=int, default=FrictionConfig.stride, help="steps between frames")
    parser.add_argument("--epochs", type=int, default=FrictionConfig.epochs)
    parser.add_argument("--patience", type=int, default=FrictionConfig.patience,
                        help="epochs without improvement before stopping")
    parser.add_argument("--bias-side", choices=("both", "over"),
                        help=f"end-of-lap errors entering the bias spread b (default {FrictionConfig.bias_side})")
    parser.add_argument("--overestimate-limit", type=float,
                        help="largest mu_hat / mu - 1 at the held-out steps "
                             f"(default {FrictionConfig.overestimate_limit})")
    parser.add_argument("--margin-floor", type=float,
                        help=f"smallest margin k of the lower bound (default {FrictionConfig.margin_floor})")
    parser.add_argument("--recalibrate", type=Path, metavar="DIR", help="recalibrate this estimator, no training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = {name: value for name, value in (("bias_side", args.bias_side),
                                                ("overestimate_limit", args.overestimate_limit),
                                                ("margin_floor", args.margin_floor)) if value is not None}
    if args.recalibrate is not None:
        spec = recalibrate(args.recalibrate, list(args.dataset), args.output, **settings)
    else:
        config = FrictionConfig(seed=args.seed, n_frames=args.frames, stride=args.stride, epochs=args.epochs,
                                patience=args.patience, **settings)
        spec = train_friction(list(args.dataset), args.output, config)
    report = {k: v for k, v in spec["report"]["validation"].items() if k != "per_lap_max_ratio"}
    logging.info("calibration: %s", json.dumps(spec["calibration"]))
    logging.info("validation: %s", json.dumps(report))


if __name__ == "__main__":
    main()
