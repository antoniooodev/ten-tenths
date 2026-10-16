"""Train a PPO policy through the speed stages, each stage starting from the weights of the previous one.

    python scripts/train_curriculum.py --seed 0 --stages 1 --budget 20000000
    python scripts/train_curriculum.py --seed 0 --stages 2 3 --budget 12000000 \
        --init outputs/models/nominal/s0/stage1.zip

Models go to <models>/<tag>/s<seed>/stage<k>.zip and the training curves to <logs>/<tag>/s<seed>_stage<k>.jsonl.
``--b-range`` and ``--oracle`` select the vehicle family and the oracle input (default: the nominal car).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.env import make_vec_env
from contextual_bicycle_pacejka.learning.monitor import TrainingCurves
from contextual_bicycle_pacejka.learning.ppo import PPOConfig, make_ppo
from contextual_bicycle_pacejka.spec.constants import TRAINING_EDGE_MARGIN_M

CATALUNYA = "data/tracks/catalunya/reference.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, nargs="+", required=True, help="decisions per stage (one, or one each)")
    parser.add_argument("--init", type=Path, help="model whose weights start the first stage (default: random)")
    parser.add_argument("--tag", default="nominal")
    parser.add_argument("--b-range", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--edge-margin", type=float, default=TRAINING_EDGE_MARGIN_M,
                        help="[m] the car counts as off track this far inside the edge (training only)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--logs", type=Path, default=Path("outputs/logs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    budgets = args.budget * len(args.stages) if len(args.budget) == 1 else args.budget
    if len(budgets) != len(args.stages):
        parser.error("--budget takes one value or one per stage")

    init = args.init
    for stage, budget in zip(args.stages, budgets, strict=True):
        env = make_vec_env(CATALUNYA, args.workers * args.envs_per_worker, seed=100_000 * args.seed + 1_000 * stage,
                           stage=stage, b_range=tuple(args.b_range), oracle=args.oracle,
                           edge_margin=args.edge_margin, workers=args.workers)
        model = make_ppo(env, PPOConfig(), seed=args.seed, init_from=init)
        curves = args.logs / args.tag / f"s{args.seed}_stage{stage}.jsonl"
        logging.info("seed %d, stage %d: %d decisions from %s", args.seed, stage, budget, init or "random weights")
        model.learn(total_timesteps=budget, callback=TrainingCurves(curves))
        path = args.models / args.tag / f"s{args.seed}" / f"stage{stage}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        env.close()
        logging.info("seed %d, stage %d: saved %s", args.seed, stage, path)
        init = path


if __name__ == "__main__":
    main()
