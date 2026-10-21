"""Train a PPO policy through the speed stages, each stage starting from the weights of the previous one.

    python scripts/train_curriculum.py --seed 0 --stages 1 --budget 20000000
    python scripts/train_curriculum.py --seed 0 --stages 2 3 --budget 12000000 \
        --init outputs/models/nominal/s0/stage1.zip --lr-start 3e-4 --lr-end 3e-5

Models go to <models>/<tag>/s<seed>/stage<k>.zip and the training curves to <logs>/<tag>/s<seed>_stage<k>.jsonl.
``--b-range`` and ``--oracle`` select the vehicle family and the oracle input (default: the nominal car);
``--b-held-out`` keeps a band of half-width ``HELD_OUT_HALF_WIDTH`` around each of its values out of training.
``--lr-start`` and ``--lr-end`` set the linear learning rate of every stage (default: 1e-3 to 1e-4), ``--gamma`` the
discount per decision (default 0.99). ``--curriculum`` trains the free stage with the automatic speed curriculum
(learning/curriculum.py): the top speed starts at 10 m/s and rises while the training failures are rare; the changes
go to <logs>/<tag>/s<seed>_curriculum.jsonl. ``--checkpoint N ...`` also saves the model when the decisions of a
stage reach N, at <models>/<tag>/s<seed>/stage<k>_<N/1e6>M.zip.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.env import held_out_bands, make_vec_env
from contextual_bicycle_pacejka.learning.curriculum import SpeedCurriculum
from contextual_bicycle_pacejka.learning.monitor import Snapshots, TrainingCurves
from contextual_bicycle_pacejka.learning.ppo import PPOConfig, make_ppo
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import STAGE_TOP_SPEEDS_MPS, TRAINING_EDGE_MARGIN_M

CATALUNYA = "data/tracks/catalunya/reference.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the arguments of the command line (``argv``, or ``sys.argv``)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, nargs="+", required=True, help="decisions per stage (one, or one each)")
    parser.add_argument("--init", type=Path, help="model whose weights start the first stage (default: random)")
    parser.add_argument("--tag", default="nominal")
    parser.add_argument("--b-range", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--b-held-out", type=float, nargs="+", default=[], help="held-out values of b")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--edge-margin", type=float, default=TRAINING_EDGE_MARGIN_M,
                        help="[m] the car counts as off track this far inside the edge (training only)")
    parser.add_argument("--lr-start", type=float, default=PPOConfig.learning_rate_start,
                        help="learning rate at the start of every stage")
    parser.add_argument("--lr-end", type=float, default=PPOConfig.learning_rate_end,
                        help="learning rate at the end of every stage")
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma, help="discount per decision")
    parser.add_argument("--curriculum", action="store_true", help="automatic speed curriculum on the free stage")
    parser.add_argument("--checkpoint", type=int, nargs="+", default=[], help="decisions at which to save a copy")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--logs", type=Path, default=Path("outputs/logs"))
    args = parser.parse_args(argv)
    args.budgets = args.budget * len(args.stages) if len(args.budget) == 1 else args.budget
    if len(args.budgets) != len(args.stages):
        parser.error("--budget takes one value or one per stage")
    if args.curriculum and args.stages != [len(STAGE_TOP_SPEEDS_MPS)]:
        parser.error(f"--curriculum trains the free stage alone: --stages {len(STAGE_TOP_SPEEDS_MPS)}")
    return args


def ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Return the PPO configuration with the learning rate and the discount of the command line."""
    return PPOConfig(learning_rate_start=args.lr_start, learning_rate_end=args.lr_end, gamma=args.gamma)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = ppo_config(args)

    init = args.init
    for stage, budget in zip(args.stages, args.budgets, strict=True):
        env = make_vec_env(CATALUNYA, args.workers * args.envs_per_worker, seed=100_000 * args.seed + 1_000 * stage,
                           stage=stage, b_range=tuple(args.b_range), oracle=args.oracle, edge_margin=args.edge_margin,
                           b_exclude=held_out_bands(tuple(args.b_held_out)), workers=args.workers)
        model = make_ppo(env, config, seed=args.seed, init_from=init)
        curves = args.logs / args.tag / f"s{args.seed}_stage{stage}.jsonl"
        logging.info("seed %d, stage %d: %d decisions from %s, learning rate %g to %g", args.seed, stage, budget,
                     init or "random weights", config.learning_rate_start, config.learning_rate_end)
        stage_dir = args.models / args.tag / f"s{args.seed}"
        callbacks = [TrainingCurves(curves),
                     Snapshots(lambda n, k=stage: stage_dir / f"stage{k}_{n / 1e6:g}M.zip", args.checkpoint)]
        if args.curriculum:
            callbacks.append(SpeedCurriculum(RealTrack(CATALUNYA).track_length,
                                             args.logs / args.tag / f"s{args.seed}_curriculum.jsonl"))
        model.learn(total_timesteps=budget, callback=callbacks)
        path = args.models / args.tag / f"s{args.seed}" / f"stage{stage}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        env.close()
        logging.info("seed %d, stage %d: saved %s", args.seed, stage, path)
        init = path


if __name__ == "__main__":
    main()
