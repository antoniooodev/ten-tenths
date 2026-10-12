"""Train a policy by behaviour cloning and run the shielded gate laps.

    python scripts/train_bc.py --dataset datasets/all4.npz --output outputs/models/all4
    python scripts/train_bc.py --dataset datasets/treatment_a_mdp.npz --output outputs/models/S2_label_s0 \
        --level S2 --target label --seed 0
    python scripts/train_bc.py --dataset datasets/treatment_a_mdp.npz --output outputs/models/S1_label_h50_s0 \
        --level S1 --stride 50 --seed 0                       # 8 frames over the last 3.5 s
    python scripts/train_bc.py --dataset datasets/treatment_a_mdp.npz --output outputs/models/S1_label_mu_s0 \
        --level S1 --oracle mu --seed 0                       # diagnostic: true friction appended

The racing line only provides the environment spaces and the gate track (default: Monza R0). A student level
(S1, S2) or the executed-action target needs an offline (MDP) dataset (scripts/build_dataset.py --mdp). The frame
history and the oracle entries apply to S1 and S2.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, gate, train_bc
from contextual_bicycle_pacejka.learning.policy import TARGETS
from contextual_bicycle_pacejka.spec.layout import STUDENT_LEVELS, STUDENT_ORACLES

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--epochs", type=int, default=BCConfig.epochs)
    parser.add_argument("--seed", type=int, default=BCConfig.seed)
    parser.add_argument("--level", choices=STUDENT_LEVELS, default=BCConfig.level, help="observation level")
    parser.add_argument("--target", choices=TARGETS, default=BCConfig.target, help="regression target")
    parser.add_argument("--frames", type=int, default=BCConfig.n_frames, help="student frames")
    parser.add_argument("--stride", type=int, default=BCConfig.stride, help="steps between student frames")
    parser.add_argument("--oracle", nargs="*", choices=STUDENT_ORACLES, default=[],
                        help="diagnostic: privileged entries appended to the student observation")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = BCConfig(seed=args.seed, epochs=args.epochs, level=args.level, target=args.target,
                      n_frames=args.frames, stride=args.stride, oracle=tuple(args.oracle))
    result = train_bc(args.dataset, args.output, args.racing_line, config)
    logging.info("training: %s", json.dumps(result))
    if not args.skip_gate:
        summary = gate(args.output, args.dataset, args.racing_line)
        logging.info("gate: %d of %d laps passed", sum(lap["passed"] for lap in summary["laps"]), len(summary["laps"]))


if __name__ == "__main__":
    main()
