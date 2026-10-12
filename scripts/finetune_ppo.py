"""PPO fine-tuning of a behaviour-cloning policy with a KL anchor (paused line of work; see learning/ppo_finetune.py).

    python scripts/finetune_ppo.py --model models/all4 --output outputs/ppo_all4_monza
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.learning.ppo_finetune import PPOConfig, finetune

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="behaviour-cloning policy directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R1.npz")
    parser.add_argument("--timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    final = finetune(args.model, args.racing_line, args.output,
                     PPOConfig(total_timesteps=args.timesteps, seed=args.seed))
    logging.info("final policy -> %s", final)


if __name__ == "__main__":
    main()
