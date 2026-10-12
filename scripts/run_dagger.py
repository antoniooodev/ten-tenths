"""Weighted DAgger on pre-failure windows (paused line of work; see learning/dagger.py).

    python scripts/run_dagger.py --model models/treatment_a --output outputs/dagger
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.learning.dagger import DAggerConfig, run_dagger

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="starting policy directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dataset", type=Path, default=REPO / "datasets/clean_laps_monza.npz")
    parser.add_argument("--racing-line", type=Path, default=REPO / "data/tracks/monza/racing_line_R0.npz")
    parser.add_argument("--iterations", type=int, default=DAggerConfig.iterations)
    parser.add_argument("--window-m", type=float, default=DAggerConfig.window_m)
    parser.add_argument("--critical-weight", type=float, default=DAggerConfig.critical_weight)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = DAggerConfig(iterations=args.iterations, window_m=args.window_m, critical_weight=args.critical_weight)
    summary = run_dagger(args.base_dataset, args.base_dataset, args.model, args.racing_line, args.output,
                         MATv0.from_config(REPO / "data/teacher/matv0.json"), config)
    for entry in summary:
        logging.info("iteration %d: gate %d/%d", entry["iteration"], entry["gate_passes"], entry["gate_total"])


if __name__ == "__main__":
    main()
