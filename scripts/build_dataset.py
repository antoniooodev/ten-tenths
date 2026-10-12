"""Assemble a behaviour-cloning dataset, or an offline (MDP) dataset, from generated demonstration laps.

    python scripts/build_dataset.py --name all4
    python scripts/build_dataset.py --name treatment_a --mdp --kinds speed lateral none speed_k1.40 --include-failed
    python scripts/build_dataset.py --name treatment_a --mdp --kinds speed lateral speed_wide --include-failed \
        --max-failed 2 --out datasets/phase4_monza_mdp.npz

Reads outputs/demos/<circuit>/<kind>/ and writes datasets/<name>.npz (behaviour cloning) or datasets/<name>_mdp.npz
(offline, see learning/datasets.py). A kind is a folder name of outputs/demos/<circuit>/, such as speed_k1.40.
--max-failed N keeps the first N failed attempts of each lap (a wide perturbation can fail all ten).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from contextual_bicycle_pacejka.learning.datasets import DATASETS, KINDS, assemble, assemble_mdp, kept_laps, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=sorted(DATASETS))
    parser.add_argument("--demos", type=Path, default=REPO / "outputs/demos")
    parser.add_argument("--mdp", action="store_true", help="assemble an offline (MDP) dataset")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), help="demonstration folders, in order (--mdp)")
    parser.add_argument("--include-failed", action="store_true", help="add the saved failed attempts (--mdp)")
    parser.add_argument("--max-failed", type=int, help="failed attempts kept per lap (default: all; --mdp)")
    parser.add_argument("--out", type=Path, help="output file (default: datasets/<name>.npz or <name>_mdp.npz)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    circuits = DATASETS[args.name]
    specs = {c: episode_specs(REPO / "data/tracks" / c / "racing_line_R0.npz") for c in circuits}
    if args.mdp:
        laps = [lap for c in circuits for kind in args.kinds
                for lap in mdp_laps(args.demos / c / kind, args.include_failed, args.max_failed)]
        dataset = assemble_mdp(laps, specs)
        out = args.out or REPO / "datasets" / f"{args.name}_mdp.npz"
        logging.info("%s: %d failures, %d timeouts", args.name, int(dataset["terminals"].sum()),
                     int(dataset["timeouts"].sum()))
    else:
        laps = {c: {k: kept_laps(args.demos / c / k) for k in KINDS} for c in circuits}
        dataset = assemble(laps, specs, circuits)
        out = args.out or REPO / "datasets" / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **dataset)
    logging.info("%s: %d laps, %d transitions -> %s", args.name, len(dataset["episode_ends"]),
                 len(dataset["observations"]), out)


if __name__ == "__main__":
    main()
