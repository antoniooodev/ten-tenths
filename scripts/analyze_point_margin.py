"""Diagnostic a (docs/REDESIGN.md §25.5): margin of an executor at one point, by plan scale and friction error.

    python scripts/analyze_point_margin.py --racing-line data/tracks/catalunya/racing_line_R1.npz \
        --model outputs/models/multi/S2_td3bc_mu_all4_s0 --point-s 3950 --workers 8 \
        --output outputs/analysis/point_margin_catalunya_R1.json

Every scale x error pair drives the same short episodes (frictions x leads x setup seeds); the rules are in the
docstring of ``evaluation/point_margin.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.point_margin import margin_episode, margin_specs, margins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--racing-line", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="student policy reading the mu entry")
    parser.add_argument("--point-s", type=float, required=True, help="arc length of the point [m]")
    parser.add_argument("--leads", type=float, nargs="+", default=[800.0, 600.0, 400.0])
    parser.add_argument("--beyond", type=float, default=150.0, help="distance past the point to pass [m]")
    parser.add_argument("--mus", type=float, nargs="+", default=[0.95, 1.05, 1.15])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.00, 1.04, 1.08, 1.10, 1.12, 1.14])
    parser.add_argument("--errors", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06],
                        help="relative errors p of the mu entry (mu + p mu)")
    parser.add_argument("--first-seed", type=int, default=300_000)
    parser.add_argument("--setups", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seeds = [args.first_seed + i for i in range(args.setups)]
    specs = margin_specs(args.racing_line, args.point_s, args.leads, args.mus, seeds, args.max_steps)
    jobs = [{"spec": spec, "racing_line": str(args.racing_line), "model_dir": str(args.model), "scale": scale,
             "error": error, "beyond_m": args.beyond}
            for scale, error, spec in itertools.product(args.scales, args.errors, specs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(margin_episode, jobs, chunksize=4))
    summary = margins(results)
    out = {"racing_line": str(args.racing_line), "model": str(args.model), "point_s_m": args.point_s,
           "leads_m": args.leads, "beyond_m": args.beyond, "mus": args.mus, "seeds": seeds, **summary,
           "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for error, margin in summary["margin"].items():
        logging.info("error %+.0f %%: margin %s; passed per scale %s", 100 * float(error), margin,
                     {scale: f"{row['passed']}/{row['n']}" for scale, row in summary["table"][error].items()})


if __name__ == "__main__":
    main()
