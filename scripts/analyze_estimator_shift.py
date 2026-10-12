"""Diagnostic b (docs/REDESIGN.md §25.5): why a friction estimate leaves its calibration under a plan coach.

    python scripts/analyze_estimator_shift.py --protocol data/protocols/catalunya_R1_reduced.json \
        --model outputs/models/plan_coach/P_catalunya_s1 --mu-estimator outputs/estimators/friction_all4_cl2_s0 \
        --circuit catalunya --workers 4 --output outputs/analysis/estimator_shift_catalunya_R1_reduced.json

Replays every protocol episode of the coach on the estimate, checks the offline estimate against the online one,
splits the episodes above ``--overshoot`` into their stages (network measurement, filtered mean, estimate) and
compares their measured windows of the ``--lead-s`` seconds before the first overshoot with the estimator's training
windows of the circuit (nearest-neighbour distance), against the windows of the robust episodes whose estimate stays
within ``--reference-max`` on the same stretch of track. Out of distribution: the median query distance exceeds the
95th percentile of the reference distances.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.evaluation.estimator_shift import (
    measured_windows,
    nearest_distances,
    record_inputs,
    stages,
    training_windows,
)
from contextual_bicycle_pacejka.evaluation.protocol import load_protocol
from contextual_bicycle_pacejka.learning.friction import FrictionEstimator
from contextual_bicycle_pacejka.spec.constants import DT


def _replay(job: dict[str, Any]) -> dict[str, Any]:
    """One protocol episode: online and offline estimates, stages and measured windows at the measuring steps."""
    spec = job["spec"]
    inputs = record_inputs(spec, job["racing_line"], job["model_dir"], job["estimator_dir"])
    estimator = FrictionEstimator(job["estimator_dir"])
    stage = stages(estimator, inputs, spec["mu"])
    steps = stage["measured_steps"]
    record = inputs["record"]
    return {"start_label": spec["start_label"], "mu": spec["mu"], "seed": spec["seed"], "robust": record["robust"],
            "terminal_s_m": record["terminal_s_m"], "online_max": record["oracle_mu"]["max"],
            "offline_max": float(stage["estimate_ratio"].max()), "estimate_ratio": stage["estimate_ratio"],
            "mean_ratio": stage["mean_ratio"], "measured_ratio": stage["measured_ratio"], "measured_steps": steps,
            "s": inputs["s"], "windows": measured_windows(estimator, inputs).astype(np.float32)}


def _first(values: np.ndarray, threshold: float) -> int | None:
    above = np.flatnonzero(values > threshold)
    return int(above[0]) if above.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="plan coach")
    parser.add_argument("--mu-estimator", type=Path, required=True)
    parser.add_argument("--circuit", required=True, help="circuit of the training windows")
    parser.add_argument("--every", type=int, default=10, help="one training window every EVERY steps")
    parser.add_argument("--overshoot", type=float, default=1.06, help="estimate / mu above the calibration")
    parser.add_argument("--lead-s", type=float, default=10.0, help="seconds before the first overshoot")
    parser.add_argument("--reference-max", type=float, default=1.03, help="largest estimate / mu of a reference")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    protocol = load_protocol(args.protocol)
    jobs = [{"spec": spec, "racing_line": protocol["racing_line_path"], "model_dir": str(args.model),
             "estimator_dir": str(args.mu_estimator)} for spec in protocol["episodes"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(_replay, jobs))
    agreement = max(abs(e["online_max"] - e["offline_max"]) for e in episodes)
    logging.info("offline vs online maximum estimate / mu: largest difference %.2e", agreement)

    lead_steps = int(round(args.lead_s / DT))
    over, queries = [], []
    for e in episodes:
        first = _first(e["estimate_ratio"], args.overshoot)
        if first is None:
            continue
        steps = e["measured_steps"]
        measured_first = _first(e["measured_ratio"], args.overshoot)
        mean_first = _first(e["mean_ratio"], args.overshoot)
        chosen = (steps >= first - lead_steps) & (steps <= first)
        queries.append(e["windows"][chosen])
        over.append({"start_label": e["start_label"], "mu": e["mu"], "seed": e["seed"], "robust": e["robust"],
                     "terminal_s_m": e["terminal_s_m"], "max_ratio": e["offline_max"],
                     "first_estimate_step": first, "first_estimate_s_m": float(e["s"][first]),
                     "first_measurement_step": None if measured_first is None else int(steps[measured_first]),
                     "first_measurement_s_m": None if measured_first is None else float(e["s"][steps[measured_first]]),
                     "first_mean_step": mean_first, "first_mean_s_m": None if mean_first is None
                     else float(e["s"][mean_first]),
                     "measured_ratio_max_before": float(e["measured_ratio"][steps <= first].max()),
                     "query_s_m": [float(e["s"][steps[chosen]].min()), float(e["s"][steps[chosen]].max())],
                     "n_queries": int(chosen.sum())})
    out: dict[str, Any] = {"protocol": str(args.protocol), "model": str(args.model),
                           "estimator": str(args.mu_estimator), "offline_online_max_difference": agreement,
                           "overshoot": args.overshoot, "overshoot_episodes": over}
    if over:
        low = min(o["query_s_m"][0] for o in over)
        high = max(o["query_s_m"][1] for o in over)
        reference = [e["windows"][(e["s"][e["measured_steps"]] >= low) & (e["s"][e["measured_steps"]] <= high)]
                     for e in episodes if e["robust"] and e["offline_max"] <= args.reference_max]
        training = training_windows(args.mu_estimator, args.circuit, args.every)
        query_d = nearest_distances(np.concatenate(queries), training)
        reference_d = nearest_distances(np.concatenate(reference), training)
        median_query, p95_reference = float(np.median(query_d)), float(np.percentile(reference_d, 95))
        out.update({"stretch_s_m": [low, high], "n_training_windows": int(len(training)),
                    "n_reference_episodes": len(reference), "n_reference_windows": int(len(reference_d)),
                    "query_distance": {"median": median_query, "p95": float(np.percentile(query_d, 95)),
                                       "max": float(query_d.max())},
                    "reference_distance": {"median": float(np.median(reference_d)), "p95": p95_reference,
                                           "max": float(reference_d.max())},
                    "out_of_distribution": median_query > p95_reference})
        logging.info("overshoot episodes %d; stretch s %.0f-%.0f m; query distance median %.3f vs reference p95 %.3f "
                     "-> out of distribution %s", len(over), low, high, median_query, p95_reference,
                     out["out_of_distribution"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
