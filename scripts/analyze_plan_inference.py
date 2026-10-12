"""Measure 0c (docs/REDESIGN.md §25.5): how much of a plan coach's scale the student's sensors can infer.

    python scripts/analyze_plan_inference.py --protocol data/protocols/monza_R0_reduced.json \
        --coach outputs/models/plan_coach/P_monza_s1 --mu-estimator outputs/estimators/friction_all4_cl2_s0 \
        --samples datasets/plan_inference_monza_R0_reduced.npz --workers 4 \
        --output outputs/analysis/plan_inference_monza_R0_reduced.json

Collects the decision samples (``evaluation/plan_inference.py``) of ``--episodes`` training episodes and of the
protocol episodes, unless ``--samples`` already exists; fits the regressor on the training episodes and measures it
on the protocol episodes.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.evaluation.plan_inference import (
    RegressionConfig,
    decision_samples,
    fit_regressor,
    inference_gap,
    predict,
    training_specs,
)
from contextual_bicycle_pacejka.evaluation.protocol import load_protocol


def _samples(job: dict[str, Any]) -> dict[str, Any]:
    return decision_samples(job["spec"], job["racing_line"], job["coach"], job["estimator"])


def collect(args: argparse.Namespace, racing_line: str, protocol_specs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Decision samples of the training and protocol episodes (split 0 and 1), one row per decision."""
    train = training_specs(racing_line, args.episodes, args.first_seed, args.max_steps)
    specs = [(0, spec) for spec in train] + [(1, spec) for spec in protocol_specs]
    jobs = [{"spec": spec, "racing_line": racing_line, "coach": str(args.coach),
             "estimator": None if args.mu_estimator is None else str(args.mu_estimator)} for _, spec in specs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(_samples, jobs, chunksize=2))
    rows = [(split, i, e) for i, ((split, _), e) in enumerate(zip(specs, episodes, strict=True)) if len(e["scales"])]
    return {"inputs": np.concatenate([e["inputs"] for _, _, e in rows]),
            "scales": np.concatenate([e["scales"] for _, _, e in rows]),
            "s": np.concatenate([e["s"] for _, _, e in rows]),
            "mu": np.concatenate([np.full(len(e["scales"]), e["mu"]) for _, _, e in rows]),
            "episode": np.concatenate([np.full(len(e["scales"]), i) for _, i, e in rows]),
            "split": np.concatenate([np.full(len(e["scales"]), split) for split, _, e in rows]),
            "robust": np.concatenate([np.full(len(e["scales"]), e["robust"]) for _, _, e in rows])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--coach", type=Path, required=True)
    parser.add_argument("--mu-estimator", type=Path, help="friction estimator of the executor (true mu if absent)")
    parser.add_argument("--episodes", type=int, default=600, help="training episodes")
    parser.add_argument("--first-seed", type=int, default=3_000_000)
    parser.add_argument("--max-steps", type=int, default=3_000, help="steps of a training episode")
    parser.add_argument("--seed", type=int, default=0, help="seed of the regression")
    parser.add_argument("--samples", type=Path, required=True, help="NPZ of the decision samples (reused if present)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    protocol = load_protocol(args.protocol)
    if args.samples.exists():
        with np.load(args.samples) as data:
            samples = {k: data[k] for k in data.files}
    else:
        samples = collect(args, protocol["racing_line_path"], protocol["episodes"])
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.samples, **samples)
    train, test = samples["split"] == 0, samples["split"] == 1
    config = RegressionConfig(seed=args.seed)
    model = fit_regressor(samples["inputs"][train], samples["scales"][train], samples["episode"][train], config)
    gap = inference_gap(samples["scales"][test], predict(model, samples["inputs"][test]), samples["s"][test],
                        samples["mu"][test])
    out = {"protocol": str(args.protocol), "coach": str(args.coach),
           "estimator": None if args.mu_estimator is None else str(args.mu_estimator), "samples": str(args.samples),
           "config": asdict(config), "training_episodes": int(len(np.unique(samples["episode"][train]))),
           "training_decisions": int(train.sum()), "protocol_decisions": int(test.sum()),
           "regression": {k: model[k] for k in ("validation_mse", "n_train", "n_validation", "validation_episodes")},
           "epochs": len(model["curve"]), "protocol_gap": gap}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logging.info("%s: R2 %.3f, |c - c_hat| mean %.4f, p90 %.4f, above %.2f in %.0f %% of decisions; sectors above: %s",
                 args.protocol.stem, gap["r2"] if gap["r2"] is not None else float("nan"), gap["mean_abs"],
                 gap["p90_abs"], gap["threshold"], 100 * gap["share_above"], gap["sectors_above"])


if __name__ == "__main__":
    main()
