"""Compare the protocol laps of a policy with the minimum-time reference and apply the criterion of the limit.

    python scripts/compare_with_reference.py --evaluation outputs/evaluation/limit/nominal_limit.json

Reads the laps of scripts/evaluate_balance.py (at b = 0), the minimum-time laps from the protocol starts and the
flying minimum-time lap (scripts/min_time_lap.py), and writes <output>/<name>.json:
- per seed and start, the distance from the limit (T - T_opt(s_i)) / T_opt(s_i), and the criterion
  (evaluation/limit.py);
- per corner window, the policy's mean measurements against the reference (entry, minimum and exit speed, braking
  point, lateral offset at the apex, slip peaks and the time near and beyond the peak);
- the completion and the distance from the limit with the stochastic policy.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.evaluation.corners import CORNER_WINDOWS_M
from contextual_bicycle_pacejka.evaluation.limit import lap_gaps, limit_criterion

MEASURES = ("entry_speed", "min_speed", "exit_speed", "braking_s", "apex_ey", "peak_slip_front", "peak_slip_rear",
            "near_peak_front", "near_peak_rear", "beyond_peak_front", "beyond_peak_rear")


def matrices(laps: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    completed = np.array([[lap["completed"] for lap in seed_laps] for seed_laps in laps])
    times = np.array([[lap["lap_time"] if lap["completed"] else np.nan for lap in seed_laps] for seed_laps in laps])
    return completed, times


def corner_table(laps: list[list[dict]], reference: list[dict]) -> list[dict[str, Any]]:
    """Mean corner measurements of the completed laps against the reference lap, window by window."""
    rows = []
    for window, ref in zip(CORNER_WINDOWS_M, reference, strict=True):
        measured = [c for seed_laps in laps for lap in seed_laps if lap["completed"] for c in lap["corners"]
                    if tuple(c["window"]) == window]
        row: dict[str, Any] = {"window": list(window), "laps": len(measured)}
        for name in MEASURES:
            values = [c[name] for c in measured if c.get(name) is not None]
            mean = float(np.mean(values)) if values else None
            row[name] = {"policy": mean, "reference": ref.get(name),
                         "difference": None if mean is None or ref.get(name) is None else mean - ref[name]}
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True, help="output of scripts/evaluate_balance.py")
    parser.add_argument("--standing", type=Path, default=Path("outputs/reference/min_time_standing_b+0.00.json"))
    parser.add_argument("--flying", type=Path, default=Path("outputs/reference/min_time_b+0.00.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/limit"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluation = json.loads(args.evaluation.read_text())
    standing = json.loads(args.standing.read_text())
    flying = json.loads(args.flying.read_text())
    reference = np.array([lap["lap_time"] for lap in standing["laps"]])

    laps = evaluation["deterministic"]["+0.00"]
    completed, times = matrices(laps)
    gaps = lap_gaps(times, reference)
    stochastic = [rep for seed_reps in evaluation["stochastic"]["+0.00"] for rep in seed_reps]
    report: dict[str, Any] = {
        "evaluation": str(args.evaluation), "seeds": evaluation["seeds"], "model": evaluation.get("model"),
        "reference_lap_times": reference.tolist(), "flying_lap_time": flying["lap_time"],
        "gaps": gaps.tolist(), "criterion": limit_criterion(completed, gaps),
        "corners": corner_table(laps, flying["corners"]),
        "inference_ms_per_decision": evaluation["inference_ms_per_decision"],
    }
    if stochastic:
        s_completed, s_times = matrices(stochastic)
        report["stochastic"] = {"completion": float(s_completed.mean()),
                                "median_gap": float(np.nanmedian(lap_gaps(s_times, reference)))
                                if np.any(s_completed) else None}
    name = args.evaluation.stem
    out = args.output / f"{name}_vs_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    criterion = report["criterion"]
    logging.info("%s: laps %s, median gap %s, at the limit %s -> passed %s", name, criterion["completed"],
                 [None if g is None else round(100 * g, 2) for g in criterion["median_gap"]],
                 criterion["seeds_at_limit"], criterion["passed"])
    logging.info("saved %s", out)


if __name__ == "__main__":
    main()
