"""Measure how much of the track width the policies use, and how much of their lap time a narrower line explains.

    python scripts/edge_usage.py

Reads the flying laps of scripts/figures/record_laps.py (``--only flying``: the second lap of each run, b = 0; by
default those of the Nominal at the limit and continued, ``--runs`` picks other files) and
the flying minimum-time laps of scripts/min_time_lap.py, without and with an edge margin d (both edges d inside the
track's own). Writes <output>/summary.json:
- per run, the flying lap time, the smallest distance from an edge, and the distance from the edge where the
  minimum-time lap touches it (within ``TOUCH_M``): median, 10th and 90th percentile, and the median split into the
  inside edge (the apex: the edge on the side the centreline turns to) and the outside edge (corner entry and exit);
- the minimum-time lap T(d) for every margin solved, with the minimum corner speeds and apex offsets;
- per run, the cost of the training margin T(0.3) - T(0) as a share of the gap T_policy - T(0), and the share of the
  gap explained by a line as narrow as the policy's, (T(d_run) - T(0)) / (T_policy - T(0)), where d_run is the
  median distance above and T(d) is interpolated linearly over the margins;
- the reading of the median share: at least 0.8 "main cause", 0.3 to 0.8 "partial cause", below 0.3 "not the cause".
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import TRAINING_EDGE_MARGIN_M

CATALUNYA = "data/tracks/catalunya/reference.npz"
TOUCH_M: float = 0.05  # [m] the minimum-time lap touches an edge where it is closer than this
MAIN_CAUSE, PARTIAL_CAUSE = 0.8, 0.3  # shares of the gap explained by the line


def edge_distance(track: RealTrack, s: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Distance [m] from the edge on the side of the car (negative: off track)."""
    left, right = track.boundaries_many(s)
    return np.where(ey >= 0.0, left - ey, right + ey)


def references(directory: Path) -> dict[float, dict[str, Any]]:
    """Flying minimum-time laps at b = 0 by edge margin [m]."""
    laps = {0.0: json.loads((directory / "min_time_b+0.00.json").read_text())}
    for path in sorted(directory.glob("min_time_b+0.00_margin*.json")):
        laps[float(re.search(r"margin([\d.]+)\.json$", path.name).group(1))] = json.loads(path.read_text())
    return dict(sorted(laps.items()))


def run_usage(track: RealTrack, path: Path, ref_s: np.ndarray, touch: np.ndarray, inside: np.ndarray,
              ) -> dict[str, Any]:
    """Edge distances of the flying (second) lap of a recorded run; ``inside`` marks the touches at the apex."""
    run = np.load(path)
    row: dict[str, Any] = {"run": path.stem, "laps": run["lap_times"].tolist()}
    if len(run["lap_times"]) < 2:
        return row
    s, ey = run["s"], run["ey"]
    lap = (s >= track.track_length) & (s <= 2.0 * track.track_length)
    s_lap, distance = s[lap] - track.track_length, edge_distance(track, s[lap], ey[lap])
    on_grid = np.interp(ref_s, s_lap, distance)
    at_touch = on_grid[touch]
    row.update({"flying_lap_time": float(run["lap_times"][1]), "min_edge_distance": float(distance.min()),
                "where_reference_touches": {"median": float(np.median(at_touch)),
                                            "p10": float(np.percentile(at_touch, 10)),
                                            "p90": float(np.percentile(at_touch, 90)),
                                            "inside_median": float(np.median(on_grid[touch & inside])),
                                            "outside_median": float(np.median(on_grid[touch & ~inside]))}})
    return row


def reading(share: float) -> str:
    if share >= MAIN_CAUSE:
        return "main cause"
    return "partial cause" if share >= PARTIAL_CAUSE else "not the cause"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, default=Path("outputs/trajectories"))
    parser.add_argument("--runs", nargs="+", help="file patterns of the flying laps",
                        default=["flying_nominal_limit_s*.npz", "flying_nominal_limit_cont_s*.npz"])
    parser.add_argument("--references", type=Path, default=Path("outputs/reference"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/edge"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    track = RealTrack(CATALUNYA)
    laps = references(args.references)
    ref_s = np.asarray(laps[0.0]["s"])
    ref_ey = np.asarray(laps[0.0]["states"]["ey"])
    touch = edge_distance(track, ref_s, ref_ey) < TOUCH_M
    kappa = track.kappa_many(ref_s)
    inside = (np.sign(ref_ey) == np.sign(kappa)) & (np.abs(kappa) > 1e-3)
    margins = np.array([m for m, lap in laps.items() if lap["status"] == "solved"])
    times = np.array([laps[m]["lap_time"] for m in margins])
    t0 = laps[0.0]["lap_time"]

    runs = [run_usage(track, Path(p), ref_s, touch, inside)
            for pattern in args.runs for p in sorted(glob.glob(str(args.trajectories / pattern)))]
    shares = []
    for row in runs:
        if "flying_lap_time" not in row:
            continue
        gap = row["flying_lap_time"] - t0
        d_run = row["where_reference_touches"]["median"]
        row["gap_s"] = gap
        if TRAINING_EDGE_MARGIN_M in laps and laps[TRAINING_EDGE_MARGIN_M]["status"] == "solved":
            row["training_margin_share"] = (laps[TRAINING_EDGE_MARGIN_M]["lap_time"] - t0) / gap
        if margins.min() <= d_run <= margins.max():
            row["line_share"] = (float(np.interp(d_run, margins, times)) - t0) / gap
            shares.append(row["line_share"])
        logging.info("%-28s lap %.2f s (gap %.2f s), min edge %.2f m, edge where the optimum touches %.2f m "
                     "(inside %.2f, outside %.2f), line share %s", row["run"], row["flying_lap_time"], gap,
                     row["min_edge_distance"], d_run, row["where_reference_touches"]["inside_median"],
                     row["where_reference_touches"]["outside_median"],
                     f"{row['line_share']:.2f}" if "line_share" in row else "-")

    summary = {
        "touch_m": TOUCH_M, "touch_share_of_lap": float(touch.mean()),
        "touch_share_inside": float((touch & inside).mean()), "touch_share_outside": float((touch & ~inside).mean()),
        "references": [{"edge_margin": m, "status": lap["status"], "lap_time": lap["lap_time"],
                        "solver_seconds": lap.get("solver_seconds"),
                        "min_speed": [c.get("min_speed") for c in lap["corners"]],
                        "apex_ey": [c.get("apex_ey") for c in lap["corners"]]} for m, lap in laps.items()],
        "runs": runs,
        "median_line_share": float(np.median(shares)) if shares else None,
        "reading": reading(float(np.median(shares))) if shares else None,
    }
    for ref in summary["references"]:
        logging.info("margin %.1f m: %s, %.2f s (+%.2f s)", ref["edge_margin"], ref["status"], ref["lap_time"],
                     ref["lap_time"] - t0)
    logging.info("median share of the gap explained by the line: %s (%s)", summary["median_line_share"],
                 summary["reading"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
