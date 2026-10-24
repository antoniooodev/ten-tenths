"""Compute the minimum-time lap reference of the plant car on Catalunya (reference/min_time.py; needs casadi).

    python scripts/min_time_lap.py
    python scripts/min_time_lap.py --b -0.2 0 0.17 --ds 2
    python scripts/min_time_lap.py --standing-starts
    python scripts/min_time_lap.py --edge-margin 0.3 1 2

With ``--standing-starts`` it also solves the lap from each start of the protocol laps (evaluation/laps.py: on the
centreline at 3 m/s), the reference for protocol lap times, into <output>/min_time_standing_b<b>.json.

With ``--edge-margin`` the flying lap is solved with both edges moved inwards by each margin d, into
<output>/min_time_b<b>_margin<d>.json (d > 0): how much a narrower line costs the optimal lap.

Writes <output>/min_time_b<b>.json: the lap time, the solver time, the solver status, the profile along the lap (s, states, controls,
slip ratios) and, for each grip-limited corner of evaluation/corners.py, the measurements the policies are compared
with (entry, minimum and exit speed, braking point, peak steering and slip ratios). Used only to measure policies.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.evaluation.corners import CORNER_WINDOWS_M, measure_window
from contextual_bicycle_pacejka.evaluation.laps import START_SPEED_MPS, protocol_starts
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.reference.min_time import STATE_NAMES, MinTimeLap, slip_ratios, solve_min_time_lap

CATALUNYA = "data/tracks/catalunya/reference.npz"


def corner_measurements(lap: MinTimeLap, track_length: float) -> list[dict[str, Any]]:
    """Measurements of the reference lap in each corner window, as ``CornerRecorder`` measures a policy's lap."""
    slip_f, slip_r = slip_ratios(lap)
    return [measure_window(lap.s, lap.state("vx"), lap.state("tau"), np.abs(lap.state("delta")), lap.state("ey"),
                           slip_f, slip_r, start, end, track_length) for start, end in CORNER_WINDOWS_M]


def solve_flying(track: RealTrack, b: float, margin: float, args: argparse.Namespace) -> MinTimeLap:
    """Solve and save the flying lap of balance index ``b`` with the edges ``margin`` [m] inside the track's own."""
    started = time.perf_counter()
    lap = solve_min_time_lap(track, b=b, ds=args.ds, max_iter=args.max_iter, edge_margin=margin)
    slip_f, slip_r = slip_ratios(lap)
    report = {
        "b": b, "ds": args.ds, "edge_margin": margin, "status": lap.status, "lap_time": lap.lap_time,
        "solver_seconds": time.perf_counter() - started,
        "s": lap.s.tolist(), "states": {name: lap.state(name).tolist() for name in STATE_NAMES},
        "tau_rate": lap.controls[:, 1].tolist(), "slip_front": slip_f.tolist(), "slip_rear": slip_r.tolist(),
        "corners": corner_measurements(lap, track.track_length),
    }
    name = f"min_time_b{b:+.2f}" + (f"_margin{margin:.2f}" if margin > 0.0 else "")
    out = args.output / f"{name}.json"
    out.write_text(json.dumps(report))
    logging.info("b %+.2f, edge margin %.2f m: %s, lap %.2f s, top speed %.1f m/s, %.0f s of solver; saved %s", b,
                 margin, lap.status, lap.lap_time, lap.state("vx").max(), report["solver_seconds"], out)
    return lap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=float, nargs="+", default=[0.0])
    parser.add_argument("--ds", type=float, default=2.0, help="[m] step along the centreline")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--standing-starts", action="store_true", help="also the laps from the protocol starts")
    parser.add_argument("--edge-margin", type=float, nargs="+", default=[0.0], help="[m] edges moved inwards")
    parser.add_argument("--output", type=Path, default=Path("outputs/reference"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    track = RealTrack(CATALUNYA)
    args.output.mkdir(parents=True, exist_ok=True)
    for b in args.b:
        flying = {margin: solve_flying(track, b, margin, args) for margin in args.edge_margin}
        if not args.standing_starts:
            continue
        lap = flying[0.0] if 0.0 in flying else solve_flying(track, b, 0.0, args)
        laps = []
        for start in protocol_starts(track):
            standing = solve_min_time_lap(track, b=b, ds=args.ds, max_iter=args.max_iter, start_s=start,
                                          start_vx=START_SPEED_MPS)
            laps.append({"start_s": start, "status": standing.status, "lap_time": standing.lap_time})
            logging.info("b %+.2f, start %.0f m: %s, lap %.2f s", b, start, standing.status, standing.lap_time)
        out = args.output / f"min_time_standing_b{b:+.2f}.json"
        out.write_text(json.dumps({"b": b, "ds": args.ds, "flying_lap_time": lap.lap_time, "laps": laps}, indent=1))


if __name__ == "__main__":
    main()
