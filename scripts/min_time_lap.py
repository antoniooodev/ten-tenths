"""Compute the minimum-time lap reference of the plant car on Catalunya (reference/min_time.py; needs casadi).

    python scripts/min_time_lap.py
    python scripts/min_time_lap.py --b -0.2 0 0.17 --ds 2
    python scripts/min_time_lap.py --standing-starts

With ``--standing-starts`` it also solves the lap from each start of the protocol laps (evaluation/laps.py: on the
centreline at 3 m/s), the reference for protocol lap times, into <output>/min_time_standing_b<b>.json.

Writes <output>/min_time_b<b>.json: the lap time, the solver status, the profile along the lap (s, states, controls,
slip ratios) and, for each grip-limited corner of evaluation/corners.py, the measurements the policies are compared
with (entry, minimum and exit speed, braking point, peak steering and slip ratios). Used only to measure policies.
"""

from __future__ import annotations

import argparse
import json
import logging
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=float, nargs="+", default=[0.0])
    parser.add_argument("--ds", type=float, default=2.0, help="[m] step along the centreline")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--standing-starts", action="store_true", help="also the laps from the protocol starts")
    parser.add_argument("--output", type=Path, default=Path("outputs/reference"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    track = RealTrack(CATALUNYA)
    args.output.mkdir(parents=True, exist_ok=True)
    for b in args.b:
        lap = solve_min_time_lap(track, b=b, ds=args.ds, max_iter=args.max_iter)
        slip_f, slip_r = slip_ratios(lap)
        report = {
            "b": b, "ds": args.ds, "status": lap.status, "lap_time": lap.lap_time,
            "s": lap.s.tolist(), "states": {name: lap.state(name).tolist() for name in STATE_NAMES},
            "tau_rate": lap.controls[:, 1].tolist(), "slip_front": slip_f.tolist(), "slip_rear": slip_r.tolist(),
            "corners": corner_measurements(lap, track.track_length),
        }
        out = args.output / f"min_time_b{b:+.2f}.json"
        out.write_text(json.dumps(report))
        logging.info("b %+.2f: %s, lap %.2f s, top speed %.1f m/s; saved %s", b, lap.status, lap.lap_time,
                     lap.state("vx").max(), out)
        if not args.standing_starts:
            continue
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
