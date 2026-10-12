"""Rebuild the track files of one circuit in data/tracks/<circuit>/ (see contextual_bicycle_pacejka.tracks).

    python scripts/prepare_tracks.py --circuit spa --steps reference racing-line speed-profile r0 r1

The racing-line step does not apply to Monza (its time-optimal line is stored data) and r1 not to Budapest.
The speed-profile step drives 40 full laps and takes about an hour.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.tracks.raceline import build_racing_line
from contextual_bicycle_pacejka.tracks.reference import build_reference, download_centerline
from contextual_bicycle_pacejka.tracks.repair import repair_racing_line
from contextual_bicycle_pacejka.tracks.speed_profile import R0_FACTOR, fit_speed_profile, scale_profile

REPO = Path(__file__).resolve().parents[1]
STEPS = ("reference", "racing-line", "speed-profile", "r0", "r1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", required=True, choices=sorted(R0_FACTOR))
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--out-dir", type=Path, help="output directory (default: data/tracks/<circuit>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out_dir or REPO / "data/tracks" / args.circuit
    out.mkdir(parents=True, exist_ok=True)
    line = out / ("time_optimal_line.npz" if args.circuit == "monza" else "minimum_curvature_line.npz")
    if "reference" in args.steps:
        build_reference(download_centerline(args.circuit, out / "tum_centerline.csv"), out / "reference.npz")
    if "racing-line" in args.steps:
        if args.circuit == "monza":
            raise SystemExit("the Monza time-optimal line is stored data and cannot be regenerated")
        build_racing_line(out / "reference.npz", line)
    if "speed-profile" in args.steps:
        report = fit_speed_profile(line, out / "speed_profile_probed.npz")
        logging.info("best robust mean lap time %.2f s", report["best_time_s"])
    if "r0" in args.steps:
        scale_profile(out / "speed_profile_probed.npz", out / "racing_line_R0.npz", R0_FACTOR[args.circuit])
    if "r1" in args.steps:
        if args.circuit == "budapest":
            raise SystemExit("Budapest has no curvature kink: R1 is not defined")
        log = repair_racing_line(out / "racing_line_R0.npz", out / "reference.npz", out / "racing_line_R1.npz")
        logging.info(json.dumps({k: v for k, v in log.items() if k != "kinks"}, indent=1))


if __name__ == "__main__":
    main()
