"""Build the interactive lap viewer: one self-contained HTML page with the recorded laps
(scripts/figures/record_laps.py).

    python scripts/figures/build_lap_viewer.py

Fills scripts/figures/lap_viewer.html with the track edges and four scenarios, resampled at 10 Hz, and writes
<output>/lap_viewer.html:
- flying:     the flying lap at b = 0 of every limit configuration (seed 12) and the minimum-time lap;
- oversteer:  b = +0.05, the three policies of the balance runs from the standing start at s = 0;
- understeer: b = -0.21, the same;
- noise:      Nominal continued, seed 14: the mean action and five runs of the stochastic policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from thesis_plots import (
    COLOURS,
    LABELS,
    NOMINAL,
    ORACLE,
    RANDOMISED,
    TAGS,
    flying,
    load,
    load_track,
    reference,
)

CHANNELS = ("t", "s", "x", "y", "psi", "vx", "ax", "ay", "delta", "tau", "slip_f", "slip_r")
DIGITS = {"t": 2, "s": 1, "x": 2, "y": 2, "psi": 3, "vx": 2, "ax": 2, "ay": 2, "delta": 4, "tau": 3, "slip_f": 3,
          "slip_r": 3}
RATE_HZ = 10.0


def resample(run: dict[str, np.ndarray], end: float | None = None) -> dict[str, list[float]]:
    end = float(run["t"][-1]) if end is None else end
    t = np.arange(0.0, end, 1.0 / RATE_HZ)
    t = np.append(t, end)
    out = {}
    for key in CHANNELS:
        values = np.unwrap(run[key]) if key == "psi" else run[key]
        out[key] = np.round(np.interp(t, run["t"], values), DIGITS[key]).tolist()
    return out


def car(name: str, colour: str, run: dict[str, np.ndarray], note: str, end: float | None = None,
        kind: str = "policy", failed: bool = False) -> dict:
    return {"name": name, "colour": colour, "kind": kind, "note": note, "failed": failed,
            "data": resample(run, end)}


def scenarios(length: float) -> list[dict]:
    ref = reference(0.0)
    optimum = car("Minimum-time lap", "optimum", ref, f"{float(ref['lap_time']):.2f} s", kind="optimum")
    flying_cars = [optimum]
    for tag in TAGS:
        lap = flying(load(f"flying_{tag}_s12"), length)
        if lap is not None:
            flying_cars.append(car(LABELS[tag], COLOURS[tag], lap, f"{float(lap['lap_time']):.2f} s"))
    result = [{"id": "flying", "title": "Flying lap, b = 0",
               "caption": "Second lap from s = 0 of every policy at the limit (seed 12) against the minimum-time "
                          "lap. Gaps are measured at the same position along the lap.",
               "reference": True, "cars": flying_cars}]
    colours = {"nominal_limit_cont": NOMINAL, "randomised_limit_w1": RANDOMISED, "oracle_limit_w1": ORACLE}
    for b, sid, kind in ((0.05, "oversteer", "oversteering"), (-0.21, "understeer", "understeering")):
        cars = []
        for tag, colour in colours.items():
            run = load(f"balance_{tag}_s12_b{b:+.2f}")
            done = run["lap_times"]
            end = float(done[0]) if len(done) else float(run["t"][-1])
            note = f"{end:.2f} s" if len(done) else f"off track at {float(run['s'][-1]):.0f} m"
            cars.append(car(LABELS[tag], colour, run, note, end, failed=not len(done)))
        result.append({"id": sid, "title": f"{kind.capitalize()} car, b = {b:+.2f}",
                       "caption": f"First lap from a standing start at s = 0 with an {kind} car (seed 12). The "
                                  "Nominal was trained only on b = 0; the Randomised and the Oracle on b in "
                                  "[−0.21, +0.03], and the Oracle also observes b.",
                       "reference": False, "cars": cars})
    mean = load("stochastic_nominal_limit_cont_s14_mean")
    cars = [car("Mean action", "#e34948", mean, f"off track at {float(mean['s'][-1]):.0f} m", failed=True)]
    for k in range(5):
        run = load(f"stochastic_nominal_limit_cont_s14_noise{k}")
        end = float(run["lap_times"][0])
        cars.append(car(f"Stochastic run {k + 1}", ("#3987e5", "#256abf", "#5598e7", "#184f95", "#6da7ec")[k], run,
                        f"{end:.2f} s", end))
    result.append({"id": "noise", "title": "Mean action vs noise",
                   "caption": "Nominal continued, seed 14, b = 0, first lap from s = 0. The mean action leaves the "
                              "track in corner C3; with the training noise every run completes the lap.",
                   "reference": False, "cars": cars})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=Path("scripts/figures/lap_viewer.html"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/thesis"))
    args = parser.parse_args()
    track = load_track(3.0)
    payload = {
        "length": round(track.length, 2),
        "left": np.round(track.left, 2).tolist(),
        "right": np.round(track.right, 2).tolist(),
        "centre": np.round(track.centre, 2).tolist(),
        "scenarios": scenarios(track.length),
    }
    page = args.template.read_text().replace("/*LAP_DATA*/null", json.dumps(payload, separators=(",", ":")))
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "lap_viewer.html"
    out.write_text(page)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
