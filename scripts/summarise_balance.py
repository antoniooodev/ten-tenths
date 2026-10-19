"""Summarise the evaluation over b of the Nominal, Randomised and Oracle policies (scripts/evaluate_balance.py).

    python scripts/summarise_balance.py

Reads <evaluation>/<tag>.json for the Nominal policy and the Randomised and Oracle policies of every width, and writes
<evaluation>/summary.json and the figures in <figures>:
- per policy and b: completion, lap time, lap time over the point-mass lap of that b, completion with the stochastic
  policy, failures by type and place, corner and tyre measurements, inference time and training cost;
- question 1, cost of randomisation: Randomised - Nominal, paired by seed and start, at b = 0 and per class of b;
- question 2, generalisation: completion and lap time per class of b, and held-out - interpolation;
- question 3, value of information: regret Randomised - Oracle and completion Oracle - Randomised per b, and the rule
  of a clear gap over the interpolation and held-out values (evaluation/balance_study.py).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from evaluate_stage import CATALUNYA

from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.evaluation.balance_study import (
    GAP_CLASSES,
    WIDTHS,
    classify,
    clear_gap,
    completion_difference,
    paired_time_difference,
    pool,
    summarise,
)
from contextual_bicycle_pacejka.evaluation.corners import CORNER_WINDOWS_M
from contextual_bicycle_pacejka.evaluation.laps import point_mass_lap_time
from contextual_bicycle_pacejka.model.track import RealTrack

NOMINAL = "nominal_b2"
LABELS = {"w13": "W1/3", "w23": "W2/3", "w1": "W1"}
COLOURS = {"nominal": "#6b6a64", "randomised": "#2a78d6", "oracle": "#eb6834"}
CORNER_KEYS = ("entry_speed", "min_speed", "exit_speed", "braking_s", "peak_steer", "peak_slip_front",
               "peak_slip_rear")


def tag(policy: str, width: str) -> str:
    return f"{policy}_{width}_b2"


def matrices(laps: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """(seeds x starts) completion and lap times (NaN where the lap failed)."""
    completed = np.array([[lap["completed"] for lap in seed_laps] for seed_laps in laps])
    times = np.array([[lap["lap_time"] if lap["completed"] else np.nan for lap in seed_laps] for seed_laps in laps])
    return completed, times


def load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    report["matrices"] = {key: matrices(laps) for key, laps in report["deterministic"].items()}
    report["stochastic_completed"] = {
        key: np.concatenate([matrices([seed_reps[r] for seed_reps in reps])[0] for r in range(len(reps[0]))], axis=1)
        for key, reps in report["stochastic"].items() if reps and reps[0]}
    return report


def key(b: float) -> str:
    return f"{b:+.2f}"


def per_b(report: dict[str, Any], point_mass: dict[str, float]) -> dict[str, Any]:
    """Completion, lap time, failures and corners of one policy at every b."""
    out = {}
    for k, (completed, times) in report["matrices"].items():
        laps = [lap for seed_laps in report["deterministic"][k] for lap in seed_laps]
        failed = [lap for lap in laps if not lap["completed"]]
        time = summarise(times)
        corners = {}
        for window in CORNER_WINDOWS_M:
            rows = [c for lap in laps for c in lap["corners"] if tuple(c["window"]) == window]
            corners[f"{window[0]:.0f}-{window[1]:.0f}"] = {
                "laps": len(rows),
                **{name: (float(np.mean([r[name] for r in rows if r[name] is not None]))
                          if any(r[name] is not None for r in rows) else None) for name in CORNER_KEYS}}
        stochastic = report["stochastic_completed"].get(k)
        out[k] = {
            "completion": summarise(completed.astype(float)),
            "lap_time": time,
            "lap_time_over_point_mass": None if time["mean"] is None else time["mean"] / point_mass[k],
            "stochastic_completion": None if stochastic is None else float(stochastic.mean()),
            "failures": dict(Counter(lap["failure"] for lap in failed)),
            "failure_places_m": dict(Counter(f"{10 * round(lap['failure_s'] / 10):.0f}" for lap in failed)),
            "corners": corners,
        }
    return out


def paired(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Paired lap time difference a - b and completion difference a - b over the values ``keys`` pooled."""
    return {
        "time": summarise(pool([paired_time_difference(a["matrices"][k][1], b["matrices"][k][1]) for k in keys])),
        "completion": summarise(pool([completion_difference(a["matrices"][k][0], b["matrices"][k][0])
                                      for k in keys])),
    }


def classes(width: str, grid: list[float]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for b in grid:
        groups.setdefault(classify(b, width), []).append(key(b))
    return groups


def training_cost(logs: Path, name: str, seeds: list[int]) -> dict[str, float]:
    """Decisions and wall time of the two stage-3 budgets of a policy, summed over the seeds."""
    decisions = hours = 0.0
    for budget in (name.removesuffix("_b2"), name):
        for seed in seeds:
            last = json.loads((logs / budget / f"s{seed}_stage3.jsonl").read_text().splitlines()[-1])
            decisions += last["timesteps"]
            hours += last["timesteps"] / last["decisions_per_second"] / 3600.0
    return {"decisions": decisions, "hours": hours}


def figure(summary: dict[str, Any], grid: list[float], path: Path) -> None:
    """Completion, lap time and regret against b, one column per width."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey="row")
    for col, width in enumerate(WIDTHS):
        lo, hi = WIDTHS[width]["range"]
        for row in range(3):
            ax = axes[row, col]
            ax.axvspan(lo, hi, color="#000000", alpha=0.05, lw=0)
            for h in WIDTHS[width]["held_out"]:
                ax.axvline(h, color="#9a9890", lw=0.8, ls=":")
            ax.grid(axis="y", color="#e4e3dd", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        for policy, name in (("nominal", NOMINAL), ("randomised", tag("randomised", width)),
                             ("oracle", tag("oracle", width))):
            rows = summary["policies"][name]
            rate = [rows[key(b)]["completion"]["mean"] for b in grid]
            axes[0, col].plot(grid, rate, color=COLOURS[policy], lw=2, marker="o", ms=5, label=policy.capitalize())
            means = [rows[key(b)]["lap_time"]["mean"] for b in grid]
            ok = [i for i, m in enumerate(means) if m is not None]
            low = [means[i] - rows[key(grid[i])]["lap_time"]["interval"][0] for i in ok]
            high = [rows[key(grid[i])]["lap_time"]["interval"][1] - means[i] for i in ok]
            axes[1, col].errorbar([grid[i] for i in ok], [means[i] for i in ok], yerr=[low, high],
                                  color=COLOURS[policy], lw=2, marker="o", ms=5, capsize=3)
        regret = {k: v["regret"] for k, v in summary["question3"][width]["per_b"].items()}
        ok = [b for b in grid if regret[key(b)]["mean"] is not None]
        means = [regret[key(b)]["mean"] for b in ok]
        low = [m - regret[key(b)]["interval"][0] for b, m in zip(ok, means, strict=True)]
        high = [regret[key(b)]["interval"][1] - m for b, m in zip(ok, means, strict=True)]
        axes[2, col].errorbar(ok, means, yerr=[low, high], color="#1c1c1a", lw=2, marker="o", ms=5, capsize=3)
        axes[2, col].axhline(0.0, color="#9a9890", lw=0.8)
        axes[0, col].set_title(f"{LABELS[width]}: training range [{lo:+.2f}, {hi:+.2f}]", fontsize=11)
        axes[2, col].set_xlabel("balance index b")
    axes[0, 0].set_ylabel("laps completed (share of 50)")
    axes[1, 0].set_ylabel("lap time [s], completed laps")
    axes[2, 0].set_ylabel("regret Randomised - Oracle [s]")
    axes[0, 0].legend(frameon=False, loc="lower center")
    fig.suptitle("Stage 3 (20 m/s): mean action, 5 seeds x 10 starts; shaded: training range, dotted: held-out",
                 fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation/balance"))
    parser.add_argument("--logs", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--balance", type=Path, default=Path("outputs/analysis/balance_index.json"))
    parser.add_argument("--figures", type=Path, default=Path("outputs/figures/balance"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    names = [NOMINAL] + [tag(policy, width) for width in WIDTHS for policy in ("randomised", "oracle")]
    reports = {name: load(args.evaluation / f"{name}.json") for name in names}
    grid = reports[NOMINAL]["grid"]
    seeds = reports[NOMINAL]["seeds"]
    rows = json.loads(args.balance.read_text())["rows"]
    ay = {round(row["b"], 2): row["ay_max_g"] for row in rows}
    track = RealTrack(CATALUNYA)
    stage = make_stage(reports[NOMINAL]["stage"], track)
    point_mass = {key(b): point_mass_lap_time(track, stage, lateral_ratio=ay[round(b, 2)] / ay[0.0]) for b in grid}

    summary: dict[str, Any] = {"grid": grid, "point_mass_lap_time": point_mass, "policies": {}, "cost": {}}
    for name, report in reports.items():
        summary["policies"][name] = per_b(report, point_mass)
        summary["cost"][name] = {**training_cost(args.logs, name, seeds),
                                 "inference_ms_per_decision": report["inference_ms_per_decision"]}

    nominal = reports[NOMINAL]
    summary["question1"], summary["question2"], summary["question3"] = {}, {}, {}
    for width in WIDTHS:
        groups = classes(width, grid)
        randomised, oracle = reports[tag("randomised", width)], reports[tag("oracle", width)]
        summary["question1"][width] = {cls: paired(randomised, nominal, keys) for cls, keys in groups.items()}
        q2: dict[str, Any] = {}
        for policy, report in (("nominal", nominal), ("randomised", randomised), ("oracle", oracle)):
            q2[policy] = {cls: {"completion": summarise(pool([report["matrices"][k][0] for k in keys])),
                                "lap_time": summarise(pool([report["matrices"][k][1] for k in keys]))}
                          for cls, keys in groups.items()}
            mean_of = {cls: np.mean([report["matrices"][k][0] for k in groups[cls]], axis=0) for cls in GAP_CLASSES}
            q2[policy]["held_out_minus_interpolation"] = summarise(mean_of["held_out"] - mean_of["interpolation"])
        summary["question2"][width] = q2
        gap_keys = [k for cls in GAP_CLASSES for k in groups[cls]]
        regret = summarise(pool([paired_time_difference(randomised["matrices"][k][1], oracle["matrices"][k][1])
                                 for k in gap_keys]))
        completion = summarise(pool([completion_difference(oracle["matrices"][k][0], randomised["matrices"][k][0])
                                     for k in gap_keys]))
        summary["question3"][width] = {
            "per_b": {key(b): {"regret": summarise(paired_time_difference(randomised["matrices"][key(b)][1],
                                                                           oracle["matrices"][key(b)][1])),
                               "completion": summarise(completion_difference(oracle["matrices"][key(b)][0],
                                                                             randomised["matrices"][key(b)][0]))}
                      for b in grid},
            "gap": {"regret": regret, "completion": completion, "clear": clear_gap(completion, regret)},
        }
    summary["memory_extension"] = any(summary["question3"][w]["gap"]["clear"] for w in WIDTHS)

    out = args.evaluation / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    figure(summary, grid, args.figures / "balance_stage3.png")
    for width in WIDTHS:
        gap = summary["question3"][width]["gap"]
        logging.info("%s: completion Oracle - Randomised %s, regret %s, clear gap %s", width,
                     gap["completion"], gap["regret"], gap["clear"])
    logging.info("memory extension: %s; saved %s", summary["memory_extension"], out)


if __name__ == "__main__":
    main()
