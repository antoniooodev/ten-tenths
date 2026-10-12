"""Figures of results/final_results.json (written to results/figures/).

    python scripts/make_figures.py

robustness.png   robust-lap rate (Wilson 95 % interval) of every policy and the teacher, per circuit and racing line,
                 unfiltered; "0" marks an evaluated agent with no robust lap, a missing bar an agent not evaluated
lap_time.png     mean lap time of the robust laps relative to the teacher on the same episodes, unfiltered (none on
                 Spa R0 and Catalunya R0, where the unfiltered teacher has no robust lap)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AGENTS = ("teacher", "treatment_a", "all4", "loco_spa", "loco_catalunya", "loco_budapest", "ppo_pilot_all4_monza")
LABELS = {"teacher": "teacher (MATv0)", "treatment_a": "Treatment A (Monza)", "all4": "ALL4",
          "loco_spa": "LOCO Spa", "loco_catalunya": "LOCO Catalunya", "loco_budapest": "LOCO Budapest",
          "ppo_pilot_all4_monza": "PPO pilot (ALL4)"}
TRACKS = (("monza", "R0"), ("monza", "R1"), ("spa", "R0"), ("spa", "R1"), ("catalunya", "R0"), ("catalunya", "R1"),
          ("budapest", "R0"))


def main() -> None:
    results = json.loads((REPO / "results/final_results.json").read_text())["results"]
    table = {(e["agent"], e["circuit"], e["racing_line"]): e for e in results if e["condition"] == "U"}
    out = REPO / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TRACKS))
    width = 0.8 / len(AGENTS)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, agent in enumerate(AGENTS):
        for j, track in enumerate(TRACKS):
            entry = table.get((agent, *track))
            if entry is None:
                continue
            robust = entry["summary"]["robust"]
            low, high = robust["wilson95"]
            position = x[j] + (i - len(AGENTS) / 2 + 0.5) * width
            if robust["k"] == 0:
                ax.text(position, 0.01, "0", ha="center", va="bottom", fontsize=7, color=f"C{i}")
            ax.bar(position, robust["rate"], width, color=f"C{i}",
                   yerr=[[robust["rate"] - low], [high - robust["rate"]]], capsize=2,
                   label=LABELS[agent] if j == next(k for k, t in enumerate(TRACKS) if (agent, *t) in table) else None)
    ax.set_xticks(x, [f"{c.capitalize()} {line}" for c, line in TRACKS])
    ax.set_ylabel("robust laps (of 396)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness on the frozen protocols, unfiltered")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, agent in enumerate(AGENTS[1:], start=1):
        for j, track in enumerate(TRACKS):
            paired = (table.get((agent, *track)) or {}).get("paired_lap_time_vs_teacher")
            if paired:
                ax.bar(x[j] + (i - len(AGENTS) / 2 + 0.5) * width, 100 * (paired["ratio_of_means"] - 1), width,
                       color=f"C{i}", label=LABELS[agent])
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{c.capitalize()} {line}" for c, line in TRACKS])
    ax.set_ylabel("lap time vs teacher [%]")
    ax.set_title("Mean lap time relative to the teacher (episodes robust for both), unfiltered")
    fig.tight_layout()
    fig.savefig(out / "lap_time.png", dpi=150)


if __name__ == "__main__":
    main()
