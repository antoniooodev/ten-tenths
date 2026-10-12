"""Margin of an executor at one point of a circuit (docs/REDESIGN.md §25.5, diagnostic a).

Short episodes of a student policy at a constant plan scale c, reading the true mu plus an error p mu (clipped to
the range an estimate can take, ``oracle_clip``). Every episode starts ``lead`` metres before the point at the
teacher's speed (``start_speed`` with fraction 1) and passes if it gets ``beyond_m`` past the point without failing:

    passed         progress >= lead + beyond_m (a failure after that distance does not count)
    margin(p)      the highest scale of the grid at which every episode with error p passes (None if none does)

The grid is scales x errors x frictions x leads x setup seeds; every (scale, error) pair drives the same episodes.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning.closed_loop import start_speed


def margin_specs(racing_line: str | Path, point_s: float, leads: list[float], mus: list[float], seeds: list[int],
                 max_steps: int) -> list[dict[str, Any]]:
    """Protocol-like episodes starting ``lead`` metres before ``point_s`` (one per friction, lead and seed)."""
    specs = []
    for mu, lead, seed in itertools.product(mus, leads, seeds):
        start_s = float(point_s) - float(lead)
        specs.append({"block": "margin", "start_label": f"lead{lead:.0f}", "start_s": start_s,
                      "start_vx": start_speed(racing_line, start_s, mu, 1.0), "mu": float(mu), "seed": int(seed),
                      "max_steps": int(max_steps), "stall_rule": False, "lead_m": float(lead)})
    return specs


def margin_episode(job: dict[str, Any]) -> dict[str, Any]:
    """Drive one episode of the grid; return its scale, error, friction, lead, seed, progress and outcome."""
    spec, error = job["spec"], float(job["error"])
    record = run_episode(spec, job["racing_line"], "policy", False, model_dir=job["model_dir"],
                         plan_scale=float(job["scale"]), oracle_offset={"mu": error * spec["mu"]}, oracle_clip=True)
    distance = spec["lead_m"] + float(job["beyond_m"])
    return {"scale": float(job["scale"]), "error": error, "mu": spec["mu"], "lead_m": spec["lead_m"],
            "seed": spec["seed"], "progress_m": record["progress_m"], "passed": record["progress_m"] >= distance,
            "termination_reason": record["termination_reason"], "terminal_s_m": record["terminal_s_m"]}


def margins(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass counts per (error, scale) and per (error, scale, mu), and the margin of every error."""
    errors = sorted({r["error"] for r in results})
    scales = sorted({r["scale"] for r in results})
    table: dict[str, dict[str, dict[str, Any]]] = {}
    margin: dict[str, float | None] = {}
    for error in errors:
        rows = {}
        for scale in scales:
            cell = [r for r in results if r["error"] == error and r["scale"] == scale]
            by_mu: dict[str, list[int]] = {}
            for r in cell:
                by_mu.setdefault(str(r["mu"]), [0, 0])
                by_mu[str(r["mu"])][0] += int(r["passed"])
                by_mu[str(r["mu"])][1] += 1
            rows[str(scale)] = {"passed": sum(r["passed"] for r in cell), "n": len(cell),
                                "by_mu": {mu: {"passed": k, "n": n} for mu, (k, n) in by_mu.items()},
                                "timeouts": sum(r["termination_reason"] == "step_limit" and not r["passed"]
                                                for r in cell)}
        table[str(error)] = rows
        full = [scale for scale in scales if rows[str(scale)]["passed"] == rows[str(scale)]["n"]]
        margin[str(error)] = max(full) if full else None
    return {"table": table, "margin": margin}
