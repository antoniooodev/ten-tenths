"""Aggregation of protocol results: rates with confidence intervals, lap times, failure causes, several seeds.

Rules:

* rates use the Wilson 95 % score interval; paired outcomes the exact McNemar test;
* lap times are compared only on episodes that are robust for both controllers (same episode);
* the failure mechanism is the first matching rule:
      stall / low speed / step limit                                   longitudinal: stall/low speed
      track excursion                                                  lateral: track excursion
      slip with overspeed (vx > 1.05 v_target) starting no later than
      the first slip-saturation step, or terminal vx - v_target > 1 m/s  longitudinal: overspeed into slip
      other slip                                                       lateral/yaw: slip without overspeed
  where v_target is the target speed recorded by the episode diagnostics.

Several training seeds of one configuration (``seed_summary``): the robust rate of each seed with its Wilson
interval, the mean and sample standard deviation of the rates across seeds, the pooled rate over all the runs,
the episodes robust in every seed and in none, and the paired lap time against a reference per seed.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BLOCKS = ("in_sample_gate", "seed_only", "mu_only", "start_only", "held_out")
WILSON_Z = 1.959963984540054  # two-sided 95 %
OVERSPEED_TERMINAL_MPS = 1.0
Records = dict[tuple, dict[str, Any]]


def wilson(k: int, n: int) -> list[float] | None:
    """Wilson 95 % score interval of k successes in n trials."""
    if n == 0:
        return None
    p = k / n
    d = 1 + WILSON_Z * WILSON_Z / n
    c = (p + WILSON_Z * WILSON_Z / (2 * n)) / d
    h = WILSON_Z * math.sqrt(p * (1 - p) / n + WILSON_Z * WILSON_Z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def episode_key(record: dict[str, Any]) -> tuple:
    """Key pairing the same protocol episode across controllers and conditions."""
    return (record["block"], record["start_label"], round(float(record["mu"]), 3), int(record["seed"]))


def load_records(path: str | Path) -> Records:
    """Records of a JSONL result file keyed by ``episode_key`` (empty if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {episode_key(r): r for r in records}


def rate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    k = sum(bool(r[field]) for r in records)
    n = len(records)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def lap_stats(times: list[float]) -> dict[str, float] | None:
    if not times:
        return None
    a = np.asarray(times)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def failure_mechanism(record: dict[str, Any]) -> str:
    """Failure mechanism of a non-robust episode (see the module docstring)."""
    reason = record["termination_reason"]
    if reason in ("low_speed", "stalled", "step_limit"):
        return "longitudinal: stall/low speed"
    if reason == "track_excursion":
        return "lateral: track excursion"
    if reason == "slip_saturation":
        diagnostics = record["diagnostics"]
        onsets = diagnostics.get("pre_failure_last5s") or {}
        terminal = diagnostics.get("terminal_state") or {}
        overspeed = onsets.get("overspeed")
        slips = [x for x in (onsets.get("rear_slip_saturated"), onsets.get("front_slip_saturated")) if x]
        first_slip = min([x["seconds_before_end"] for x in slips] or [0.0])
        terminal_overspeed = terminal.get("vx", 0.0) - terminal.get("v_target", 1e9) > OVERSPEED_TERMINAL_MPS
        if (overspeed and overspeed["seconds_before_end"] >= first_slip) or terminal_overspeed:
            return "longitudinal: overspeed into slip"
        return "lateral/yaw: slip without overspeed"
    return reason


def summarise(records: Records) -> dict[str, Any]:
    """Robust and completion rates (overall and per block), termination reasons, robust lap times."""
    rs = list(records.values())
    return {"n": len(rs), "robust": rate(rs, "robust"), "completed": rate(rs, "completed"),
            "by_block": {b: {"robust": rate([r for r in rs if r["block"] == b], "robust")} for b in BLOCKS},
            "termination": dict(collections.Counter(r["termination_reason"] for r in rs)),
            "lap_time_robust": lap_stats([r["lap_time_s"] for r in rs if r["robust"]])}


def paired_lap_time(student: Records, teacher: Records) -> dict[str, Any] | None:
    """Lap-time comparison on the episodes robust for both controllers."""
    keys = [k for k in student if k in teacher and student[k]["robust"] and teacher[k]["robust"]]
    if not keys:
        return None
    s = np.array([student[k]["lap_time_s"] for k in keys])
    t = np.array([teacher[k]["lap_time_s"] for k in keys])
    return {"n": len(keys), "student_mean": float(s.mean()), "teacher_mean": float(t.mean()),
            "ratio_mean": float((s / t).mean()), "ratio_of_means": float(s.mean() / t.mean()),
            "diff_s_min_max": [float((s - t).min()), float((s - t).max())]}


def joint_outcome(student: Records, teacher: Records) -> dict[str, Any]:
    """Robustness of student and teacher on identical episodes, and which of the two fails first."""
    keys = [k for k in student if k in teacher]
    return {"n": len(keys),
            "both_robust": sum(student[k]["robust"] and teacher[k]["robust"] for k in keys),
            "student_only": sum(student[k]["robust"] and not teacher[k]["robust"] for k in keys),
            "teacher_only": sum(teacher[k]["robust"] and not student[k]["robust"] for k in keys),
            "neither": sum(not student[k]["robust"] and not teacher[k]["robust"] for k in keys),
            "student_progress_median_m": float(np.median([student[k]["progress_m"] for k in keys])),
            "teacher_progress_median_m": float(np.median([teacher[k]["progress_m"] for k in keys])),
            "student_fails_before_teacher": sum(student[k]["progress_m"] < teacher[k]["progress_m"] - 1.0
                                                for k in keys),
            "student_fails_after_teacher": sum(student[k]["progress_m"] > teacher[k]["progress_m"] + 1.0
                                               for k in keys)}


def failure_groups(records: Records, teacher: Records | None = None) -> dict[str, dict[str, int]]:
    """Counts of the non-robust episodes by zone, reason, mechanism, mu, seed, start and block."""
    fails = [r for r in records.values() if not r["robust"]]
    getters = {"zone": lambda r: r["diagnostics"].get("terminal_zone"), "reason": lambda r: r["termination_reason"],
               "mechanism": failure_mechanism, "mu": lambda r: r["mu"], "seed": lambda r: r["seed"],
               "start": lambda r: r["start_label"], "block": lambda r: r["block"]}
    groups = {name: dict(collections.Counter(str(get(r)) for r in fails).most_common())
              for name, get in getters.items()}
    if teacher:
        groups["teacher_same_spec_robust"] = dict(collections.Counter(
            str(teacher[episode_key(r)]["robust"]) if episode_key(r) in teacher else "n/a" for r in fails))
    return groups


def failure_rows(records: Records, agent: str, condition: str, twin: Records | None = None,
                 teacher: Records | None = None) -> list[dict[str, Any]]:
    """One row per non-robust episode, for the failure table."""
    rows = []
    for key, r in records.items():
        if r["robust"]:
            continue
        terminal = r["diagnostics"].get("terminal_state") or {}
        rows.append({
            "circuit": r.get("circuit"), "agent": agent, "cond": condition, "id": r.get("id"), "block": r["block"],
            "reason": r["termination_reason"], "mechanism": failure_mechanism(r),
            "terminal_s_m": round(r["terminal_s_m"], 1), "lap_fraction": round(r["lap_fraction"], 4),
            "zone": r["diagnostics"].get("terminal_zone"), "mu": r["mu"], "seed": r["seed"],
            "start": r["start_label"], "start_s_m": round(r["start_s_m"], 1),
            "twin_other_cond_robust": None if not twin or key not in twin else twin[key]["robust"],
            "teacher_same_spec_robust": None if not teacher or key not in teacher else teacher[key]["robust"],
            "terminal_vx": round(terminal.get("vx", float("nan")), 2),
            "terminal_v_target": round(terminal.get("v_target", float("nan")), 2),
            "terminal_ey": round(terminal.get("ey", float("nan")), 3),
            "terminal_rear_slip_ratio": round(terminal.get("rear_slip_ratio", float("nan")), 3),
            "no_safe_steps": r["no_safe_steps"], "projected_steps": r["projected_steps"]})
    return rows


def seed_summary(runs: dict[str, Records], teacher: Records | None = None) -> dict[str, Any]:
    """Robustness and lap time of several training seeds of one configuration on the same protocol episodes.

    ``runs`` maps a run name (one per seed) to its records; every run must cover the same episodes.
    """
    if not runs:
        raise ValueError("no runs")
    keys = set(next(iter(runs.values())))
    if any(set(records) != keys for records in runs.values()):
        raise ValueError("the runs do not cover the same episodes")
    per_seed = {}
    for name, records in runs.items():
        rs = list(records.values())
        per_seed[name] = {"robust": rate(rs, "robust"), "completed": rate(rs, "completed"),
                          "lap_time_robust": lap_stats([r["lap_time_s"] for r in rs if r["robust"]]),
                          "paired_lap_time": paired_lap_time(records, teacher) if teacher else None,
                          "termination": dict(collections.Counter(r["termination_reason"] for r in rs)),
                          "mechanism": dict(collections.Counter(failure_mechanism(r) for r in rs if not r["robust"])),
                          "traction_cost_sum_mean": float(np.mean([r.get("traction_cost_sum", 0.0) for r in rs]))}
    rates = np.array([s["robust"]["rate"] for s in per_seed.values()], dtype=np.float64)
    pooled = [r for records in runs.values() for r in records.values()]
    ratios = [s["paired_lap_time"]["ratio_of_means"] for s in per_seed.values()
              if s["paired_lap_time"] is not None]
    return {
        "n_seeds": len(runs), "n_episodes": len(keys), "per_seed": per_seed,
        "robust_rate_mean": float(rates.mean()), "robust_rate_sd": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
        "robust_rate_min": float(rates.min()), "robust_rate_max": float(rates.max()),
        "robust_pooled": rate(pooled, "robust"),
        "robust_in_every_seed": sum(all(records[k]["robust"] for records in runs.values()) for k in keys),
        "robust_in_no_seed": sum(not any(records[k]["robust"] for records in runs.values()) for k in keys),
        "lap_time_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "teacher": summarise(teacher) if teacher else None,
    }
