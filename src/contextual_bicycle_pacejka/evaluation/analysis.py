"""Aggregation of protocol results: rates with confidence intervals, filter dependence, lap times, failure causes.

Rules (fixed before the final evaluation):

* rates use the Wilson 95 % score interval;
* filter dependence compares the U and F twins of each episode (exact McNemar test on completion):
      FILTER-DEPENDENT    completion rate U < 0.8 x completion rate F
      POLICY-DOMINATED    filter rescues <= 1 % of the twins and fewer than 0.1 % of filtered steps are projected
      FILTER-ASSISTED     otherwise;
* lap times are compared only on episodes that are robust for both controllers (same spec, same condition);
* the failure mechanism is the first matching rule:
      stall / low speed / step limit                                   longitudinal: stall/low speed
      track excursion                                                  lateral: track excursion
      slip with overspeed (vx > 1.05 v_target) starting no later than
      the first slip-saturation step, or terminal vx - v_target > 1 m/s  longitudinal: overspeed into slip
      other slip                                                       lateral/yaw: slip without overspeed
  where v_target is the teacher's target speed on the controller's own state.

Several training seeds of one configuration (``seed_summary``): the robust rate of each seed with its Wilson
interval, the mean and sample standard deviation of the rates across seeds, the pooled rate over all the runs,
the episodes robust in every seed and in none, and the paired lap time against the teacher per seed.

Cost of the friction estimate under a plan coach (``estimate_cost``, docs/REDESIGN.md §25, measure 0a; fixed
before the results). The same coach drives with its executor reading the true mu and the estimate; ``executor`` is
the executor on the estimate at the teacher's plan (scale 1). Robust counts R; lap-time ratio estimate / true mu as
the mean per-episode ratio on the episodes robust for both. The extra failures are the episodes robust with the
true mu and not with the estimate; ``mu_ratio_final`` is the estimate over the true mu at their end. Outcome, first
matching rule:
      R(estimate) < R(executor) - 1                                          C   worse than the teacher's plan
      R(estimate) >= R(true mu) - 1 and lap-time ratio <= 1.005               A   the estimate costs nothing
      no extra failure                                                        B3  the estimate costs lap time only
      more than half of the extra failures end with mu_ratio_final > 1        B1  failures on an overestimate
      otherwise                                                               B2  failures without overestimate
The robustness floor of the sensor plan student (§23.6, criterion 1) is max(R(estimate), R(executor)) - 1.

Constant-scale reference (``plan_reference``, docs/REDESIGN.md §25.5, measure 0b): an executor driven on the teacher's
plan multiplied by constant scales; c* is the highest scale whose robust count is at least that of scale 1 (1.0 if
none), the reference lap time of criterion 2 of §23.6.

Sensor plan student (``plan_student_criterion``, docs/REDESIGN.md §26.3; fixed before the results), on the pairs
(seed, protocol episode) of several training seeds against the coach and the constant scale c*, all on the estimate.
Intervals are 95 % percentile intervals of a two-level bootstrap: the seeds and, independently, the protocol episodes
(with all their pairs) are resampled with replacement. On the aggregate:
      1. robustness not inferior to the coach: upper bound of d <= margin, d = failure share of the student minus
         failure share of the coach
      2. faster than c*: upper bound of r < 1, r = mean lap-time ratio student / c* on the pairs robust for both
The circuit passes when both hold. Reported, not deciding: the lap-time ratio student / coach (the cost of
perception), per-seed robust counts with Wilson intervals and McNemar against the coach, an inclusive ratio to c* in
which a student failure counts with the worst ratio of the robust pairs, the scale per sector of 500 m and the
failures.
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


def summarise(records: Records, filtered: bool) -> dict[str, Any]:
    """Robust and completion rates (overall and per block), termination reasons, robust lap times."""
    rs = list(records.values())
    out = {"n": len(rs), "robust": rate(rs, "robust"), "completed": rate(rs, "completed"),
           "by_block": {b: {"robust": rate([r for r in rs if r["block"] == b], "robust")} for b in BLOCKS},
           "termination": dict(collections.Counter(r["termination_reason"] for r in rs)),
           "lap_time_robust": lap_stats([r["lap_time_s"] for r in rs if r["robust"]])}
    if filtered:
        steps = sum(r["steps"] for r in rs)
        projected = sum(r["projected_steps"] for r in rs)
        out.update({"empty_safe_set_episodes": sum(r["no_safe_steps"] > 0 for r in rs),
                    "empty_safe_set_steps": sum(r["no_safe_steps"] for r in rs),
                    "projected_steps": projected, "projected_fraction": projected / steps if steps else None,
                    "episodes_with_projection": sum(r["projected_steps"] > 0 for r in rs)})
    return out


def filter_dependence(unfiltered: Records, filtered: Records) -> dict[str, Any]:
    """Paired U/F comparison and the filter-dependence class (see the module docstring)."""
    keys = [k for k in unfiltered if k in filtered]
    rescues = sum(filtered[k]["completed"] and not unfiltered[k]["completed"] for k in keys)
    hurts = sum(unfiltered[k]["completed"] and not filtered[k]["completed"] for k in keys)
    completed_u = sum(unfiltered[k]["completed"] for k in keys) / len(keys) if keys else 0
    completed_f = sum(filtered[k]["completed"] for k in keys) / len(keys) if keys else 0
    steps = sum(filtered[k]["steps"] for k in keys)
    projected = sum(filtered[k]["projected_steps"] for k in keys) / steps if steps else 0
    if completed_u < 0.8 * completed_f:
        classification = "FILTER-DEPENDENT"
    elif rescues <= 0.01 * len(keys) and projected < 1e-3:
        classification = "POLICY-DOMINATED"
    else:
        classification = "FILTER-ASSISTED"
    return {"paired": len(keys), "b_filter_rescues": rescues, "c_filter_hurts": hurts,
            "mcnemar_p": mcnemar(rescues, hurts), "p_proj": projected, "classification": classification,
            "robust_twins_differ": sum(unfiltered[k]["robust"] != filtered[k]["robust"] for k in keys)}


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


TRACKING_FIELDS = ("speed_error_mean_abs_mps", "frac_overspeed", "speed_error_mean_mps", "ey_rms_m",
                   "rear_slip_ratio_p99", "teacher_tau_mae", "teacher_delta_mae")


def tracking_medians(records: Records) -> dict[str, float | None]:
    """Median over episodes of the per-episode tracking diagnostics (teacher target on the controller's state)."""
    out = {}
    for field in TRACKING_FIELDS:
        values = [r["diagnostics"][field] for r in records.values() if r["diagnostics"].get(field) is not None]
        out[f"{field}_median"] = float(np.median(values)) if values else None
    return out


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


def compare(student: dict[str, Records], teacher: dict[str, Records]) -> dict[str, Any]:
    """Full comparison of a student with the teacher on one protocol; inputs are keyed by condition "U"/"F"."""
    out: dict[str, Any] = {
        "student": {c: summarise(student[c], c == "F") for c in student if student[c]},
        "teacher": {c: summarise(teacher[c], c == "F") for c in teacher if teacher[c]},
        "paired_lap_time": {c: paired_lap_time(student[c], teacher[c])
                            for c in student if student[c] and teacher.get(c)},
        "joint": {c: joint_outcome(student[c], teacher[c]) for c in student if student[c] and teacher.get(c)},
        "tracking_student_U": tracking_medians(student["U"]) if student.get("U") else None,
        "tracking_teacher_U": tracking_medians(teacher["U"]) if teacher.get("U") else None,
        "failure_groups": {f"student_{c}": failure_groups(student[c], teacher.get(c)) for c in student if student[c]},
    }
    if student.get("U") and student.get("F"):
        out["filter_student"] = filter_dependence(student["U"], student["F"])
    if teacher.get("U") and teacher.get("F"):
        out["filter_teacher"] = filter_dependence(teacher["U"], teacher["F"])
    return out


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
        "teacher": summarise(teacher, False) if teacher else None,
    }


ESTIMATE_ROBUST_SLACK = 1
ESTIMATE_TIME_SLACK = 0.005


def _agent_summary(records: Records, teacher: Records) -> dict[str, Any]:
    """Robust count, lap-time ratio to the teacher (overall, per mu, worst mu), plan scale and mu-entry ratio."""
    rs = list(records.values())
    keys = [k for k in records if k in teacher and records[k]["robust"] and teacher[k]["robust"]]
    by_mu: dict[float, list[float]] = collections.defaultdict(list)
    for k in keys:
        by_mu[float(records[k]["mu"])].append(records[k]["lap_time_s"] / teacher[k]["lap_time_s"])
    ratio_by_mu = {str(mu): float(np.mean(v)) for mu, v in sorted(by_mu.items())}
    oracle = [r["oracle_mu"] for r in rs if r.get("oracle_mu")]
    scales = [r["plan_scale"]["mean"] for r in rs if "plan_scale" in r]  # records older than the plan scale lack it
    return {"robust": sum(r["robust"] for r in rs), "n": len(rs),
            "teacher_ratio": paired_lap_time(records, teacher)["ratio_mean"] if keys else None,
            "teacher_ratio_by_mu": ratio_by_mu, "teacher_ratio_worst": max(ratio_by_mu.values(), default=None),
            "plan_scale_mean": float(np.mean(scales)) if scales else None,
            "mu_ratio_max": max((o["max"] for o in oracle), default=None),
            "mu_ratio_mean": float(np.mean([o["mean"] for o in oracle])) if oracle else None}


def estimate_cost(on_estimate: Records, true_mu: Records, executor: Records, teacher: Records) -> dict[str, Any]:
    """Cost of the friction estimate under a plan coach and the outcome of measure 0a (see the module docstring)."""
    if not set(on_estimate) == set(true_mu) == set(executor):
        raise ValueError("the runs do not cover the same episodes")
    robust_est, robust_true = (sum(r["robust"] for r in rs.values()) for rs in (on_estimate, true_mu))
    robust_exec = sum(r["robust"] for r in executor.values())
    paired = paired_lap_time(on_estimate, true_mu)
    ratio = paired["ratio_mean"] if paired else None
    extra = []
    for key, r in on_estimate.items():
        if r["robust"] or not true_mu[key]["robust"]:
            continue
        oracle = r.get("oracle_mu") or {}
        extra.append({"mu": r["mu"], "seed": r["seed"], "start": r["start_label"],
                      "terminal_s_m": round(r["terminal_s_m"], 1), "mechanism": failure_mechanism(r),
                      "mu_ratio_final": oracle.get("final"), "mu_ratio_max": oracle.get("max"),
                      "last_plan_scale": r["plan_decisions"][-1][1] if r.get("plan_decisions") else None})
    over = sum((row["mu_ratio_final"] or 0.0) > 1.0 for row in extra)
    if robust_est < robust_exec - ESTIMATE_ROBUST_SLACK:
        outcome = "C"
    elif robust_est >= robust_true - ESTIMATE_ROBUST_SLACK and (ratio is None or ratio <= 1 + ESTIMATE_TIME_SLACK):
        outcome = "A"
    elif not extra:
        outcome = "B3"
    else:
        outcome = "B1" if over > len(extra) / 2 else "B2"
    return {"outcome": outcome,
            "on_estimate": _agent_summary(on_estimate, teacher), "true_mu": _agent_summary(true_mu, teacher),
            "executor": _agent_summary(executor, teacher),
            "estimate_over_true_mu": paired, "extra_failures": extra, "extra_failures_on_overestimate": over,
            "robust_lost": sum(true_mu[k]["robust"] and not on_estimate[k]["robust"] for k in on_estimate),
            "robust_gained": sum(on_estimate[k]["robust"] and not true_mu[k]["robust"] for k in on_estimate),
            "student_robust_floor": max(robust_est, robust_exec) - ESTIMATE_ROBUST_SLACK}


def plan_reference(base: Records, scaled: dict[float, Records], teacher: Records) -> dict[str, Any]:
    """Summary of an executor at scale 1 and at constant scales, and the reference scale c* (module docstring)."""
    if any(set(records) != set(base) for records in scaled.values()):
        raise ValueError("the runs do not cover the same episodes")
    robust_base = sum(r["robust"] for r in base.values())
    rows = {"1.00": _agent_summary(base, teacher)}
    rows.update({f"{scale:.2f}": _agent_summary(records, teacher) for scale, records in sorted(scaled.items())})
    eligible = [scale for scale, records in scaled.items() if sum(r["robust"] for r in records.values()) >= robust_base]
    c_star = max(eligible) if eligible else 1.0
    return {"scales": rows, "c_star": c_star, "c_star_summary": rows[f"{c_star:.2f}"]}


STUDENT_ROBUST_MARGIN = 0.05
BOOTSTRAP_RESAMPLES = 10_000
SECTOR_M = 500.0


def _sector_scales(runs: list[Records]) -> dict[str, float]:
    """Mean plan scale of the decisions in every sector of ``SECTOR_M`` metres, over all episodes of all runs."""
    by_sector: dict[float, list[float]] = collections.defaultdict(list)
    for records in runs:
        for r in records.values():
            for s, scale in r.get("plan_decisions", []):
                by_sector[math.floor(s / SECTOR_M) * SECTOR_M].append(scale)
    return {f"{start:.0f}": float(np.mean(v)) for start, v in sorted(by_sector.items())}


def _outcomes(records: Records, keys: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Robust flags and lap times (NaN when not robust) of ``keys``."""
    robust = np.array([bool(records[k]["robust"]) for k in keys])
    times = np.array([records[k]["lap_time_s"] if records[k]["robust"] else np.nan for k in keys], dtype=np.float64)
    return robust, times


def _ratio(times: np.ndarray, reference: np.ndarray) -> float:
    """Mean ratio over the pairs where both lap times exist (NaN if none)."""
    ratios = times / reference
    return float(np.nanmean(ratios)) if np.isfinite(ratios).any() else float("nan")


def _bootstrap(statistic: Any, n_seeds: int, n_episodes: int, resamples: int, seed: int) -> list[float]:
    """95 % percentile interval of ``statistic(seed_index, episode_index)`` over a two-level bootstrap."""
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n_seeds, n_seeds), rng.integers(0, n_episodes, n_episodes))
                       for _ in range(resamples)])
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def plan_student_criterion(students: dict[str, Records], coach: Records, c_star: Records, teacher: Records,
                           margin: float = STUDENT_ROBUST_MARGIN, resamples: int = BOOTSTRAP_RESAMPLES,
                           bootstrap_seed: int = 0) -> dict[str, Any]:
    """Criterion of a sensor plan student over several seeds (see the module docstring)."""
    keys = sorted(coach)
    if any(set(records) != set(keys) for records in [*students.values(), c_star]):
        raise ValueError("the runs do not cover the same episodes")
    names = list(students)
    pairs = [_outcomes(students[name], keys) for name in names]
    robust, times = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    coach_robust, coach_times = _outcomes(coach, keys)
    star_robust, star_times = _outcomes(c_star, keys)

    def difference(si: np.ndarray, ei: np.ndarray) -> float:
        return float(np.mean(~robust[np.ix_(si, ei)]) - np.mean(~coach_robust[ei]))

    def versus(reference: np.ndarray) -> Any:
        return lambda si, ei: _ratio(times[np.ix_(si, ei)], reference[ei][None, :])

    every_seed, every_episode = np.arange(len(names)), np.arange(len(keys))
    interval = {"n_seeds": len(names), "n_episodes": len(keys), "resamples": resamples, "seed": bootstrap_seed}
    d = {"point": difference(every_seed, every_episode),
         "ci95": _bootstrap(difference, len(names), len(keys), resamples, bootstrap_seed)}
    r_star = {"point": versus(star_times)(every_seed, every_episode),
              "ci95": _bootstrap(versus(star_times), len(names), len(keys), resamples, bootstrap_seed),
              "n_pairs": int(np.sum(np.isfinite(times / star_times[None, :])))}
    r_coach = {"point": versus(coach_times)(every_seed, every_episode),
               "ci95": _bootstrap(versus(coach_times), len(names), len(keys), resamples, bootstrap_seed),
               "n_pairs": int(np.sum(np.isfinite(times / coach_times[None, :])))}
    ratios = times / star_times[None, :]
    worst = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else float("nan")
    inclusive = np.where(~robust & star_robust[None, :], worst, ratios)
    checks = {"robust_non_inferior": d["ci95"][1] <= margin, "faster_than_c_star": r_star["ci95"][1] < 1.0}
    seeds = {}
    for i, name in enumerate(names):
        k = int(robust[i].sum())
        better, worse = int(np.sum(robust[i] & ~coach_robust)), int(np.sum(~robust[i] & coach_robust))
        seeds[name] = {**_agent_summary(students[name], teacher), "wilson95": wilson(k, len(keys)),
                       "mcnemar_vs_coach": {"student_only": better, "coach_only": worse, "p": mcnemar(better, worse)},
                       "versus_c_star": _ratio(times[i], star_times), "versus_coach": _ratio(times[i], coach_times)}
    failures = [{"seed": name, "mu": r["mu"], "start": r["start_label"], "terminal_s_m": round(r["terminal_s_m"], 1),
                 "mechanism": failure_mechanism(r), "mu_ratio_final": (r.get("oracle_mu") or {}).get("final"),
                 "last_plan_scale": r["plan_decisions"][-1][1] if r.get("plan_decisions") else None}
                for name in names for r in students[name].values() if not r["robust"]]
    pooled = int(robust.sum())
    return {"passed": all(checks.values()), "checks": checks, "margin": margin, "bootstrap": interval,
            "robust_difference": d, "versus_c_star": r_star, "versus_coach": r_coach,
            "inclusive_versus_c_star": float(np.nanmean(inclusive)),
            "pooled": {"robust": pooled, "n": int(robust.size), "wilson95": wilson(pooled, int(robust.size))},
            "seeds": seeds, "coach": _agent_summary(coach, teacher), "c_star": _agent_summary(c_star, teacher),
            "failures": failures,
            "sector_scale": {"student": _sector_scales(list(students.values())), "coach": _sector_scales([coach])}}
