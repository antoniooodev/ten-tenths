"""Run one evaluation episode of a controller (learned policy or teacher) on a protocol specification.

Conditions: unfiltered (U) clips the proposed action to the action box; filtered (F) passes it through the
safety filter. Outcome definitions:

    completed      wrap-corrected progress >= lap length
    reason         first of: completed, filter_exception, environment event, stalled, step_limit
    robust         completed and no environment event and no empty-safe-set step (always 0 in U)
    stall rule     (non-Monza protocols) stop if progress over the last 2000 steps is below 10 m

Per step the runner also accumulates per-axle tyre-utilisation histograms (0.002 bins) for the regions
all / curve (|kappa| >= 1e-3) / tight_curve (|kappa| >= 1e-2) / braking / the 100 m before a failure.

A student policy (``student.json`` in its directory) reads the student observation of its level through
``learning.policy.policy_env``; the teacher, the diagnostics and ``on_step["obs"]`` keep the privileged
observation, ``on_step["student_obs"]`` holds the policy's and ``on_step["imu"]`` the accelerometer channels of the
step; ``on_reset`` receives the same quantities of the reset. The record also sums the traction cost.
``oracle_offset`` adds a constant error to the oracle entries of a diagnostic student (recorded as given; with
``oracle_clip``, recorded too, the entries are clipped to the range an estimate can take), and
``mu_estimator_dir`` replaces the true mu of its oracle entry by the estimate of a friction estimator
(``learning/friction.py``), optionally with its rise limited to ``mu_rise_limit`` per second; ``oracle_mu`` then
summarises the ratio of the mu entry to the true mu over the episode.

``plan_scale`` multiplies the v_ref preview a student reads (a constant scale of the teacher's plan). A plan coach
in ``model_dir`` (``learning/plan_coach.py``) chooses that scale every ``decision_steps`` steps and its executor
policy drives; ``plan_decisions`` lists the decisions as [s, scale]. A sensor plan student
(``learning/plan_student.py``) decides on the student observation and drives with its own friction estimator.
``plan_scale`` in the record summarises the scale over the steps in both cases.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from contextual_bicycle_pacejka.control.safety_filter import filter_action
from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.diagnostics import EpisodeRecorder
from contextual_bicycle_pacejka.learning.friction import FrictionEstimator
from contextual_bicycle_pacejka.learning.policy import PlanCoach, Policy, is_plan_coach, policy_env, policy_layout
from contextual_bicycle_pacejka.spec.constants import DT, V_MIN
from contextual_bicycle_pacejka.spec.layout import student_layout

CURVE_KAPPA_MIN = 1.0e-3  # [1/m]
TIGHT_KAPPA_MIN = 1.0e-2  # [1/m]
PRE_FAILURE_WINDOW_M = 100.0  # [m]
HIST_MAX = 1.3
HIST_BIN = 0.002
HIST_EDGES = np.linspace(0.0, HIST_MAX, int(round(HIST_MAX / HIST_BIN)) + 1)
METRICS = ("af_ratio", "ar_ratio", "fyf_util", "fyr_util", "comb_f", "comb_r")
STALL_WINDOW_STEPS = 2_000
STALL_PROGRESS_M = 10.0
MAX_PROJECTION_RECORDS = 20_000


def progress_increment(previous_s: float, current_s: float, track_length: float) -> float:
    """Arc-length increment [m] between two positions, corrected for the lap wrap."""
    ds = float(current_s - previous_s)
    if ds < -0.5 * track_length:
        ds += track_length
    elif ds > 0.5 * track_length:
        ds -= track_length
    return ds


def run_episode(
    spec: dict[str, Any],
    racing_line: str | Path,
    controller: str,
    filtered: bool,
    model_dir: str | Path | None = None,
    teacher: MATv0 | None = None,
    corner_zones: list[dict] | None = None,
    max_steps: int | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_reset: Callable[[dict[str, Any]], None] | None = None,
    oracle_offset: dict[str, float] | None = None,
    mu_estimator_dir: str | Path | None = None,
    mu_rise_limit: float | None = None,
    oracle_clip: bool = False,
    plan_scale: float = 1.0,
) -> dict[str, Any]:
    """Run one protocol episode and return its outcome record.

    Parameters
    ----------
    spec : dict
        Protocol episode (``block, start_label, start_s, start_vx, mu, seed, max_steps, stall_rule, in_sample``).
    racing_line : str or Path
        Racing-line NPZ of the protocol.
    controller : {"policy", "teacher"}
        Learned policy (or plan coach) in ``model_dir`` or the MATv0 teacher.
    filtered : bool
        Condition F (safety filter) instead of U.
    teacher : MATv0, optional
        Teacher evaluated on every observation (as controller and for the failure diagnostics); default certified.
    corner_zones : list of dict, optional
        Corner zones of the circuit, used to label the failure site.
    max_steps : int, optional
        Override of ``spec["max_steps"]`` (shorter demonstrations).
    on_step : callable, optional
        Called after every step with a dict of the step quantities (used by the demo).
    on_reset : callable, optional
        Called once after the reset with the first observations (``obs``, ``student_obs``, ``imu``).
    oracle_offset : dict, optional
        Constant error added to named oracle entries of a diagnostic student policy.
    mu_estimator_dir : str or Path, optional
        Friction estimator whose estimate replaces the true mu of the policy's oracle entry.
    mu_rise_limit : float, optional
        Largest rise of the estimate per second.
    oracle_clip : bool
        Clip the oracle entries to the range an estimate can take (after the offset).
    plan_scale : float
        Constant scale of the v_ref preview a student policy reads.
    """
    teacher = teacher or MATv0()
    if controller not in ("policy", "teacher"):
        raise ValueError(f"unknown controller {controller!r}")
    coach = PlanCoach(model_dir) if controller == "policy" and is_plan_coach(model_dir) else None
    if coach is not None and plan_scale != 1.0:
        raise ValueError("a plan coach chooses the plan scale itself")
    if coach is not None and coach.mu_estimator is not None:  # a sensor plan student's executor reads its estimator
        if mu_estimator_dir is not None and Path(mu_estimator_dir) != coach.mu_estimator:
            raise ValueError(f"the plan student drives with the estimator {coach.mu_estimator}, not {mu_estimator_dir}")
        mu_estimator_dir = coach.mu_estimator
    policy_dir = coach.executor if coach is not None else model_dir
    layout = policy_layout(policy_dir) if controller == "policy" else student_layout("S0")
    if mu_rise_limit is not None and mu_estimator_dir is None:
        raise ValueError("a rise limit needs a friction estimator")
    estimator = None if mu_estimator_dir is None else FrictionEstimator(mu_estimator_dir, mu_rise_limit)
    env = policy_env(racing_line, layout, oracle_offset=oracle_offset, mu_estimator=estimator, oracle_clip=oracle_clip,
                     plan_scale=plan_scale)
    base = env.unwrapped
    policy = Policy(policy_dir, env) if controller == "policy" else None
    mu, seed = float(spec["mu"]), int(spec["seed"])
    start_s, start_vx = float(spec["start_s"]), float(spec["start_vx"])
    limit = int(max_steps if max_steps is not None else spec["max_steps"])
    setup = randomized_setup(mu, np.random.default_rng(seed))
    student_obs, reset_info = env.reset(seed=seed, options={"setup": setup, "initial_s": start_s,
                                                            "initial_vx": max(V_MIN, start_vx)})
    obs = reset_info.get("privileged_obs", student_obs)
    if on_reset is not None:
        on_reset({"obs": obs, "student_obs": student_obs, "imu": reset_info.get("imu")})
    oracle_mu = [reset_info["oracle"]["mu"]] if "mu" in layout.oracle else []
    traction_cost = 0.0
    track = base.track
    length = float(track.track_length)
    alpha_sat_f, alpha_sat_r = float(reset_info["alpha_sat_f"]), float(reset_info["alpha_sat_r"])
    recorder = EpisodeRecorder(length, (alpha_sat_f, alpha_sat_r), mu, corner_zones)

    previous = np.zeros(2, dtype=np.float64)
    previous_s, current_s, current_vx = start_s, start_s, start_vx
    progress = progress_checkpoint = 0.0
    no_safe = projections = 0
    projection_records: list[list[float]] = []
    flags: dict[str, bool] = {}
    stalled = False
    filter_exception = None
    rows = np.zeros((limit, 13), dtype=np.float32)
    scale, scales, decisions = float(plan_scale), np.full(limit, float(plan_scale)), []
    wall = time.perf_counter()
    step = -1
    try:
        for step in range(limit):
            if coach is not None and step % coach.decision_steps == 0:
                scale = coach.decide(obs, student_obs, mu, scale)
                student_obs = env.set_plan_scale(scale)
                decisions.append([round(float(base.state.s) % length, 2), round(scale, 4)])
            scales[step] = scale
            teacher_action, v_target, _ = teacher.act_with_diagnostics(obs)
            proposed = policy(student_obs) if policy is not None else teacher_action
            empty = projected = False
            if filtered:
                try:
                    result = filter_action(env, proposed, previous)
                except RuntimeError as exc:
                    filter_exception = str(exc)
                    break
                action, projected, empty = result.action, result.projected, result.empty
            else:
                action = np.clip(proposed, env.action_space.low, env.action_space.high).astype(np.float32)
            no_safe += int(empty)
            projections += int(projected)
            s_before = float(base.state.s) % length
            if (projected or empty) and len(projection_records) < MAX_PROJECTION_RECORDS:
                clipped = np.clip(proposed, env.action_space.low, env.action_space.high)
                projection_records.append([step, round(s_before, 2), round(progress, 2), float(action[0] - clipped[0]),
                                           float(action[1] - clipped[1]), int(projected), int(empty)])
            student_obs, _, terminated, truncated, info = env.step(action)
            obs = info.get("privileged_obs", student_obs)
            if oracle_mu:
                oracle_mu.append(info["oracle"]["mu"])
            traction_cost += float(info["traction_cost"])
            state = np.asarray(info["state"], dtype=np.float64)
            current_s, current_vx = float(state[0]), float(state[3])
            progress += progress_increment(previous_s, current_s, length)
            previous_s = current_s
            previous = np.asarray(action, dtype=np.float64)
            diag = info["diag"]
            mu_fz_f, mu_fz_r = mu * float(diag["Fz_f"]), mu * float(diag["Fz_r"])
            fx_f, fx_r, fy_f, fy_r = (float(diag[k]) for k in ("Fx_f", "Fx_r", "Fy_f", "Fy_r"))
            kappa = float(track.kappa(current_s))
            rows[step] = (progress, abs(float(diag["alpha_f"])) / alpha_sat_f,
                          abs(float(diag["alpha_r"])) / alpha_sat_r,
                          abs(fy_f) / mu_fz_f, abs(fy_r) / mu_fz_r, math.hypot(fx_f, fy_f) / mu_fz_f,
                          math.hypot(fx_r, fy_r) / mu_fz_r, abs(kappa), fx_f + fx_r, current_s % length, current_vx,
                          float(projected), float(empty))
            recorder.record(state, action, teacher_action, v_target, kappa, diag, track)
            if on_step is not None:
                on_step({"step": step, "obs": obs, "student_obs": student_obs, "imu": info.get("imu"),
                         "state": state, "action": action, "proposed": proposed,
                         "teacher_action": teacher_action, "v_target": v_target, "progress": progress,
                         "projected": projected, "empty": empty, "diag": diag})
            flags = dict(info.get("event_flags", {}))
            if spec["stall_rule"] and (step + 1) % STALL_WINDOW_STEPS == 0:
                stalled = progress - progress_checkpoint < STALL_PROGRESS_M
                progress_checkpoint = progress
            if progress >= length or terminated or truncated or stalled:
                break
    finally:
        if policy is not None:
            policy.close()
        env.close()

    n = step + 1 if filter_exception is None else step
    rows = rows[: max(n, 0)]
    completed = bool(progress >= length)
    event = next((name for name, active in flags.items() if active), None)
    if completed:
        reason = "completed"
    elif filter_exception is not None:
        reason = "filter_exception"
    elif event is not None:
        reason = event
    elif stalled:
        reason = "stalled"
    else:
        reason = "step_limit"
    record = {
        "block": spec["block"], "start_label": spec["start_label"], "in_sample": bool(spec.get("in_sample", False)),
        "controller": controller, "filtered": bool(filtered), "start_s_m": start_s, "start_vx_mps": start_vx,
        "mu": mu, "seed": seed, "completed": completed,
        "robust": bool(completed and no_safe == 0 and event is None),
        "termination_reason": reason, "events": flags, "steps": n, "lap_time_s": n * DT if completed else None,
        "progress_m": progress, "lap_fraction": min(progress / length, 1.0), "terminal_s_m": current_s % length,
        "terminal_vx_mps": current_vx, "no_safe_steps": no_safe, "projected_steps": projections,
        "projected_fraction": projections / n if n else 0.0, "projection_records": projection_records,
        "alpha_sat_f_rad": alpha_sat_f, "alpha_sat_r_rad": alpha_sat_r, "setup": [float(x) for x in setup],
        "filter_exception": filter_exception, "traction_cost_sum": traction_cost,
        "student_level": layout.level, "oracle_offset": dict(oracle_offset or {}), "oracle_clip": bool(oracle_clip),
        "mu_estimator": None if mu_estimator_dir is None else str(mu_estimator_dir), "mu_rise_limit": mu_rise_limit,
        "oracle_mu": _oracle_summary(oracle_mu, mu),
        "plan_coach": None if coach is None else str(model_dir),
        "plan_scale": {"mean": float(scales[:max(n, 1)].mean()), "min": float(scales[:max(n, 1)].min()),
                       "max": float(scales[:max(n, 1)].max())},
        "plan_decisions": decisions,
    }
    record.update(_utilisation(rows, progress, completed))
    record["diagnostics"] = recorder.summary(completed)
    record["wall_s"] = time.perf_counter() - wall
    return record


def _oracle_summary(values: list[float], mu: float) -> dict[str, float | None]:
    """Ratio of the mu oracle entry to the true mu over an episode (empty without the entry)."""
    if not values:
        return {}
    ratio = np.asarray(values, dtype=np.float64) / mu
    reached = np.flatnonzero(ratio >= 0.95)
    return {"initial": float(ratio[0]), "final": float(ratio[-1]), "mean": float(ratio.mean()),
            "min": float(ratio.min()), "max": float(ratio.max()),
            "fraction_above_1.06": float(np.mean(ratio > 1.06)),
            "time_to_0.95_s": float(reached[0]) * DT if reached.size else None}


def _sparse_histogram(values: np.ndarray) -> dict[str, int]:
    """Sparse histogram {bin index: count} on HIST_EDGES; values beyond the range go to the last bin."""
    if values.size == 0:
        return {}
    clipped = np.minimum(values, HIST_MAX - 0.5 * HIST_BIN)
    counts = np.bincount(np.searchsorted(HIST_EDGES, clipped, side="right") - 1, minlength=len(HIST_EDGES) - 1)
    return {str(i): int(c) for i, c in enumerate(counts) if c}


def _utilisation(rows: np.ndarray, progress: float, completed: bool) -> dict[str, Any]:
    """Tyre-utilisation histograms and maxima by region, plus the step counts of each region."""
    metrics = rows[:, 1:7]
    pre_failure = rows[:, 0] >= progress - PRE_FAILURE_WINDOW_M if not completed else np.zeros(len(rows), dtype=bool)
    masks = {"all": np.ones(len(rows), dtype=bool), "curve": rows[:, 7] >= CURVE_KAPPA_MIN,
             "tight_curve": rows[:, 7] >= TIGHT_KAPPA_MIN, "braking": rows[:, 8] < 0.0, "pre_failure": pre_failure}
    hist: dict[str, dict[str, dict[str, int]]] = {}
    maxima: dict[str, dict[str, float | None]] = {}
    for region, mask in masks.items():
        hist[region] = {name: _sparse_histogram(metrics[mask, j]) for j, name in enumerate(METRICS)}
        maxima[region] = {name: float(metrics[mask, j].max()) if mask.any() else None for j, name in enumerate(METRICS)}
    return {"hist": hist, "max": maxima,
            "projections_in_pre_failure_window": int(np.sum(rows[pre_failure, 11])),
            "empty_in_pre_failure_window": int(np.sum(rows[pre_failure, 12])),
            "steps_curve": int(masks["curve"].sum()), "steps_braking": int(masks["braking"].sum()),
            "steps_pre_failure": int(pre_failure.sum())}
