"""Closed-loop laps of a student policy, logged as training data for the friction estimator.

A friction estimator trained on the teacher's laps only sees the states the teacher visits. In closed loop its
errors move the car elsewhere: an overestimate makes the policy push harder, the states leave the training data and
the estimate rises further (docs/REDESIGN.md, section 13.26). The remedy is the one of DAgger: drive with the
student and label the states it visits. For the estimator the label is the episode friction, always known in
simulation, so no expert is needed.

A lap is driven by a student with the mu oracle entry (level S2 + mu) under one variant:

    variant           the policy reads
    muhat             the estimate of a friction estimator (the deployed loop)
    mu<+/-b>          the true mu plus a constant error b (an error of either sign moves the car off the
                      teacher's pace: -b drives slowly, +b drives into the limit and often fails)

Laps use the setups of the demonstration specifications (the friction estimator's held-out setups stay held out)
and a start drawn per lap, variant and seed: the arc length uniform on the lap and the speed a uniform fraction in
[START_FRACTION_LO, 1] of the teacher's speed there. A lap ends when it is completed, at a termination event, or at
``max_steps``. The kind of a lap is ``<label>_<variant>`` (label ``closed_loop`` by default; a later DAgger iteration
gets its own label, e.g. ``closed_loop2``, so its laps are reported apart). The file written by ``collect`` has the
keys the friction estimator reads from an offline dataset (``observations``, ``imu``, ``actions_exec``,
``episode_ends``, ``mu_per_lap``, ``setup_per_lap``, ``kind_per_lap``, ``circuit_per_lap``) plus the start, the
outcome and the oracle entry the policy read at every step (``oracle_mu``). The circuit is the name of the folder of
the racing line (``data/tracks/<circuit>/``).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.evaluation.episode import progress_increment
from contextual_bicycle_pacejka.learning.friction import FrictionEstimator
from contextual_bicycle_pacejka.learning.policy import Policy, policy_env, policy_layout
from contextual_bicycle_pacejka.spec.constants import V_MIN, VREF_REFERENCE_MU

START_FRACTION_LO = 0.5
MAX_STEPS = 30_000  # 300 s, about twice a lap of Monza


DEFAULT_LABEL = "closed_loop"


def variant_kind(variant: str, label: str = DEFAULT_LABEL) -> str:
    """Kind label of the laps of one variant (``kind_per_lap``)."""
    return f"{label}_{variant}"


def parse_variant(variant: str) -> tuple[float | None, bool]:
    """(oracle offset, uses the estimator) of a variant name: ``muhat`` or ``mu<+/-b>``."""
    if variant == "muhat":
        return None, True
    if variant.startswith("mu") and variant[2:3] in "+-" and len(variant) > 3:
        return float(variant[2:]), False
    raise ValueError(f"unknown variant {variant!r}; expected 'muhat' or 'mu<+/-b>', e.g. 'mu+0.10'")


def lap_start(racing_line: str | Path, mu: float, seed: int) -> tuple[float, float]:
    """Start arc length [m] and speed [m/s] of one lap (see the module docstring)."""
    with np.load(racing_line, allow_pickle=False) as data:
        length = float(data["track_length"])
    rng = np.random.default_rng(seed)
    start_s = float(rng.uniform(0.0, length))
    fraction = float(rng.uniform(START_FRACTION_LO, 1.0))
    return start_s, start_speed(racing_line, start_s, mu, fraction)


def start_speed(racing_line: str | Path, start_s: float, mu: float, fraction: float) -> float:
    """Start speed [m/s] at ``start_s``: ``fraction`` of the teacher's speed there, at least V_MIN."""
    with np.load(racing_line, allow_pickle=False) as data:
        s_m = np.asarray(data["s_m"], dtype=np.float64)
        v_ref = np.asarray(data["v_ref_mps"], dtype=np.float64)
        length = float(data["track_length"])
    v_teacher = float(np.interp(start_s, s_m, v_ref, period=length)) * math.sqrt(mu / VREF_REFERENCE_MU)
    return max(fraction * v_teacher * MATv0().speed_safety, V_MIN)


def closed_loop_lap(job: dict[str, Any]) -> dict[str, Any]:
    """Drive one lap of a job (policy, variant, setup, start); return the logged channels and the outcome."""
    offset, uses_estimator = parse_variant(job["variant"])
    layout = policy_layout(job["policy_dir"])
    if "mu" not in layout.oracle:
        raise ValueError(f"policy {job['policy_dir']} has no mu oracle entry")
    estimator = FrictionEstimator(job["estimator_dir"]) if uses_estimator else None
    env = policy_env(job["racing_line"], layout, max_episode_steps=job["max_steps"],
                     oracle_offset=None if offset is None else {"mu": offset}, mu_estimator=estimator)
    policy = Policy(job["policy_dir"], env)
    log: dict[str, list] = {"obs": [], "imu": [], "exec": [], "oracle": []}
    terminated = truncated = False
    try:
        student, info = env.reset(seed=job["reset_seed"], options={"setup": np.asarray(job["setup"]),
                                                                   "initial_s": job["start_s"],
                                                                   "initial_vx": job["start_vx"]})
        length = float(env.unwrapped.track.track_length)
        progress, previous_s = 0.0, job["start_s"]
        for _ in range(job["max_steps"]):
            action = np.clip(policy(student), env.action_space.low, env.action_space.high).astype(np.float32)
            log["obs"].append(info["privileged_obs"])
            log["imu"].append(info["imu"])
            log["exec"].append(action)
            log["oracle"].append(info["oracle"]["mu"])
            student, _, terminated, truncated, info = env.step(action)
            current_s = float(info["state"][0])
            progress += progress_increment(previous_s, current_s, length)
            previous_s = current_s
            if progress >= length or terminated or truncated:
                break
    finally:
        policy.close()
        env.close()
    return {"obs": np.asarray(log["obs"], dtype=np.float32), "imu": np.asarray(log["imu"], dtype=np.float32),
            "exec": np.asarray(log["exec"], dtype=np.float32), "oracle": np.asarray(log["oracle"], dtype=np.float32),
            "mu": job["mu"], "setup": np.asarray(job["setup"], dtype=np.float64),
            "kind": variant_kind(job["variant"], job["label"]), "circuit": Path(job["racing_line"]).parent.name,
            "start_s": job["start_s"], "start_vx": job["start_vx"], "terminated": bool(terminated),
            "completed": bool(progress >= length)}


def collect(racing_line: str | Path, policy_dir: str | Path, specs: list[Any], variants: list[str],
            output: str | Path, estimator_dir: str | Path | None = None, seed: int = 0, workers: int = 1,
            max_steps: int = MAX_STEPS, label: str = DEFAULT_LABEL) -> dict[str, Any]:
    """Drive one lap per specification and variant; write the logged laps to ``output`` (npz) and return a summary.

    ``specs`` are demonstration specifications (``demonstrations.episode_specs``); ``estimator_dir`` is needed by
    the ``muhat`` variant; ``label`` prefixes the kind of the laps.
    """
    jobs = []
    for v, variant in enumerate(variants):
        _, uses_estimator = parse_variant(variant)
        if uses_estimator and estimator_dir is None:
            raise ValueError("the 'muhat' variant needs a friction estimator")
        for spec in specs:
            start_s, start_vx = lap_start(racing_line, spec.mu, seed + 1000 * v + int(spec.lap))
            jobs.append({"racing_line": str(racing_line), "policy_dir": str(policy_dir), "variant": variant,
                         "estimator_dir": None if estimator_dir is None else str(estimator_dir),
                         "setup": np.asarray(spec.setup).tolist(), "mu": float(spec.mu), "reset_seed": spec.reset_seed,
                         "start_s": start_s, "start_vx": start_vx, "max_steps": int(max_steps), "label": label})
    if workers <= 1:
        laps = [closed_loop_lap(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            laps = list(pool.map(closed_loop_lap, jobs))
    arrays = assemble(laps)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    summary: dict[str, Any] = {}
    for variant in variants:
        kind = [lap for lap in laps if lap["kind"] == variant_kind(variant, label)]
        summary[variant] = {"laps": len(kind), "completed": sum(lap["completed"] for lap in kind),
                            "terminated": sum(lap["terminated"] for lap in kind),
                            "steps": sum(len(lap["obs"]) for lap in kind)}
    return summary


def assemble(laps: list[dict[str, Any]]) -> dict[str, NDArray]:
    """Concatenate closed-loop laps into the arrays written by ``collect``."""
    return {
        "observations": np.concatenate([lap["obs"] for lap in laps]),
        "imu": np.concatenate([lap["imu"] for lap in laps]),
        "actions_exec": np.concatenate([lap["exec"] for lap in laps]),
        "oracle_mu": np.concatenate([lap["oracle"] for lap in laps]),
        "episode_ends": np.cumsum([len(lap["obs"]) for lap in laps]).astype(np.int64),
        "mu_per_lap": np.asarray([lap["mu"] for lap in laps], dtype=np.float64),
        "setup_per_lap": np.asarray([lap["setup"] for lap in laps], dtype=np.float64),
        "kind_per_lap": np.asarray([lap["kind"] for lap in laps]),
        "circuit_per_lap": np.asarray([lap["circuit"] for lap in laps]),
        "start_s_per_lap": np.asarray([lap["start_s"] for lap in laps], dtype=np.float64),
        "start_vx_per_lap": np.asarray([lap["start_vx"] for lap in laps], dtype=np.float64),
        "terminated_per_lap": np.asarray([lap["terminated"] for lap in laps]),
        "completed_per_lap": np.asarray([lap["completed"] for lap in laps]),
    }
