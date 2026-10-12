"""Speed profile fitted by probing the plant with the teacher, and the scaled racing lines R0.

The profile is v(s) = scale(s) v_point(s), with v_point the point-mass profile of
``minimum_curvature_line.npz`` and scale starting at 0.7. Each iteration drives one lap per friction
mu in {0.8, 0.9, 1.0, 1.1, 1.2} (randomised setup, setup seed 42 + 5 iteration + index) with the teacher at k = 1
and no extensions, through the safety filter, starting at the slowest point at 0.9 v. Per arc-length bin it records
the worst rear-slip ratio |alpha_r| / (1.25 alpha_sat_r).

    all laps robust  ->  scale += 0.05 (0.65 - ratio)        in the bins with ratio < 0.65
    otherwise        ->  for every failed lap, scale -= 0.02 w(d) in the window 800 m before to 30 m after the
                         terminal point, w rising linearly to 1 at the terminal point
    then             ->  50 passes of the periodic [1/4, 1/2, 1/4] smoother, clip to [0.1, 1]

The fastest robust iteration is kept (``speed_profile_probed.npz``, history in ``speed_profile_probe_log.json``).
The racing line R0 used for training and evaluation is that profile scaled by k = 0.9 (Monza: 0.83 of its probed
profile). Every rewrite recomputes ax_i = (v_{i+1}^2 - v_i^2) / (2 ds_i) with periodic forward differences.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.control.safety_filter import filter_action
from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.spec.constants import DT, SLIP_TERMINATION_RATIO, V_MIN

INITIAL_SCALE = 0.70
LOWER_STEP = 0.02
REAR_TARGET_RATIO = 0.65
RAISE_STEP = 0.05
BACKWARD_WINDOW_M = 800.0
FORWARD_WINDOW_M = 30.0
SMOOTHING_PASSES = 50
SCALE_BOUNDS = (0.1, 1.0)
PROBE_MUS = (0.8, 0.9, 1.0, 1.1, 1.2)
PROBE_ITERATIONS = 8
PROBE_SEED = 42
PROBE_MAX_STEPS = 60_000
START_SPEED_FACTOR = 0.9
PROBE_LOG_FILE = "speed_profile_probe_log.json"
R0_FACTOR = {"monza": 0.83, "spa": 0.9, "catalunya": 0.9, "budapest": 0.9}


@dataclass(frozen=True)
class ProbeLap:
    """Outcome of one probe lap."""

    completed: bool
    terminal_s_m: float
    time_s: float
    empty_steps: int
    rear_ratio_by_bin: NDArray[np.float64]  # worst rear-slip ratio per arc-length bin (NaN where not visited)


def write_profile(source: str | Path, output: str | Path, speed: NDArray[np.float64]) -> None:
    """Copy a track NPZ with its speed profile replaced by ``speed`` [m/s] (and ax recomputed)."""
    with np.load(source, allow_pickle=False) as data:
        payload: dict[str, Any] = {key: data[key] for key in data.files}
    s = np.asarray(payload["s_m"], dtype=np.float64)
    ds = np.diff(np.r_[s, float(payload["track_length"])])
    ax = np.diff(np.r_[speed * speed, speed[0] * speed[0]]) / (2.0 * ds)
    payload.update({"v_ref_mps": speed, "v_ref": speed, "ax_mps2": ax, "ax": ax,
                    "speed_profile_method": np.asarray("bicycle_axle_probe")})
    np.savez(output, **payload)


def scale_profile(source: str | Path, output: str | Path, factor: float) -> Path:
    """Write ``source`` with its speed profile multiplied by ``factor`` (racing line R0 from the probed profile)."""
    with np.load(source, allow_pickle=False) as data:
        speed = np.asarray(data["v_ref_mps"], dtype=np.float64)
    write_profile(source, output, factor * speed)
    return Path(output)


def smooth(scale: NDArray[np.float64]) -> NDArray[np.float64]:
    result = scale.copy()
    for _ in range(SMOOTHING_PASSES):
        result = 0.25 * np.roll(result, 1) + 0.5 * result + 0.25 * np.roll(result, -1)
    return result


def update_scale(scale: NDArray[np.float64], s: NDArray[np.float64], length: float, lap: ProbeLap,
                 rear_target_ratio: float = REAR_TARGET_RATIO, raise_step: float = RAISE_STEP,
                 ) -> NDArray[np.float64]:
    """Raise the scale where the rear axle has slack (robust lap) or lower it before the failure point."""
    updated = scale.copy()
    if lap.completed and lap.empty_steps == 0:
        ratio = lap.rear_ratio_by_bin
        slack = np.isfinite(ratio) & (ratio < rear_target_ratio)
        updated[slack] += raise_step * (rear_target_ratio - ratio[slack])
    else:
        delta = (s - lap.terminal_s_m + 0.5 * length) % length - 0.5 * length
        affected = (delta >= -BACKWARD_WINDOW_M) & (delta <= FORWARD_WINDOW_M)
        weight = np.where(delta <= 0.0, 1.0 + delta / BACKWARD_WINDOW_M, 1.0 - delta / FORWARD_WINDOW_M)
        updated[affected] -= LOWER_STEP * np.maximum(weight[affected], 0.0)
    return np.clip(smooth(updated), *SCALE_BOUNDS)


def probe_lap(profile: str | Path, mu: float, setup_seed: int, max_steps: int = PROBE_MAX_STEPS) -> ProbeLap:
    """Drive one filtered teacher lap (k = 1, no extensions) on ``profile`` from its slowest point."""
    with np.load(profile, allow_pickle=False) as data:
        s_grid = np.asarray(data["s_m"], dtype=np.float64)
        speed = np.asarray(data["v_ref_mps"], dtype=np.float64)
        length = float(data["track_length"])
    start = int(np.argmin(speed))
    env = make_env(profile, max_episode_steps=max_steps)
    teacher = MATv0(speed_safety=1.0, max_speed=100.0, T=False, L=False, Y=False)
    obs, reset_info = env.reset(seed=setup_seed, options={
        "setup": randomized_setup(mu, np.random.default_rng(setup_seed)), "initial_s": float(s_grid[start]),
        "initial_vx": max(START_SPEED_FACTOR * float(speed[start]), V_MIN)})
    rear_limit = SLIP_TERMINATION_RATIO * float(reset_info["alpha_sat_r"])
    rear_ratio = np.full(s_grid.size, np.nan)
    previous = np.zeros(2, dtype=np.float64)
    previous_s = terminal_s = float(s_grid[start])
    progress, empty, steps = 0.0, 0, 0
    try:
        for step in range(max_steps):
            result = filter_action(env, teacher.act(obs), previous)
            empty += int(result.empty)
            steps = step + 1
            obs, _, terminated, truncated, info = env.step(result.action)
            previous = np.asarray(result.action, dtype=np.float64)
            terminal_s = float(np.asarray(info["state"])[0]) % length
            ds = terminal_s - previous_s
            if ds < -0.5 * length:
                ds += length
            elif ds > 0.5 * length:
                ds -= length
            progress += max(ds, 0.0)
            previous_s = terminal_s
            index = int(np.searchsorted(s_grid, terminal_s, side="right") - 1) % s_grid.size
            ratio = abs(float(info["diag"]["alpha_r"])) / rear_limit
            rear_ratio[index] = ratio if not np.isfinite(rear_ratio[index]) else max(rear_ratio[index], ratio)
            if progress >= length:
                return ProbeLap(True, terminal_s, steps * DT, empty, rear_ratio)
            if terminated or truncated:
                break
    finally:
        env.close()
    return ProbeLap(False, terminal_s, steps * DT, empty, rear_ratio)


def fit_speed_profile(source: str | Path, output: str | Path, iterations: int = PROBE_ITERATIONS,
                      mus: tuple[float, ...] = PROBE_MUS, seed: int = PROBE_SEED) -> dict[str, Any]:
    """Fit the probed speed profile of ``source`` (``minimum_curvature_line.npz``); write the fastest robust one
    to ``output``."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        s = np.asarray(data["s_m"], dtype=np.float64)
        point_speed = np.asarray(data["v_ref_mps"], dtype=np.float64)
        length = float(data["track_length"])
    scale = np.full_like(point_speed, INITIAL_SCALE)
    best_scale, best_time = scale.copy(), float("inf")
    history = []
    with tempfile.TemporaryDirectory(prefix="speed_profile_") as tmp:
        candidate = Path(tmp) / "candidate.npz"
        for iteration in range(iterations):
            write_profile(source, candidate, point_speed * scale)
            laps = [probe_lap(candidate, mu, seed + iteration * len(mus) + index) for index, mu in enumerate(mus)]
            robust = all(lap.completed and lap.empty_steps == 0 for lap in laps)
            ratios = np.vstack([np.where(np.isfinite(lap.rear_ratio_by_bin), lap.rear_ratio_by_bin, -np.inf)
                                for lap in laps])
            worst = np.max(ratios, axis=0)
            worst[~np.isfinite(worst)] = np.nan
            mean_time = float(np.mean([lap.time_s for lap in laps]))
            history.append({"iteration": iteration, "robust": robust, "completed": sum(lap.completed for lap in laps),
                            "time_s_mean": mean_time, "terminal_s_m": [lap.terminal_s_m for lap in laps],
                            "empty_steps": sum(lap.empty_steps for lap in laps),
                            "scale_mean": float(np.mean(scale)), "scale_min": float(np.min(scale)),
                            "scale_max": float(np.max(scale)),
                            "rear_ratio_max": float(np.nanmax(worst)) if np.any(np.isfinite(worst)) else None})
            if robust and mean_time < best_time:
                best_scale, best_time = scale.copy(), mean_time
            if robust:
                scale = update_scale(scale, s, length, ProbeLap(True, 0.0, mean_time, 0, worst))
            else:
                for lap in laps:
                    if not lap.completed or lap.empty_steps:
                        scale = update_scale(scale, s, length, lap)
    write_profile(source, output, point_speed * best_scale)
    report = {"source": Path(source).name, "history": history, "best_time_s": best_time}
    (output.parent / PROBE_LOG_FILE).write_text(json.dumps(report, indent=2))
    return report
