"""Teacher demonstrations with perturbed execution (the training data of behaviour cloning).

Episode specifications (20 laps): lap i uses the setup ``randomized_setup(mu_i, rng)`` drawn sequentially from
``rng = default_rng(42)``, a friction mu_i from a seeded balanced permutation of {0.8, 0.9, 1.0, 1.1, 1.2}
(4 laps each), the reset seed 42 + i, and starts at the minimum of the circuit's v_ref profile with speed
max(0.9 v_ref_min, V_MIN) sqrt(mu_i / 0.9). The same 20 (setup, mu) pairs are used on every circuit.

One demonstration lap: at every step the teacher is evaluated twice,

    label_t     = filter(MATv0(o_t))      the stored target (teacher on the true observation)
    executed_t  = filter(MATv0'(o'_t))    the applied action (teacher on a perturbed observation)

with ``filter`` the safety filter (previous executed action as rate reference) and o'_t equal to o_t except

    speed        : v_ref preview * (1 + d_t),  d_t Ornstein-Uhlenbeck (tau_c = 2 s, sigma 0.08, clipped to +/- 0.16)
    lateral      : ey - n_t,                   n_t Ornstein-Uhlenbeck (tau_c = 2 s, sigma 0.25 m, clipped to +/- 0.5 m)
    speed_wide   : as speed, sigma 0.16 clipped to +/- 0.32
    lateral_wide : as lateral, sigma 0.5 m clipped to +/- 1.0 m
    none         : unperturbed (the clean teacher lap)

The vehicle therefore visits off-nominal states and the label is the teacher's correction there. MATv0' is the
labelling teacher unless an execution teacher is given (``exec_teacher``): a teacher with another speed factor k
drives the lap at another speed while the labels stay those of the labelling teacher (mixed-speed laps). The wide
kinds draw their noise from their own seeds, so they are not scaled copies of the narrow ones.

A lap is kept ("healthy") if it completes with no termination event and no empty-safe-set step; otherwise it is
repeated with the next noise draw, up to 10 attempts. A lap without noise (kind ``none``) is deterministic and gets
a single attempt. With ``keep_failed`` the attempts that are not kept are saved as well: they are the failures of
the offline datasets.

Per step a lap also logs what offline reinforcement learning and the student observation need (``datasets.py``):

    imu             accelerometer channels of the observation (zeros at the first step)
    reward          reward of the transition, with OFFLINE_REWARD_WEIGHTS (no lateral and no traction term)
    traction_cost   traction cost of the transition (weighted at training time)
    label_projected the safety filter replaced the teacher's action in the label

and, once per lap, the observation and the accelerometer channels after the last step (``final_obs``,
``final_imu``), the next state of the last transition. ``summary["terminated"]`` tells a failure (a terminal
transition) from a lap that ended by completion or by the step limit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.control.safety_filter import filter_action
from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import progress_increment
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import DT, V_MIN, VREF_REFERENCE_MU
from contextual_bicycle_pacejka.spec.layout import OBS

N_LAPS = 20
SPEC_SEED = 42
DEMO_MUS = (0.8, 0.9, 1.0, 1.1, 1.2)
START_SPEED_FACTOR = 0.9  # [-] start speed as a fraction of the minimum of v_ref
TAU_CORRELATION_S = 2.0  # [s] Ornstein-Uhlenbeck correlation time
MAX_ATTEMPTS = 10
PERTURBATIONS: dict[str, dict[str, float]] = {
    "speed": {"speed_std": 0.08, "speed_clip": 0.16, "lateral_std": 0.0, "lateral_clip": 0.5},
    "lateral": {"speed_std": 0.0, "speed_clip": 0.16, "lateral_std": 0.25, "lateral_clip": 0.5},
    "speed_wide": {"speed_std": 0.16, "speed_clip": 0.32, "lateral_std": 0.0, "lateral_clip": 1.0},
    "lateral_wide": {"speed_std": 0.0, "speed_clip": 0.32, "lateral_std": 0.5, "lateral_clip": 1.0},
    "none": {"speed_std": 0.0, "speed_clip": 0.16, "lateral_std": 0.0, "lateral_clip": 0.5},
}
# Offset of the noise seeds of a kind (the narrow kinds keep the seeds of the original demonstrations).
NOISE_SEED_OFFSET = {"speed_wide": 100_000, "lateral_wide": 100_000}
# Circuit index used to offset the noise seeds, so that the same lap gets different noise on each circuit.
CIRCUIT_INDEX = {"monza": 0, "spa": 1, "catalunya": 2, "budapest": 3}


@dataclass(frozen=True)
class LapSpec:
    """Initial condition of one demonstration lap."""

    lap: int
    setup: NDArray[np.float64]
    mu: float
    start_s: float  # [m]
    start_vx: float  # [m/s]
    reset_seed: int


def start_point(racing_line: str | Path) -> tuple[float, float]:
    """Start of the demonstration laps: arc length of the v_ref minimum [m] and the start speed at mu = 0.9 [m/s]."""
    with np.load(racing_line, allow_pickle=False) as data:
        s_m = np.asarray(data["s_m"], dtype=np.float64)
        v_ref = np.asarray(data["v_ref_mps"], dtype=np.float64)
    idx = int(np.argmin(v_ref))
    start_s = float(s_m[idx]) % RealTrack(str(racing_line)).track_length
    return start_s, max(float(v_ref[idx]) * START_SPEED_FACTOR, V_MIN)


def episode_specs(racing_line: str | Path, n_laps: int = N_LAPS, seed: int = SPEC_SEED,
                  mus: tuple[float, ...] = DEMO_MUS) -> list[LapSpec]:
    """Specifications of the demonstration laps on one racing line (see the module docstring)."""
    start_s, base_vx = start_point(racing_line)
    schedule = np.random.default_rng(seed).permutation(np.resize(np.asarray(mus, dtype=np.float64), n_laps))
    setup_rng = np.random.default_rng(seed)
    specs = []
    for lap in range(n_laps):
        mu = float(schedule[lap])
        specs.append(LapSpec(lap, randomized_setup(mu, setup_rng), mu, start_s,
                             max(base_vx * math.sqrt(mu / VREF_REFERENCE_MU), V_MIN), seed + lap))
    return specs


def run_lap(racing_line: str | Path, spec: LapSpec, kind: str, attempt: int, circuit: str,
            teacher: MATv0 | None = None, max_steps: int | None = None, exec_teacher: MATv0 | None = None,
            ) -> tuple[dict[str, Any], dict[str, NDArray]]:
    """Drive one perturbed demonstration lap; return (summary, per-step arrays).

    ``teacher`` gives the labels and ``exec_teacher`` (default: ``teacher``) the executed actions.
    """
    teacher = teacher or MATv0()
    exec_teacher = exec_teacher or teacher
    noise = PERTURBATIONS[kind]
    if max_steps is None:
        env = make_env(racing_line, reward_weights=OFFLINE_REWARD_WEIGHTS)
    else:
        env = make_env(racing_line, max_episode_steps=max_steps, reward_weights=OFFLINE_REWARD_WEIGHTS)
    base = env.unwrapped
    length = float(base.track.track_length)
    obs, info = env.reset(seed=spec.reset_seed, options={"setup": spec.setup, "initial_s": spec.start_s,
                                                         "initial_vx": spec.start_vx})
    imu = info["imu"]
    offset = 10_000 * CIRCUIT_INDEX[circuit] + NOISE_SEED_OFFSET.get(kind, 0)
    rng_speed = np.random.default_rng(1000 + spec.lap + 100 * attempt + offset)
    rng_lateral = np.random.default_rng(5000 + spec.lap + 100 * attempt + offset)
    decay = math.exp(-DT / TAU_CORRELATION_S)
    gain_speed = noise["speed_std"] * math.sqrt(1.0 - decay * decay)
    gain_lateral = noise["lateral_std"] * math.sqrt(1.0 - decay * decay)
    perturbed = kind != "none"
    separate = perturbed or exec_teacher is not teacher
    speed_noise = lateral_noise = 0.0
    previous = np.zeros(2, dtype=np.float64)
    progress, previous_s = 0.0, spec.start_s
    rec: dict[str, list] = {k: [] for k in ("obs", "label", "exec", "speed_noise", "lateral_noise", "s",
                                             "label_empty", "exec_projected", "imu", "reward", "traction_cost",
                                             "label_projected")}
    empty_steps = 0
    done = completed = terminated = False
    step = 0
    try:
        while not done:
            label_result = filter_action(env, teacher.act_with_diagnostics(obs)[0], previous)
            if separate:
                obs_perturbed = np.array(obs, copy=True)
                obs_perturbed[OBS.VREF_START:OBS.VREF_END] *= 1.0 + speed_noise
                obs_perturbed[OBS.EY] = obs_perturbed[OBS.EY] - lateral_noise
                executed = filter_action(env, exec_teacher.act_with_diagnostics(obs_perturbed)[0], previous)
            else:
                executed = label_result
            empty_steps += int(executed.empty)
            rec["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            rec["label"].append(np.asarray(label_result.action).copy())
            rec["exec"].append(np.asarray(executed.action, dtype=np.float32).copy())
            rec["speed_noise"].append(speed_noise)
            rec["lateral_noise"].append(lateral_noise)
            rec["s"].append(float(base.state.s) % length)
            rec["label_empty"].append(label_result.empty)
            rec["exec_projected"].append(executed.projected if separate else False)
            rec["imu"].append(np.asarray(imu, dtype=np.float32).copy())
            rec["label_projected"].append(label_result.projected)
            obs, reward, terminated, truncated, info = env.step(executed.action)
            rec["reward"].append(reward)
            rec["traction_cost"].append(info["traction_cost"])
            imu = info["imu"]
            previous = np.asarray(executed.action, dtype=np.float64)
            s_now = float(np.asarray(info["state"])[0])
            progress += progress_increment(previous_s, s_now, length)
            previous_s = s_now
            step += 1
            if progress >= length:
                completed = done = True
            if terminated or truncated:
                done = True
            if perturbed:
                speed_noise = float(np.clip(decay * speed_noise + gain_speed * rng_speed.standard_normal(),
                                            -noise["speed_clip"], noise["speed_clip"]))
                lateral_noise = float(np.clip(decay * lateral_noise + gain_lateral * rng_lateral.standard_normal(),
                                              -noise["lateral_clip"], noise["lateral_clip"]))
    finally:
        env.close()
    events = {k: bool(v) for k, v in dict(info.get("event_flags", {})).items()}
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    arrays["imu"] = arrays["imu"].reshape(-1, 2).astype(np.float32)
    arrays["reward"] = arrays["reward"].astype(np.float32)
    arrays["traction_cost"] = arrays["traction_cost"].astype(np.float32)
    arrays["final_obs"] = np.asarray(obs, dtype=np.float32).copy()
    arrays["final_imu"] = np.asarray(imu, dtype=np.float32).copy()
    summary = {"circuit": circuit, "kind": kind, "lap": spec.lap, "attempt": attempt, "mu": spec.mu, "steps": step,
               "completed": completed, "events": events, "empty_steps": empty_steps,
               "healthy": bool(completed and not any(events.values()) and empty_steps == 0), "progress_m": progress,
               "projected_steps": int(arrays["exec_projected"].sum()), "terminal_s": float(previous_s % length),
               "terminated": bool(terminated), "speed_safety": float(teacher.speed_safety),
               "exec_speed_safety": float(exec_teacher.speed_safety)}
    return summary, arrays


def attempt_file(lap: int, attempt: int) -> str:
    """File name of a saved attempt that was not kept."""
    return f"lap{lap:02d}_attempt{attempt:02d}.npz"


def generate_laps(racing_line: str | Path, circuit: str, kind: str, out_dir: str | Path,
                  specs: list[LapSpec], laps: list[int] | None = None, teacher: MATv0 | None = None,
                  keep_failed: bool = False, max_steps: int | None = None, exec_teacher: MATv0 | None = None,
                  max_attempts: int | None = None) -> dict[str, Any]:
    """Generate demonstration laps with retries; save every kept lap as ``lapNN.npz`` and return a report.

    With ``keep_failed`` every attempt that is not kept is saved as ``attempt_file(lap, attempt)`` and listed in
    the report under ``failed_files``. ``exec_teacher`` drives the laps (``run_lap``). ``max_attempts`` (default
    ``MAX_ATTEMPTS``) bounds the attempts per lap (a wide perturbation can fail them all). ``max_steps`` shortens
    the laps (tests).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for spec in specs if laps is None else [specs[i] for i in laps]:
        attempts, failed_files = [], []
        for attempt in range(MAX_ATTEMPTS if max_attempts is None else max_attempts):
            summary, arrays = run_lap(racing_line, spec, kind, attempt, circuit, teacher, max_steps, exec_teacher)
            attempts.append(summary)
            if summary["healthy"]:
                np.savez_compressed(out_dir / f"lap{spec.lap:02d}.npz", **arrays, meta=json.dumps(summary))
                break
            if keep_failed:
                name = attempt_file(spec.lap, attempt)
                np.savez_compressed(out_dir / name, **arrays, meta=json.dumps(summary))
                failed_files.append(name)
            if kind == "none":
                break
        entry = {"lap": spec.lap, "attempts": attempts,
                 "kept_attempt": attempts[-1]["attempt"] if attempts[-1]["healthy"] else None}
        if keep_failed:
            entry["failed_files"] = failed_files
        report.append(entry)
    (out_dir / "generation.json").write_text(json.dumps(report, indent=1))
    return {"laps": report}
