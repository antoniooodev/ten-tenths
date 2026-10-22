"""Mean action against the stochastic policy: protocol laps with a fraction of the action noise, and clipping.

    python scripts/diagnose_mean_action.py
    python scripts/diagnose_mean_action.py --scales 0 1 --workers 4

For every model of ``MODELS`` (b = 0, free stage) and every noise scale k of ``--scales``, the 10 protocol laps
driven with clip(mu + k sigma eps) (evaluation/action_noise.py): once for k = 0 (the mean action), ``--repeats``
times otherwise, the noise seeded with 1000 seed + 100 (index of k) + repeat. Every lap is compared with the
minimum-time lap from the same standing start (``--standing``) and carries the clipping measurements: the share of
decisions with the mean action beyond [-1, 1] and with the longitudinal command tau at its bound, and the bias
E[clip(mu + sigma eps)] - clip(mu) of the training noise, over the lap and over the ``BEFORE_FAILURE_M`` before a
failure.

Writes <output>/laps.json (one row per lap, the wall and CPU time of every run and of the whole diagnosis) and, per
decision, <output>/traces/<model>_s<seed>_k<k>_r<repeat>.npz for the mean action of every model and for every scale
of the models of ``TRACED`` (columns ``TRACE_COLUMNS``, plus the lap index).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_stage import CATALUNYA, protocol_env
from stable_baselines3 import PPO

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.evaluation.action_noise import ScaledNoisePolicy
from contextual_bicycle_pacejka.evaluation.laps import drive_lap, protocol_starts
from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.model.tires import (
    compute_saturation_threshold,
    compute_slip_angles,
)
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import V_MIN

STAGE = 7
SEEDS = (10, 11, 12, 13, 14)
MODELS = tuple((tag, "stage7", seed) for tag in ("nominal_limit", "nominal_limit_cont") for seed in SEEDS) + (
    ("nominal_limit", "stage7_300M", 10),)
TRACED = (("nominal_limit_cont", "stage7", 14), ("nominal_limit", "stage7_300M", 10))
SCALES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
BEFORE_FAILURE_M = 200.0  # [m] stretch before a failure where the clipping is measured on its own
STATE_COLUMNS = ("s", "vx", "ey", "delta", "tau", "slip_f", "slip_r")
TRACE_COLUMNS = STATE_COLUMNS + ("mu_delta", "mu_tau", "sigma_delta", "sigma_tau", "bias_delta", "bias_tau")


def model_name(tag: str, model: str) -> str:
    return tag if model == "stage7" else f"{tag}_{model.removeprefix('stage7_')}"


class StateRecorder:
    """Observer of ``drive_lap``: one row of ``STATE_COLUMNS`` after the reset and after every decision."""

    def __init__(self) -> None:
        self.rows: list[list[float]] = []
        self.saturation: tuple[float, float] | None = None

    def __call__(self, env: RacingEnv) -> None:
        if self.saturation is None:
            self.saturation = (compute_saturation_threshold(env.setup, "front"),
                               compute_saturation_threshold(env.setup, "rear"))
        state: VehicleState = env.state
        slip = (0.0, 0.0)
        if state.vx >= V_MIN:
            alpha_f, alpha_r = compute_slip_angles(state)
            slip = (abs(alpha_f) / self.saturation[0], abs(alpha_r) / self.saturation[1])
        self.rows.append([state.s % env.track.track_length, state.vx, state.ey, state.delta, env.tau, *slip])


def clipping(rows: np.ndarray) -> dict[str, list[float]]:
    """Clipping measurements over trace rows (``TRACE_COLUMNS``), per channel (steering, longitudinal)."""
    if len(rows) == 0:
        return {}
    mu, bias = rows[:, 7:9], rows[:, 11:13]
    return {
        "mean_beyond_bound": np.mean(np.abs(mu) > 1.0, axis=0).tolist(),
        "tau_at_bound": float(np.mean(np.abs(rows[:, 4]) >= 1.0 - 1e-9)),
        "bias_mean_abs": np.mean(np.abs(bias), axis=0).tolist(),
        "bias_max_abs": np.max(np.abs(bias), axis=0).tolist(),
    }


def drive(job: tuple[str, str, int, int, float, int, list[float], Path | None]) -> dict[str, Any]:
    """Drive the protocol laps of one model at one noise scale and repeat (in a worker process)."""
    tag, model_file, seed, scale_index, scale, repeat, reference, traces = job
    wall, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    track = RealTrack(CATALUNYA)
    model = PPO.load(Path("outputs/models") / tag / f"s{seed}" / f"{model_file}.zip", device="cpu")
    env = protocol_env(model, track, STAGE, 0.0)
    policy = ScaledNoisePolicy(model, scale, np.random.default_rng(1000 * seed + 100 * scale_index + repeat))
    laps, trace_rows = [], []
    for index, start in enumerate(protocol_starts(track)):
        recorder = StateRecorder()
        record = drive_lap(policy, env, start, observer=recorder)
        decisions = policy.reset_records()
        rows = np.hstack([np.array(recorder.rows)[: len(decisions)], decisions[: len(recorder.rows)]])
        lap = {"start_s": start, "completed": record.completed, "lap_time": record.lap_time,
               "gap": None if record.lap_time is None else (record.lap_time - reference[index]) / reference[index],
               "failure": record.failure, "failure_s": record.failure_s, "decisions": record.decisions,
               "clipping": clipping(rows)}
        if record.failure_s is not None:
            behind = (record.failure_s - rows[:, 0]) % track.track_length
            lap["clipping_before_failure"] = clipping(rows[behind <= BEFORE_FAILURE_M])
        laps.append(lap)
        trace_rows.append(np.hstack([rows, np.full((len(rows), 1), index)]))
    if traces is not None:
        traces.mkdir(parents=True, exist_ok=True)
        data = np.vstack(trace_rows)
        np.savez_compressed(traces / f"{model_name(tag, model_file)}_s{seed}_k{scale:g}_r{repeat}.npz",
                            **{name: data[:, i] for i, name in enumerate(TRACE_COLUMNS)}, lap=data[:, -1])
    return {"model": model_name(tag, model_file), "seed": seed, "scale": scale, "repeat": repeat, "laps": laps,
            "wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu,
            "decisions": int(sum(lap["decisions"] for lap in laps))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--standing", type=Path, default=Path("outputs/reference/min_time_standing_b+0.00.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/mean_action"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reference = [lap["lap_time"] for lap in json.loads(args.standing.read_text())["laps"]]
    jobs = []
    for tag, model_file, seed in MODELS:
        for index, scale in enumerate(args.scales):
            traced = scale == 0.0 or (tag, model_file, seed) in TRACED
            for repeat in range(1 if scale == 0.0 else args.repeats):
                jobs.append((tag, model_file, seed, index, scale, repeat, reference,
                             args.output / "traces" if traced else None))
    wall, start = time.perf_counter(), time.strftime("%Y-%m-%d %H:%M:%S")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(drive, jobs))
    report = {
        "scales": args.scales, "repeats": args.repeats, "standing_reference": reference, "runs": runs,
        "cost": {"started": start, "wall_s": time.perf_counter() - wall, "cpu_s": sum(r["cpu_s"] for r in runs),
                 "decisions": sum(r["decisions"] for r in runs), "workers": args.workers,
                 "machine": f"{platform.platform()} {platform.processor()}, {os.cpu_count()} CPUs"},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "laps.json").write_text(json.dumps(report))
    for run in runs:
        done = sum(lap["completed"] for lap in run["laps"])
        logging.info("%s s%d k %g r%d: %d/10", run["model"], run["seed"], run["scale"], run["repeat"], done)
    logging.info("%.0f s wall, %.0f s CPU; saved %s", report["cost"]["wall_s"], report["cost"]["cpu_s"],
                 args.output / "laps.json")


if __name__ == "__main__":
    main()
