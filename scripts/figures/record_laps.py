"""Record the trajectories of policies and of the minimum-time laps, for figures and animations.

    python scripts/figures/record_laps.py
    python scripts/figures/record_laps.py --only flying --workers 8

Every policy run starts at s = 0 on the centreline at 3 m/s, as the protocol laps, and drives ``LAPS`` laps or until
the first failure; one sample is stored per decision (20 Hz). The second lap is a flying lap, comparable with the
flying minimum-time lap. A run is stored as <output>/<name>.npz with the arrays of ``CHANNELS`` and the scalars
``tag``, ``seed``, ``b``, ``deterministic``, ``failure``, ``lap_times`` (one per completed lap).

Runs (``--only`` picks one group):
- flying:     every limit configuration at b = 0, seeds 10-14, mean action;
- balance:    Nominal continued, Randomised W1 and Oracle W1, seed 12, over five values of b, mean action;
- stochastic: Nominal continued, seed 14, b = 0, the mean action and five stochastic runs (the case where the mean
              action leaves the track at s = 2167 m);
- reference:  the flying minimum-time lap of every b of the grid, resampled in time, as reference_b<b>.npz.

The accelerations are the body-frame resultants over the mass (F_body_x / m, F_body_y / m) of the next plant step,
and the slips |alpha| / alpha_sat of each axle; x, y and psi are world coordinates of the centre of mass.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from thesis_plots import CATALUNYA

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.evaluation.balance_study import LIMIT_GRID
from contextual_bicycle_pacejka.model.dynamics import compute_vehicle_rates
from contextual_bicycle_pacejka.model.powertrain import limit_top_speed
from contextual_bicycle_pacejka.model.state import VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.tires import (
    compute_saturation_threshold,
    compute_slip_angles,
)
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    DECISION_DT,
    DELTA_MAX,
    OBS_DIM,
    TAU_RATE_MAX,
)

STAGE = 7
LAPS = 2
START_SPEED_MPS = 3.0
LIMIT_TAGS = ("nominal_limit", "nominal_limit_cont", "randomised_limit_w13", "oracle_limit_w13",
              "randomised_limit_w23", "oracle_limit_w23", "randomised_limit_w1", "oracle_limit_w1")
SEEDS = (10, 11, 12, 13, 14)
BALANCE_TAGS = ("nominal_limit_cont", "randomised_limit_w1", "oracle_limit_w1")
BALANCE_VALUES = (-0.21, -0.11, 0.0, 0.03, 0.05)
STOCHASTIC_RUNS = 5
CHANNELS = ("t", "s", "ey", "epsi", "vx", "vy", "r", "delta", "tau", "delta_ref", "tau_rate", "x", "y", "psi",
            "ax", "ay", "slip_f", "slip_r", "fz_f", "fz_r", "std_delta", "std_tau")


def world_pose(track: RealTrack, s: np.ndarray, ey: np.ndarray, epsi: np.ndarray) -> tuple[np.ndarray, ...]:
    """World coordinates (x, y) [m] and heading psi [rad] of Frenet points (ey positive to the left)."""
    centre = np.array([track.centerline_point(v) for v in s])
    heading = np.array([track.psi_ref(v) for v in s])
    x = centre[:, 0] - ey * np.sin(heading)
    y = centre[:, 1] + ey * np.cos(heading)
    return x, y, heading + epsi


def sample(env: RacingEnv, t: float, std: np.ndarray) -> dict[str, float]:
    """One sample of the environment after a decision."""
    state = env.state
    tau_eff = limit_top_speed(env.tau, state.vx, env.stage.top_speed)
    _, _, _, diag = compute_vehicle_rates(state, tau_eff, env.setup, env.memory)
    alpha_f, alpha_r = compute_slip_angles(state)
    return {"t": t, "s": state.s, "ey": state.ey, "epsi": state.epsi, "vx": state.vx, "vy": state.vy,
            "r": state.yaw_rate, "delta": state.delta, "tau": env.tau, "delta_ref": env.delta_ref,
            "tau_rate": env.tau_rate, "ax": diag["F_body_x"] / env.setup.m, "ay": diag["F_body_y"] / env.setup.m,
            "slip_f": abs(alpha_f) / compute_saturation_threshold(env.setup, "front"),
            "slip_r": abs(alpha_r) / compute_saturation_threshold(env.setup, "rear"),
            "fz_f": diag["Fz_f"], "fz_r": diag["Fz_r"], "std_delta": float(std[0]), "std_tau": float(std[1])}


def drive(job: dict[str, Any]) -> str:
    """Drive one run (in a worker process) and save it."""
    torch.set_num_threads(1)
    track = RealTrack(CATALUNYA)
    model = PPO.load(job["model"], device="cpu")
    if job.get("noise_seed") is not None:
        torch.manual_seed(job["noise_seed"])
    oracle = model.observation_space.shape[0] == OBS_DIM + 1
    env = RacingEnv(track, make_stage(STAGE, track), b_range=(job["b"], job["b"]), oracle=oracle)
    obs, _ = env.reset(options={"initial_s": job["start_s"], "initial_vx": START_SPEED_MPS})
    finish = [job["start_s"] + (k + 1) * track.track_length for k in range(LAPS)]
    rows = [sample(env, 0.0, np.zeros(2))]
    lap_times, failure, decisions = [], None, 0
    while len(lap_times) < LAPS and decisions * DECISION_DT < 6.0 * LAPS * 140.0:
        with torch.no_grad():
            dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
            std = dist.distribution.stddev.numpy()[0]
        action = model.predict(obs, deterministic=job["deterministic"])[0]
        s_before = env.state.s
        obs, _, terminated, _, info = env.step(action)
        decisions += 1
        rows.append(sample(env, decisions * DECISION_DT, std))
        if terminated:
            failure = info["failure"]
            break
        target = finish[len(lap_times)]
        if env.state.s >= target:
            fraction = (target - s_before) / (env.state.s - s_before)
            lap_times.append((decisions - 1 + fraction) * DECISION_DT - sum(lap_times))
    data = {key: np.array([row[key] for row in rows if key in row]) for key in CHANNELS if key not in ("x", "y", "psi")}
    data["x"], data["y"], data["psi"] = world_pose(track, data["s"], data["ey"], data["epsi"])
    out = Path(job["output"]) / f"{job['name']}.npz"
    np.savez_compressed(out, **data, tag=job["tag"], seed=job["seed"], b=job["b"],
                        deterministic=job["deterministic"], failure=str(failure), lap_times=np.array(lap_times),
                        delta_max=DELTA_MAX, tau_rate_max=TAU_RATE_MAX)
    return f"{job['name']}: laps {np.round(lap_times, 2).tolist()}, failure {failure}"


def record_reference(b: float, output: Path, track: RealTrack) -> str:
    """Resample the flying minimum-time lap of ``b`` in time, with its accelerations and slips."""
    ref = json.loads(Path(f"outputs/reference/min_time_b{b:+.2f}.json").read_text())
    s = np.asarray(ref["s"])
    st = {k: np.asarray(v) for k, v in ref["states"].items()}
    kappa = track.kappa_many(s)
    s_dot = (st["vx"] * np.cos(st["epsi"]) - st["vy"] * np.sin(st["epsi"])) / (1.0 - kappa * st["ey"])
    ds = np.diff(np.concatenate([s, [track.track_length]]))
    t = np.concatenate([[0.0], np.cumsum(ds * 0.5 * (1.0 / s_dot + 1.0 / np.roll(s_dot, -1)))])[:-1]
    setup = sample_setup(np.random.default_rng(0), (b, b))
    peak = compute_saturation_threshold(setup, "front"), compute_saturation_threshold(setup, "rear")
    ax, ay, fz_f, fz_r, slip_f, slip_r = (np.zeros_like(s) for _ in range(6))
    for i in range(len(s)):
        state = VehicleState(s[i], st["ey"][i], st["epsi"][i], st["vx"][i], st["vy"][i], st["r"][i], st["delta"][i])
        memory = VehicleMemory.zeros()
        for _ in range(8):  # the reference has no one-step delay: iterate the load transfer to its fixed point
            _, _, _, diag = compute_vehicle_rates(state, float(st["tau"][i]), setup, memory)
            memory = VehicleMemory(F_body_x_prev=diag["F_body_x"])
        ax[i], ay[i] = diag["F_body_x"] / setup.m, diag["F_body_y"] / setup.m
        fz_f[i], fz_r[i] = diag["Fz_f"], diag["Fz_r"]
        alpha_f, alpha_r = compute_slip_angles(state)
        slip_f[i], slip_r[i] = abs(alpha_f) / peak[0], abs(alpha_r) / peak[1]
    grid = np.arange(0.0, ref["lap_time"], DECISION_DT)
    data = {"t": grid}
    columns = {"s": s, "ey": st["ey"], "epsi": st["epsi"], "vx": st["vx"], "vy": st["vy"], "r": st["r"],
               "delta": st["delta"], "tau": st["tau"], "ax": ax, "ay": ay, "slip_f": slip_f, "slip_r": slip_r,
               "fz_f": fz_f, "fz_r": fz_r}
    for key, values in columns.items():
        data[key] = np.interp(grid, t, values)
    data["x"], data["y"], data["psi"] = world_pose(track, data["s"], data["ey"], data["epsi"])
    np.savez_compressed(output / f"reference_b{b:+.2f}.npz", **data, b=b, lap_time=ref["lap_time"],
                        s_grid=s, t_grid=t)
    return f"reference b {b:+.2f}: {ref['lap_time']:.2f} s"


def jobs(group: str, models: Path, output: Path) -> list[dict[str, Any]]:
    def job(tag: str, seed: int, b: float, name: str, deterministic: bool = True,
            noise_seed: int | None = None) -> dict[str, Any]:
        return {"model": str(models / tag / f"s{seed}" / "stage7.zip"), "tag": tag, "seed": seed, "b": b,
                "start_s": 0.0, "deterministic": deterministic, "noise_seed": noise_seed, "name": name,
                "output": str(output)}

    if group == "flying":
        return [job(tag, seed, 0.0, f"flying_{tag}_s{seed}") for tag in LIMIT_TAGS for seed in SEEDS]
    if group == "balance":
        return [job(tag, 12, b, f"balance_{tag}_s12_b{b:+.2f}") for tag in BALANCE_TAGS for b in BALANCE_VALUES]
    if group == "stochastic":
        runs = [job("nominal_limit_cont", 14, 0.0, "stochastic_nominal_limit_cont_s14_mean")]
        return runs + [job("nominal_limit_cont", 14, 0.0, f"stochastic_nominal_limit_cont_s14_noise{k}", False,
                           14000 + k) for k in range(STOCHASTIC_RUNS)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("flying", "balance", "stochastic", "reference"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectories"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [args.only] if args.only else ["reference", "flying", "balance", "stochastic"]
    if "reference" in groups:
        track = RealTrack(CATALUNYA)
        for b in LIMIT_GRID:
            logging.info(record_reference(b, args.output, track))
    runs = [j for group in groups for j in jobs(group, args.models, args.output)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(drive, runs):
            logging.info(line)


if __name__ == "__main__":
    main()
