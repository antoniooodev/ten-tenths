"""Weighted DAgger on pre-failure windows (an alternative to perturbed demonstrations; not used by the final policies).

One iteration:

1. Collection. For every (mu, seed), the current policy drives a lap through the safety filter while the teacher
   labels every visited state: label = filter(MATv0(o_t)). If the lap fails, the labels of the last ``window_m``
   metres before the failure are kept; a second rollout lets the teacher take over at the window start (reported
   as "teacher recovery", not used for training).
2. Aggregation. The windows of all iterations so far are appended to the base demonstrations.
3. Weighting. Window rows get weight ``critical_weight``; base rows keep weight 1 but their labels are replaced by
   the current policy's own outputs ("anchors"), so that only the windows carry new teacher information.
4. Update. Behaviour cloning continues from the current policy (lr 3e-5, 12 epochs, patience 4).
5. Gate. The shielded gate decides whether to stop; otherwise the next iteration starts from the new policy.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from contextual_bicycle_pacejka.control.safety_filter import filter_action
from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import progress_increment
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, GateConfig, gate, train_bc
from contextual_bicycle_pacejka.learning.demonstrations import start_point
from contextual_bicycle_pacejka.learning.policy import NORMALIZER_FILE, POLICY_FILE, Policy, actor_mean
from contextual_bicycle_pacejka.spec.constants import DT, SLIP_TERMINATION_RATIO, V_MIN, VREF_REFERENCE_MU


@dataclass
class DAggerConfig:
    """Settings of weighted DAgger."""

    iterations: int = 3
    collect_mus: tuple[float, ...] = (1.0, 1.1, 1.2)
    collect_seeds: tuple[int, ...] = (42, 45)
    window_m: float = 100.0  # [m] pre-failure window
    critical_weight: float = 80.0
    max_steps: int = 60_000
    learning_rate: float = 3.0e-5
    epochs: int = 12
    patience: int = 4
    gate: GateConfig | None = None


@dataclass(frozen=True)
class StartCondition:
    """Start of the collection laps: arc length [m], speed at the reference friction [m/s], reference friction."""

    start_s: float
    start_vx_at_mu_ref: float
    mu_ref: float

    @classmethod
    def from_racing_line(cls, racing_line: str | Path) -> StartCondition:
        """The start of the demonstration laps on ``racing_line``."""
        start_s, start_vx = start_point(racing_line)
        return cls(start_s, start_vx, VREF_REFERENCE_MU)


def rollout(model_dir: Path, racing_line: str | Path, teacher: MATv0, start: StartCondition, mu: float, seed: int,
            max_steps: int, switch_progress_m: float | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray,
                                                                           np.ndarray]:
    """Policy lap with teacher labels; with ``switch_progress_m`` the teacher drives from that progress on.

    Returns (report, observations, teacher labels, progress) of every step (labels only when not switching).
    """
    env = make_env(racing_line)
    base = env.unwrapped
    policy = Policy(model_dir, env)
    obs, reset_info = env.reset(seed=seed, options={
        "setup": randomized_setup(mu, np.random.default_rng(seed)), "initial_s": start.start_s,
        "initial_vx": max(V_MIN, start.start_vx_at_mu_ref * math.sqrt(mu / start.mu_ref))})
    rear_limit = SLIP_TERMINATION_RATIO * float(reset_info["alpha_sat_r"])
    length = float(base.track.track_length)
    previous = np.zeros(2, dtype=np.float64)
    previous_s = current_s = start.start_s
    progress, empty, max_rear_ratio, step = 0.0, 0, 0.0, -1
    switched, switch_state = False, None
    observations, labels, progresses = [], [], []
    flags: dict[str, bool] = {}
    try:
        for step in range(max_steps):
            use_teacher = switch_progress_m is not None and progress >= switch_progress_m
            if use_teacher and not switched:
                switched, switch_state = True, base.state.to_array().tolist()
            label = filter_action(env, teacher.act(obs), previous)
            if switch_progress_m is None:
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                labels.append(np.asarray(label.action, dtype=np.float32))
                progresses.append(progress)
            executed = label if use_teacher else filter_action(env, policy(obs), previous)
            empty += int(executed.empty)
            obs, _, terminated, truncated, info = env.step(executed.action)
            current_s = float(np.asarray(info["state"])[0])
            progress += progress_increment(previous_s, current_s, length)
            previous_s = current_s
            previous = np.asarray(executed.action, dtype=np.float32).astype(np.float64)
            max_rear_ratio = max(max_rear_ratio, abs(float(info["diag"]["alpha_r"])) / rear_limit)
            flags = dict(info.get("event_flags", {}))
            if progress >= length or terminated or truncated:
                break
    finally:
        policy.close()
    report = {"mu": mu, "seed": seed, "completed": progress >= length, "steps": step + 1, "progress_m": progress,
              "terminal_s_m": current_s, "events": flags, "empty_steps": empty, "max_rear_ratio": max_rear_ratio,
              "switched": switched, "switch_progress_m": switch_progress_m, "switch_state": switch_state}
    return report, np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.float32), \
        np.asarray(progresses, dtype=np.float64)


def collect_windows(model_dir: Path, racing_line: str | Path, teacher: MATv0, start: StartCondition,
                    config: DAggerConfig) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Teacher-labelled pre-failure windows of the current policy, over all (mu, seed) collection laps."""
    obs_windows, label_windows, reports = [], [], []
    for mu in config.collect_mus:
        for seed in config.collect_seeds:
            failure, obs, labels, progress = rollout(model_dir, racing_line, teacher, start, mu, seed, config.max_steps)
            if failure["completed"]:
                reports.append({"failure": failure, "selected_labels": 0})
                continue
            window_start = max(0.0, float(failure["progress_m"]) - float(config.window_m))
            selected = progress >= window_start
            if not selected.any():
                raise RuntimeError(f"empty pre-failure window for mu={mu}, seed={seed}")
            recovery = rollout(model_dir, racing_line, teacher, start, mu, seed, config.max_steps,
                               switch_progress_m=window_start)[0]
            obs_windows.append(obs[selected])
            label_windows.append(labels[selected])
            reports.append({"failure": failure, "window_start_progress_m": window_start,
                            "selected_labels": int(selected.sum()), "teacher_recovery": recovery})
    return obs_windows, label_windows, reports


def anchor_actions(model_dir: Path, racing_line: str | Path, observations: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Actor mean of the policy in ``model_dir`` on ``observations``."""
    policy = Policy(model_dir, make_env(racing_line))
    try:
        parts = []
        for start in range(0, len(observations), batch):
            normalized = policy.normalizer.normalize_obs(observations[start:start + batch])
            with th.no_grad():
                parts.append(actor_mean(policy.model.policy, normalized).cpu().numpy().astype(np.float32))
    finally:
        policy.close()
    return np.concatenate(parts)


def write_weighted_dataset(base_dataset: Path, output: Path, obs_windows: list[np.ndarray],
                           label_windows: list[np.ndarray], critical_weight: float, base_actions: np.ndarray) -> dict:
    """Base observations with anchor labels (weight 1) followed by the critical windows (weight critical_weight)."""
    with np.load(base_dataset, allow_pickle=False) as data:
        observations = [np.asarray(data["observations"], dtype=np.float32)]
        ends = np.asarray(data["episode_ends"], dtype=np.int64).tolist()
    actions = [np.asarray(base_actions, dtype=np.float32)]
    weights = [np.ones(len(observations[0]), dtype=np.float32)]
    offset = len(observations[0])
    for window_obs, window_labels in zip(obs_windows, label_windows, strict=True):
        observations.append(window_obs)
        actions.append(window_labels)
        weights.append(np.full(len(window_obs), critical_weight, dtype=np.float32))
        offset += len(window_obs)
        ends.append(offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, observations=np.concatenate(observations), actions=np.concatenate(actions),
             episode_ends=np.asarray(ends, dtype=np.int64), sample_weights=np.concatenate(weights))
    critical = sum(len(w) for w in obs_windows)
    return {"base_labels": len(observations[0]), "critical_labels": critical, "critical_weight": critical_weight,
            "effective_critical_labels": critical * critical_weight}


def run_dagger(base_dataset: Path, gate_dataset: Path, model_dir: Path, racing_line: str | Path, output: Path,
               teacher: MATv0 | None = None, config: DAggerConfig | None = None) -> list[dict[str, Any]]:
    """Run weighted DAgger from the policy in ``model_dir``; every iteration is written to ``output/iteration_k``.

    ``gate_dataset`` provides the start of the gate laps (its first lap), normally the clean Monza laps.
    """
    teacher = teacher or MATv0()
    config = config or DAggerConfig()
    start = StartCondition.from_racing_line(racing_line)
    output.mkdir(parents=True, exist_ok=True)
    current = Path(model_dir)
    all_obs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    summary: list[dict[str, Any]] = []
    for iteration in range(1, config.iterations + 1):
        obs_windows, label_windows, collection = collect_windows(current, racing_line, teacher, start, config)
        (output / f"collection_{iteration}.json").write_text(json.dumps(collection, indent=2))
        all_obs.extend(obs_windows)
        all_labels.extend(label_windows)
        if not all_obs:
            break
        iteration_dir = output / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        with np.load(base_dataset, allow_pickle=False) as data:
            base_obs = np.asarray(data["observations"], dtype=np.float32)
        anchors = anchor_actions(current, racing_line, base_obs)
        dataset = iteration_dir / "weighted_dataset.npz"
        balance = write_weighted_dataset(base_dataset, dataset, all_obs, all_labels, config.critical_weight, anchors)
        shutil.copy2(current / NORMALIZER_FILE, iteration_dir / NORMALIZER_FILE)
        training = train_bc(dataset, iteration_dir, racing_line, BCConfig(
            learning_rate=config.learning_rate, epochs=config.epochs, patience=config.patience,
            base_model=current / POLICY_FILE))
        current = iteration_dir
        result = gate(iteration_dir, gate_dataset, racing_line, config.gate)
        passed = [lap for lap in result["laps"] if lap["passed"]]
        summary.append({"iteration": iteration, "balance": balance, "training": training,
                        "gate_passes": len(passed), "gate_total": len(result["laps"]), "gate_passed": result["passed"],
                        "passed_mean_time_s": (float(np.mean([lap["steps"] * DT for lap in passed]))
                                               if passed else None),
                        "model": str(iteration_dir)})
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        if result["passed"]:
            break
    return summary
