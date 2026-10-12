"""Behaviour cloning: supervised regression of the actor mean onto the teacher labels, and the shielded lap gate.

Training (defaults used for every final policy):

    normaliser   VecNormalize fitted once on the dataset observations, then frozen; mean and variance are
                 accumulated by chunks in the order of numpy's (``statistics.column_moments``), so a large
                 dataset needs no float64 copy and the normaliser is the same to the last bit
    parameters   actor MLP and action head only (the critic is untouched)
    loss         sum_b w_b sum_j lambda_j (mu_theta,j(o_b) - y_b,j)^2 / (2 sum_b w_b),  lambda = [20, 1]
    optimiser    Adam, lr 3e-4, gradient-norm clip 1.0, batch 1024, at most 80 epochs
    validation   every 10th transition of each lap; early stopping with patience 12; best state kept
    output       policy.zip (actor log-std set to -4), vecnormalize.pkl, training.json

Per-sample weights w_b (all 1 unless the dataset provides ``sample_weights``) are used by DAgger.

Students: ``level`` selects the observation (S0 privileged; S1, S2 student observations built from an offline
dataset by ``datasets.student_dataset_observations``) and ``target`` the regression target, the teacher label
(``label``, the dataset's ``actions``) or the executed action (``exec``, ``actions_exec``). The normaliser is
fitted on the observations of the level, and ``student.json`` records level, history, oracle and target. The
history (``n_frames`` x ``stride``) and the diagnostic oracle entries (``spec/layout.py``) apply to S1 and S2. Level
S0 with the label target is the original training, unchanged.

Gate: 20 shielded laps from the start of the first dataset lap, mu in {0.8, 0.9, 1.0, 1.1, 1.2} x setup seeds
42-45; reported, not used to select the model.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from contextual_bicycle_pacejka.control.safety_filter import filter_action
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import progress_increment
from contextual_bicycle_pacejka.learning.datasets import student_dataset_observations
from contextual_bicycle_pacejka.learning.policy import (
    NORMALIZER_FILE,
    POLICY_FILE,
    TARGETS,
    Policy,
    actor_mean,
    load_model,
    new_bc_model,
    policy_env,
    policy_layout,
    write_student_spec,
)
from contextual_bicycle_pacejka.learning.statistics import CHUNK_ROWS, column_moments
from contextual_bicycle_pacejka.spec.constants import STUDENT_HISTORY_FRAMES, STUDENT_HISTORY_STRIDE, V_MIN
from contextual_bicycle_pacejka.spec.layout import StudentLayout, student_layout

LOGGER = logging.getLogger(__name__)
VALIDATION_STRIDE = 10
GATE_MUS = (0.8, 0.9, 1.0, 1.1, 1.2)
GATE_SEEDS = (42, 43, 44, 45)


@dataclass
class BCConfig:
    """Hyper-parameters of behaviour cloning."""

    seed: int = 42
    learning_rate: float = 3.0e-4
    steering_weight: float = 20.0
    epochs: int = 80
    batch_size: int = 1024
    patience: int = 12
    log_std: float = -4.0
    base_model: Path | None = None  # continue from this model instead of a fresh initialisation (DAgger)
    level: str = "S0"  # observation level: S0 (privileged), S1 or S2
    target: str = "label"  # regression target: "label" (teacher) or "exec" (executed action)
    n_frames: int = STUDENT_HISTORY_FRAMES  # student frames (S1, S2)
    stride: int = STUDENT_HISTORY_STRIDE  # steps between student frames (S1, S2)
    oracle: tuple[str, ...] = ()  # diagnostic oracle entries appended to the student observation


@dataclass
class GateConfig:
    """Laps of the behaviour-cloning gate."""

    mus: tuple[float, ...] = GATE_MUS
    seeds: tuple[int, ...] = GATE_SEEDS
    max_steps: int = 100_000
    extra: dict[str, Any] = field(default_factory=dict)


def stratified_split(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation indices: every 10th transition of each lap is held out."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    train, validation = [], []
    for start, end in zip(starts, ends, strict=True):
        lap = np.arange(start, end)
        held_out = (lap - start) % VALIDATION_STRIDE == VALIDATION_STRIDE - 1
        train.append(lap[~held_out])
        validation.append(lap[held_out])
    return np.concatenate(train), np.concatenate(validation)


def normalise_in_place(vec: VecNormalize, rows: np.ndarray, chunk: int = CHUNK_ROWS) -> np.ndarray:
    """The frozen normaliser applied to float32 rows in place, by chunks (as ``vec.normalize_obs`` row by row)."""
    for start in range(0, len(rows), chunk):
        rows[start:start + chunk] = vec.normalize_obs(rows[start:start + chunk])
    return rows


def load_or_fit_normalizer(output: Path, racing_line: str | Path, observations: np.ndarray,
                           layout: StudentLayout | None = None) -> VecNormalize:
    """Load ``output/vecnormalize.pkl`` if it exists, otherwise fit it on ``observations`` and save it.

    ``layout`` is the observation of the policy (default: the privileged one). The fit is that of
    ``RunningMeanStd.update`` on the whole array, with the moments of ``column_moments``.
    """
    saved = output / NORMALIZER_FILE
    layout = layout or student_layout("S0")
    vec_env = DummyVecEnv([lambda: policy_env(racing_line, layout)])
    if saved.exists():
        vec = VecNormalize.load(str(saved), vec_env)
    else:
        vec = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        vec.obs_rms.update_from_moments(*column_moments(observations), len(observations))
        output.mkdir(parents=True, exist_ok=True)
        vec.save(str(saved))
    vec.training = False
    vec.norm_reward = False
    return vec


def train_bc(dataset: str | Path, output: str | Path, racing_line: str | Path, config: BCConfig | None = None,
             ) -> dict[str, float | int]:
    """Train a policy on ``dataset`` and write it to ``output``; ``racing_line`` only provides the env spaces."""
    config = config or BCConfig()
    output = Path(output)
    layout = student_layout(config.level, config.n_frames, config.stride, config.oracle)
    if config.target not in TARGETS:
        raise ValueError(f"unknown target {config.target!r}; expected one of {TARGETS}")
    with np.load(dataset, allow_pickle=False) as data:
        if layout.privileged:
            raw_obs = np.asarray(data["observations"], dtype=np.float32)
        else:
            raw_obs = student_dataset_observations(data, layout)
        targets = np.asarray(data["actions" if config.target == "label" else "actions_exec"], dtype=np.float32)
        ends = np.asarray(data["episode_ends"], dtype=np.int64)
        weights = (np.asarray(data["sample_weights"], dtype=np.float32) if "sample_weights" in data.files
                   else np.ones(len(raw_obs), dtype=np.float32))
    if len(ends) < 2 or int(ends[-1]) != len(raw_obs):
        raise ValueError("behaviour cloning needs at least two complete laps with valid episode_ends")
    if weights.shape != (len(raw_obs),) or np.any(weights <= 0.0):
        raise ValueError("sample_weights must be positive with one value per transition")

    vec = load_or_fit_normalizer(output, racing_line, raw_obs, layout)
    model = new_bc_model(vec, config.seed)
    if config.base_model is not None:
        model = load_model(config.base_model, env=vec)
    obs = normalise_in_place(vec, raw_obs)
    train_idx, val_idx = stratified_split(ends)
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    loss_weights = th.as_tensor([float(config.steering_weight), 1.0], device=model.device)
    optimizer = th.optim.Adam(params, lr=float(config.learning_rate))
    rng = np.random.default_rng(config.seed)

    def weighted_loss(index: np.ndarray) -> th.Tensor:
        mean = actor_mean(policy, obs[index])
        w = th.as_tensor(weights[index], device=model.device)
        error = (mean - th.as_tensor(targets[index], device=model.device)) ** 2
        return th.sum(w[:, None] * loss_weights * error) / (th.sum(w) * len(loss_weights))

    best_loss, best_state, stale, epoch = math.inf, None, 0, -1
    try:
        for epoch in range(int(config.epochs)):
            losses = []
            shuffled = rng.permutation(train_idx)
            for start in range(0, len(train_idx), int(config.batch_size)):
                loss = weighted_loss(shuffled[start:start + int(config.batch_size)])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            with th.no_grad():
                val_loss = float(weighted_loss(val_idx).item())
            if val_loss < best_loss - 1.0e-7:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            else:
                stale += 1
            if epoch % 10 == 0:
                LOGGER.info("BC epoch %03d: train %.6g, validation %.6g", epoch, np.mean(losses), val_loss)
            if stale >= int(config.patience):
                break
        if best_state is None:
            raise RuntimeError("behaviour cloning produced no checkpoint")
        policy.load_state_dict(best_state)
        policy.log_std.data.fill_(float(config.log_std))
        output.mkdir(parents=True, exist_ok=True)
        model.save(str(output / POLICY_FILE))
        write_student_spec(output, layout, config.target)
    finally:
        vec.close()
    result = {"best_validation_mse": best_loss, "epochs": epoch + 1}
    (output / "training.json").write_text(json.dumps(result, indent=2))
    return result


def gate(model_dir: str | Path, dataset: str | Path, racing_line: str | Path, config: GateConfig | None = None,
         ) -> dict[str, Any]:
    """Run the shielded gate laps of the policy in ``model_dir``; write and return ``gate.json``."""
    config = config or GateConfig()
    model_dir = Path(model_dir)
    with np.load(dataset, allow_pickle=False) as data:
        start_s = float(data["start_s_per_lap"][0])
        start_vx = float(data["start_vx_per_lap"][0])
        start_mu = float(data["mu_per_lap"][0])
    env = policy_env(racing_line, policy_layout(model_dir), max_episode_steps=config.max_steps)
    policy = Policy(model_dir, env)
    length = float(env.unwrapped.track.track_length)
    laps = []
    try:
        for mu in config.mus:
            for seed in config.seeds:
                vx0 = start_vx * math.sqrt(mu / start_mu)
                obs, _ = env.reset(seed=seed, options={"setup": randomized_setup(mu, np.random.default_rng(seed)),
                                                       "initial_s": start_s, "initial_vx": max(V_MIN, vx0)})
                previous = np.zeros(2, dtype=np.float64)
                progress, previous_s, empty, projected, flags, step = 0.0, start_s, 0, 0, {}, -1
                for step in range(config.max_steps):
                    result = filter_action(env, policy(obs), previous)
                    projected += int(result.projected)
                    empty += int(result.empty)
                    obs, _, terminated, truncated, info = env.step(result.action)
                    s_now = float(np.asarray(info["state"])[0])
                    progress += progress_increment(previous_s, s_now, length)
                    previous_s, previous = s_now, np.asarray(result.action, dtype=np.float64)
                    flags = dict(info.get("event_flags", {}))
                    if progress >= length or terminated or truncated:
                        break
                laps.append({"passed": bool(progress >= length and not any(flags.values()) and empty == 0),
                             "mu": mu, "seed": seed, "steps": step + 1, "progress_m": progress, "events": flags,
                             "empty_steps": empty, "projected_steps": projected})
                LOGGER.info("gate mu=%.2f seed=%d: passed=%s progress=%.1f m", mu, seed, laps[-1]["passed"], progress)
    finally:
        policy.close()
    summary = {"passed": all(lap["passed"] for lap in laps), "laps": laps}
    (model_dir / "gate.json").write_text(json.dumps(summary, indent=2))
    return summary
