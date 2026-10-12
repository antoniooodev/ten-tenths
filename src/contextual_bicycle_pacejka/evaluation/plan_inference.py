"""How much of a plan coach's scale can be inferred from the student's sensors (docs/REDESIGN.md §25.5, measure 0c).

A plan coach decides the scale c every ``decision_steps`` steps from the privileged observation. ``decision_samples``
replays an episode of the coach and keeps, for every decision after the first, the student observation the executor
read just before it, with two changes:

    preview        the v_ref preview at scale 1 (the teacher's plan) instead of the scale in use
    no scale       the scale in use is not an input: with CAPS it changes by 0.013-0.017 between decisions, and a
                   regression that reads it would copy it (latching, Swamy et al. 2022)

so the input is what a sensor-based plan student would know (S2 channels with their history, previews, mu_hat when
the executor runs on an estimate), and the target is the coach's scale. ``training_specs`` draws episodes as the coach
was trained (mu uniform in [MU_LO, MU_HI], start ``lap_start``, setup from the seed), with seeds of their own.

``fit_regressor`` fits an MLP (MSE on the unit value of the scale, Adam, early stopping on held-out episodes);
``inference_gap`` measures it on other episodes: R^2 of c, |c - c_hat| mean, 90th percentile and share above a
threshold, overall, per sector of the lap and per friction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning.closed_loop import lap_start
from contextual_bicycle_pacejka.learning.policy import (
    PlanCoach,
    plan_scale_of,
    plan_unit_of,
    policy_layout,
    teacher_plan_observation,
)
from contextual_bicycle_pacejka.spec.constants import MU_HI, MU_LO


def training_specs(racing_line: str | Path, n: int, first_seed: int, max_steps: int) -> list[dict[str, Any]]:
    """``n`` episodes drawn as the plan coach was trained, with seeds ``first_seed`` ... ``first_seed + n - 1``."""
    specs = []
    for i in range(n):
        seed = first_seed + i
        mu = float(np.random.default_rng((seed, 1)).uniform(MU_LO, MU_HI))
        start_s, start_vx = lap_start(racing_line, mu, seed)
        specs.append({"block": "train", "start_label": f"t{i}", "start_s": start_s, "start_vx": start_vx, "mu": mu,
                      "seed": seed, "max_steps": int(max_steps), "stall_rule": False})
    return specs


def decision_samples(spec: dict[str, Any], racing_line: str | Path, coach_dir: str | Path,
                     mu_estimator_dir: str | Path | None = None) -> dict[str, Any]:
    """Student inputs (preview at scale 1, no scale in use) and coach scales of the decisions of one episode."""
    coach = PlanCoach(coach_dir)
    layout = policy_layout(coach.executor)
    captured: list[NDArray[np.float32]] = []

    def on_step(step: dict[str, Any]) -> None:
        if (step["step"] + 1) % coach.decision_steps == 0:  # the observation the next decision is taken on
            captured.append(teacher_plan_observation(step["student_obs"], step["obs"], spec["mu"], layout))

    record = run_episode(spec, racing_line, "policy", False, model_dir=coach_dir, mu_estimator_dir=mu_estimator_dir,
                         on_step=on_step)
    decisions = record["plan_decisions"][1:]  # the first decision is taken on the reset observation
    n = min(len(captured), len(decisions))
    return {"inputs": np.asarray(captured[:n], dtype=np.float32).reshape(n, layout.dim),
            "scales": np.asarray([d[1] for d in decisions[:n]], dtype=np.float64),
            "s": np.asarray([d[0] for d in decisions[:n]], dtype=np.float64),
            "mu": float(spec["mu"]), "robust": bool(record["robust"])}


@dataclass(frozen=True)
class RegressionConfig:
    hidden: tuple[int, ...] = (256, 256)
    learning_rate: float = 3.0e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10
    validation_share: float = 0.1
    seed: int = 0


def _network(n_inputs: int, hidden: tuple[int, ...]) -> th.nn.Sequential:
    layers: list[th.nn.Module] = []
    width = n_inputs
    for h in hidden:
        layers += [th.nn.Linear(width, h), th.nn.ReLU()]
        width = h
    layers.append(th.nn.Linear(width, 1))
    return th.nn.Sequential(*layers)


def fit_regressor(inputs: NDArray[np.floating], scales: NDArray[np.floating], episodes: NDArray[np.integer],
                  config: RegressionConfig | None = None) -> dict[str, Any]:
    """Fit the scale (as a unit value) from the inputs; hold out whole episodes for early stopping."""
    config = config or RegressionConfig()
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    unique = np.unique(episodes)
    held = rng.choice(unique, size=max(1, int(round(config.validation_share * len(unique)))), replace=False)
    validation = np.isin(episodes, held)
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray([plan_unit_of(float(c)) for c in scales], dtype=np.float64)
    mean, std = x[~validation].mean(axis=0), x[~validation].std(axis=0) + 1e-8
    xt = th.as_tensor((x - mean) / std, dtype=th.float32)
    yt = th.as_tensor(y[:, None], dtype=th.float32)
    train_idx, val_idx = np.flatnonzero(~validation), np.flatnonzero(validation)
    network = _network(x.shape[1], config.hidden)
    optimizer = th.optim.Adam(network.parameters(), lr=config.learning_rate)
    best, best_state, waited, curve = np.inf, copy.deepcopy(network.state_dict()), 0, []
    for epoch in range(config.max_epochs):
        network.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            loss = th.mean((network(xt[batch]) - yt[batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with th.no_grad():
            val_loss = float(th.mean((network(xt[val_idx]) - yt[val_idx]) ** 2))
        curve.append({"epoch": epoch, "validation_mse": val_loss})
        if val_loss < best:
            best, best_state, waited = val_loss, copy.deepcopy(network.state_dict()), 0
        else:
            waited += 1
            if waited >= config.patience:
                break
    network.load_state_dict(best_state)
    return {"network": network, "mean": mean, "std": std, "config": config, "curve": curve,
            "validation_mse": best, "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)),
            "validation_episodes": int(len(held))}


def predict(model: dict[str, Any], inputs: NDArray[np.floating]) -> NDArray[np.float64]:
    """Predicted scales of a fitted regressor (unit values clipped to [-1, 1], mapped back to scales)."""
    x = th.as_tensor((np.asarray(inputs, dtype=np.float64) - model["mean"]) / model["std"], dtype=th.float32)
    with th.no_grad():
        units = model["network"](x).numpy()[:, 0].astype(np.float64)
    return np.asarray([plan_scale_of(float(u)) for u in np.clip(units, -1.0, 1.0)])


def inference_gap(scales: NDArray[np.floating], predicted: NDArray[np.floating], s: NDArray[np.floating],
                  mus: NDArray[np.floating], sector_m: float = 500.0, threshold: float = 0.03) -> dict[str, Any]:
    """R^2 of the scale and |c - c_hat| overall, per sector of ``sector_m`` metres and per friction."""
    c, c_hat = np.asarray(scales, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    gap = np.abs(c - c_hat)
    total = float(np.sum((c - c.mean()) ** 2))

    def summary(mask: NDArray[np.bool_]) -> dict[str, Any]:
        return {"n": int(mask.sum()), "mean_abs": float(gap[mask].mean()), "scale_mean": float(c[mask].mean()),
                "predicted_mean": float(c_hat[mask].mean()), "share_above": float(np.mean(gap[mask] > threshold))}

    sectors = np.floor(np.asarray(s) / sector_m) * sector_m
    return {"n": int(len(c)), "r2": 1.0 - float(np.sum((c - c_hat) ** 2)) / total if total > 0 else None,
            "mean_abs": float(gap.mean()), "p90_abs": float(np.percentile(gap, 90)),
            "share_above": float(np.mean(gap > threshold)), "threshold": threshold, "scale_std": float(c.std()),
            "by_sector": {f"{start:.0f}": summary(sectors == start) for start in np.unique(sectors)},
            "by_mu": {f"{mu:.2f}": summary(np.isclose(mus, mu)) for mu in np.unique(np.round(mus, 2))},
            "sectors_above": [f"{start:.0f}" for start in np.unique(sectors)
                              if gap[sectors == start].mean() > threshold]}
