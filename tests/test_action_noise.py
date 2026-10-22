"""A policy with a fraction of its own action noise, and the closed-form mean of a clipped Gaussian."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.env import make_vec_env
from contextual_bicycle_pacejka.evaluation.action_noise import (
    ScaledNoisePolicy,
    clipped_gaussian_mean,
)
from contextual_bicycle_pacejka.learning.ppo import PPOConfig, make_ppo

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"


def test_clipped_gaussian_mean() -> None:
    """Monte Carlo agrees with the closed form; no noise gives clip(mu); far inside the bounds, mu."""
    rng = np.random.default_rng(0)
    mu, sigma = np.array([0.95, -1.1]), np.array([0.1, 0.3])
    samples = np.clip(mu + sigma * rng.standard_normal((400_000, 2)), -1.0, 1.0).mean(axis=0)
    assert clipped_gaussian_mean(mu, sigma) == pytest.approx(samples, abs=1e-3)
    assert clipped_gaussian_mean(np.array([1.5, -0.2]), np.zeros(2)) == pytest.approx([1.0, -0.2])
    assert clipped_gaussian_mean(np.array([0.1, -0.3]), np.array([0.02, 0.04])) == pytest.approx([0.1, -0.3])


def test_noise_scale_of_the_policy() -> None:
    """Scale 0 is the mean action of PPO; scale 1 has the standard deviation of the policy; records per decision."""
    env = make_vec_env(CATALUNYA, 1, seed=0)
    model = make_ppo(env, PPOConfig(n_steps=64, batch_size=64, n_epochs=1), seed=0)
    obs = env.reset()[0]
    mean = ScaledNoisePolicy(model, 0.0, np.random.default_rng(0))
    assert mean(obs) == pytest.approx(model.predict(obs, deterministic=True)[0], abs=1e-6)
    noisy = ScaledNoisePolicy(model, 1.0, np.random.default_rng(1))
    actions = np.array([noisy(obs) for _ in range(4000)])
    sigma = model.policy.log_std.exp().detach().numpy()
    assert actions.std(axis=0) == pytest.approx(sigma, rel=0.05)
    assert actions.mean(axis=0) == pytest.approx(mean(obs), abs=4 * sigma.max() / np.sqrt(4000))
    rows = noisy.reset_records()
    assert rows.shape == (4000, 6) and noisy.records == []
    assert rows[0, 2:4] == pytest.approx(sigma)
    env.close()
