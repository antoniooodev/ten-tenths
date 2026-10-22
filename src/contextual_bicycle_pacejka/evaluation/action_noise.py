"""A policy that drives with a fraction of its own action noise, and the bias of the clipped actions.

A PPO policy is a diagonal Gaussian N(mu(o), sigma^2) over the normalised actions; the environment clips every
action to [-1, 1]. In training the car executes clip(mu + sigma eps), in the protocol laps clip(mu). With a noise
scale k the policy executes clip(mu + k sigma eps): k = 0 is the mean action, k = 1 the stochastic policy of
training, k > 1 more noise than training.

Clipping bias: the mean command executed in training, E[clip(mu + sigma eps)], differs from the command executed
with the mean action, clip(mu), when mu lies near or beyond a bound. ``clipped_gaussian_mean`` gives the first in
closed form; ``ScaledNoisePolicy`` records mu, sigma and that bias at every decision.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3 import PPO

from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, ACTION_LOW

__all__ = ["ScaledNoisePolicy", "clipped_gaussian_mean"]


def _normal_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def clipped_gaussian_mean(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], low: NDArray[np.float64] = ACTION_LOW,
    high: NDArray[np.float64] = ACTION_HIGH,
) -> NDArray[np.float64]:
    """Return E[clip(X, low, high)] for X ~ N(mu, sigma^2), element-wise; clip(mu) where sigma is 0."""
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    low, high = np.broadcast_to(low, mu.shape), np.broadcast_to(high, mu.shape)
    safe = np.where(sigma > 0.0, sigma, 1.0)
    a, b = (low - mu) / safe, (high - mu) / safe
    cdf_a, cdf_b = _normal_cdf(a), _normal_cdf(b)
    inside = mu * (cdf_b - cdf_a) + safe * (_normal_pdf(a) - _normal_pdf(b))
    mean = low * cdf_a + high * (1.0 - cdf_b) + inside
    return np.where(sigma > 0.0, mean, np.clip(mu, low, high))


class ScaledNoisePolicy:
    """Drive ``model`` with actions mu + scale sigma eps (eps from ``rng``); record mu, sigma and the clipping bias.

    ``records`` holds one row per decision: mu (2), sigma (2) and the bias E[clip(mu + sigma eps)] - clip(mu) (2) of
    the training noise (scale 1), whatever the scale used to drive.
    """

    def __init__(self, model: PPO, scale: float, rng: np.random.Generator) -> None:
        self.model, self.scale, self.rng = model, scale, rng
        self.records: list[NDArray[np.float64]] = []

    def __call__(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(obs)
            distribution = self.model.policy.get_distribution(tensor).distribution
            mu = distribution.mean.numpy()[0].astype(np.float64)
            sigma = distribution.stddev.numpy()[0].astype(np.float64)
        bias = clipped_gaussian_mean(mu, sigma) - np.clip(mu, ACTION_LOW, ACTION_HIGH)
        self.records.append(np.concatenate([mu, sigma, bias]))
        noise = self.rng.standard_normal(mu.shape) if self.scale > 0.0 else np.zeros_like(mu)
        return (mu + self.scale * sigma * noise).astype(np.float32)

    def reset_records(self) -> NDArray[np.float64]:
        """Return the records so far as an array (decisions x 6) and start a new list."""
        rows = np.array(self.records).reshape(-1, 6)
        self.records = []
        return rows
