"""PPO of Stable-Baselines3 configured as in Czechmanowski et al. (IROS 2025).

Separate actor (256 x 2) and critic (512 x 2) MLPs with LeakyReLU of slope 0.2, gamma 0.99 per decision, rollouts of
1024 decisions per environment, minibatches of 1024, and a learning rate decaying linearly from 1e-3 to 1e-4 over
each call to ``learn``. The other settings are the defaults of Stable-Baselines3, except 5 epochs per update and a
target KL divergence of 0.02: the epochs of an update stop once the approximate KL exceeds 1.5 times it, which keeps
a policy that has already narrowed from jumping when a new stage restarts the learning rate at 1e-3.

A model can start from the weights of another one (the previous speed stage): actor, critic and log-std are copied,
while the optimiser and the learning-rate schedule start afresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

LEAKY_RELU_SLOPE: float = 0.2


class LeakyReLU(nn.LeakyReLU):
    """LeakyReLU with the slope of Czechmanowski et al.; a class, as Stable-Baselines3 expects."""

    def __init__(self) -> None:
        super().__init__(negative_slope=LEAKY_RELU_SLOPE)


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters of PPO."""

    actor: tuple[int, ...] = (256, 256)
    critic: tuple[int, ...] = (512, 512)
    gamma: float = 0.99
    learning_rate_start: float = 1e-3
    learning_rate_end: float = 1e-4
    n_steps: int = 1024  # decisions per environment and rollout
    batch_size: int = 1024
    n_epochs: int = 5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02


def make_ppo(env: VecEnv, config: PPOConfig, seed: int, init_from: str | Path | None = None) -> PPO:
    """Return a PPO model on ``env``; with ``init_from`` (a saved model) its policy starts from those weights."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LinearSchedule(config.learning_rate_start, config.learning_rate_end, 1.0),
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs={"net_arch": {"pi": list(config.actor), "vf": list(config.critic)}, "activation_fn": LeakyReLU},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    if init_from is not None:
        previous = PPO.load(str(init_from), device="cpu")
        model.policy.load_state_dict(previous.policy.state_dict())
    return model
