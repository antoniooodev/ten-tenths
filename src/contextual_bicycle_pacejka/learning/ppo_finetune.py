"""PPO fine-tuning of a behaviour-cloning policy with a decaying KL anchor to it (paused line of work).

    loss = PPO clipped loss + vf_coef * value loss + w_KL(t) * KL(pi_theta || pi_BC)
    w_KL(t) = max(kl_floor, kl_coef * (1 - t / kl_decay_steps))

Initialisation: actor weights and log-std are copied from the BC policy (the critic starts fresh); the BC
observation normaliser is loaded and frozen, rewards are scaled by its (frozen) return statistics.
Training episodes start at a uniform position and speed with a normally distributed friction and end at the
first completed lap; the reward is defined in ``model/reward.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.learning.policy import NET_ARCH, NORMALIZER_FILE, POLICY_FILE, load_model
from contextual_bicycle_pacejka.model.dynamics import DIAGNOSTIC_KEYS
from contextual_bicycle_pacejka.model.observation import build_observation
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold

MIN_AVERAGE_SPEED_MPS = 6.0  # [m/s] episode time limit = 1.5 laps at this speed


@dataclass
class PPOConfig:
    """Settings of the fine-tuning run (values of the Monza pilot)."""

    total_timesteps: int = 1_000_000
    seed: int = 42
    n_steps: int = 65_536
    batch_size: int = 1024
    n_epochs: int = 3
    learning_rate: float = 3e-5
    learning_rate_final: float = 1e-5
    gamma: float = 0.9999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    kl_coef: float = 0.1
    kl_floor: float = 0.02
    kl_decay_steps: int = 1_000_000
    mu_mean: float = 1.0
    mu_std: float = 0.15
    mu_min: float = 0.8
    mu_max: float = 1.2
    start_vx_min: float = 5.0  # [m/s]
    start_vx_max: float = 15.0  # [m/s]
    checkpoint_every: int = 100_000


class TrainingEpisodes(gym.Wrapper):
    """Random start (s ~ U[0, L), vx ~ U[vx_min, vx_max]), random friction (normal, clipped), end at the first lap."""

    def __init__(self, env: gym.Env, config: PPOConfig, seed: int) -> None:
        super().__init__(env)
        self.config = config
        self._start_rng = np.random.default_rng(seed)
        self._mu_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        base = self.env.unwrapped
        s0 = float(self._start_rng.uniform(0.0, base.track.track_length))
        vx0 = float(self._start_rng.uniform(self.config.start_vx_min, self.config.start_vx_max))
        if seed is not None:
            self._mu_rng = np.random.default_rng(int(seed))
        mu = float(np.clip(self._mu_rng.normal(self.config.mu_mean, self.config.mu_std),
                           self.config.mu_min, self.config.mu_max))
        _, info = self.env.reset(seed=seed, options={**(options or {}), "initial_s": s0, "initial_vx": vx0})
        base.setup.mu = mu
        base.alpha_sat = (compute_saturation_threshold(base.setup, "front"),
                          compute_saturation_threshold(base.setup, "rear"))
        obs = build_observation(base.state, base.setup, dict.fromkeys(DIAGNOSTIC_KEYS, 0.0), base.track)
        return obs.astype(np.float32), {**info, "friction_mu": mu, "initial_s": s0, "initial_vx": vx0}

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("lap_complete", False) and not terminated:
            terminated = True
            info = {**info, "is_success": True}
        return obs, float(reward), bool(terminated), bool(truncated), info


class AnchoredPPO(PPO):
    """PPO whose loss includes a decaying KL divergence to a frozen reference (BC) policy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.reference_policy: ActorCriticPolicy | None = None
        self.kl_coef = self.kl_floor = 0.0
        self.kl_decay_steps = 1
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["reference_policy"]

    def initialize_from(self, path: str | Path, kl_coef: float, kl_floor: float, kl_decay_steps: int) -> None:
        """Copy the actor and log-std of the policy at ``path`` and keep it frozen as the KL reference."""
        if not 0.0 <= kl_floor <= kl_coef or kl_decay_steps < 1:
            raise ValueError("the KL anchor needs 0 <= kl_floor <= kl_coef and kl_decay_steps >= 1")
        source = load_model(path).policy
        self.policy.features_extractor.load_state_dict(source.features_extractor.state_dict())
        self.policy.mlp_extractor.policy_net.load_state_dict(source.mlp_extractor.policy_net.state_dict())
        self.policy.action_net.load_state_dict(source.action_net.state_dict())
        with th.no_grad():
            self.policy.log_std.copy_(source.log_std)
        self.kl_coef, self.kl_floor, self.kl_decay_steps = float(kl_coef), float(kl_floor), int(kl_decay_steps)
        source.set_training_mode(False)
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        self.reference_policy = source

    def kl_weight(self) -> float:
        """Current weight of the KL anchor."""
        remaining = max(1.0 - self.num_timesteps / self.kl_decay_steps, 0.0)
        return max(self.kl_floor, self.kl_coef * remaining)

    def reference_kl(self, observations: th.Tensor) -> th.Tensor:
        """KL(pi_theta || pi_reference) of the diagonal Gaussian actors, summed over actions, averaged over states."""
        distribution = self.policy.get_distribution(observations).distribution
        with th.no_grad():
            reference = self.reference_policy.get_distribution(observations).distribution
        mean, std = distribution.mean, distribution.stddev
        kl = th.log(reference.stddev / std) + (std.square() + (mean - reference.mean).square()) \
            / (2.0 * reference.stddev.square()) - 0.5
        return kl.sum(dim=1).mean()

    def train(self) -> None:
        """One PPO update (SB3 algorithm) with the KL anchor added to the loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        weight = self.kl_weight() if self.reference_policy is not None else 0.0
        pg_losses, value_losses, kl_terms, clip_fractions, approx_kls = [], [], [], [], []
        loss = th.tensor(0.0)
        continue_training = True
        for _ in range(self.n_epochs):
            approx_kls = []
            for batch in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, batch.actions)
                values = values.flatten()
                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_ratio = log_prob - batch.old_log_prob
                ratio = th.exp(log_ratio)
                policy_loss = -th.min(advantages * ratio,
                                      advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                value_loss = F.mse_loss(batch.returns, values)
                entropy_loss = -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if weight > 0.0:
                    kl = self.reference_kl(batch.observations)
                    kl_terms.append(float(kl.detach()))
                    loss = loss + weight * kl
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                with th.no_grad():
                    approx_kl = th.mean((ratio - 1) - log_ratio).item()
                approx_kls.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kls) if approx_kls else 0.0)
        self.logger.record("train/reference_kl", float(np.mean(kl_terms)) if kl_terms else 0.0)
        self.logger.record("train/kl_weight", weight)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_variance(self.rollout_buffer.values.flatten(),
                                                                          self.rollout_buffer.returns.flatten()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def linear_schedule(initial: float, final: float):  # noqa: ANN201
    """SB3 schedule decaying linearly from ``initial`` to ``final`` over the run."""
    return lambda progress_remaining: final + (initial - final) * float(progress_remaining)


def finetune(bc_model_dir: str | Path, racing_line: str | Path, output: str | Path,
             config: PPOConfig | None = None) -> Path:
    """Fine-tune the BC policy in ``bc_model_dir`` on ``racing_line``; return the final model directory."""
    config = config or PPOConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bc_model_dir = Path(bc_model_dir)

    def make() -> gym.Env:
        env = make_env(racing_line)
        length = env.unwrapped.track.track_length
        env = gym.wrappers.TimeLimit(TrainingEpisodes(env.env, config, config.seed),
                                     max_episode_steps=int(length / MIN_AVERAGE_SPEED_MPS * 100.0 * 1.5))
        return Monitor(env, str(output / "monitor"))

    (output / "monitor").mkdir(exist_ok=True)
    venv = VecNormalize.load(str(bc_model_dir / NORMALIZER_FILE), DummyVecEnv([make]))
    venv.training = False  # observation statistics stay those of the BC data
    venv.norm_reward = True
    model = AnchoredPPO(ActorCriticPolicy, venv, learning_rate=linear_schedule(config.learning_rate,
                                                                               config.learning_rate_final),
                        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                        gamma=config.gamma, gae_lambda=config.gae_lambda, clip_range=config.clip_range,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm,
                        target_kl=config.target_kl, policy_kwargs={"net_arch": list(NET_ARCH)}, seed=config.seed,
                        device="cpu", verbose=1)
    model.initialize_from(bc_model_dir / POLICY_FILE, config.kl_coef, config.kl_floor, config.kl_decay_steps)
    checkpoints = CheckpointCallback(save_freq=config.checkpoint_every, save_path=str(output / "checkpoints"),
                                     name_prefix="ppo", save_vecnormalize=True)
    model.learn(total_timesteps=config.total_timesteps, callback=checkpoints)
    final = output / "final"
    final.mkdir(exist_ok=True)
    model.save(str(final / POLICY_FILE))
    venv.save(str(final / NORMALIZER_FILE))
    (output / "config.json").write_text(json.dumps(asdict(config), indent=1))
    return final
