"""PPO fine-tuning (paused): initialisation from the BC policy, KL anchor, training episodes and a short run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th
from conftest import REPO

from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.learning.policy import NORMALIZER_FILE, POLICY_FILE, Policy, load_model
from contextual_bicycle_pacejka.learning.ppo_finetune import (
    AnchoredPPO,
    PPOConfig,
    TrainingEpisodes,
    finetune,
    linear_schedule,
)

MONZA_R1 = REPO / "data/tracks/monza/racing_line_R1.npz"
BC_MODEL = REPO / "models/all4"


def test_training_episodes_randomise_start_and_friction() -> None:
    config = PPOConfig()
    env = TrainingEpisodes(make_env(MONZA_R1).env, config, seed=7)
    starts, mus = [], []
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        base = env.unwrapped
        assert obs.shape == (116,) and obs.dtype == np.float32
        assert config.mu_min <= info["friction_mu"] <= config.mu_max
        assert base.setup.mu == info["friction_mu"]
        assert config.start_vx_min <= info["initial_vx"] <= config.start_vx_max
        starts.append(info["initial_s"])
        mus.append(info["friction_mu"])
    assert len(set(starts)) == 5
    _, info = env.reset(seed=3)
    assert info["friction_mu"] == mus[3]  # friction depends only on the reset seed


def test_initialisation_copies_the_bc_actor() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = VecNormalize.load(str(BC_MODEL / NORMALIZER_FILE), DummyVecEnv([lambda: make_env(MONZA_R1)]))
    venv.training = False
    model = AnchoredPPO("MlpPolicy", venv, n_steps=64, batch_size=64, policy_kwargs={"net_arch": [512, 512]},
                        device="cpu", seed=0)
    model.initialize_from(BC_MODEL / POLICY_FILE, kl_coef=0.1, kl_floor=0.02, kl_decay_steps=1000)
    reference = load_model(BC_MODEL / POLICY_FILE).policy
    obs = th.as_tensor(np.random.default_rng(0).normal(size=(32, 116)), dtype=th.float32)
    with th.no_grad():
        np.testing.assert_array_equal(model.policy.get_distribution(obs).distribution.mean.numpy(),
                                      reference.get_distribution(obs).distribution.mean.numpy())
        assert float(model.reference_kl(obs)) == pytest.approx(0.0, abs=1e-6)
    assert model.kl_weight() == pytest.approx(0.1)
    model.num_timesteps = 900
    assert model.kl_weight() == pytest.approx(0.02)  # floor reached
    with pytest.raises(ValueError):
        model.initialize_from(BC_MODEL / POLICY_FILE, kl_coef=0.01, kl_floor=0.02, kl_decay_steps=1000)
    assert linear_schedule(3e-5, 1e-5)(0.0) == pytest.approx(1e-5)


@pytest.mark.slow
def test_short_finetune_run(tmp_path: Path) -> None:
    config = PPOConfig(total_timesteps=256, n_steps=128, batch_size=64, n_epochs=1, checkpoint_every=128)
    final = finetune(BC_MODEL, MONZA_R1, tmp_path, config)
    assert (final / POLICY_FILE).exists() and (final / NORMALIZER_FILE).exists()
    assert (tmp_path / "config.json").exists()
    assert any((tmp_path / "checkpoints").iterdir())
    env = make_env(MONZA_R1)
    obs, _ = env.reset(seed=42)
    action = Policy(final, env)(obs)
    assert action.shape == (2,) and np.all(np.isfinite(action))
