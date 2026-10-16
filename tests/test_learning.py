"""PPO configuration, starting from the weights of another model, and the training curves."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import REPO

from contextual_bicycle_pacejka.env import make_vec_env
from contextual_bicycle_pacejka.learning.monitor import TrainingCurves
from contextual_bicycle_pacejka.learning.ppo import PPOConfig, make_ppo
from contextual_bicycle_pacejka.model.termination import FAILURES
from contextual_bicycle_pacejka.spec.constants import OBS_DIM

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"
SMALL = PPOConfig(n_steps=64, batch_size=64, n_epochs=1)


def linear_sizes(module: torch.nn.Module) -> list[tuple[int, int]]:
    return [(layer.in_features, layer.out_features) for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def test_architecture_and_schedule() -> None:
    env = make_vec_env(CATALUNYA, 2, seed=0)
    model = make_ppo(env, PPOConfig(), seed=0)
    policy = model.policy
    assert linear_sizes(policy.mlp_extractor.policy_net) == [(OBS_DIM, 256), (256, 256)]
    assert linear_sizes(policy.mlp_extractor.value_net) == [(OBS_DIM, 512), (512, 512)]
    assert linear_sizes(policy.action_net) == [(256, 2)] and linear_sizes(policy.value_net) == [(512, 1)]
    activations = [m for m in policy.mlp_extractor.modules() if isinstance(m, torch.nn.LeakyReLU)]
    assert len(activations) == 4 and all(m.negative_slope == 0.2 for m in activations)
    assert model.lr_schedule(1.0) == pytest.approx(1e-3) and model.lr_schedule(0.0) == pytest.approx(1e-4)
    assert (model.gamma, model.n_steps, model.batch_size, model.n_epochs) == (0.99, 1024, 1024, 5)
    assert model.target_kl == 0.02
    env.close()


def test_start_from_another_model_and_write_curves(tmp_path) -> None:
    env = make_vec_env(CATALUNYA, 2, seed=0, stage=2)
    first = make_ppo(env, SMALL, seed=0)
    curves = tmp_path / "curves.jsonl"
    first.learn(total_timesteps=256, callback=TrainingCurves(curves))
    first.save(tmp_path / "first.zip")
    lines = [json.loads(line) for line in curves.read_text().splitlines()]
    assert len(lines) == 2 and lines[-1]["timesteps"] == 256 and lines[-1]["decisions"] == 128
    assert set(lines[-1]["failures_per_1000"]) == set(FAILURES) and 0.0 < lines[-1]["progress_per_decision"] < 1.5
    assert any(key.startswith("train/") for key in lines[-1]["losses"])

    second = make_ppo(make_vec_env(CATALUNYA, 2, seed=1, stage=3), SMALL, seed=1, init_from=tmp_path / "first.zip")
    for (name, a), b in zip(first.policy.state_dict().items(), second.policy.state_dict().values(), strict=True):
        assert torch.equal(a, b), name
    assert len(second.policy.optimizer.state) == 0  # the optimiser starts afresh
    env.close()
    second.env.close()

