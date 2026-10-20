"""PPO configuration, starting from the weights of another model, and the training curves."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import torch
from conftest import REPO

from contextual_bicycle_pacejka.env import make_vec_env
from contextual_bicycle_pacejka.learning.curriculum import SpeedCurriculum, failures_per_lap, next_top_speed
from contextual_bicycle_pacejka.learning.monitor import TrainingCurves
from contextual_bicycle_pacejka.learning.ppo import PPOConfig, make_ppo, widen_inputs
from contextual_bicycle_pacejka.model.termination import FAILURES
from contextual_bicycle_pacejka.spec.constants import OBS_DIM

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"
TRAIN_SCRIPT = REPO / "scripts/train_curriculum.py"
EVALUATE_SCRIPT = REPO / "scripts/evaluate_stage.py"
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
    assert 0.0 < lines[-1]["mean_speed"] < 15.0 and set(lines[-1]["slip_beyond_peak"]) == {"front", "rear"}
    assert any(key.startswith("train/") for key in lines[-1]["losses"])

    second = make_ppo(make_vec_env(CATALUNYA, 2, seed=1, stage=3), SMALL, seed=1, init_from=tmp_path / "first.zip")
    for (name, a), b in zip(first.policy.state_dict().items(), second.policy.state_dict().values(), strict=True):
        assert torch.equal(a, b), name
    assert len(second.policy.optimizer.state) == 0  # the optimiser starts afresh
    env.close()
    second.env.close()



def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_learning_rate_of_the_command_line() -> None:
    """--lr-start, --lr-end and --gamma of scripts/train_curriculum.py reach PPO; by default, those of PPOConfig."""
    script = load_script(TRAIN_SCRIPT)
    required = ["--seed", "0", "--stages", "3", "--budget", "1000"]
    assert script.ppo_config(script.parse_args(required)) == PPOConfig()
    options = ["--lr-start", "3e-4", "--lr-end", "3e-5", "--gamma", "0.995"]
    config = script.ppo_config(script.parse_args([*required, *options]))
    assert (config.learning_rate_start, config.learning_rate_end, config.gamma) == (3e-4, 3e-5, 0.995)
    env = make_vec_env(CATALUNYA, 2, seed=0)
    model = make_ppo(env, config, seed=0)
    assert model.lr_schedule(1.0) == pytest.approx(3e-4) and model.lr_schedule(0.0) == pytest.approx(3e-5)
    assert model.lr_schedule(0.5) == pytest.approx(1.65e-4)
    env.close()


def test_oracle_starts_as_the_nominal_policy(tmp_path) -> None:
    """An Oracle started from a Nominal model gets a zero column for b: same actions and values whatever b is."""
    nominal = make_ppo(make_vec_env(CATALUNYA, 2, seed=0, stage=2), SMALL, seed=0)
    nominal.save(tmp_path / "nominal.zip")
    oracle_env = make_vec_env(CATALUNYA, 2, seed=0, stage=3, b_range=(-0.2, 0.17), oracle=True)
    oracle = make_ppo(oracle_env, SMALL, seed=1, init_from=tmp_path / "nominal.zip")
    first = oracle.policy.mlp_extractor.policy_net[0].weight
    assert first.shape == (256, OBS_DIM + 1) and torch.all(first[:, -1] == 0.0)
    obs = torch.as_tensor(np.random.default_rng(0).normal(size=(8, OBS_DIM)), dtype=torch.float32)
    for b in (-1.0, 0.0, 0.85):
        with_b = torch.cat([obs, torch.full((8, 1), b)], dim=1)
        with torch.no_grad():
            assert torch.allclose(oracle.policy.predict_values(with_b), nominal.policy.predict_values(obs))
            assert torch.allclose(oracle.policy.get_distribution(with_b).distribution.mean,
                                  nominal.policy.get_distribution(obs).distribution.mean)
    with pytest.raises(ValueError):
        widen_inputs({"w": torch.zeros(3, 5)}, {"w": torch.zeros(3, 4)})
    nominal.env.close()
    oracle_env.close()


def test_held_out_values_and_protocol_balance_of_the_command_line(tmp_path) -> None:
    """--b-held-out of the training script and --b of the evaluation script; the evaluation reads an Oracle from the
    size of its observation."""
    args = load_script(TRAIN_SCRIPT).parse_args(["--seed", "0", "--stages", "3", "--budget", "1000",
                                                 "--b-held-out", "-0.1", "0.09"])
    assert args.b_held_out == [-0.1, 0.09]
    evaluate = load_script(EVALUATE_SCRIPT)
    env = make_vec_env(CATALUNYA, 2, seed=0, oracle=True)
    make_ppo(env, SMALL, seed=0).save(tmp_path / "oracle.zip")
    env.close()
    model = evaluate.PPO.load(tmp_path / "oracle.zip", device="cpu")
    track = evaluate.RealTrack(str(CATALUNYA))
    protocol = evaluate.protocol_env(model, track, 3, -0.13)
    obs, info = protocol.reset(seed=0)
    assert protocol.oracle and info["setup"][-1] == pytest.approx(-0.13) and obs[-1] == pytest.approx(-0.65)
    assert not evaluate.protocol_env(make_ppo(make_vec_env(CATALUNYA, 2, seed=0), SMALL, seed=0), track, 3, 0.0).oracle


def test_speed_curriculum_rule() -> None:
    """Failures per equivalent lap, the rise of 2.5 m/s, and no bound once the rise reaches the sizing speed."""
    assert failures_per_lap(1, 46_498, 10.0, 4649.8) == pytest.approx(0.2, rel=1e-3)  # 5 laps of 9300 decisions
    assert next_top_speed(10.0) == 12.5 and next_top_speed(65.0) == 67.5 and next_top_speed(67.5) is None


def test_speed_curriculum_raises_the_top_speed(tmp_path) -> None:
    """With a threshold no rollout can miss, the top speed of every environment rises after each window of rollouts,
    and every change is logged; the free stage keeps its observation."""
    env = make_vec_env(CATALUNYA, 2, seed=0, stage=7, workers=0)
    log = tmp_path / "curriculum.jsonl"
    curriculum = SpeedCurriculum(4649.8, log, threshold=1e9, window=2)
    make_ppo(env, SMALL, seed=0).learn(total_timesteps=6 * 128, callback=curriculum)
    changes = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["top_speed"] for c in changes] == [10.0, 12.5, 15.0, 17.5]
    assert [s.top_speed for s in env.get_attr("stage")] == [17.5, 17.5] and curriculum.top_speed == 17.5
    assert env.get_attr("stage")[0].sizing_speed == 70.0
    env.close()
