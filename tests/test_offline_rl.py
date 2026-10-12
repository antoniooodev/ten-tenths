"""Offline reinforcement learning: IQL and TD3+BC with an asymmetric critic on a small offline dataset.

Covers the training of both algorithms on short demonstration laps, the files of the policy directory, the
student layout of the actor, an evaluation episode driven by the trained actor, the critic diagnostics, the
failure ablation (failures turned into truncations without the crash penalty) and the traction-cost constraint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th
from conftest import REPO

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.offline_rl import (
    CRITIC_FILE,
    REPORT_FILE,
    OfflineRLConfig,
    strip_failures,
    train_offline_rl,
)
from contextual_bicycle_pacejka.learning.policy import STUDENT_FILE, policy_layout
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS, compute_reward
from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.spec.layout import OBS, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("offline_rl")
    specs = episode_specs(MONZA_R0)
    previous = demonstrations.MAX_ATTEMPTS
    demonstrations.MAX_ATTEMPTS = 1
    try:
        generate_laps(MONZA_R0, "monza", "speed", root / "demos", specs, laps=[0, 1, 2], keep_failed=True,
                      max_steps=200)
    finally:
        demonstrations.MAX_ATTEMPTS = previous
    path = root / "mdp.npz"
    np.savez(path, **assemble_mdp(mdp_laps(root / "demos", include_failed=True), {"monza": specs}))
    return path


@pytest.mark.parametrize("algorithm, target", [("iql", "exec"), ("td3bc", "label")])
def test_offline_rl(dataset: Path, tmp_path: Path, algorithm: str, target: str) -> None:
    config = OfflineRLConfig(algorithm=algorithm, steps=20, batch_size=64, log_every=10, critic_hidden=(32,),
                             threads=1)
    report = train_offline_rl(dataset, tmp_path / algorithm, MONZA_R0, config)
    assert len(report["curve"]) == 2 and np.isfinite(report["curve"][-1]["critic_loss"])
    assert 0.0 <= report["diagnostics"]["fraction_label_preferred"] <= 1.0
    for name in ("policy.zip", "vecnormalize.pkl", STUDENT_FILE, CRITIC_FILE, REPORT_FILE):
        assert (tmp_path / algorithm / name).exists(), name
    assert json.loads((tmp_path / algorithm / STUDENT_FILE).read_text())["target"] == target
    assert policy_layout(tmp_path / algorithm) == student_layout("S2", oracle=("mu",))
    specs = episode_specs(MONZA_R0)
    spec = {"block": "held_out", "start_label": "h1", "start_s": specs[0].start_s, "start_vx": specs[0].start_vx,
            "mu": 1.0, "seed": 42, "max_steps": 30, "stall_rule": False}
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / algorithm)
    assert record["student_level"] == "S2" and record["steps"] > 0


def test_offline_rl_rejects_privileged_actor(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_offline_rl(dataset, tmp_path / "s0", MONZA_R0, OfflineRLConfig(level="S0", oracle=(), steps=1))
    with pytest.raises(ValueError):
        train_offline_rl(dataset, tmp_path / "bad", MONZA_R0, OfflineRLConfig(algorithm="cql", steps=1))


def test_strip_failures() -> None:
    state = VehicleState(vx=30.0, vy=2.0)
    flags = {"slip_saturation": True}
    failed = compute_reward(state, 30.0, 5.0, 0.01, OFFLINE_REWARD_WEIGHTS, True, flags, False)
    cut = compute_reward(state, 30.0, 5.0, 0.01, OFFLINE_REWARD_WEIGHTS, False, {}, False)
    rewards = np.array([0.2, 0.2, 0.2, 0.2, failed], dtype=np.float32)
    terminals = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    final = np.zeros((2, 116), dtype=np.float32)
    final[1, OBS.VX], final[1, OBS.VY] = state.vx, state.vy
    stripped, dones = strip_failures(rewards, terminals, np.array([3, 5]), final)
    assert not dones.any() and stripped[:4].tolist() == rewards[:4].tolist()
    assert stripped[4] == pytest.approx(cut, abs=1e-5)
    assert terminals[4] == 1.0  # the input is not modified


def test_offline_rl_without_failures(dataset: Path, tmp_path: Path) -> None:
    config = OfflineRLConfig(algorithm="td3bc", steps=4, batch_size=64, log_every=2, critic_hidden=(32,),
                             threads=1, failures=False)
    report = train_offline_rl(dataset, tmp_path / "td3bc", MONZA_R0, config)
    assert report["config"]["failures"] is False and np.isfinite(report["curve"][-1]["critic_loss"])


def test_offline_rl_with_cost_constraint(dataset: Path, tmp_path: Path) -> None:
    """TD3+BC with a cost critic: the multiplier stays in its range, the cost networks are saved, and the
    constraint is refused for IQL or a negative budget."""
    config = OfflineRLConfig(algorithm="td3bc", steps=8, batch_size=64, log_every=4, critic_hidden=(32,),
                             threads=1, cost_limit=0.5, cost_lr=0.1)
    report = train_offline_rl(dataset, tmp_path / "td3bc", MONZA_R0, config)
    last = report["curve"][-1]
    assert np.isfinite(last["cost_critic_loss"]) and 0.0 <= last["multiplier"] <= config.cost_multiplier_max
    assert 0.0 <= report["cost_diagnostics"]["multiplier"] <= config.cost_multiplier_max
    assert "cost_critic" in th.load(tmp_path / "td3bc" / CRITIC_FILE, weights_only=False)
    for bad in (OfflineRLConfig(algorithm="iql", steps=1, cost_limit=0.5),
                OfflineRLConfig(algorithm="td3bc", steps=1, cost_limit=-1.0)):
        with pytest.raises(ValueError):
            train_offline_rl(dataset, tmp_path / "bad", MONZA_R0, bad)
