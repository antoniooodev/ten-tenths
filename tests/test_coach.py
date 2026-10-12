"""Coach: the observation without the plan, the windows of the prior data, the jump-start schedule and a short
online training on a small dataset.

Covers the coach layout (level C: the privileged observation without the v_ref and reference-acceleration previews,
offline and online), the decision windows cut from an offline dataset (mean action, discounted reward, terminal,
bootstrap discount, next row), the falling teacher prefix of the episodes, the files of a trained coach and an
evaluation episode driven by it like a student, and a coach started from a policy of level C trained offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.env import make_student_env, randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, train_bc
from contextual_bicycle_pacejka.learning.coach import (
    REPORT_FILE,
    CoachConfig,
    PriorWindows,
    guide_horizon,
    perceived_friction,
    train_coach,
)
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.policy import POLICY_FILE, STUDENT_FILE, policy_layout
from contextual_bicycle_pacejka.model.student import coach_observation, student_observations
from contextual_bicycle_pacejka.spec.constants import OBS_DIM
from contextual_bicycle_pacejka.spec.layout import COACH_INDEX, OBS, PLAN_SLICES, SETUP, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"


def test_coach_layout() -> None:
    """Level C keeps every privileged entry except the v_ref and reference-acceleration previews."""
    layout = student_layout("C")
    assert layout.coach and not layout.privileged and layout.dim == OBS_DIM - 24 == len(COACH_INDEX)
    plan = np.concatenate([np.arange(s.start, s.stop) for s in PLAN_SLICES])
    np.testing.assert_array_equal(np.sort(np.concatenate([COACH_INDEX, plan])), np.arange(OBS_DIM))
    assert OBS.VREF_START in plan and OBS.AX_PREVIEW_START in plan and OBS.KAPPA_START in COACH_INDEX
    obs = np.random.default_rng(0).normal(size=(5, OBS_DIM)).astype(np.float32)
    altered = obs.copy()
    altered[:, plan] = 7.0
    np.testing.assert_array_equal(coach_observation(altered), coach_observation(obs))
    rows = student_observations(obs, np.zeros((5, 2)), np.zeros((5, 2)), 1.0, layout)
    np.testing.assert_array_equal(rows, obs[:, COACH_INDEX])
    with pytest.raises(ValueError):
        student_layout("C", oracle=("mu",))


def test_coach_environment_observation() -> None:
    """The student wrapper of level C returns the coach observation of the privileged one."""
    env = make_student_env(MONZA_R0, student_layout("C"))
    options = {"setup": randomized_setup(1.0, np.random.default_rng(3)), "initial_s": 935.35833801, "initial_vx": 30.0}
    obs, info = env.reset(seed=3, options=options)
    np.testing.assert_array_equal(obs, coach_observation(info["privileged_obs"]))
    obs, _, _, _, info = env.step(np.array([0.0, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(obs, coach_observation(info["privileged_obs"]))


def test_prior_windows(tmp_path: Path) -> None:
    """A window of the prior data averages the executed actions, discounts the rewards and ends with its lap."""
    n, high = 7, np.array([0.5, 1.0])
    data = {"observations": np.arange(n * OBS_DIM, dtype=np.float32).reshape(n, OBS_DIM),
            "final_observations": -np.ones((2, OBS_DIM), dtype=np.float32), "episode_ends": np.array([4, 7]),
            "actions_exec": np.column_stack([np.linspace(-0.3, 0.3, n), np.linspace(0.0, 0.6, n)]) * high,
            "rewards": np.arange(1, n + 1, dtype=np.float32), "terminals": np.array([0, 0, 0, 0, 0, 0, 1.0]),
            "actions": np.column_stack([np.zeros(n), np.linspace(0.1, 0.7, n)]).astype(np.float32)}
    np.savez(tmp_path / "mdp.npz", **data)
    windows = PriorWindows(tmp_path / "mdp.npz", repeat=3, step_gamma=0.5)
    assert windows.n == n
    np.testing.assert_allclose(windows.rewards[[0, 2, 5]], [1 + 0.5 * 2 + 0.25 * 3, 3 + 0.5 * 4, 6 + 0.5 * 7])
    np.testing.assert_allclose(windows.discounts[[0, 2, 5, 6]], [0.125, 0.25, 0.25, 0.5])
    np.testing.assert_allclose(windows.actions[2], (data["actions_exec"][2:4] / [0.5, 1.0]).mean(axis=0), rtol=1e-6)
    np.testing.assert_array_equal(windows.next[[0, 1, 2, 4, 5]], [3, n, n, n + 1, n + 1])
    np.testing.assert_array_equal(windows.dones, [0, 0, 0, 0, 1, 1, 1])  # windows 4-6 reach the failure at 6
    np.testing.assert_array_equal(windows.labels, data["actions"])  # the label of a window is that of its first step


def test_perceived_friction() -> None:
    """The exploration scales only the friction entry the coach reads, within the friction range."""
    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    index = OBS.XI_START + SETUP.MU
    obs[index] = 0.0  # mu = 1.0
    perceived = perceived_friction(obs, 0.05)
    assert perceived[index] == pytest.approx(0.25) and obs[index] == 0.0
    np.testing.assert_array_equal(np.delete(perceived, index), np.delete(obs, index))
    obs[index] = 1.0  # mu = MU_HI
    assert perceived_friction(obs, 0.1)[index] == pytest.approx(1.0)


def test_guide_horizon() -> None:
    """The teacher prefix (in decisions) falls linearly from guide_steps to zero over guide_fraction."""
    config = CoachConfig(env_steps=1000, guide_steps=200, guide_fraction=0.5, action_repeat=10)
    assert guide_horizon(0, config) == 20 and guide_horizon(250, config) == 10
    assert guide_horizon(500, config) == 0 == guide_horizon(900, config)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An offline dataset of two short perturbed laps of Monza."""
    root = tmp_path_factory.mktemp("coach")
    specs = episode_specs(MONZA_R0)
    previous = demonstrations.MAX_ATTEMPTS
    demonstrations.MAX_ATTEMPTS = 1
    try:
        generate_laps(MONZA_R0, "monza", "speed", root / "demos", specs, laps=[0, 1], keep_failed=True,
                      max_steps=200)
    finally:
        demonstrations.MAX_ATTEMPTS = previous
    dataset = root / "mdp.npz"
    np.savez(dataset, **assemble_mdp(mdp_laps(root / "demos", include_failed=True), {"monza": specs}))
    return dataset


def test_train_coach(small_dataset: Path, tmp_path: Path) -> None:
    """A short online training writes a coach of level C that evaluates like a student."""
    specs = episode_specs(MONZA_R0)
    dataset = small_dataset
    config = CoachConfig(env_steps=300, learning_starts=100, batch_size=32, critic_hidden=(16,), episode_steps=120,
                         guide_steps=40, log_every=100, validate_every=10_000, validation_steps=30, threads=1)
    report = train_coach(dataset, MONZA_R0, tmp_path / "coach", config)
    steps = [entry["step"] for entry in report["curve"]]
    assert len(steps) == 3 and steps == sorted(steps) and steps[-1] >= 300
    assert report["curve"][-1]["updates"] > 0 and report["curve"][-1]["updates"] % config.updates_per_decision == 0
    assert len(report["validations"]) == 1 and report["validations"][0]["laps"] == 6
    assert set(report["validations"][0]) >= {"every_step", "held"}
    for name in (POLICY_FILE, STUDENT_FILE, REPORT_FILE, "critic.pt", "vecnormalize.pkl"):
        assert (tmp_path / "coach" / name).exists()
    assert json.loads((tmp_path / "coach" / STUDENT_FILE).read_text())["target"] == "reward"
    assert policy_layout(tmp_path / "coach").coach
    spec = {"block": "held_out", "start_label": "h1", "start_s": specs[0].start_s, "start_vx": specs[0].start_vx,
            "mu": 1.0, "seed": 42, "max_steps": 40, "stall_rule": False}
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "coach")
    assert record["student_level"] == "C" and record["steps"] > 0


def test_coach_from_offline_policy(small_dataset: Path, tmp_path: Path) -> None:
    """A coach started from a policy of level C keeps its normaliser, pretrains its critic at the decision rate with
    the actor fixed, validates its start, keeps the actor fixed during the online warm-up and then trains it with the
    fixed lambda, the trust term and the falling BC weight."""
    train_bc(small_dataset, tmp_path / "init", MONZA_R0, BCConfig(epochs=1, level="C"))
    config = CoachConfig(env_steps=200, learning_starts=50, batch_size=32, critic_hidden=(16,), episode_steps=100,
                         log_every=100, validate_every=10_000, validation_steps=20, threads=1,
                         init_policy=str(tmp_path / "init"), pretrain_updates=4, bc_decay_fraction=0.5,
                         actor_warmup_steps=100)
    report = train_coach(small_dataset, MONZA_R0, tmp_path / "coach", config)
    assert set(report["pretrain"]) == {"critic_loss", "q", "fixed_lambda"}  # the actor stays fixed meanwhile
    assert report["curve"][-1]["trust"] >= 0.0  # the actor trains after the warm-up
    assert report["curve"][-1]["lambda"] == pytest.approx(report["pretrain"]["fixed_lambda"], rel=1e-5)
    assert [v["step"] for v in report["validations"]][0] == 0 and len(report["validations"]) == 2
    assert report["curve"][0]["bc_weight"] == pytest.approx(0.0) and report["curve"][-1]["bc_weight"] == 0.0
    init = (tmp_path / "init" / "vecnormalize.pkl").read_bytes()
    assert (tmp_path / "coach" / "vecnormalize.pkl").read_bytes() == init
    with pytest.raises(ValueError):
        train_coach(small_dataset, MONZA_R0, tmp_path / "bad", CoachConfig(init_policy=str(tmp_path / "coach_s2")))
