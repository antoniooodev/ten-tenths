"""Learning pipeline: exact agreement with the original implementation.

Covers the demonstration-lap specifications and perturbed laps, the transitions logged for the offline datasets and
the rows of their next states, the column statistics by chunks (``learning/statistics.py``), behaviour cloning
(initialisation, trained weights, normaliser), the behaviour-cloning gate and one weighted-DAgger iteration.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import GOLDEN, REPO, array_digest, golden_json, golden_npz

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.factory import make_env, make_student_env
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, GateConfig, gate, train_bc
from contextual_bicycle_pacejka.learning.dagger import DAggerConfig, StartCondition, rollout, run_dagger
from contextual_bicycle_pacejka.learning.datasets import (
    MDP_LAP_KEYS,
    assemble_mdp,
    mdp_laps,
    next_index,
    student_dataset,
)
from contextual_bicycle_pacejka.learning.demonstrations import (
    LapSpec,
    attempt_file,
    episode_specs,
    generate_laps,
    run_lap,
)
from contextual_bicycle_pacejka.learning.policy import (
    NORMALIZER_FILE,
    POLICY_FILE,
    STUDENT_FILE,
    Policy,
    load_model,
    new_bc_model,
    policy_env,
    policy_layout,
)
from contextual_bicycle_pacejka.learning.statistics import column_moments, standardised
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.spec.layout import student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"
SPECS = GOLDEN / "specs_mat_teacher_20laps.npz"


def _racing_line_R0(circuit: str) -> Path:
    return REPO / f"data/tracks/{circuit}/racing_line_R0.npz"


def _assert_weights(state_dict: dict, expected: dict) -> None:
    for name, tensor in state_dict.items():
        assert array_digest(tensor.detach().cpu().numpy()) == expected[name]["sha256"], name


# --------------------------------------------------------------------------------------------- demonstrations
def test_episode_specs_monza() -> None:
    golden = golden_npz("specs_mat_teacher_20laps")
    specs = episode_specs(MONZA_R0)
    np.testing.assert_array_equal(np.stack([s.setup for s in specs]), golden["setup_per_lap"])
    np.testing.assert_array_equal([s.mu for s in specs], golden["mu_per_lap"])
    np.testing.assert_array_equal([s.start_s for s in specs], golden["start_s_per_lap"])
    np.testing.assert_array_equal([s.start_vx for s in specs], golden["start_vx_per_lap"])


@pytest.mark.parametrize("circuit", ["spa", "catalunya", "budapest"])
def test_episode_specs_other_circuits(circuit: str) -> None:
    golden = golden_npz("specs_mat_teacher_20laps")
    specs = episode_specs(_racing_line_R0(circuit))
    np.testing.assert_array_equal([s.start_s for s in specs], golden[f"{circuit}_start_s"])
    np.testing.assert_array_equal([s.start_vx for s in specs], golden[f"{circuit}_start_vx"])
    np.testing.assert_array_equal(np.stack([s.setup for s in specs]), golden["setup_per_lap"])


DEMO_CASES = golden_json("demo_cases")
DEMO_NAMES = [name for name in DEMO_CASES if name.startswith("demo_")]


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_demonstration_lap(name: str) -> None:
    case = DEMO_CASES[name]
    golden = golden_npz(name)
    spec = episode_specs(_racing_line_R0(case["circuit"]))[case["lap"]]
    summary, arrays = run_lap(_racing_line_R0(case["circuit"]), spec, case["kind"], case["attempt"], case["circuit"],
                              max_steps=case["steps"])
    for new, legacy in (("obs", "obs"), ("label", "label"), ("exec", "exec"), ("speed_noise", "delta"),
                        ("lateral_noise", "eta"), ("s", "s"), ("label_empty", "label_no_safe"),
                        ("exec_projected", "exec_projected")):
        np.testing.assert_array_equal(arrays[new], golden[legacy], err_msg=new)
    for key in ("steps", "completed", "events", "healthy", "progress_m", "terminal_s"):
        assert summary[key] == case[key], key
    assert summary["empty_steps"] == case["no_safe"]
    assert summary["projected_steps"] == case["exec_projected_steps"]


def test_demonstration_generators_were_equivalent() -> None:
    """The original Monza-only and multi-circuit generators produced identical laps (recorded check)."""
    assert DEMO_CASES["equiv_speed_gen_vs_gen_mc"] and DEMO_CASES["equiv_lateral_gen_vs_gen_mc"]


# --------------------------------------------------------------------------------------------- offline datasets
def _short_lap(name: str = "demo_monza_speed_l0_a0") -> tuple[LapSpec, dict, dict]:
    case = DEMO_CASES[name]
    spec = episode_specs(_racing_line_R0(case["circuit"]))[case["lap"]]
    summary, arrays = run_lap(_racing_line_R0(case["circuit"]), spec, case["kind"], case["attempt"], case["circuit"],
                              max_steps=case["steps"])
    return spec, summary, arrays


def test_logged_transitions_are_environment_transitions() -> None:
    """Replaying the executed actions reproduces the next observations, rewards, traction costs and IMU channels;
    the logged student observations of the dataset equal those of the student environment."""
    spec, summary, arrays = _short_lap()
    assert not summary["terminated"] and not summary["completed"]  # ended by the step limit: a timeout
    data = assemble_mdp([{**{k: arrays[k] for k in MDP_LAP_KEYS}, "summary": summary}], {"monza": [spec] * 20})
    layout = student_layout("S1")
    student, following = student_dataset(data, layout)
    assert len(student) == summary["steps"] + 1  # the transitions, then the lap's final observation
    env = make_student_env(MONZA_R0, layout, max_episode_steps=summary["steps"], reward_weights=OFFLINE_REWARD_WEIGHTS)
    obs, info = env.reset(seed=spec.reset_seed, options={"setup": spec.setup, "initial_s": spec.start_s,
                                                         "initial_vx": spec.start_vx})
    for t, action in enumerate(data["actions_exec"]):
        np.testing.assert_array_equal(info["privileged_obs"], data["observations"][t])
        np.testing.assert_array_equal(np.asarray(info["imu"], dtype=np.float32), data["imu"][t])
        np.testing.assert_array_equal(obs, student[t])
        obs, reward, _, _, info = env.step(action)
        assert np.float32(reward) == data["rewards"][t]
        assert np.float32(info["traction_cost"]) == data["traction_costs"][t]
        np.testing.assert_array_equal(obs, student[following[t]])
    np.testing.assert_array_equal(info["privileged_obs"], data["final_observations"][0])
    np.testing.assert_array_equal(data["timeouts"], np.eye(1, summary["steps"], summary["steps"] - 1)[0])
    assert not data["terminals"].any()


def test_generate_laps_keeps_failed_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demonstrations, "MAX_ATTEMPTS", 2)
    specs = episode_specs(MONZA_R0)
    report = generate_laps(MONZA_R0, "monza", "speed", tmp_path, specs, laps=[3], keep_failed=True, max_steps=150)
    entry = report["laps"][0]
    assert entry["kept_attempt"] is None and entry["failed_files"] == [attempt_file(3, 0), attempt_file(3, 1)]
    assert mdp_laps(tmp_path) == []  # nothing kept
    laps = mdp_laps(tmp_path, include_failed=True)
    assert [lap["summary"]["attempt"] for lap in mdp_laps(tmp_path, include_failed=True, max_failed=1)] == [0]
    data = assemble_mdp(laps, {"monza": specs})
    np.testing.assert_array_equal(data["episode_ends"], [150, 300])
    np.testing.assert_array_equal(data["attempt_per_lap"], [0, 1])
    np.testing.assert_array_equal(data["actions"], np.concatenate([lap["label"] for lap in laps]))
    np.testing.assert_array_equal(data["actions_exec"], np.concatenate([lap["exec"] for lap in laps]))
    assert data["timeouts"][[149, 299]].all() and data["timeouts"].sum() == 2 and not data["terminals"].any()
    assert not data["healthy_per_lap"].any() and (data["speed_safety_per_lap"] == 1.25).all()
    rows = np.concatenate([data["observations"], data["final_observations"]])
    following = rows[next_index(data["episode_ends"])]
    np.testing.assert_array_equal(following[:149], data["observations"][1:150])
    np.testing.assert_array_equal(following[149], data["final_observations"][0])
    np.testing.assert_array_equal(following[150:299], data["observations"][151:300])
    np.testing.assert_array_equal(following[299], data["final_observations"][1])


def test_execution_teacher(tmp_path: Path) -> None:
    spec = episode_specs(MONZA_R0)[0]
    teacher = MATv0()
    # A distinct execution teacher with the same speed factor reproduces the clean lap.
    _, clean = run_lap(MONZA_R0, spec, "none", 0, "monza", teacher, max_steps=150)
    _, same = run_lap(MONZA_R0, spec, "none", 0, "monza", teacher, max_steps=150,
                      exec_teacher=dataclasses.replace(teacher))
    np.testing.assert_array_equal(same["exec"], clean["exec"])
    np.testing.assert_array_equal(same["exec"], same["label"])
    # A faster execution teacher drives the lap, the labels stay those of the labelling teacher.
    fast_teacher = dataclasses.replace(teacher, speed_safety=1.4)
    summary, fast = run_lap(MONZA_R0, spec, "none", 0, "monza", teacher, max_steps=150, exec_teacher=fast_teacher)
    assert summary["speed_safety"] == 1.25 and summary["exec_speed_safety"] == 1.4
    assert (fast["exec"][:, 1] > fast["label"][:, 1] + 1e-6).any()
    # A noiseless lap is deterministic: one attempt, saved when it is not kept.
    report = generate_laps(MONZA_R0, "monza", "none", tmp_path, episode_specs(MONZA_R0), laps=[0], teacher=teacher,
                           keep_failed=True, max_steps=150, exec_teacher=fast_teacher)
    assert report["laps"][0]["failed_files"] == [attempt_file(0, 0)]
    data = assemble_mdp(mdp_laps(tmp_path, include_failed=True), {"monza": episode_specs(MONZA_R0)})
    assert data["speed_safety_per_lap"].tolist() == [1.25] and data["exec_speed_safety_per_lap"].tolist() == [1.4]


def test_wide_perturbations_have_their_own_noise() -> None:
    spec = episode_specs(MONZA_R0)[0]
    _, narrow = run_lap(MONZA_R0, spec, "speed", 0, "monza", max_steps=300)
    _, wide = run_lap(MONZA_R0, spec, "speed_wide", 0, "monza", max_steps=300)
    assert np.abs(wide["speed_noise"]).max() > 0.0
    assert not np.allclose(wide["speed_noise"], 2.0 * narrow["speed_noise"])


# --------------------------------------------------------------------------------------------- behaviour cloning
def test_fresh_initialisation() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec = VecNormalize(DummyVecEnv([lambda: make_env(MONZA_R0)]), norm_obs=True, norm_reward=False)
    _assert_weights(new_bc_model(vec, 42).policy.state_dict(), golden_json("bc_fresh_init_hashes"))


def test_column_moments() -> None:
    """The moments and the standardisation by chunks equal numpy's whole-array expressions bit for bit, so the
    normalisers do not change."""
    rows = np.random.default_rng(0).normal(3.0, 2.0, size=(10_001, 7)).astype(np.float32)
    mean, var = column_moments(rows, chunk=999)
    np.testing.assert_array_equal(mean, np.mean(rows.astype(np.float64), axis=0))
    np.testing.assert_array_equal(var, np.var(rows.astype(np.float64), axis=0))
    np.testing.assert_array_equal(mean, rows.mean(axis=0, dtype=np.float64))
    np.testing.assert_array_equal(np.sqrt(var), rows.std(axis=0, dtype=np.float64))
    std = np.sqrt(var)
    np.testing.assert_array_equal(standardised(rows, mean, std, chunk=999), ((rows - mean) / std).astype(np.float32))


@pytest.mark.slow
def test_train_bc(tmp_path: Path) -> None:
    expected = golden_json("bc_trained_hashes")
    result = train_bc(GOLDEN / "bc_tiny_dataset.npz", tmp_path, MONZA_R0, BCConfig(epochs=3))
    assert result == golden_json("bc_result")
    _assert_weights(load_model(tmp_path / POLICY_FILE).policy.state_dict(), expected)
    import pickle

    with (tmp_path / NORMALIZER_FILE).open("rb") as handle:
        rms = pickle.load(handle).obs_rms
    assert array_digest(rms.mean) == expected["obs_rms_mean"]["sha256"]
    assert array_digest(rms.var) == expected["obs_rms_var"]["sha256"]
    assert rms.count == expected["obs_rms_count"]["sum"]


def test_student_behaviour_cloning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A student policy trained on an offline dataset reads the student observation of its level, records its
    level and target, and drives an evaluation episode through the student environment."""
    monkeypatch.setattr(demonstrations, "MAX_ATTEMPTS", 1)
    specs = episode_specs(MONZA_R0)
    generate_laps(MONZA_R0, "monza", "speed", tmp_path / "demos", specs, laps=[0, 1], keep_failed=True,
                  max_steps=200)
    dataset = tmp_path / "mdp.npz"
    np.savez(dataset, **assemble_mdp(mdp_laps(tmp_path / "demos", include_failed=True), {"monza": specs}))
    model_dir = tmp_path / "S2_exec"
    result = train_bc(dataset, model_dir, MONZA_R0, BCConfig(epochs=2, level="S2", target="exec"))
    assert result["epochs"] == 2
    assert json.loads((model_dir / STUDENT_FILE).read_text()) == {"level": "S2", "n_frames": 8, "stride": 5,
                                                                   "target": "exec"}
    layout = policy_layout(model_dir)
    assert layout == student_layout("S2") and policy_layout(REPO / "models/all4").privileged
    oracle_dir = tmp_path / "S1_mu"
    train_bc(dataset, oracle_dir, MONZA_R0, BCConfig(epochs=1, level="S1", stride=3, oracle=("mu",)))
    assert policy_layout(oracle_dir) == student_layout("S1", stride=3, oracle=("mu",))
    policy = Policy(model_dir, policy_env(MONZA_R0, layout))
    assert policy.model.observation_space.shape == (layout.dim,)
    policy.close()
    spec = {"block": "held_out", "start_label": "h1", "start_s": specs[0].start_s, "start_vx": specs[0].start_vx,
            "mu": 1.0, "seed": 42, "max_steps": 50, "stall_rule": False}
    seen: list[tuple[int, int]] = []
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=model_dir,
                         on_step=lambda step: seen.append((len(step["obs"]), len(step["student_obs"]))))
    assert record["student_level"] == "S2" and record["steps"] == len(seen)
    assert set(seen) == {(116, layout.dim)}


@pytest.mark.slow
@pytest.mark.datasets
def test_gate(tmp_path: Path) -> None:
    for name in (POLICY_FILE, NORMALIZER_FILE):
        shutil.copy(REPO / "models/all4" / name, tmp_path / name)
    result = gate(tmp_path, REPO / "datasets/all4.npz", MONZA_R0, GateConfig(mus=(0.9,), seeds=(43,)))
    expected = golden_json("gate_result")
    assert result["passed"] == expected["passed"]
    for lap, legacy in zip(result["laps"], expected["laps"], strict=True):
        for key in ("passed", "mu", "seed", "steps", "progress_m", "events"):
            assert lap[key] == legacy[key], key
        assert lap["empty_steps"] == legacy["no_safe"]
        assert lap["projected_steps"] == legacy["shield_projections"]


# --------------------------------------------------------------------------------------------- DAgger
def _failure_report_matches(report: dict, legacy: dict) -> None:
    for key in ("mu", "seed", "completed", "steps", "progress_m", "terminal_s_m", "events", "max_rear_ratio",
                "switched", "switch_progress_m", "switch_state"):
        assert report[key] == legacy[key], key
    assert report["empty_steps"] == legacy["no_safe_actions"]


@pytest.mark.slow
def test_dagger_rollout() -> None:
    golden = golden_npz("dagger_rollout")
    recorded = golden_json("dagger_rollout")
    start = StartCondition.from_racing_line(MONZA_R0)
    with np.load(SPECS) as specs:
        assert start == StartCondition(float(specs["start_s_per_lap"][0]), float(specs["v_ref_at_start"]) * 0.9,
                                       float(specs["mu_reference"]))
    report, obs, labels, progress = rollout(REPO / "models/all4", MONZA_R0, MATv0(), start, 1.1, 43, 300)
    _failure_report_matches(report, recorded["report"])
    np.testing.assert_array_equal(obs, golden["obs"])
    np.testing.assert_array_equal(labels, golden["act"])
    np.testing.assert_array_equal(progress, golden["prog"])
    report, obs, _, _ = rollout(REPO / "models/all4", MONZA_R0, MATv0(), start, 1.1, 43, 300, switch_progress_m=20.0)
    _failure_report_matches(report, recorded["report_switch"])
    assert obs.size == golden["obs2"].size == 0


@pytest.mark.slow
def test_dagger_iteration(tmp_path: Path) -> None:
    config = DAggerConfig(iterations=1, collect_mus=(1.0,), collect_seeds=(42,), max_steps=400, window_m=30.0,
                          critical_weight=10.0, epochs=1, patience=1, gate=GateConfig(mus=(1.0,), seeds=(42,)))
    summary = run_dagger(GOLDEN / "dagger_tiny_dataset.npz", SPECS, REPO / "models/all4", MONZA_R0, tmp_path,
                         config=config)
    expected = golden_json("dagger_summary")[0]
    hashes = golden_json("dagger_iteration1_hashes")
    collection = json.loads((tmp_path / "collection_1.json").read_text())
    for entry, legacy in zip(collection, expected["collection"], strict=True):
        _failure_report_matches(entry["failure"], legacy["failure"])
        _failure_report_matches(entry["teacher_recovery"], legacy["teacher_recovery"])
        assert entry["window_start_progress_m"] == legacy["window_start_progress_m"]
        assert entry["selected_labels"] == legacy["selected_labels"]
    for key in ("iteration", "balance", "training", "gate_passes", "gate_total", "gate_passed", "passed_mean_time_s"):
        assert summary[0][key] == expected[key], key
    with np.load(tmp_path / "iteration_1/weighted_dataset.npz") as data:
        for key in data.files:
            assert array_digest(data[key]) == hashes[f"ds_{key}"]["sha256"], key
    _assert_weights(load_model(tmp_path / "iteration_1" / POLICY_FILE).policy.state_dict(), hashes)
