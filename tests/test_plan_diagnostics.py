"""Diagnostics of the sensor plan student (docs/REDESIGN.md §25.5): margin at a point, estimator shift, plan inference.

Covers the episodes and the margin rule of ``evaluation/point_margin.py``; the replay of an episode on the estimate
with its offline estimate equal to the online one, the memory-mapped dataset arrays, the training windows of one
circuit and the nearest-neighbour distances of ``evaluation/estimator_shift.py``; the training episodes, the decision
samples (preview at scale 1, targets equal to the coach's decisions), the regression and the gap measures of
``evaluation/plan_inference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.evaluation import estimator_shift, plan_inference, point_margin
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, train_bc
from contextual_bicycle_pacejka.learning.closed_loop import lap_start, start_speed
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.friction import FrictionConfig, FrictionEstimator, train_friction
from contextual_bicycle_pacejka.learning.plan_coach import PlanCoachConfig, train_plan_coach
from contextual_bicycle_pacejka.learning.policy import PlanCoach, policy_layout
from contextual_bicycle_pacejka.model.student import channel_frames, previous_actions, stack_window, student_previews
from contextual_bicycle_pacejka.spec.constants import MU_HI, MU_LO
from contextual_bicycle_pacejka.spec.layout import FRICTION_CHANNELS, STUDENT_PREVIEW_DIM

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A small estimator, an S2 + mu policy and a plan coach driving it, from four short laps of Monza (two setups
    for each of two frictions, so that the estimator keeps one setup per friction for training)."""
    root = tmp_path_factory.mktemp("plan_diagnostics")
    specs = episode_specs(MONZA_R0)
    by_mu: dict[float, list[int]] = {}
    for spec in specs:
        by_mu.setdefault(spec.mu, []).append(spec.lap)
    laps = sorted(lap for mu in sorted(by_mu)[:2] for lap in by_mu[mu][:2])
    previous = demonstrations.MAX_ATTEMPTS
    demonstrations.MAX_ATTEMPTS = 1
    try:
        generate_laps(MONZA_R0, "monza", "speed", root / "demos", specs, laps=laps, keep_failed=True,
                      max_steps=300)
    finally:
        demonstrations.MAX_ATTEMPTS = previous
    dataset = root / "mdp.npz"
    np.savez(dataset, **assemble_mdp(mdp_laps(root / "demos", include_failed=True), {"monza": specs}))
    train_friction(dataset, root / "estimator", FrictionConfig(hidden=(16,), epochs=1, batch_size=256))
    train_bc(dataset, root / "policy", MONZA_R0, BCConfig(epochs=1, level="S2", oracle=("mu",)))
    config = PlanCoachConfig(env_steps=300, decision_steps=20, learning_starts=5, batch_size=8, hidden=(16,),
                             episode_steps=150, log_every=300, validate_every=10_000, validation_steps=40, threads=1)
    train_plan_coach(root / "policy", MONZA_R0, root / "coach", config)
    return {"dataset": dataset, "estimator": root / "estimator", "policy": root / "policy", "coach": root / "coach"}


def _spec(max_steps: int, mu: float = 1.0) -> dict:
    start_s, start_vx = lap_start(MONZA_R0, mu, 7)
    return {"block": "held_out", "start_label": "h1", "start_s": start_s, "start_vx": start_vx, "mu": mu, "seed": 7,
            "max_steps": max_steps, "stall_rule": False}


# --------------------------------------------------------------------------------------------- margin at a point
def test_margin_specs_and_episode(trained: dict[str, Path]) -> None:
    specs = point_margin.margin_specs(MONZA_R0, 1000.0, [300.0, 100.0], [0.9, 1.1], [5, 6], 40)
    assert len(specs) == 8 and {s["start_s"] for s in specs} == {700.0, 900.0}
    spec = next(s for s in specs if s["mu"] == 1.1 and s["lead_m"] == 300.0)
    assert spec["start_vx"] == pytest.approx(start_speed(MONZA_R0, 700.0, 1.1, 1.0))
    result = point_margin.margin_episode({"spec": spec, "racing_line": str(MONZA_R0), "model_dir": trained["policy"],
                                          "scale": 1.04, "error": 0.06, "beyond_m": 10.0})
    assert result["scale"] == 1.04 and result["error"] == 0.06 and result["mu"] == 1.1
    assert result["passed"] == (result["progress_m"] >= 310.0)


def test_margin_rule() -> None:
    def row(scale: float, error: float, passed: bool, mu: float = 1.0) -> dict:
        return {"scale": scale, "error": error, "mu": mu, "passed": passed,
                "termination_reason": "step_limit" if passed else "slip_saturation"}

    results = [row(1.0, 0.0, True), row(1.0, 0.0, True, 1.2), row(1.1, 0.0, True), row(1.1, 0.0, False, 1.2),
               row(1.0, 0.04, False), row(1.0, 0.04, True, 1.2), row(1.1, 0.04, False), row(1.1, 0.04, False, 1.2)]
    out = point_margin.margins(results)
    assert out["margin"] == {"0.0": 1.0, "0.04": None}
    assert out["table"]["0.0"]["1.1"] == {"passed": 1, "n": 2, "timeouts": 0,
                                          "by_mu": {"1.0": {"passed": 1, "n": 1}, "1.2": {"passed": 0, "n": 1}}}


# --------------------------------------------------------------------------------------------- estimator shift
def test_replay_on_the_estimate(trained: dict[str, Path]) -> None:
    """The offline estimate of a replayed episode equals the online one; the inputs include the reset row."""
    spec = _spec(120)
    inputs = estimator_shift.record_inputs(spec, MONZA_R0, trained["policy"], trained["estimator"])
    record = inputs["record"]
    assert len(inputs["obs"]) == len(inputs["imu"]) == len(inputs["prev_actions"]) == record["steps"] + 1
    assert np.all(inputs["prev_actions"][0] == 0.0) and inputs["s"][0] == spec["start_s"]
    estimator = FrictionEstimator(trained["estimator"])
    stage = estimator_shift.stages(estimator, inputs, spec["mu"])
    # the online estimate reaches the record through the float32 observation, the offline one stays float64
    assert stage["estimate_ratio"].max() == pytest.approx(record["oracle_mu"]["max"], abs=1e-6)
    assert stage["estimate_ratio"][-1] == pytest.approx(record["oracle_mu"]["final"], abs=1e-6)
    windows = estimator_shift.measured_windows(estimator, inputs)
    assert windows.shape == (len(stage["measured_steps"]), estimator.input_mean.size)


def test_dataset_windows_and_distances(trained: dict[str, Path]) -> None:
    with np.load(trained["dataset"]) as data:
        observations, imu = data["observations"], data["imu"]
        executed, ends = data["actions_exec"], data["episode_ends"]
    assert np.array_equal(estimator_shift.stored_array(trained["dataset"], "observations"), observations)
    windows = estimator_shift.training_windows(trained["estimator"], "monza", every=10)
    estimator = FrictionEstimator(trained["estimator"])
    first = slice(0, int(ends[0]))
    frames = channel_frames(observations[first], imu[first], previous_actions(executed[first]), FRICTION_CHANNELS)
    expected = stack_window(frames, estimator.config.n_frames, estimator.config.stride)[::10]
    assert np.allclose(windows[:len(expected)], (expected - estimator.input_mean) / estimator.input_std, atol=1e-5)
    rng = np.random.default_rng(0)
    queries, reference = rng.normal(size=(7, 5)), rng.normal(size=(50, 5))
    brute = np.sqrt(((queries[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    assert np.allclose(estimator_shift.nearest_distances(queries, reference, chunk=8), brute)


# --------------------------------------------------------------------------------------------- plan inference
def test_training_specs() -> None:
    specs = plan_inference.training_specs(MONZA_R0, 3, 3_000_000, 500)
    assert [s["seed"] for s in specs] == [3_000_000, 3_000_001, 3_000_002]
    assert all(MU_LO <= s["mu"] <= MU_HI and s["max_steps"] == 500 for s in specs)
    assert specs == plan_inference.training_specs(MONZA_R0, 3, 3_000_000, 500)
    assert (specs[0]["start_s"], specs[0]["start_vx"]) == lap_start(MONZA_R0, specs[0]["mu"], 3_000_000)


def test_decision_samples(trained: dict[str, Path]) -> None:
    """One sample per decision after the first, with the preview of the teacher's plan and the coach's scale."""
    spec = _spec(65)
    out = plan_inference.decision_samples(spec, MONZA_R0, trained["coach"], trained["estimator"])
    layout = policy_layout(PlanCoach(trained["coach"]).executor)
    assert out["inputs"].shape == (3, layout.dim) and len(out["scales"]) == len(out["s"]) == 3  # steps 20, 40, 60
    captured = []
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=trained["coach"],
                         mu_estimator_dir=trained["estimator"],
                         on_step=lambda step: captured.append(step) if (step["step"] + 1) % 20 == 0 else None)
    assert np.allclose(out["scales"], [d[1] for d in record["plan_decisions"][1:]])
    preview = slice(layout.dynamic_dim, layout.dynamic_dim + STUDENT_PREVIEW_DIM)
    for x, step in zip(out["inputs"], captured, strict=False):
        assert np.allclose(x[preview], student_previews(step["obs"], spec["mu"], 1.0)[0])
        assert np.allclose(x[:layout.dynamic_dim], step["student_obs"][:layout.dynamic_dim])
        assert x[-1] == pytest.approx(step["student_obs"][-1])  # mu_hat, not the true mu


def test_regression_and_gap() -> None:
    rng = np.random.default_rng(1)
    inputs = rng.normal(size=(400, 3))
    scales = 1.05 + 0.1 * np.tanh(inputs[:, 0])
    episodes = np.repeat(np.arange(40), 10)
    model = plan_inference.fit_regressor(inputs, scales, episodes,
                                         plan_inference.RegressionConfig(hidden=(16,), max_epochs=30, patience=5))
    predicted = plan_inference.predict(model, inputs)
    assert predicted.shape == scales.shape and model["validation_episodes"] == 4
    s, mus = np.tile(np.arange(10) * 300.0, 40), np.repeat([0.8, 1.2], 200)
    gap = plan_inference.inference_gap(scales, predicted, s, mus)
    assert gap["n"] == 400 and set(gap["by_mu"]) == {"0.80", "1.20"} and len(gap["by_sector"]) == 6
    exact = plan_inference.inference_gap(scales, scales, s, mus)
    assert exact["r2"] == pytest.approx(1.0) and exact["mean_abs"] == 0.0 and exact["sectors_above"] == []
    shifted = plan_inference.inference_gap(scales, scales + 0.05, s, mus)
    assert shifted["share_above"] == 1.0 and len(shifted["sectors_above"]) == 6
