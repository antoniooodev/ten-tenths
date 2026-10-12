"""Friction estimator: filter, lower bound, calibration, training, inputs and online use.

Covers the Kalman fusion and the lower bound at the prior, the calibrated margin and its one-sided bias, the
recalibration of a trained estimator, the rise limit, a small estimator trained on short demonstration laps, the
absence of privileged information in its inputs, the agreement of the estimate built online by the student wrapper
with the one built offline (with and without the rise limit), an evaluation episode driven on the estimate, a plan
coach whose executor reads the estimate, and closed-loop laps logged, with their circuit, and added to the training
data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env import make_student_env, randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, train_bc
from contextual_bicycle_pacejka.learning.closed_loop import collect, parse_variant
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.friction import (
    ESTIMATOR_FILE,
    PRIOR_MEAN,
    PRIOR_STD,
    Calibration,
    FrictionConfig,
    FrictionEstimator,
    calibrate,
    fuse,
    limit_rise,
    lower_bound,
    recalibrate,
    train_friction,
)
from contextual_bicycle_pacejka.learning.plan_coach import PlanCoachConfig, train_plan_coach
from contextual_bicycle_pacejka.model.student import previous_actions
from contextual_bicycle_pacejka.spec.constants import DT, MU_HI, MU_LO
from contextual_bicycle_pacejka.spec.layout import OBS, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"
LAP_STEPS = 300
ROLLOUT_STEPS = 250


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A small estimator and an S2 + mu policy trained on four short laps (two setups for each of two frictions)."""
    root = tmp_path_factory.mktemp("friction")
    specs = episode_specs(MONZA_R0)
    by_mu: dict[float, list[int]] = {}
    for spec in specs:
        by_mu.setdefault(spec.mu, []).append(spec.lap)
    laps = sorted(lap for mu in sorted(by_mu)[:2] for lap in by_mu[mu][:2])
    previous = demonstrations.MAX_ATTEMPTS
    demonstrations.MAX_ATTEMPTS = 1
    try:
        generate_laps(MONZA_R0, "monza", "speed", root / "demos", specs, laps=laps, keep_failed=True,
                      max_steps=LAP_STEPS)
    finally:
        demonstrations.MAX_ATTEMPTS = previous
    dataset = root / "mdp.npz"
    np.savez(dataset, **assemble_mdp(mdp_laps(root / "demos", include_failed=True), {"monza": specs}))
    train_friction(dataset, root / "estimator", FrictionConfig(hidden=(16,), epochs=2, batch_size=256))
    train_bc(dataset, root / "policy", MONZA_R0, BCConfig(epochs=1, level="S2", oracle=("mu",)))
    return {"dataset": dataset, "estimator": root / "estimator", "policy": root / "policy",
            "specs": [specs[lap] for lap in laps]}


# --------------------------------------------------------------------------------------------- filter and bound
def test_fusion_and_lower_bound() -> None:
    """Uninformative measurements keep the prior (mu_hat = MU_LO); precise ones pull the mean to their value."""
    calibration = Calibration(inflation=1.0, bias_std=0.0, margin=1.0)
    mean, var = fuse(np.full(3, 1.15), np.full(3, 1.0e6), 10, 25, 1.0)
    assert mean[-1] == pytest.approx(PRIOR_MEAN, abs=1e-6) and var[-1] == pytest.approx(PRIOR_STD ** 2, rel=1e-5)
    assert lower_bound(mean[0], var[0], calibration) == pytest.approx(MU_LO, abs=1e-6)
    mean, var = fuse(np.full(3, 1.15), np.full(3, 1.0e-8), 10, 25, 1.0)
    assert mean[0] == pytest.approx(1.15, abs=1e-5) and mean[-1] == pytest.approx(1.15, abs=1e-7)
    assert lower_bound(5.0, 0.0, calibration) == MU_HI


def test_calibrated_margin_bounds_the_overestimate() -> None:
    """With the calibrated constants no held-out step exceeds the overestimate limit."""
    rng = np.random.default_rng(0)
    runs = [{"measured": mu + 0.05 * rng.standard_normal(30), "variance": np.full(30, 0.01), "n": 300, "mu": mu}
            for mu in (0.8, 0.9, 1.0, 1.1, 1.2)]
    config = FrictionConfig()
    calibration = calibrate(runs, config)
    assert calibration.margin >= 1.0 and calibration.inflation in config.inflations
    for run in runs:
        mean, var = fuse(run["measured"], run["variance"], config.update_steps, run["n"], calibration.inflation)
        assert np.all(lower_bound(mean, var, calibration) <= (1.0 + config.overestimate_limit) * run["mu"] + 1e-12)


def test_margin_floor() -> None:
    """Below the default floor the overestimate limit sets k: a looser limit gives a smaller margin, still bounded."""
    rng = np.random.default_rng(0)
    runs = [{"measured": mu + 0.01 * rng.standard_normal(30), "variance": np.full(30, 1.0e-4), "n": 300, "mu": mu}
            for mu in (0.8, 0.9, 1.0, 1.1, 1.2)]
    floor = calibrate(runs, FrictionConfig())
    strict = calibrate(runs, FrictionConfig(margin_floor=0.0))
    loose = calibrate(runs, FrictionConfig(margin_floor=0.0, overestimate_limit=0.06))
    assert floor.margin == 1.0 and loose.margin <= strict.margin < 1.0
    for config, calibration in ((FrictionConfig(margin_floor=0.0), strict),
                                (FrictionConfig(margin_floor=0.0, overestimate_limit=0.06), loose)):
        for run in runs:
            mean, var = fuse(run["measured"], run["variance"], config.update_steps, run["n"], calibration.inflation)
            bound = lower_bound(mean, var, calibration)
            assert np.all(bound <= (1.0 + config.overestimate_limit) * run["mu"] + 1e-12)


def test_rise_limit() -> None:
    """The estimate rises by at most the limit per second and falls freely."""
    estimate = np.array([0.8, 1.2, 1.2, 0.9, 1.0, 1.0])
    limited = limit_rise(estimate, 5.0)
    np.testing.assert_allclose(limited, [0.8, 0.85, 0.9, 0.9, 0.95, 1.0], rtol=0, atol=1e-12)
    assert np.all(np.diff(limited) <= 5.0 * DT + 1e-12)
    assert limit_rise(estimate, None) is estimate


def test_one_sided_bias() -> None:
    """With bias_side="over" only the overestimates at the end of a lap enter b."""
    runs = [{"measured": np.full(30, mu + error), "variance": np.full(30, 1.0e-4), "n": 300, "mu": mu}
            for mu, error in ((0.9, -0.10), (1.0, 0.02), (1.1, -0.05))]
    both = calibrate(runs, FrictionConfig())
    over = calibrate(runs, FrictionConfig(bias_side="over"))
    assert over.inflation == both.inflation and over.bias_std < both.bias_std
    final = [fuse(r["measured"], r["variance"], 10, r["n"], over.inflation)[0][-1] - r["mu"] for r in runs]
    assert final[1] > 0.0 > final[0]
    assert over.bias_std == pytest.approx(np.sqrt(np.mean(np.maximum(final, 0.0) ** 2)), rel=1e-12)


# --------------------------------------------------------------------------------------------- training
def test_train_friction(trained: dict[str, Path]) -> None:
    spec = json.loads((trained["estimator"] / ESTIMATOR_FILE).read_text())
    assert spec["report"]["n_train_laps"] == 2 and spec["report"]["n_validation_laps"] == 2
    assert spec["calibration"]["margin"] >= 1.0
    assert spec["report"]["validation"]["estimate_ratio_max"] <= 1.0 + FrictionConfig.overestimate_limit + 1e-9
    estimator = FrictionEstimator(trained["estimator"])
    assert len(estimator.weights) == 2 and estimator.window == 36


def test_recalibrate(trained: dict[str, Path], tmp_path: Path) -> None:
    """Recalibration keeps the network and changes only the calibration settings."""
    spec = recalibrate(trained["estimator"], trained["dataset"], tmp_path / "over", bias_side="over")
    original = json.loads((trained["estimator"] / ESTIMATOR_FILE).read_text())
    assert spec["config"]["bias_side"] == "over" and spec["recalibrated_from"] == str(trained["estimator"])
    assert spec["calibration"]["inflation"] == original["calibration"]["inflation"]
    assert spec["calibration"]["bias_std"] <= original["calibration"]["bias_std"] + 1e-12
    np.testing.assert_array_equal(FrictionEstimator(tmp_path / "over").weights[0][0],
                                  FrictionEstimator(trained["estimator"]).weights[0][0])
    loose = recalibrate(trained["estimator"], trained["dataset"], tmp_path / "loose", overestimate_limit=0.06,
                        margin_floor=0.0)
    assert loose["config"]["margin_floor"] == 0.0 and loose["calibration"]["margin"] >= 0.0
    assert loose["report"]["validation"]["estimate_ratio_max"] <= 1.06 + 1e-9
    with pytest.raises(ValueError):
        recalibrate(trained["estimator"], trained["dataset"], tmp_path / "bad", hidden=(8,))


def test_estimator_reads_no_privileged_information(trained: dict[str, Path]) -> None:
    """Changing the setup, vy, the slip angles, the forces, the previews and the actions leaves the estimate."""
    with np.load(trained["dataset"]) as data:
        obs = np.asarray(data["observations"][:LAP_STEPS])
        imu = np.asarray(data["imu"][:LAP_STEPS])
        prev = previous_actions(data["actions_exec"][:LAP_STEPS])
    estimator = FrictionEstimator(trained["estimator"])
    altered = obs.copy()
    hidden = [OBS.VY, OBS.AX, OBS.RHO_LAT, OBS.RHO_LON, OBS.ALPHA_F, OBS.ALPHA_R, OBS.FY_F, OBS.FY_R]
    altered[:, hidden] = 0.3
    for start, end in ((OBS.XI_START, OBS.XI_END), (OBS.VREF_START, OBS.VREF_END), (OBS.TRAJ_START, OBS.TRAJ_END),
                       (OBS.KAPPA_START, OBS.KAPPA_END), (OBS.AX_PREVIEW_START, OBS.AX_PREVIEW_END)):
        altered[:, start:end] = -0.2
    reference = estimator.estimate_episode(obs, imu, prev)["estimate"]
    np.testing.assert_array_equal(estimator.estimate_episode(altered, imu, prev * 0.5)["estimate"], reference)


# --------------------------------------------------------------------------------------------- online use
@pytest.mark.parametrize("rise_limit", [None, 0.05])
def test_online_estimate_equals_offline(trained: dict[str, Path], rise_limit: float | None) -> None:
    """The estimate in the student observation equals the one rebuilt offline from the logged channels."""
    layout = student_layout("S2", oracle=("mu",))
    env = make_student_env(MONZA_R0, layout, mu_estimator=FrictionEstimator(trained["estimator"], rise_limit))
    teacher = MATv0()
    options = {"setup": randomized_setup(1.05, np.random.default_rng(101)), "initial_s": 935.35833801,
               "initial_vx": 6.6}
    student, info = env.reset(seed=101, options=options)
    log: dict[str, list] = {"obs": [], "imu": [], "exec": [], "oracle": [], "entry": []}
    for _ in range(ROLLOUT_STEPS):
        action = teacher.act(info["privileged_obs"])
        log["obs"].append(info["privileged_obs"])
        log["imu"].append(info["imu"])
        log["exec"].append(np.clip(action, env.action_space.low, env.action_space.high))
        log["oracle"].append(info["oracle"]["mu"])
        log["entry"].append(student[-1])
        student, _, _, _, info = env.step(action)
    offline = FrictionEstimator(trained["estimator"], rise_limit).estimate_episode(
        np.asarray(log["obs"]), np.asarray(log["imu"]), previous_actions(np.asarray(log["exec"])))["estimate"]
    np.testing.assert_allclose(log["oracle"], offline.astype(np.float32), rtol=1e-6, atol=0)
    np.testing.assert_array_equal(log["entry"], np.asarray(log["oracle"], dtype=np.float32))
    assert log["oracle"][0] == pytest.approx(MU_LO)
    if rise_limit is not None:
        assert np.all(np.diff(offline) <= rise_limit * DT + 1e-12)


def test_estimator_needs_the_mu_entry(trained: dict[str, Path]) -> None:
    with pytest.raises(ValueError):
        make_student_env(MONZA_R0, student_layout("S2"), mu_estimator=FrictionEstimator(trained["estimator"]))


def test_episode_on_the_estimate(trained: dict[str, Path]) -> None:
    specs = episode_specs(MONZA_R0)
    spec = {"block": "held_out", "start_label": "h1", "start_s": specs[0].start_s, "start_vx": specs[0].start_vx,
            "mu": 1.0, "seed": 42, "max_steps": 60, "stall_rule": False}
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=trained["policy"],
                         mu_estimator_dir=trained["estimator"])
    summary = record["oracle_mu"]
    assert record["mu_estimator"] == str(trained["estimator"])
    assert summary["initial"] == pytest.approx(MU_LO / 1.0, rel=1e-6) and summary["max"] <= MU_HI
    exact = run_episode(spec, MONZA_R0, "policy", False, model_dir=trained["policy"])
    assert exact["oracle_mu"]["min"] == pytest.approx(1.0) and exact["mu_estimator"] is None


def test_plan_coach_on_the_estimate(trained: dict[str, Path], tmp_path: Path) -> None:
    """A plan coach decides on the privileged observation while its executor reads the estimate; a new plan scale
    keeps the estimate of the step (docs/REDESIGN.md §25, measure 0a)."""
    specs = episode_specs(MONZA_R0)
    env = make_student_env(MONZA_R0, student_layout("S2", oracle=("mu",)),
                           mu_estimator=FrictionEstimator(trained["estimator"]))
    try:
        obs, info = env.reset(seed=0, options={"setup": randomized_setup(1.0, np.random.default_rng(0)),
                                               "initial_s": specs[0].start_s, "initial_vx": specs[0].start_vx})
        for _ in range(30):
            obs, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        rescaled = env.set_plan_scale(1.1)
    finally:
        env.close()
    assert abs(info["oracle"]["mu"] - 1.0) > 1e-6  # the entry is the estimate, not the true mu
    assert rescaled[-1] == obs[-1] == pytest.approx(info["oracle"]["mu"])

    config = PlanCoachConfig(env_steps=300, decision_steps=20, learning_starts=5, batch_size=8, hidden=(16,),
                             episode_steps=150, log_every=300, validate_every=10_000, validation_steps=40, threads=1)
    train_plan_coach(trained["policy"], MONZA_R0, tmp_path / "coach", config)
    spec = {"block": "held_out", "start_label": "h1", "start_s": specs[0].start_s, "start_vx": specs[0].start_vx,
            "mu": 1.0, "seed": 42, "max_steps": 45, "stall_rule": False}
    on_estimate = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "coach",
                              mu_estimator_dir=trained["estimator"])
    exact = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "coach")
    assert on_estimate["plan_coach"] == exact["plan_coach"] == str(tmp_path / "coach")
    assert on_estimate["mu_estimator"] == str(trained["estimator"]) and exact["mu_estimator"] is None
    assert on_estimate["oracle_mu"]["initial"] == pytest.approx(MU_LO / 1.0, rel=1e-6)
    assert exact["oracle_mu"]["min"] == pytest.approx(1.0)
    assert len(on_estimate["plan_decisions"]) == 3  # steps 0, 20 and 40
    assert on_estimate["plan_decisions"][0] == exact["plan_decisions"][0]  # same state: the coach ignores the estimate


# --------------------------------------------------------------------------------------------- closed-loop data
def test_closed_loop_laps(trained: dict[str, Path], tmp_path: Path) -> None:
    """Closed-loop laps log the channels the estimator reads, labelled with the true mu, and extend its data."""
    output = tmp_path / "closed_loop.npz"
    summary = collect(MONZA_R0, trained["policy"], trained["specs"], ["muhat", "mu+0.10"], output,
                      estimator_dir=trained["estimator"], max_steps=80)
    assert summary["muhat"]["laps"] == summary["mu+0.10"]["laps"] == 4
    with np.load(output) as data:
        ends, mus = data["episode_ends"], data["mu_per_lap"]
        assert len(data["observations"]) == len(data["imu"]) == len(data["oracle_mu"]) == ends[-1] <= 8 * 80
        assert list(data["kind_per_lap"][[0, 4]]) == ["closed_loop_muhat", "closed_loop_mu+0.10"]
        assert set(data["circuit_per_lap"]) == {"monza"}
        np.testing.assert_allclose(data["oracle_mu"][ends[3]], np.float32(mus[4] + 0.10), rtol=1e-6)
        assert data["oracle_mu"][0] == pytest.approx(MU_LO)
        np.testing.assert_array_equal(data["setup_per_lap"][:4], data["setup_per_lap"][4:])
    spec = train_friction([trained["dataset"], output], tmp_path / "estimator",
                          FrictionConfig(hidden=(16,), epochs=1, batch_size=256))
    assert set(spec["report"]["validation_by_kind"]) == {"speed", "closed_loop_muhat", "closed_loop_mu+0.10"}
    assert set(spec["report"]["validation_by_circuit"]) == {"monza"}
    assert spec["report"]["n_validation_laps"] == 6
    assert parse_variant("mu-0.05") == (-0.05, False) and parse_variant("muhat") == (None, True)
    second = collect(MONZA_R0, trained["policy"], trained["specs"][:1], ["mu-0.10"], tmp_path / "second.npz",
                     max_steps=20, seed=1, label="closed_loop2")
    assert second["mu-0.10"]["laps"] == 1
    with np.load(tmp_path / "second.npz") as data:
        assert list(data["kind_per_lap"]) == ["closed_loop2_mu-0.10"]
    with pytest.raises(ValueError):
        parse_variant("mu")
