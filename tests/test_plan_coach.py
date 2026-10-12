"""Plan coach: the scale of the student's v_ref preview, the actor's start at the teacher's plan, the episodes that
start before a failure, a short online training on a small executor and the evaluation of the coach with its
executor.

Covers the plan scale of the student previews and of the student wrapper (constant, or changed during an episode
with ``set_plan_scale``), the levels that reject it, the map between scale and unit value, the actor that starts at
the scale 1, the files of a trained plan coach and an evaluation episode driven by it through ``run_episode``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th
from conftest import REPO

from contextual_bicycle_pacejka.env import make_student_env, randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, train_bc
from contextual_bicycle_pacejka.learning.closed_loop import lap_start, start_speed
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.plan_coach import REPORT_FILE, PlanCoachConfig, failure_start, train_plan_coach
from contextual_bicycle_pacejka.learning.policy import (
    PLAN_COACH_FILE,
    PLAN_SCALE_RANGE,
    PlanCoach,
    is_plan_coach,
    plan_actor,
    plan_scale_of,
    plan_unit_of,
    policy_env,
)
from contextual_bicycle_pacejka.model.student import student_previews
from contextual_bicycle_pacejka.spec.constants import OBS_DIM
from contextual_bicycle_pacejka.spec.layout import OBS, SETUP, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"
VREF = slice(0, OBS.VREF_END - OBS.VREF_START)  # the v_ref entries of a student preview


def test_plan_scale_of_previews() -> None:
    """The plan scale multiplies the v_ref preview only, and the scale 1 leaves it bit for bit."""
    obs = np.abs(np.random.default_rng(0).normal(size=(3, OBS_DIM))).astype(np.float32) + 1.0
    base = student_previews(obs, 0.9)
    np.testing.assert_array_equal(student_previews(obs, 0.9, 1.0), base)
    scaled = student_previews(obs, 0.9, 1.1)
    np.testing.assert_allclose(scaled[:, VREF], (base[:, VREF].astype(np.float64) * 1.1), rtol=1e-6)
    np.testing.assert_array_equal(scaled[:, VREF.stop:], base[:, VREF.stop:])


def test_plan_scale_of_student_wrapper() -> None:
    """``set_plan_scale`` returns the observation a wrapper with that scale builds, from the current step on."""
    layout = student_layout("S2", oracle=("mu",))
    options = {"setup": randomized_setup(1.0, np.random.default_rng(3)), "initial_s": 935.35833801, "initial_vx": 30.0}
    plain, scaled = make_student_env(MONZA_R0, layout), make_student_env(MONZA_R0, layout, plan_scale=1.08)
    obs_plain, _ = plain.reset(seed=3, options=options)
    obs_scaled, _ = scaled.reset(seed=3, options=options)
    np.testing.assert_array_equal(plain.set_plan_scale(1.08), obs_scaled)
    assert not np.array_equal(obs_plain, obs_scaled)
    action = np.array([0.01, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(plain.step(action)[0], scaled.step(action)[0])
    with pytest.raises(ValueError):
        make_student_env(MONZA_R0, student_layout("C"), plan_scale=1.08)
    with pytest.raises(ValueError):
        policy_env(MONZA_R0, student_layout("S0"), plan_scale=1.08)


def test_plan_actor_starts_at_teacher_plan() -> None:
    """The scale and its unit value map onto each other, and a new actor gives the scale 1 for any input."""
    low, high = PLAN_SCALE_RANGE
    assert plan_scale_of(-1.0) == pytest.approx(low) and plan_scale_of(1.0) == pytest.approx(high)
    assert plan_scale_of(plan_unit_of(1.07)) == pytest.approx(1.07)
    actor = plan_actor(OBS_DIM + 1, (16,))
    with th.no_grad():
        units = actor(th.as_tensor(np.random.default_rng(1).normal(size=(5, OBS_DIM + 1)).astype(np.float32)))
    np.testing.assert_allclose(plan_scale_of(units.numpy()), 1.0, rtol=1e-6)


def test_failure_start() -> None:
    """An episode from a failure starts 400-800 m before it (across the lap end), at its friction, with the start
    speed rule of the closed-loop laps."""
    with np.load(MONZA_R0) as data:
        length = float(data["track_length"])
    rng = np.random.default_rng(0)
    for failure_s in (925.0, 200.0):
        for _ in range(20):
            options, seed = failure_start(MONZA_R0, (failure_s, 1.2), rng, (400.0, 800.0))
            assert 400.0 <= (failure_s - options["initial_s"]) % length <= 800.0
            assert float(np.asarray(options["setup"])[SETUP.MU]) == pytest.approx(1.2)
            assert np.array_equal(options["setup"], randomized_setup(1.2, np.random.default_rng(seed)))
            assert options["initial_vx"] >= start_speed(MONZA_R0, options["initial_s"], 1.2, 0.5) - 1e-9
    start_s, start_vx = lap_start(MONZA_R0, 1.0, 7)
    draws = np.random.default_rng(7)
    draws.uniform(0.0, length)  # the start arc length
    fraction = float(draws.uniform(0.5, 1.0))
    assert start_vx == pytest.approx(start_speed(MONZA_R0, start_s, 1.0, fraction))


@pytest.fixture(scope="module")
def executor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small S2 + mu policy trained on two short perturbed laps of Monza."""
    root = tmp_path_factory.mktemp("plan_coach")
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
    train_bc(dataset, root / "policy", MONZA_R0, BCConfig(epochs=1, level="S2", oracle=("mu",)))
    return root / "policy"


def test_train_plan_coach(executor: Path, tmp_path: Path) -> None:
    """A short training writes a plan coach that ``run_episode`` evaluates with its executor."""
    config = PlanCoachConfig(env_steps=600, decision_steps=20, learning_starts=10, batch_size=8, hidden=(16,),
                             episode_steps=150, log_every=300, validate_every=10_000, validation_steps=40, threads=1)
    report = train_plan_coach(executor, MONZA_R0, tmp_path / "coach", config)
    assert [entry["step"] for entry in report["curve"]] == sorted(entry["step"] for entry in report["curve"])
    assert report["curve"][-1]["updates"] > 0
    assert {"actor_loss", "smooth_time", "smooth_space", "bound", "failure_start_share"} <= set(report["curve"][-1])
    assert report["validations"][0]["step"] == 0 and len(report["validations"]) == 2
    assert report["validations"][0]["plan_scale"] == pytest.approx(1.0)  # the start is the teacher's plan
    assert len(report["validations"][-1]["laps"]) == 6 and report["validations"][-1]["teacher_completed"] <= 6
    for name in (PLAN_COACH_FILE, REPORT_FILE, "critic.pt"):
        assert (tmp_path / "coach" / name).exists()
    assert json.loads((tmp_path / "coach" / REPORT_FILE).read_text())["executor"] == str(executor)
    assert is_plan_coach(tmp_path / "coach") and not is_plan_coach(executor)
    coach = PlanCoach(tmp_path / "coach")
    assert coach.executor == executor and coach.decision_steps == 20
    spec = {"block": "held_out", "start_label": "h1", "start_s": 935.35833801, "start_vx": 30.0, "mu": 1.0,
            "seed": 42, "max_steps": 45, "stall_rule": False}
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "coach")
    assert record["plan_coach"] == str(tmp_path / "coach") and record["student_level"] == "S2"
    assert len(record["plan_decisions"]) == 3  # steps 0, 20 and 40
    low, high = PLAN_SCALE_RANGE
    assert low <= record["plan_scale"]["min"] <= record["plan_scale"]["max"] <= high
    with pytest.raises(ValueError):
        run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "coach", plan_scale=1.05)
    constant = run_episode(spec, MONZA_R0, "policy", False, model_dir=executor, plan_scale=1.05)
    assert constant["plan_coach"] is None and constant["plan_decisions"] == []
    assert constant["plan_scale"] == pytest.approx({"mean": 1.05, "min": 1.05, "max": 1.05})
