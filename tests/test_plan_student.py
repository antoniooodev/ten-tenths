"""Sensor plan student: its input, a short online training with a guide, and its evaluation with its own estimator.

Covers the student observation with the teacher's plan and the plan student input; a short training on a small
executor reading a small estimator, guided by a small plan coach, with its files, curve and validation; the loading
of the student as a plan coach that decides on the student observation; an evaluation episode that drives with the
student's estimator, and the rejection of another estimator, of a sensor student as guide and of a privileged call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.env import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning import demonstrations
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, train_bc
from contextual_bicycle_pacejka.learning.closed_loop import lap_start
from contextual_bicycle_pacejka.learning.datasets import assemble_mdp, mdp_laps
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, generate_laps
from contextual_bicycle_pacejka.learning.friction import FrictionConfig, FrictionEstimator, train_friction
from contextual_bicycle_pacejka.learning.plan_coach import PlanCoachConfig, train_plan_coach
from contextual_bicycle_pacejka.learning.plan_student import REPORT_FILE, PlanStudentConfig, train_plan_student
from contextual_bicycle_pacejka.learning.policy import (
    PLAN_COACH_FILE,
    PlanCoach,
    plan_student_input,
    plan_unit_of,
    policy_env,
    policy_layout,
    teacher_plan_observation,
)
from contextual_bicycle_pacejka.model.student import student_previews
from contextual_bicycle_pacejka.spec.layout import STUDENT_PREVIEW_DIM, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"
SMALL = {"decision_steps": 20, "learning_starts": 5, "batch_size": 8, "hidden": (16,), "episode_steps": 150,
         "log_every": 300, "validate_every": 10_000, "validation_steps": 40, "threads": 1}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A small estimator, an S2 + mu policy and a plan coach driving it, from four short laps of Monza (two setups
    for each of two frictions, so that the estimator keeps one setup per friction for training)."""
    root = tmp_path_factory.mktemp("plan_student")
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
    train_plan_coach(root / "policy", MONZA_R0, root / "coach", PlanCoachConfig(env_steps=300, **SMALL))
    return {"estimator": root / "estimator", "policy": root / "policy", "coach": root / "coach"}


def test_plan_student_input() -> None:
    """The student observation with the preview at scale 1, and the scale in use appended."""
    layout = student_layout("S2", oracle=("mu",))
    env = policy_env(MONZA_R0, layout, plan_scale=1.1)
    try:
        student_obs, info = env.reset(seed=3, options={"setup": randomized_setup(0.9, np.random.default_rng(3))})
    finally:
        env.close()
    privileged = info["privileged_obs"]
    teacher_plan = teacher_plan_observation(student_obs, privileged, 0.9, layout)
    preview = slice(layout.dynamic_dim, layout.dynamic_dim + STUDENT_PREVIEW_DIM)
    assert np.allclose(teacher_plan[preview], student_previews(privileged, 0.9, 1.0)[0])
    assert not np.allclose(student_obs[preview], teacher_plan[preview])  # the scale in use is 1.1
    assert np.array_equal(np.delete(teacher_plan, np.arange(preview.start, preview.stop)),
                          np.delete(student_obs, np.arange(preview.start, preview.stop)))
    inputs = plan_student_input(student_obs, privileged, 0.9, 1.1, layout)
    assert inputs.shape == (layout.dim + 1,) and inputs[-1] == pytest.approx(plan_unit_of(1.1))


def test_train_plan_student(trained: dict[str, Path], tmp_path: Path) -> None:
    """A short guided training writes a student that decides on its sensors and drives with its estimator."""
    config = PlanStudentConfig(env_steps=300, guide_share=0.5, **SMALL)
    report = train_plan_student(trained["policy"], trained["estimator"], trained["coach"], MONZA_R0,
                                tmp_path / "student", config)
    assert report["curve"][-1]["updates"] > 0 and "guided_share" in report["curve"][-1]
    assert report["validations"][0]["plan_scale"] == pytest.approx(1.0)  # the start is the teacher's plan
    for name in (PLAN_COACH_FILE, REPORT_FILE, "critic.pt"):
        assert (tmp_path / "student" / name).exists()
    student = PlanCoach(tmp_path / "student")
    layout = policy_layout(trained["policy"])
    assert student.sensors and student.mu_estimator == trained["estimator"] and student.layout == layout
    assert len(student.mean) == layout.dim + 1
    with pytest.raises(ValueError):
        student(np.zeros(116, dtype=np.float32), 1.0)  # a sensor student does not read the privileged observation

    start_s, start_vx = lap_start(MONZA_R0, 1.0, 11)
    spec = {"block": "held_out", "start_label": "h1", "start_s": start_s, "start_vx": start_vx, "mu": 1.0,
            "seed": 11, "max_steps": 45, "stall_rule": False}
    record = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "student")
    assert record["mu_estimator"] == str(trained["estimator"]) and record["oracle_mu"]["max"] <= 1.2
    assert len(record["plan_decisions"]) == 3  # steps 0, 20 and 40
    same = run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "student",
                       mu_estimator_dir=trained["estimator"])
    assert same["plan_decisions"] == record["plan_decisions"]
    with pytest.raises(ValueError):
        run_episode(spec, MONZA_R0, "policy", False, model_dir=tmp_path / "student",
                    mu_estimator_dir=tmp_path / "other")
    with pytest.raises(ValueError):
        train_plan_student(trained["policy"], trained["estimator"], tmp_path / "student", MONZA_R0, tmp_path / "x",
                           config)


def test_coach_decides_as_before(trained: dict[str, Path]) -> None:
    """A privileged coach decides through ``decide`` exactly as through its call, ignoring the student observation."""
    coach = PlanCoach(trained["coach"])
    assert not coach.sensors and coach.mu_estimator is None
    env = policy_env(MONZA_R0, policy_layout(trained["policy"]), mu_estimator=FrictionEstimator(trained["estimator"]))
    try:
        student_obs, info = env.reset(seed=5, options={"setup": randomized_setup(1.1, np.random.default_rng(5))})
    finally:
        env.close()
    privileged = info["privileged_obs"]
    assert coach.decide(privileged, student_obs, 1.1, 1.05) == coach(privileged, 1.05)
    assert coach.decide(privileged, np.zeros_like(student_obs), 0.8, 1.05) == coach(privileged, 1.05)
