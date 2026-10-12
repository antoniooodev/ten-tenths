"""Protocols and result analysis: agreement with the original aggregation on recorded episode results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO, golden_json

from contextual_bicycle_pacejka.evaluation import analysis
from contextual_bicycle_pacejka.evaluation.protocol import evaluate_protocol, load_protocol, reduce_protocol

PROTOCOLS = sorted(p for p in (REPO / "data/protocols").glob("*.json") if not p.stem.endswith("_reduced"))
REDUCED_PROTOCOLS = sorted((REPO / "data/protocols").glob("*_reduced.json"))
BLOCK_SIZES = {"in_sample_gate": 20, "seed_only": 40, "mu_only": 16, "start_only": 64, "held_out": 256}

_ONSETS = {"rear_slip_saturated": "ar_ge_1", "front_slip_saturated": "af_ge_1", "overspeed": "overspeed"}
_TERMINAL = {"rear_slip_ratio": "ar_ratio", "front_slip_ratio": "af_ratio", "tau": "tau_prop",
             "delta_cmd": "delta_prop"}
_TRACKING = {"speed_error_mean_abs_mps": ("teacher_diag", "speed_error_mean_abs_mps"),
             "frac_overspeed": ("teacher_diag", "frac_overspeed"),
             "rear_slip_ratio_p99": ("teacher_diag", "ar_ratio_p99"),
             "speed_error_mean_mps": ("stage_a_diag", "speed_err_mean_mps"),
             "ey_rms_m": ("stage_a_diag", "ey_rms_m"),
             "teacher_tau_mae": ("stage_a_diag", "student_teacher_tau_mae"),
             "teacher_delta_mae": ("stage_a_diag", "student_teacher_delta_mae")}


def _convert(legacy: dict) -> dict:
    """Legacy result record -> current record layout (field names of ``evaluation.diagnostics``)."""
    old_onsets = legacy["teacher_diag"].get("pre_failure_last5s") or {}
    old_terminal = legacy["teacher_diag"].get("terminal_state") or {}
    terminal_names = {v: k for k, v in _TERMINAL.items()}
    diagnostics = {new: legacy[src][old] for new, (src, old) in _TRACKING.items()}
    diagnostics["terminal_zone"] = legacy["teacher_diag"].get("terminal_zone")
    diagnostics["pre_failure_last5s"] = {new: old_onsets.get(old) for new, old in _ONSETS.items()} if old_onsets else {}
    diagnostics["terminal_state"] = {terminal_names.get(k, k): v for k, v in old_terminal.items()}
    record = {k: v for k, v in legacy.items() if k not in ("teacher_diag", "stage_a_diag")}
    return {**record, "diagnostics": diagnostics}


@pytest.fixture(scope="module")
def records() -> dict[str, dict]:
    return {name: {analysis.episode_key(r): r for r in map(_convert, rs)}
            for name, rs in golden_json("analysis_records").items()}


def test_statistics() -> None:
    expected = golden_json("analysis_outputs")
    for label, interval in expected["wilson"].items():
        k, n = map(int, label.split("/"))
        assert analysis.wilson(k, n) == interval
    for label, p in expected["mcnemar"].items():
        b, c = map(int, label.split(","))
        assert analysis.mcnemar(b, c) == p


def test_summaries(records: dict[str, dict]) -> None:
    expected = golden_json("analysis_outputs")
    assert analysis.summarise(records["LOCO_catalunya_catalunyaR1_U"], False) == expected["summary_LOCOcat_U"]
    assert analysis.summarise(records["MAT_catalunyaR1_F"], True) == expected["summary_MAT_F"]
    assert analysis.paired_lap_time(records["LOCO_catalunya_catalunyaR1_U"], records["MAT_catalunyaR1_U"]) \
        == expected["paired_LOCOcat_vs_MAT_U"]
    assert analysis.filter_dependence(records["MAT_catalunyaR1_U"], records["MAT_catalunyaR1_F"]) \
        == expected["filter_MAT"]


def test_failure_mechanisms(records: dict[str, dict]) -> None:
    expected = golden_json("analysis_outputs")["mechanisms_LOCOspa"]
    got = {r["id"]: analysis.failure_mechanism(r) for r in records["LOCO_spa_spaR1_U"].values() if not r["robust"]}
    assert got == expected
    assert len(got) > 0


def test_tracking_medians(records: dict[str, dict]) -> None:
    expected = golden_json("analysis_outputs")["diag_means_LOCOcat"]
    legacy_names = {"speed_error_mean_abs_mps": "speed_error_mean_abs_mps", "frac_overspeed": "frac_overspeed",
                    "speed_error_mean_mps": "speed_err_mean_mps", "ey_rms_m": "ey_rms_m",
                    "rear_slip_ratio_p99": "ar_ratio_p99", "teacher_tau_mae": "student_teacher_tau_mae",
                    "teacher_delta_mae": "student_teacher_delta_mae"}
    got = analysis.tracking_medians(records["LOCO_catalunya_catalunyaR1_U"])
    for new, old in legacy_names.items():
        assert got[f"{new}_median"] == expected[f"{old}_median"], new


def test_compare_runs(records: dict[str, dict]) -> None:
    student = {"U": records["LOCO_catalunya_catalunyaR1_U"], "F": {}}
    teacher = {"U": records["MAT_catalunyaR1_U"], "F": records["MAT_catalunyaR1_F"]}
    out = analysis.compare(student, teacher)
    assert out["filter_teacher"]["classification"] in ("POLICY-DOMINATED", "FILTER-ASSISTED", "FILTER-DEPENDENT")
    joint = out["joint"]["U"]
    assert joint["both_robust"] + joint["student_only"] + joint["teacher_only"] + joint["neither"] == joint["n"]
    rows = analysis.failure_rows(student["U"], "loco_catalunya", "U", teacher=teacher["U"])
    assert len(rows) == sum(not r["robust"] for r in student["U"].values())


@pytest.mark.parametrize("path", PROTOCOLS, ids=lambda p: p.stem)
def test_protocol_structure(path: Path) -> None:
    protocol = load_protocol(path)
    assert Path(protocol["racing_line_path"]).exists()
    assert len(protocol["episodes"]) == 396
    for block, size in BLOCK_SIZES.items():
        assert sum(e["block"] == block for e in protocol["episodes"]) == size
    keys = {analysis.episode_key(e) for e in protocol["episodes"]}
    assert len(keys) == 396


@pytest.mark.parametrize("path", REDUCED_PROTOCOLS, ids=lambda p: p.stem)
def test_reduced_protocol(path: Path) -> None:
    """A stored reduced protocol is the rule applied to its full protocol: 40 held-out-start episodes copied
    from it, five frictions per start, every held-out setup seed four times."""
    reduced = json.loads(path.read_text())
    full = json.loads((path.parent / reduced["reduced_from"]).read_text())
    assert reduced == json.loads(json.dumps(reduce_protocol(full, reduced["reduced_from"])))
    episodes = reduced["episodes"]
    assert len({analysis.episode_key(e) for e in episodes}) == 40
    assert all(e in full["episodes"] for e in episodes) and not any(e["in_sample"] for e in episodes)
    starts = {e["start_label"] for e in episodes}
    assert len(starts) == 8 and all(sum(e["start_label"] == s for e in episodes) == 5 for s in starts)
    assert all(sum(e["seed"] == seed for e in episodes) == 4 for seed in range(100, 108))


def test_seed_summary(records: dict[str, dict]) -> None:
    runs = {"a": records["LOCO_catalunya_catalunyaR1_U"], "b": records["MAT_catalunyaR1_U"]}
    out = analysis.seed_summary(runs, records["MAT_catalunyaR1_U"])
    rates = [out["per_seed"][name]["robust"]["rate"] for name in runs]
    assert out["n_seeds"] == 2 and out["robust_rate_mean"] == pytest.approx(sum(rates) / 2)
    assert out["robust_pooled"]["n"] == 2 * out["n_episodes"]
    assert out["robust_in_every_seed"] <= min(out["per_seed"][name]["robust"]["k"] for name in runs)
    assert out["per_seed"]["b"]["paired_lap_time"]["ratio_of_means"] == 1.0
    with pytest.raises(ValueError):
        analysis.seed_summary({"a": runs["a"], "c": dict(list(runs["a"].items())[:5])})


def test_evaluate_protocol_resumes(tmp_path: Path) -> None:
    protocol = json.loads((REPO / "data/protocols/budapest_R0.json").read_text())
    protocol["episodes"] = [dict(e, max_steps=150) for e in protocol["episodes"][:2]]
    protocol["racing_line_file"] = str(REPO / protocol["racing_line_file"])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    output = tmp_path / "results.jsonl"
    evaluate_protocol(path, "teacher", "teacher", True, output, workers=2)
    evaluate_protocol(path, "teacher", "teacher", True, output, workers=1)  # nothing left to run
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(lines) == 2
    assert all(r["steps"] == 150 and not r["completed"] for r in lines)
    assert lines[0]["id"] == "teacher|budapest-R0|in_sample_gate|F|orig|0.80|42"


# --------------------------------------------------------------------------------------------- estimate cost (0a)
def _episode(i: int, robust: bool = True, lap: float = 100.0, mu_final: float = 1.0) -> dict:
    """A synthetic protocol episode of a plan coach; a failure is a track excursion at s = 10 i."""
    return {"block": "held_out", "start_label": f"h{i}", "mu": 0.8 + 0.1 * (i % 5), "seed": i, "robust": robust,
            "completed": robust, "lap_time_s": lap if robust else None,
            "termination_reason": "completed" if robust else "track_excursion", "terminal_s_m": 10.0 * i,
            "diagnostics": {}, "oracle_mu": {"final": mu_final, "max": max(mu_final, 1.0), "mean": 1.0},
            "plan_scale": {"mean": 1.1, "min": 1.0, "max": 1.2}, "plan_decisions": [[0.0, 1.0], [5.0, 1.15]]}


def _run(episodes: list[dict]) -> dict:
    return {analysis.episode_key(r): r for r in episodes}


@pytest.mark.parametrize(("estimate", "executor", "outcome"), [
    ([_episode(i) for i in range(10)], [_episode(i, lap=106.0) for i in range(10)], "A"),
    ([_episode(i, lap=101.0) for i in range(10)], [_episode(i, lap=106.0) for i in range(10)], "B3"),
    ([_episode(i, robust=i > 2, mu_final=1.04) for i in range(10)], [_episode(i, lap=106.0) for i in range(10)], "C"),
    ([_episode(i, robust=i > 2, mu_final=1.04) for i in range(10)],
     [_episode(i, robust=i > 3, lap=106.0) for i in range(10)], "B1"),
    ([_episode(i, robust=i > 2, mu_final=0.97) for i in range(10)],
     [_episode(i, robust=i > 3, lap=106.0) for i in range(10)], "B2"),
])
def test_estimate_cost(estimate: list[dict], executor: list[dict], outcome: str) -> None:
    """The outcome of measure 0a follows the rules fixed before the results (``analysis`` module docstring)."""
    true_mu, teacher = _run([_episode(i) for i in range(10)]), _run([_episode(i, lap=110.0) for i in range(10)])
    out = analysis.estimate_cost(_run(estimate), true_mu, _run(executor), teacher)
    assert out["outcome"] == outcome
    lost = sum(not r["robust"] for r in estimate)
    assert out["robust_lost"] == len(out["extra_failures"]) == lost and out["robust_gained"] == 0
    assert out["true_mu"]["robust"] == 10 and out["true_mu"]["teacher_ratio"] == pytest.approx(100.0 / 110.0)
    assert out["student_robust_floor"] == max(10 - lost, sum(r["robust"] for r in executor)) - 1
    if lost:
        assert out["extra_failures"][0]["last_plan_scale"] == 1.15
        assert out["extra_failures_on_overestimate"] == (lost if estimate[0]["oracle_mu"]["final"] > 1 else 0)


def test_estimate_cost_needs_the_same_episodes() -> None:
    runs = _run([_episode(i) for i in range(3)])
    with pytest.raises(ValueError):
        analysis.estimate_cost(runs, _run([_episode(i) for i in range(4)]), runs, runs)


# --------------------------------------------------------------------------------------------- plan reference (0b)
def test_plan_reference() -> None:
    """c* is the highest constant scale as robust as scale 1; 1.0 when none is."""
    teacher, base = _run([_episode(i, lap=110.0) for i in range(10)]), _run([_episode(i) for i in range(10)])
    scaled = {1.04: _run([_episode(i, lap=97.0) for i in range(10)]),
              1.08: _run([_episode(i, robust=i > 0, lap=94.0) for i in range(10)])}
    out = analysis.plan_reference(base, scaled, teacher)
    assert out["c_star"] == 1.04 and set(out["scales"]) == {"1.00", "1.04", "1.08"}
    assert out["c_star_summary"]["teacher_ratio"] == pytest.approx(97.0 / 110.0)
    assert analysis.plan_reference(base, {1.08: scaled[1.08]}, teacher)["c_star"] == 1.0
    with pytest.raises(ValueError):
        analysis.plan_reference(base, {1.04: _run([_episode(i) for i in range(3)])}, teacher)


# --------------------------------------------------------------------------------------------- plan student (§26.3)
def test_plan_student_criterion() -> None:
    """Non-inferior robustness and faster than c* on the aggregate of the seeds, with bootstrap intervals."""
    teacher = _run([_episode(i, lap=110.0) for i in range(20)])
    coach = _run([_episode(i, lap=100.0) for i in range(20)])
    c_star = _run([_episode(i, lap=103.0) for i in range(20)])
    good = {f"s{s}": _run([_episode(i, lap=100.5 + 0.1 * s) for i in range(20)]) for s in range(3)}
    out = analysis.plan_student_criterion(good, coach, c_star, teacher, resamples=200)
    assert out["passed"] and out["checks"] == {"robust_non_inferior": True, "faster_than_c_star": True}
    assert out["robust_difference"] == {"point": 0.0, "ci95": [0.0, 0.0]}
    assert out["versus_c_star"]["point"] == pytest.approx(100.6 / 103.0) and out["versus_c_star"]["n_pairs"] == 60
    assert out["versus_coach"]["ci95"][0] <= out["versus_coach"]["point"] <= out["versus_coach"]["ci95"][1]
    assert out["pooled"] == {"robust": 60, "n": 60, "wilson95": analysis.wilson(60, 60)}
    assert out["seeds"]["s1"]["mcnemar_vs_coach"] == {"student_only": 0, "coach_only": 0, "p": 1.0}
    assert out["sector_scale"]["student"] == {"0": pytest.approx(1.075)} and out["failures"] == []

    fragile = {f"s{s}": _run([_episode(i, robust=i % 4 != s, lap=100.0) for i in range(20)]) for s in range(3)}
    out = analysis.plan_student_criterion(fragile, coach, c_star, teacher, resamples=200)
    assert out["robust_difference"]["point"] == pytest.approx(0.25) and not out["checks"]["robust_non_inferior"]
    assert len(out["failures"]) == 15 and out["seeds"]["s0"]["mcnemar_vs_coach"]["coach_only"] == 5
    assert out["inclusive_versus_c_star"] == pytest.approx(100.0 / 103.0)  # the worst robust ratio is the same

    slow = {f"s{s}": _run([_episode(i, lap=103.5) for i in range(20)]) for s in range(3)}
    out = analysis.plan_student_criterion(slow, coach, c_star, teacher, resamples=200)
    assert out["checks"] == {"robust_non_inferior": True, "faster_than_c_star": False}
    with pytest.raises(ValueError):
        analysis.plan_student_criterion({"s0": _run([_episode(i) for i in range(3)])}, coach, c_star, teacher)
