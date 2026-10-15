"""Result analysis: agreement with the original aggregation on recorded episode results."""

from __future__ import annotations

import pytest
from conftest import golden_json

from contextual_bicycle_pacejka.evaluation import analysis

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
    assert analysis.summarise(records["LOCO_catalunya_catalunyaR1_U"]) == expected["summary_LOCOcat_U"]
    assert analysis.paired_lap_time(records["LOCO_catalunya_catalunyaR1_U"], records["MAT_catalunyaR1_U"]) \
        == expected["paired_LOCOcat_vs_MAT_U"]


def test_failure_mechanisms(records: dict[str, dict]) -> None:
    expected = golden_json("analysis_outputs")["mechanisms_LOCOspa"]
    got = {r["id"]: analysis.failure_mechanism(r) for r in records["LOCO_spa_spaR1_U"].values() if not r["robust"]}
    assert got == expected
    assert len(got) > 0


def test_joint_outcome_and_failure_rows(records: dict[str, dict]) -> None:
    student, reference = records["LOCO_catalunya_catalunyaR1_U"], records["MAT_catalunyaR1_U"]
    joint = analysis.joint_outcome(student, reference)
    assert joint["both_robust"] + joint["student_only"] + joint["teacher_only"] + joint["neither"] == joint["n"]
    rows = analysis.failure_rows(student, "loco_catalunya", "U", teacher=reference)
    assert len(rows) == sum(not r["robust"] for r in student.values())


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
