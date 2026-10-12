"""Evaluation episodes: exact agreement with recorded episodes of the original implementation.

Covers the five learned policies (unfiltered), the safety filter (filtered policy and teacher episodes, including
the safe-set masks of every step) and the failure diagnostics of a failing episode.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import LEGACY_MODEL_DIRS, REPO, golden_json, golden_npz, repo_path

from contextual_bicycle_pacejka.control import safety_filter
from contextual_bicycle_pacejka.evaluation.episode import run_episode

CASES = golden_json("eval_cases")
OUTCOME_FIELDS = ("completed", "robust", "termination_reason", "events", "steps", "lap_time_s", "progress_m",
                  "lap_fraction", "terminal_s_m", "terminal_vx_mps", "no_safe_steps", "projected_steps",
                  "projected_fraction", "projection_records", "alpha_sat_f_rad", "alpha_sat_r_rad", "setup", "hist",
                  "max", "projections_in_pre_failure_window", "empty_in_pre_failure_window", "steps_curve",
                  "steps_braking", "steps_pre_failure", "filter_exception", "start_s_m", "start_vx_mps", "mu", "seed")


def _run(name: str, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    case = CASES[name]
    spec = dict(case["spec"], max_steps=case["steps_cap"], in_sample=True)
    trace: dict[str, list] = {"obs": [], "action": [], "state": [], "masks": []}
    original = safety_filter.safe_action_set

    def recording_safe_set(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        safe = original(*args, **kwargs)
        trace["masks"].append(np.packbits(safe.active.ravel()))
        return safe

    monkeypatch.setattr(safety_filter, "safe_action_set", recording_safe_set)

    def on_step(step: dict) -> None:
        trace["obs"].append(step["obs"])
        trace["action"].append(step["action"])
        trace["state"].append(step["state"])

    model_dir = REPO / "models" / LEGACY_MODEL_DIRS[case["model"]] if case["model"] else None
    record = run_episode(spec, repo_path(case["spec"]["raceline"]), case["controller"], case["cond"] == "F",
                         model_dir=model_dir, on_step=on_step)
    return record, trace


@pytest.mark.parametrize("name", sorted(n for n in CASES if not n.endswith("_fail")))
def test_episode_matches_reference(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    record, trace = _run(name, monkeypatch)
    ref = golden_npz(f"eval_{name}")
    expected = CASES[name]["record"]
    for field in OUTCOME_FIELDS:
        assert record[field] == expected[field], field
    np.testing.assert_array_equal(np.asarray(trace["action"]), ref["action"])
    np.testing.assert_array_equal(np.asarray(trace["state"]), ref["state"])
    np.testing.assert_array_equal(np.asarray(trace["obs"]), ref["obs"][1:])
    if CASES[name]["cond"] == "F":
        np.testing.assert_array_equal(np.asarray(trace["masks"]), ref["shield_masks"])


def test_failure_diagnostics_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing episode: outcome, trajectory and the failure diagnostics used by the analysis."""
    record, trace = _run("TreatmentA_spaR1_U_fail", monkeypatch)
    expected = CASES["TreatmentA_spaR1_U_fail"]["record"]
    for field in OUTCOME_FIELDS:
        assert record[field] == expected[field], field
    np.testing.assert_array_equal(np.asarray(trace["action"]), golden_npz("eval_TreatmentA_spaR1_U_fail")["action"])

    legacy = golden_json("faildiag_TreatmentA_spaR1")
    spec = dict(legacy["spec"], in_sample=True)
    record = run_episode(spec, repo_path(legacy["spec"]["raceline"]), "policy", False,
                         model_dir=REPO / "models/treatment_a", corner_zones=legacy["zones"])
    d = record["diagnostics"]
    td, sd = legacy["record"]["teacher_diag"], legacy["record"]["stage_a_diag"]
    assert d["speed_error_mean_abs_mps"] == td["speed_error_mean_abs_mps"]
    assert d["frac_overspeed"] == td["frac_overspeed"]
    assert d["rear_slip_ratio_p99"] == td["ar_ratio_p99"]
    assert d["speed_error_mean_mps"] == sd["speed_err_mean_mps"]
    assert d["ey_rms_m"] == sd["ey_rms_m"]
    assert d["teacher_tau_mae"] == sd["student_teacher_tau_mae"]
    assert d["teacher_delta_mae"] == sd["student_teacher_delta_mae"]
    assert d["terminal_zone"] == td["terminal_zone"]
    names = {"rear_slip_saturated": "ar_ge_1", "front_slip_saturated": "af_ge_1", "overspeed": "overspeed"}
    for new, old in names.items():
        assert d["pre_failure_last5s"][new] == td["pre_failure_last5s"][old], new
    terminal = {"s": "s", "ey": "ey", "vx": "vx", "v_target": "v_target", "rear_slip_ratio": "ar_ratio",
                "front_slip_ratio": "af_ratio", "kappa": "kappa", "tau": "tau_prop", "delta_cmd": "delta_prop"}
    for new, old in terminal.items():
        assert d["terminal_state"][new] == td["terminal_state"][old], new

