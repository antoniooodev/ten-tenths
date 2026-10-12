"""Track preparation: every stored track file is reproduced from its predecessor in the chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import REPO, golden_npz

from contextual_bicycle_pacejka.tracks.raceline import build_racing_line
from contextual_bicycle_pacejka.tracks.reference import build_reference
from contextual_bicycle_pacejka.tracks.repair import repair_racing_line, spikes
from contextual_bicycle_pacejka.tracks.speed_profile import (
    R0_FACTOR,
    ProbeLap,
    probe_lap,
    scale_profile,
    smooth,
    update_scale,
    write_profile,
)

TRACKS = REPO / "data/tracks"
CIRCUITS = ("monza", "spa", "catalunya", "budapest")


def _assert_same_npz(produced: Path, stored: Path, ignore: tuple[str, ...] = (), rtol: float = 0.0,
                     atol: float = 0.0) -> None:
    with np.load(produced) as a, np.load(stored) as b:
        assert sorted(set(a.files) - set(ignore)) == sorted(set(b.files) - set(ignore))
        for key in set(a.files) - set(ignore):
            if a[key].dtype.kind in "fc" and (rtol or atol):
                np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=atol, err_msg=key)
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_reference_from_tum_csv(circuit: str, tmp_path: Path) -> None:
    out = build_reference(TRACKS / circuit / "tum_centerline.csv", tmp_path / "reference.npz")
    _assert_same_npz(out, TRACKS / circuit / "reference.npz")


@pytest.mark.slow
@pytest.mark.parametrize("circuit", ["spa", "catalunya", "budapest"])
def test_racing_line_from_reference(circuit: str, tmp_path: Path) -> None:
    """The bound-constrained least-squares solution agrees to ~1e-6 m across platforms, not bit-for-bit."""
    out = build_racing_line(TRACKS / circuit / "reference.npz", tmp_path / "minimum_curvature_line.npz")
    _assert_same_npz(out, TRACKS / circuit / "minimum_curvature_line.npz", atol=1e-4)


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_racing_line_R0_from_probed_profile(circuit: str, tmp_path: Path) -> None:
    out = scale_profile(TRACKS / circuit / "speed_profile_probed.npz", tmp_path / "r0.npz", R0_FACTOR[circuit])
    _assert_same_npz(out, TRACKS / circuit / "racing_line_R0.npz")


@pytest.mark.slow
@pytest.mark.parametrize("circuit", ["monza", "spa", "catalunya"])
def test_racing_line_R1_from_R0(circuit: str, tmp_path: Path) -> None:
    log = repair_racing_line(TRACKS / circuit / "racing_line_R0.npz", TRACKS / circuit / "reference.npz",
                             tmp_path / "r1.npz")
    _assert_same_npz(tmp_path / "r1.npz", TRACKS / circuit / "racing_line_R1.npz", ignore=("repair",))
    assert log["G1_all_repaired"] and log["G2_no_spike_left"] and log["G3_kept_identical_length_ok"]
    assert log["G4_finite_geometry"]


def test_budapest_has_no_kink() -> None:
    with np.load(TRACKS / "budapest/racing_line_R0.npz") as data:
        assert spikes(data["kappa_radpm"]) == []


def test_speed_profile_update_rules(tmp_path: Path) -> None:
    golden = golden_npz("axle_profile_functions")
    scale, ratio, s, length = golden["scale_in"], golden["ratio"], golden["s"], float(golden["L"])
    np.testing.assert_array_equal(smooth(scale), golden["smooth"])
    robust = ProbeLap(True, 0.0, 100.0, 0, ratio)
    failed = ProbeLap(False, 1234.5, 50.0, 3, ratio)
    np.testing.assert_array_equal(update_scale(scale, s, length, robust), golden["upd_ok"])
    np.testing.assert_array_equal(update_scale(scale, s, length, failed), golden["upd_bad"])
    with np.load(TRACKS / "spa/minimum_curvature_line.npz") as data:
        speed = data["v_ref_mps"] * 0.8
    write_profile(TRACKS / "spa/minimum_curvature_line.npz", tmp_path / "p.npz", speed)
    with np.load(tmp_path / "p.npz") as data:
        np.testing.assert_array_equal(data["v_ref_mps"], golden["write_v"])
        np.testing.assert_array_equal(data["ax_mps2"], golden["write_ax"])


@pytest.mark.slow
def test_first_probe_lap_of_spa(tmp_path: Path) -> None:
    """Iteration 0, mu = 0.8 of the recorded Spa fit: same lap time and terminal point."""
    import json

    log = json.loads((TRACKS / "spa/speed_profile_probe_log.json").read_text())["history"][0]
    with np.load(TRACKS / "spa/minimum_curvature_line.npz") as data:
        speed = data["v_ref_mps"] * 0.7
    write_profile(TRACKS / "spa/minimum_curvature_line.npz", tmp_path / "candidate.npz", speed)
    lap = probe_lap(tmp_path / "candidate.npz", 0.8, 42)
    assert lap.completed and lap.empty_steps == 0
    assert lap.terminal_s_m == log["terminal_s_m"][0]
