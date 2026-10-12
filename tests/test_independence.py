"""Is an agent a copy of the teacher: speed profiles on a progress grid, their shape against the scaled teacher
profile, sector times, failure sites near the teacher's, the measures of a scaled copy and a recorded trace."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.evaluation.independence import (
    colocated,
    independence,
    profile,
    sector_times,
    shape_residual,
    speed_trace,
)
from contextual_bicycle_pacejka.evaluation.protocol import load_protocol
from contextual_bicycle_pacejka.spec.constants import DT

MONZA_R0 = str(REPO / "data/tracks/monza/racing_line_R0.npz")
LENGTH = 1000.0


def _trace(speed: np.ndarray, mu: float = 1.0, robust: bool = True) -> dict:
    """A lap of LENGTH metres at the speed profile ``speed`` (one value per metre)."""
    progress, speeds, s = [], [], 0.0
    while s < LENGTH:
        v = float(speed[int(s)])
        progress.append(s)
        speeds.append(v)
        s += v * DT
    return {"progress": np.asarray(progress), "speed": np.asarray(speeds), "robust": robust, "mu": mu,
            "terminal_s": 0.0, "lap_time_s": len(progress) * DT}


def test_profile_and_shape() -> None:
    """A scaled teacher profile has no shape residual; a different shape has one."""
    v = 30.0 + 10.0 * np.sin(np.linspace(0.0, 6.0, 1000))

    def lap_profile(speed: np.ndarray) -> np.ndarray:
        trace = _trace(speed)
        return profile(trace["progress"], trace["speed"], LENGTH)

    teacher = lap_profile(v)
    k, residual = shape_residual(lap_profile(1.08 * v), teacher)
    assert k == pytest.approx(1.08, rel=2e-3) and residual < 5e-3
    assert shape_residual(lap_profile(v[::-1].copy()), teacher)[1] > 0.1
    assert np.isnan(profile(np.array([5.0]), np.array([1.0]), LENGTH)[50])


def test_sector_times_and_failure_sites() -> None:
    """Sector times add up to the lap time; failure sites are near across the end of the lap."""
    trace = _trace(np.full(1000, 25.0))
    times = sector_times(trace["progress"], LENGTH, 250.0)
    assert len(times) == 4 and times.sum() == pytest.approx(trace["lap_time_s"])
    np.testing.assert_allclose(times, 10.0, atol=DT)
    assert colocated([20.0, 500.0], [980.0], LENGTH, 50.0) == [True, False]
    assert colocated([20.0], [], LENGTH) == [False]


def test_scaled_copy_measures() -> None:
    """A scaled copy gains the same in every sector, at every friction, and fails where the teacher fails."""
    v = 30.0 + 10.0 * np.sin(np.linspace(0.0, 6.0, 1000))
    teacher = [_trace(v, mu) for mu in (0.8, 1.2)] + [{**_trace(v, 1.0, False), "terminal_s": 400.0}]
    copy = [_trace(1.05 * v, mu) for mu in (0.8, 1.2)] + [{**_trace(v, 1.0, False), "terminal_s": 410.0}]
    m = independence(copy, teacher, LENGTH)
    assert m["episodes"] == 2 and m["robust"] == 2 and m["n"] == 3
    assert m["shape_residual"] < 5e-3 and m["sector_gain_std"] < 5e-3 and m["sector_slower_share"] == 0.0
    assert m["ratio_spread"] == pytest.approx(0.0, abs=1e-3)
    assert m["failures_at_teacher_sites"] == 1 and m["teacher_failures_survived"] == 0


def test_speed_trace() -> None:
    """A recorded trace has one progress and speed per step of the episode."""
    spec = {**load_protocol(REPO / "data/protocols/monza_R0_reduced.json")["episodes"][0], "max_steps": 30}
    trace = speed_trace(spec, MONZA_R0)
    assert len(trace["progress"]) == len(trace["speed"]) == 30
    assert np.all(np.diff(trace["progress"]) > 0.0) and not trace["robust"]  # 30 steps are not a lap
