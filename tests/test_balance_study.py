"""Classes of the balance-index grid, paired differences, regret and the rule of a clear gap."""

from __future__ import annotations

import numpy as np
import pytest

from contextual_bicycle_pacejka.evaluation.balance_study import (
    GRID,
    classify,
    clear_gap,
    completion_difference,
    paired_time_difference,
    pool,
    summarise,
)


def test_classes_of_the_grid() -> None:
    def members(width: str, cls: str) -> list[float]:
        return [b for b in GRID if classify(b, width) == cls]
    assert members("w13", "interpolation") == [-0.07, 0.06] and members("w13", "held_out") == [-0.04, 0.03]
    assert members("w23", "interpolation") == [-0.13, -0.10, -0.04, 0.03, 0.09, 0.11]
    assert members("w23", "held_out") == [-0.07, 0.06] and members("w23", "extrapolation") == [-0.23, -0.20, 0.17, 0.20]
    assert members("w1", "held_out") == [-0.10, 0.09] and members("w1", "extrapolation") == [-0.23, 0.20]
    assert len(members("w1", "interpolation")) == 8
    assert all(classify(0.0, width) == "nominal" for width in ("w13", "w23", "w1"))
    assert classify(0.0, None) == "nominal" and classify(-0.07, None) == "outside"


def test_paired_differences_and_pooling() -> None:
    nan = np.nan
    randomised = np.array([[231.0, nan, 232.0], [230.0, 230.5, nan]])
    oracle = np.array([[230.0, 230.0, nan], [230.5, 230.0, nan]])
    regret = paired_time_difference(randomised, oracle)
    np.testing.assert_allclose(regret, [[1.0, nan, nan], [-0.5, 0.5, nan]])
    completion = completion_difference(~np.isnan(oracle), ~np.isnan(randomised))
    np.testing.assert_array_equal(completion, [[0.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    pooled = pool([regret, regret])
    assert pooled.shape == (2, 6) and summarise(pooled)["n"] == 6
    summary = summarise(regret)
    assert summary["mean"] == pytest.approx(1.0 / 3.0) and summary["n"] == 3
    assert summary["interval"][0] <= summary["mean"] <= summary["interval"][1]
    assert summarise(np.full((2, 3), nan)) == {"mean": None, "interval": None, "n": 0}


def test_rule_of_a_clear_gap() -> None:
    def s(mean: float | None, low: float = 0.0, high: float = 1.0) -> dict:
        return {"mean": mean, "interval": None if mean is None else [low, high]}
    assert clear_gap(s(0.10, 0.01), s(None))
    assert not clear_gap(s(0.09, 0.01), s(0.2, 0.1))
    assert not clear_gap(s(0.30, 0.0), s(0.4, 0.1))  # the interval must be above 0
    assert clear_gap(s(0.0, -0.1), s(0.5, 0.05))
    assert not clear_gap(s(0.0, -0.1), s(0.8, -0.2))
