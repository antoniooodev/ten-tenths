"""Comparison of the Nominal, Randomised and Oracle policies over a grid of the balance index b.

Every policy drives the protocol laps of a stage at every b of ``GRID``. For a width of the training range, each b of
the grid is ``nominal`` (b = 0), ``interpolation`` (inside the range, outside the held-out bands), ``held_out`` (a
held-out value) or ``extrapolation`` (outside the range); for the Nominal policy every b other than 0 is ``outside``.

Laps form (seeds x starts) matrices: ``completed`` (bool) and ``times`` (NaN where the lap failed). Two policies are
paired by seed and start, because the models of one seed start from the same weights: the paired time difference is
NaN unless both complete, and the completion difference is -1, 0 or +1. Several values of b are pooled by putting
their starts side by side, so that the laps of one seed stay together in the two-level bootstrap of
``evaluation/stage_gate.py``.

The regret is R(b) = T_Randomised(b) - T_Oracle(b) at the same width. A width shows a clear gap between Oracle and
Randomised when, over its interpolation and held-out values pooled, the completion difference Oracle - Randomised has
a mean of at least ``GAP_COMPLETION`` and an interval above 0, or the regret a mean of at least ``GAP_REGRET_S`` and
an interval above 0. The memory extension is studied only if some width shows a clear gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.stage_gate import two_level_bootstrap

GRID: tuple[float, ...] = (-0.23, -0.20, -0.13, -0.10, -0.07, -0.04, 0.0, 0.03, 0.06, 0.09, 0.11, 0.17, 0.20)
WIDTHS: dict[str, dict[str, tuple[float, ...]]] = {
    "w13": {"range": (-0.07, 0.06), "held_out": (-0.04, 0.03)},
    "w23": {"range": (-0.13, 0.11), "held_out": (-0.07, 0.06)},
    "w1": {"range": (-0.20, 0.17), "held_out": (-0.10, 0.09)},
}
GAP_COMPLETION: float = 0.10  # [-] completion difference Oracle - Randomised for a clear gap
GAP_REGRET_S: float = 0.5  # [s] regret for a clear gap
GAP_CLASSES: tuple[str, ...] = ("interpolation", "held_out")
TOLERANCE: float = 1e-9


def classify(b: float, width: str | None) -> str:
    """Return the class of ``b`` for a width of ``WIDTHS`` (None: the Nominal policy)."""
    if abs(b) < TOLERANCE:
        return "nominal"
    if width is None:
        return "outside"
    lo, hi = WIDTHS[width]["range"]
    if any(abs(b - h) < TOLERANCE for h in WIDTHS[width]["held_out"]):
        return "held_out"
    return "interpolation" if lo - TOLERANCE <= b <= hi + TOLERANCE else "extrapolation"


def paired_time_difference(times_a: NDArray[np.float64], times_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``times_a - times_b`` per (seed, start), NaN unless both laps were completed."""
    return np.asarray(times_a, dtype=np.float64) - np.asarray(times_b, dtype=np.float64)


def completion_difference(completed_a: NDArray[np.bool_], completed_b: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return ``completed_a - completed_b`` per (seed, start): -1, 0 or +1."""
    return np.asarray(completed_a, dtype=np.float64) - np.asarray(completed_b, dtype=np.float64)


def pool(matrices: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Put (seeds x starts) matrices of several values of b side by side: seeds x (values x starts)."""
    return np.concatenate([np.asarray(m, dtype=np.float64) for m in matrices], axis=1)


def summarise(values: NDArray[np.float64]) -> dict[str, Any]:
    """Return the mean, its two-level bootstrap interval and the number of values (NaN: no value)."""
    values = np.asarray(values, dtype=np.float64)
    n = int(np.sum(~np.isnan(values)))
    if n == 0:
        return {"mean": None, "interval": None, "n": 0}
    return {"mean": float(np.nanmean(values)), "interval": list(two_level_bootstrap(values)), "n": n}


def clear_gap(completion: dict[str, Any], regret: dict[str, Any]) -> bool:
    """Apply the rule of a clear gap to the summaries of the completion difference and of the regret."""
    def clear(summary: dict[str, Any], threshold: float) -> bool:
        return summary["mean"] is not None and summary["mean"] >= threshold and summary["interval"][0] > 0.0
    return clear(completion, GAP_COMPLETION) or clear(regret, GAP_REGRET_S)
