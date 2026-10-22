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

A difference that two policies already show at b = 0 is not due to b. ``net_difference`` therefore takes, for every
(seed, start), the mean of a paired difference over the values of b minus the same difference at b = 0; the rule of a
clear gap can be applied to these net differences.

Two designs are defined: ``STAGE3`` (20 m/s, the policies ``<policy>_<width>_b2`` against ``nominal_b2``) and
``LIMIT`` (the free stage, the policies ``<policy>_limit_<width>`` against ``nominal_limit_cont``), each with its grid
and widths fixed before training.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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
LIMIT_GRID: tuple[float, ...] = (-0.24, -0.21, -0.17, -0.14, -0.11, -0.07, -0.04, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05)
LIMIT_WIDTHS: dict[str, dict[str, tuple[float, ...]]] = {
    "w13": {"range": (-0.07, 0.01), "held_out": (-0.04,)},
    "w23": {"range": (-0.14, 0.02), "held_out": (-0.07,)},
    "w1": {"range": (-0.21, 0.03), "held_out": (-0.11,)},
}
GAP_COMPLETION: float = 0.10  # [-] completion difference Oracle - Randomised for a clear gap
GAP_REGRET_S: float = 0.5  # [s] regret for a clear gap
GAP_CLASSES: tuple[str, ...] = ("interpolation", "held_out")
TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class BalanceDesign:
    """Grid of b, widths of the training range and policy names of one study of the balance index."""

    name: str
    grid: tuple[float, ...]
    widths: dict[str, dict[str, tuple[float, ...]]]
    nominal: str
    pattern: str  # name of a Randomised or Oracle policy, from ``policy`` and ``width``

    def tag(self, policy: str, width: str) -> str:
        return self.pattern.format(policy=policy, width=width)


STAGE3 = BalanceDesign("stage3", GRID, WIDTHS, "nominal_b2", "{policy}_{width}_b2")
LIMIT = BalanceDesign("limit", LIMIT_GRID, LIMIT_WIDTHS, "nominal_limit_cont", "{policy}_limit_{width}")
DESIGNS: dict[str, BalanceDesign] = {design.name: design for design in (STAGE3, LIMIT)}


def classify(b: float, width: str | None, widths: dict[str, dict[str, tuple[float, ...]]] = WIDTHS) -> str:
    """Return the class of ``b`` for a width of ``widths`` (None: the Nominal policy)."""
    if abs(b) < TOLERANCE:
        return "nominal"
    if width is None:
        return "outside"
    lo, hi = widths[width]["range"]
    if any(abs(b - h) < TOLERANCE for h in widths[width]["held_out"]):
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


def net_difference(by_b: list[NDArray[np.float64]], at_nominal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return, per (seed, start), the mean of the paired differences ``by_b`` over the values of b (NaN ignored)
    minus the difference at b = 0; NaN where either is missing."""
    stacked = np.stack([np.asarray(m, dtype=np.float64) for m in by_b])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a (seed, start) with no value at any b: NaN
        mean = np.nanmean(stacked, axis=0)
    return mean - np.asarray(at_nominal, dtype=np.float64)


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
