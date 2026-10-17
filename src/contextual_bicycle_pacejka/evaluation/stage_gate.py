"""Criterion of a speed stage on the laps of several training seeds, with a two-level bootstrap.

The laps form a matrix (seeds x starts). A bootstrap replicate resamples the seeds and, independently, the starts,
both with replacement, and keeps every pair of the two resamples; the interval is the 2.5-97.5 percentile range of
``N_BOOTSTRAP`` replicates (generator seed ``BOOTSTRAP_SEED``). This is the stratified bootstrap of Agarwal et al.
(NeurIPS 2021) for one task with paired episodes.

A seed passes when it completes at least ``SEED_LAP_FRACTION`` of its laps (9 of 10); a stage passes when at least
``STAGE_SEED_FRACTION`` of the seeds pass (4 of 5). With a few seeds that complete either all their laps or none, a
threshold on the lower bound of the completion interval would ask for every seed. The completion rate over all laps
is reported with its interval, and lap times are summarised over the completed laps only; neither decides.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

N_BOOTSTRAP: int = 10_000
BOOTSTRAP_SEED: int = 0
SEED_LAP_FRACTION: float = 0.9
STAGE_SEED_FRACTION: float = 0.8


def two_level_bootstrap(
    values: NDArray[np.float64], statistic: Callable[[NDArray[np.float64]], float] = np.nanmean,
) -> tuple[float, float]:
    """Return the 95 % interval of ``statistic`` over (seeds x starts) ``values`` (NaN: no value)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_seeds, n_starts = values.shape
    replicates = np.empty(N_BOOTSTRAP)
    for k in range(N_BOOTSTRAP):
        seeds = rng.integers(0, n_seeds, n_seeds)
        starts = rng.integers(0, n_starts, n_starts)
        sample = values[np.ix_(seeds, starts)]
        replicates[k] = statistic(sample) if np.any(~np.isnan(sample)) else np.nan
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return float(low), float(high)


def stage_gate(completed: NDArray[np.bool_], lap_times: NDArray[np.float64]) -> dict[str, Any]:
    """Apply the criterion to (seeds x starts) lap outcomes; ``lap_times`` holds NaN where the lap failed."""
    n_seeds, n_starts = completed.shape
    seeds_passing = int(np.sum(completed.sum(axis=1) >= math.ceil(SEED_LAP_FRACTION * n_starts - 1e-9)))
    rate = float(np.mean(completed))
    low, high = two_level_bootstrap(completed.astype(np.float64), np.mean)
    times: dict[str, Any] = {"mean": None, "interval": None}
    if np.any(completed):
        times = {"mean": float(np.nanmean(lap_times)), "interval": list(two_level_bootstrap(lap_times))}
    return {
        "passed": seeds_passing >= math.ceil(STAGE_SEED_FRACTION * n_seeds - 1e-9),
        "seeds_passing": seeds_passing,
        "completion": {"rate": rate, "interval": [low, high]},
        "lap_time": times,
        "per_seed": {"completed": completed.sum(axis=1).tolist(),
                     "mean_lap_time": [float(np.nanmean(row)) if np.any(~np.isnan(row)) else None
                                       for row in lap_times]},
    }
