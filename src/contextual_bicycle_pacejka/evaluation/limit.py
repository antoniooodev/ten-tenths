"""Distance of protocol laps from the limit, and the criterion of a policy that drives at the limit.

A completed protocol lap from start s_i is compared with the minimum-time lap from the same standing start,
T_opt(s_i) (``reference/min_time.py``): its distance from the limit is (T - T_opt(s_i)) / T_opt(s_i). Laps form
(seeds x starts) matrices with NaN for the failed laps. A seed is at the limit when it completes at least
``SEED_LAP_FRACTION`` of its laps (9 of 10) and the median distance of its completed laps is at most ``LIMIT_GAP``;
a policy is at the limit when at least ``STAGE_SEED_FRACTION`` of its seeds are (4 of 5).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.stage_gate import SEED_LAP_FRACTION, STAGE_SEED_FRACTION, two_level_bootstrap

LIMIT_GAP: float = 0.05  # [-] largest median distance from the minimum-time lap of a seed at the limit


def lap_gaps(lap_times: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (T - T_opt(s_i)) / T_opt(s_i) for (seeds x starts) lap times (NaN: failed) and per-start references."""
    reference = np.asarray(reference, dtype=np.float64)
    return (np.asarray(lap_times, dtype=np.float64) - reference) / reference


def limit_criterion(completed: NDArray[np.bool_], gaps: NDArray[np.float64]) -> dict[str, Any]:
    """Apply the criterion to (seeds x starts) completion and distances from the limit (NaN where the lap failed)."""
    n_seeds, n_starts = completed.shape
    least_laps = math.ceil(SEED_LAP_FRACTION * n_starts - 1e-9)
    medians = [float(np.nanmedian(row)) if np.any(~np.isnan(row)) else None for row in gaps]
    seeds_at_limit = [bool(done >= least_laps and median is not None and median <= LIMIT_GAP)
                      for done, median in zip(completed.sum(axis=1), medians, strict=True)]
    gap = {"mean": None, "interval": None}
    if np.any(~np.isnan(gaps)):
        gap = {"mean": float(np.nanmean(gaps)), "interval": list(two_level_bootstrap(gaps))}
    return {
        "passed": sum(seeds_at_limit) >= math.ceil(STAGE_SEED_FRACTION * n_seeds - 1e-9),
        "seeds_at_limit": seeds_at_limit,
        "completed": completed.sum(axis=1).tolist(),
        "median_gap": medians,
        "gap": gap,
    }
