"""Longitudinal load transfer from the previous step (no algebraic loop).

    ax_LT = (Fx_r_prev + Fx_f_prev cos(delta_prev)) / m + vy_prev yawrate_prev
    Fz_f  = m / L (g LR - ax_LT hcg),     Fz_r = m / L (g LF + ax_LT hcg),     L = LF + LR

ax_LT uses only the memory of step t-1; it is distinct from the acceleration ax reported by the dynamics.
"""

from __future__ import annotations

import math

from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory
from contextual_bicycle_pacejka.spec.constants import G, LF, LR

WHEELBASE_M: float = LF + LR


def compute_ax_lt(memory: VehicleMemory, setup: SetupVector) -> float:
    """Return the load-transfer acceleration ax_LT [m/s^2] from the previous step."""
    return (
        (memory.Fx_r_prev + memory.Fx_f_prev * math.cos(memory.delta_prev)) / setup.m
        + memory.vy_prev * memory.yawrate_prev
    )


def compute_vertical_loads(ax_lt: float, setup: SetupVector) -> tuple[float, float]:
    """Return the axle loads (Fz_f, Fz_r) [N]; raise AdmissibilityError if one is not positive."""
    m = setup.m
    fz_f = (m / WHEELBASE_M) * (G * LR - ax_lt * setup.hcg)
    fz_r = (m / WHEELBASE_M) * (G * LF + ax_lt * setup.hcg)
    if fz_f <= 0.0 or fz_r <= 0.0:
        raise AdmissibilityError(f"non-positive axle load: Fz_f={fz_f:.4f} N, Fz_r={fz_r:.4f} N (ax_LT={ax_lt:.4f})")
    return fz_f, fz_r
