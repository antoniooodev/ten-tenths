"""Longitudinal load transfer from the previous step (no algebraic loop).

    ax_LT = F_body_x_prev / m = dvx/dt - vy r                                 (inertial, the default)
    ax_LT = (Fx_r_prev + Fx_f_prev cos(delta_prev)) / m + vy_prev yawrate_prev   (original plant)
    Fz_f  = m / L (g LR - ax_LT hcg),     Fz_r = m / L (g LF + ax_LT hcg),     L = LF + LR

The load transfer follows the longitudinal acceleration of the centre of mass in the body frame, which is the
longitudinal resultant of every force on the body (both axles, the lateral force of the steered front axle and the
drag) divided by the mass. The original plant added vy r to the longitudinal tyre forces, which gives dvx/dt, the
rate of a body-frame velocity component rather than the acceleration of the centre of mass, and left out the lateral
force of the steered axle: in a slide (vy r of -20 m/s^2 or more) it unloaded an axle that the forces did not
unload, and declared the state inadmissible. ax_LT uses only the memory of step t-1; it is distinct from the
acceleration ax = dvx/dt reported by the dynamics.
"""

from __future__ import annotations

import math

from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory
from contextual_bicycle_pacejka.spec.constants import G, LF, LR

WHEELBASE_M: float = LF + LR


def compute_ax_lt(memory: VehicleMemory, setup: SetupVector, inertial: bool = True) -> float:
    """Return the load-transfer acceleration ax_LT [m/s^2] from the previous step (``inertial``: F_body_x / m)."""
    if inertial:
        return memory.F_body_x_prev / setup.m
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
