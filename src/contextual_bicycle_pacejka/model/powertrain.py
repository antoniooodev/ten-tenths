"""Powertrain: rear-wheel drive, braking split by a fixed front bias, friction-limited axle forces.

    traction (tau >= 0):  Fx_f = 0,                          Fx_r = tau m A_MAX
    braking  (tau < 0):   Fx_f = tau m D_MAX BETA_BRAKE,     Fx_r = tau m D_MAX (1 - BETA_BRAKE)
    saturation:           |Fx_i| <= mu Fz_i
"""

from __future__ import annotations

import math

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import A_MAX, BETA_BRAKE, D_MAX


def compute_longitudinal_commands(tau_cmd: float, setup: SetupVector) -> tuple[float, float]:
    """Return the commanded longitudinal forces (Fx_f, Fx_r) [N] for tau_cmd in [-1, 1]."""
    m = setup.m
    if tau_cmd >= 0.0:
        return 0.0, tau_cmd * m * A_MAX
    return tau_cmd * m * D_MAX * BETA_BRAKE, tau_cmd * m * D_MAX * (1.0 - BETA_BRAKE)


def saturate_longitudinal(fx_cmd: float, mu: float, fz: float) -> float:
    """Clip a longitudinal axle force [N] to the friction limit mu Fz."""
    return math.copysign(min(abs(fx_cmd), mu * fz), fx_cmd)
