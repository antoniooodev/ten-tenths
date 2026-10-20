"""Powertrain: rear-wheel drive, braking split by a fixed front bias, friction-limited axle forces.

    traction (tau >= 0):  Fx_f = 0,                          Fx_r = tau min(m A_MAX, P / max(vx, v_floor))
    braking  (tau < 0):   Fx_f = tau m D_MAX BETA_BRAKE,     Fx_r = tau m D_MAX (1 - BETA_BRAKE)
    saturation:           |Fx_i| <= mu Fz_i

The top speed of a speed stage is an actuator bound on the traction alone, applied before the plant:

    tau_eff = min(tau, clip((v_top - vx) / (SPEED_LIMIT_BAND v_top), 0, 1))   for tau > 0;  braking is free

The environment applies the same bound to its integrated command, so that the command does not wind up.
"""

from __future__ import annotations

import math

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import A_MAX, BETA_BRAKE, D_MAX, POWER_SPEED_FLOOR_MPS, SPEED_LIMIT_BAND


def compute_longitudinal_commands(
    tau_cmd: float, setup: SetupVector, vx: float = 0.0, wheel_power: float = math.inf,
) -> tuple[float, float]:
    """Return the commanded longitudinal forces (Fx_f, Fx_r) [N] for tau_cmd in [-1, 1]; the traction is bounded by
    the power at the wheels ``wheel_power`` [W] (infinite: no power limit, the original plant)."""
    m = setup.m
    if tau_cmd >= 0.0:
        return 0.0, tau_cmd * min(m * A_MAX, wheel_power / max(vx, POWER_SPEED_FLOOR_MPS))
    return tau_cmd * m * D_MAX * BETA_BRAKE, tau_cmd * m * D_MAX * (1.0 - BETA_BRAKE)


def saturate_longitudinal(fx_cmd: float, mu: float, fz: float) -> float:
    """Clip a longitudinal axle force [N] to the friction limit mu Fz."""
    return math.copysign(min(abs(fx_cmd), mu * fz), fx_cmd)


def limit_top_speed(tau_cmd: float, vx: float, top_speed: float | None) -> float:
    """Return the longitudinal command with the traction faded out near ``top_speed`` [m/s] (None: no bound)."""
    if top_speed is None or tau_cmd <= 0.0:
        return tau_cmd
    return min(tau_cmd, max(0.0, min(1.0, (top_speed - vx) / (SPEED_LIMIT_BAND * top_speed))))
