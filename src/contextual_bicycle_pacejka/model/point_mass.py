"""Point-mass speed profile along the centreline: the largest speed the friction ellipse and the powertrain allow.

At speed v the point mass accelerates and brakes with the limits of the plant (rear-wheel drive, front brake share
BETA_BRAKE, load transfer of the longitudinal resultant, drag D = 1/2 rho cdA v^2, power at the wheels P):

    traction   Fx <= min(m A_MAX, P / max(v, v_floor), mu (m g LF - D hcg) / (L - mu hcg)),   a = (Fx - D) / m
    braking    Fb <= min(m D_MAX, mu (m g LR + D hcg) / (BETA_BRAKE L - mu hcg),
                                  mu (m g LF - D hcg) / ((1 - BETA_BRAKE) L + mu hcg)),         a = -(Fb + D) / m

and turns with lateral acceleration up to mu g lateral_ratio (lateral_ratio = a_y,max(b) / a_y,max(0) for a balance
index b). A forward pass with the traction limit and a backward pass with the braking limit, twice around the lap,
give the profile. No policy uses it: it sizes the stages, bounds the start speeds of training episodes and scales
lap times.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.dynamics import PLANT, PlantOptions
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    A_MAX,
    AIR_DENSITY,
    BETA_BRAKE,
    D_MAX,
    G,
    HCG_NOM,
    LF,
    LR,
    MASS_NOM,
    MU_NOM,
    POWER_SPEED_FLOOR_MPS,
)

PROFILE_STEP_M: float = 1.0  # [m] largest step of the profile (the lap is split into equal steps)


def longitudinal_limits(mu: float = MU_NOM, v: float = 0.0, options: PlantOptions = PLANT) -> tuple[float, float]:
    """Return the largest (traction, braking) accelerations [m/s^2] of the nominal car at speed v, drag included."""
    m, wheelbase = MASS_NOM, LF + LR
    drag = 0.5 * AIR_DENSITY * options.drag_area * v * v
    traction = min(m * A_MAX, options.wheel_power / max(v, POWER_SPEED_FLOOR_MPS),
                   mu * (m * G * LF - drag * HCG_NOM) / (wheelbase - mu * HCG_NOM))
    braking = min(m * D_MAX, mu * (m * G * LR + drag * HCG_NOM) / (wheelbase * BETA_BRAKE - mu * HCG_NOM),
                  mu * (m * G * LF - drag * HCG_NOM) / (wheelbase * (1.0 - BETA_BRAKE) + mu * HCG_NOM))
    return (traction - drag) / m, (braking + drag) / m


def speed_profile(
    track: RealTrack, top_speed: float | None, mu: float = MU_NOM, lateral_ratio: float = 1.0,
    options: PlantOptions = PLANT,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the positions s [m] (equal steps over the lap) and the point-mass speed [m/s] at each."""
    n = math.ceil(track.track_length / PROFILE_STEP_M)
    step = track.track_length / n
    s = np.arange(n) * step
    kappa = np.abs(track.kappa_many(s))
    lateral_limit = mu * G * lateral_ratio
    v = np.minimum(np.sqrt(lateral_limit / np.maximum(kappa, 1e-12)), math.inf if top_speed is None else top_speed)

    def reachable(v_from: float, k: float, braking: bool) -> float:
        traction, brake = longitudinal_limits(mu, v_from, options)
        lateral = min(v_from * v_from * k / lateral_limit, 1.0)
        a = (brake if braking else traction) * math.sqrt(1.0 - lateral * lateral)
        return math.sqrt(max(v_from * v_from + 2.0 * a * step, 0.0))

    for _ in range(2):  # twice around the lap, so the closing point is consistent
        for i in range(1, 2 * n + 1):
            v[i % n] = min(v[i % n], reachable(v[(i - 1) % n], kappa[(i - 1) % n], braking=False))
        for i in range(2 * n, 0, -1):
            v[(i - 1) % n] = min(v[(i - 1) % n], reachable(v[i % n], kappa[i % n], braking=True))
    return s, v


def profile_lap_time(s: NDArray[np.float64], v: NDArray[np.float64], track_length: float) -> float:
    """Return the lap time [s] of a speed profile on equal steps."""
    return float(np.sum((track_length / len(s)) / v))
