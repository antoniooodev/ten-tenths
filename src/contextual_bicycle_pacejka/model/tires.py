"""Tyre model: slip angles, combined-slip capacity and Pacejka lateral force per axle.

    alpha_f = delta - atan2(vy + LF r, vx),     alpha_r = -atan2(vy - LR r, vx)
    D_i     = max(lambda_i mu Fz_i sqrt(1 - (Fx_i / (mu Fz_i))^2), EPS_D)       (longitudinal force has priority)
    Fy_i    = D_i sin(C_i atan(B_i a - E_i (B_i a - atan(B_i a)))),     a = alpha_i

The balance index b scales only the lateral capacity, at constant static total (static loads Fz_f0 / Fz_r0 = LR / LF):

    lambda_f = 1 + b,     lambda_r = 1 - b LR / LF,     lambda_f Fz_f0 + lambda_r Fz_r0 = Fz_f0 + Fz_r0

b > 0 moves grip to the front (oversteer), b < 0 to the rear (understeer); b = 0 is the original plant. The
longitudinal limit stays mu Fz_i, and B, C, E stay fixed, so the cornering stiffness B C D scales with lambda_i and the
peak slip angle does not depend on b.

B, C, E are fixed per axle. C and E are the identified AV-24 shape and curvature factors
(Jardali et al., "LPV-MPC for Lateral Control in Full-Scale Autonomous Racing", Table I);
B is calibrated so that the force peaks at about 0.12 rad (front) and 0.13 rad (rear) for this vehicle.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from contextual_bicycle_pacejka.model.load_transfer import compute_vertical_loads
from contextual_bicycle_pacejka.model.state import SetupVector, VehicleState
from contextual_bicycle_pacejka.spec.constants import ALPHA_SAT_XTOL, B_MAX, B_MIN, EPS_D, LF, LR, V_MIN

PACEJKA_B_FRONT: float = 8.141196706041054
PACEJKA_C_FRONT: float = 1.81
PACEJKA_E_FRONT: float = -1.00
PACEJKA_B_REAR: float = 7.534090791059767
PACEJKA_C_REAR: float = 1.96
PACEJKA_E_REAR: float = -0.26
_COEFFICIENTS = {
    "front": (PACEJKA_B_FRONT, PACEJKA_C_FRONT, PACEJKA_E_FRONT),
    "rear": (PACEJKA_B_REAR, PACEJKA_C_REAR, PACEJKA_E_REAR),
}


def pacejka_coefficients(axle: str) -> tuple[float, float, float]:
    """Return (B, C, E) of the ``"front"`` or ``"rear"`` axle."""
    try:
        return _COEFFICIENTS[axle]
    except KeyError:
        raise ValueError(f"axle must be 'front' or 'rear', got {axle!r}") from None


def compute_slip_angles(state: VehicleState) -> tuple[float, float]:
    """Return (alpha_f, alpha_r) [rad]; ``state.delta`` must already be the post-actuator angle."""
    if state.vx < V_MIN:
        raise ValueError(f"vx={state.vx:.4f} < V_MIN={V_MIN}")
    alpha_f = state.delta - math.atan2(state.vy + LF * state.yaw_rate, state.vx)
    alpha_r = -math.atan2(state.vy - LR * state.yaw_rate, state.vx)
    return alpha_f, alpha_r


def compute_lateral_scales(b: float) -> tuple[float, float]:
    """Return the lateral capacity scales (lambda_f, lambda_r) [-] of the balance index b; both are 1.0 at b = 0."""
    if not B_MIN < b < B_MAX:
        raise ValueError(f"balance index b={b} outside ({B_MIN}, {B_MAX:.6f})")
    return 1.0 + b, 1.0 - b * (LR / LF)


def compute_combined_slip_capacity(fx: float, mu: float, fz: float, scale: float = 1.0) -> float:
    """Return the lateral capacity D [N] left on an axle that transmits the longitudinal force fx.

    ``scale`` is the lateral capacity scale of the axle (``compute_lateral_scales``); the friction limit of fx stays
    mu fz.
    """
    mu_fz = mu * fz
    ratio = fx / mu_fz
    return max(scale * mu_fz * math.sqrt(1.0 - min(ratio * ratio, 1.0)), EPS_D)


def compute_lateral_force(alpha: float, b: float, c: float, d: float, e: float) -> float:
    """Return the Pacejka lateral force [N] for slip angle alpha [rad] and coefficients (B, C, D, E)."""
    ba = b * alpha
    phi = ba - e * (ba - math.atan(ba))
    return d * math.sin(c * math.atan(phi))


def compute_saturation_threshold(setup: SetupVector, axle: str) -> float:
    """Return the peak slip angle alpha_sat [rad] of an axle (static loads, no longitudinal force).

    The peak solves B a (1 - E) + E atan(B a) = tan(pi / (2 C)); it depends only on the fixed coefficients.
    It is computed once per episode at reset.
    """
    compute_vertical_loads(0.0, setup)  # admissibility check of the static loads
    b, c, e = pacejka_coefficients(axle)
    target = math.tan(math.pi / (2.0 * c))

    def residual(alpha: float) -> float:
        ba = b * alpha
        return ba * (1.0 - e) + e * math.atan(ba) - target

    asymptote = e * math.pi / 2.0
    upper = 2.0 * (target - asymptote) / (b * (1.0 - e)) + math.pi if target > asymptote else math.pi
    while residual(upper) <= 0.0:
        upper *= 2.0
    return brentq(residual, 0.0, upper, xtol=ALPHA_SAT_XTOL)
