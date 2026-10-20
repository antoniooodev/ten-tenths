"""Steady-state handling of the plant: constant-radius trim, understeer gradient, lateral limit and yaw stability.

Every quantity comes from the plant equations (``dynamics.compute_vehicle_rates``), not from linear formulas.

Trim on a circle of radius R at speed v: the unknowns (delta, vy, tau) solve

    dvx/dt = dvy/dt = dr/dt = 0,     vx = v,     r = v / R,

with the load-transfer memory equal to the trim itself (steady state). A trim is on the rising side of both tyre
curves when |alpha_i| <= alpha_sat_i.

Understeer gradient: K = d delta / d ay at constant radius (delta = L / R + K ay in the linear range), the
least-squares slope over a grid of ay; K > 0 is understeer.

Lateral limit: the largest ay of a trim on the rising side of both tyre curves. The speed folds back before the peak
slip angle, so the branch is followed in the slip angle of each axle, not in v; the axle with the larger
|alpha_i| / alpha_sat_i at the limit saturates first.

Yaw stability: eigenvalues of the Jacobian of (dvy/dt, dr/dt) with respect to (vy, r) on the straight line
(delta = 0, tau = 0, central differences). The critical speed is the speed at which an eigenvalue crosses into the
right half-plane; an understeering car has none.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve, minimize_scalar

from contextual_bicycle_pacejka.model.dynamics import PLANT, compute_vehicle_rates
from contextual_bicycle_pacejka.model.powertrain import compute_longitudinal_commands
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold
from contextual_bicycle_pacejka.spec.constants import ALPHA_SAT_XTOL, G, LF, LR

TRIM_RESIDUAL_TOL: float = 1e-9  # [m/s^2, rad/s^2] largest accepted residual of a trim
LIMIT_SLIP_TOL: float = 1e-9  # [rad] slip-angle tolerance of the lateral-limit search
JACOBIAN_STEP: float = 1e-6  # [m/s, rad/s] central-difference step of the yaw Jacobian
CRITICAL_SPEED_TOL: float = 1e-3  # [m/s] bisection tolerance of the critical speed


class Trim(NamedTuple):
    """Steady cornering on a circle: speed v [m/s], radius [m], delta [rad], vy [m/s], tau [-], slip angles [rad]."""

    v: float
    radius: float
    delta: float
    vy: float
    tau: float
    alpha_f: float
    alpha_r: float

    @property
    def ay(self) -> float:
        """Lateral acceleration v^2 / R [m/s^2]."""
        return self.v * self.v / self.radius


def _trim_rates(
    unknowns: NDArray[np.float64], v: float, radius: float, setup: SetupVector,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Return (dvx/dt, dvy/dt, dr/dt) and the diagnostics at a candidate trim (delta, vy, tau)."""
    delta, vy, tau = (float(u) for u in unknowns)
    r = v / radius
    fx_f, fx_r = compute_longitudinal_commands(tau, setup, v, PLANT.wheel_power)
    memory = VehicleMemory(Fx_r_prev=fx_r, Fx_f_prev=fx_f, delta_prev=delta, vy_prev=vy, yawrate_prev=r,
                           F_body_x_prev=-setup.m * vy * r)  # at the trim dvx/dt = 0, so F_body_x = -m vy r
    ax, ay, yaw_acc, diag = compute_vehicle_rates(VehicleState(vx=v, vy=vy, yaw_rate=r, delta=delta), tau, setup,
                                                  memory)
    return np.array([ax, ay, yaw_acc]), diag


def _accept_trim(
    unknowns: NDArray[np.float64], v: float, radius: float, setup: SetupVector, rates: NDArray[np.float64],
    diag: dict[str, float],
) -> Trim | None:
    """Return the trim if the solver converged to unsaturated longitudinal forces on the rising side of both tyres."""
    if np.max(np.abs(rates)) > TRIM_RESIDUAL_TOL:
        return None
    fx_f, fx_r = compute_longitudinal_commands(float(unknowns[2]), setup, v, PLANT.wheel_power)
    if diag["Fx_f"] != fx_f or diag["Fx_r"] != fx_r:  # the memory assumed unsaturated longitudinal forces
        return None
    if (abs(diag["alpha_f"]) > compute_saturation_threshold(setup, "front")
            or abs(diag["alpha_r"]) > compute_saturation_threshold(setup, "rear")):
        return None
    return Trim(v, radius, float(unknowns[0]), float(unknowns[1]), float(unknowns[2]), diag["alpha_f"],
                diag["alpha_r"])


def solve_trim(v: float, radius: float, setup: SetupVector, guess: Trim | None = None) -> Trim | None:
    """Return the trim at speed v [m/s] on radius [m], or None when none is found on the rising side of both tyres.

    ``guess`` warm-starts the solver (continuation); without it the kinematic trim is used.
    """
    if guess is None:
        start = np.array([(LF + LR) / radius, LR * v / radius, 0.0])
    else:
        start = np.array([guess.delta, guess.vy, guess.tau])
    try:
        solution, _, converged, _ = fsolve(lambda u: _trim_rates(u, v, radius, setup)[0], start, xtol=1e-12,
                                           full_output=True)
        rates, diag = _trim_rates(solution, v, radius, setup)
    except AdmissibilityError:  # the solver wandered into a non-positive axle load
        return None
    return _accept_trim(solution, v, radius, setup, rates, diag) if converged == 1 else None


def solve_trim_at_slip(alpha: float, axle: str, radius: float, setup: SetupVector, guess: Trim) -> Trim | None:
    """Return the trim on radius [m] whose ``axle`` has slip angle ``alpha`` [rad], the speed being an unknown.

    Unlike the speed, the slip angle grows monotonically along the trim branch, also through its fold.
    """
    key = "alpha_f" if axle == "front" else "alpha_r"

    def residual(u: NDArray[np.float64]) -> NDArray[np.float64]:
        rates, diag = _trim_rates(u[:3], float(u[3]), radius, setup)
        return np.append(rates, diag[key] - alpha)

    try:
        solution, _, converged, _ = fsolve(residual, np.array([guess.delta, guess.vy, guess.tau, guess.v]),
                                           xtol=1e-12, full_output=True)
        v = float(solution[3])
        rates, diag = _trim_rates(solution[:3], v, radius, setup)
    except AdmissibilityError:
        return None
    if converged != 1 or abs(diag[key] - alpha) > ALPHA_SAT_XTOL:
        return None
    return _accept_trim(solution[:3], v, radius, setup, rates, diag)


def understeer_gradient(setup: SetupVector, radius: float, ay_grid: NDArray[np.float64]) -> float:
    """Return K = d delta / d ay [rad / (m/s^2)] on ``radius`` [m], fitted over ``ay_grid`` [m/s^2] (ascending)."""
    trims: list[Trim] = []
    guess = None
    for ay in ay_grid:
        trim = solve_trim(math.sqrt(float(ay) * radius), radius, setup, guess)
        if trim is None:
            raise ValueError(f"no trim at ay={ay / G:.3f} g on R={radius} m for b={setup.b}")
        trims.append(trim)
        guess = trim
    return float(np.polyfit([t.ay for t in trims], [t.delta for t in trims], 1)[0])


def lateral_limit(setup: SetupVector, radius: float, ay_start: float, steps: int) -> Trim:
    """Return the trim at the largest lateral acceleration on ``radius`` [m].

    From the trim at ``ay_start`` [m/s^2], the slip angle of each axle in turn is raised towards its peak in ``steps``
    steps (continuation; the slip angle of the axle that does not saturate may stop growing, so both are followed);
    the largest ay found is then refined by a bounded scalar search in the slip angle of the axle that reached it.
    """
    start = solve_trim(math.sqrt(ay_start * radius), radius, setup)
    if start is None:
        raise ValueError(f"no trim at ay={ay_start / G:.3f} g on R={radius} m for b={setup.b}")
    best, best_axle, best_step = start, "front", 0.0
    for axle in ("front", "rear"):
        alpha_peak = compute_saturation_threshold(setup, axle)
        alphas = np.linspace(start.alpha_f if axle == "front" else start.alpha_r, alpha_peak, steps + 1)
        guess = start
        for alpha in alphas[1:]:
            nxt = solve_trim_at_slip(float(alpha), axle, radius, setup, guess)
            if nxt is None:
                break
            guess = nxt
            if nxt.ay > best.ay:
                best, best_axle, best_step = nxt, axle, float(alphas[1] - alphas[0])
    if best is start:
        return best
    alpha_best = best.alpha_f if best_axle == "front" else best.alpha_r
    alpha_peak = compute_saturation_threshold(setup, best_axle)
    found: list[Trim] = [best]

    def negative_ay(alpha: float) -> float:
        refined = solve_trim_at_slip(alpha, best_axle, radius, setup, found[0])
        if refined is None:
            return math.inf
        found.append(refined)
        return -refined.ay

    minimize_scalar(negative_ay, bounds=(alpha_best - best_step, min(alpha_best + best_step, alpha_peak)),
                    method="bounded", options={"xatol": LIMIT_SLIP_TOL})
    return max(found, key=lambda t: t.ay)


def saturating_axle(trim: Trim, setup: SetupVector) -> str:
    """Return ``"front"`` or ``"rear"``: the axle closer to its peak slip angle at ``trim``."""
    ratio_f = abs(trim.alpha_f) / compute_saturation_threshold(setup, "front")
    ratio_r = abs(trim.alpha_r) / compute_saturation_threshold(setup, "rear")
    return "front" if ratio_f >= ratio_r else "rear"


def yaw_eigenvalues(v: float, setup: SetupVector) -> NDArray[np.complex128]:
    """Return the eigenvalues [1/s] of the lateral-yaw dynamics (vy, r) on the straight line at speed v [m/s]."""

    def lateral_rates(vy: float, r: float) -> NDArray[np.float64]:
        memory = VehicleMemory(vy_prev=vy, yawrate_prev=r, F_body_x_prev=-setup.m * vy * r)
        _, ay, yaw_acc, _ = compute_vehicle_rates(VehicleState(vx=v, vy=vy, yaw_rate=r), 0.0, setup, memory)
        return np.array([ay, yaw_acc])

    h = JACOBIAN_STEP
    jacobian = np.column_stack([
        (lateral_rates(h, 0.0) - lateral_rates(-h, 0.0)) / (2.0 * h),
        (lateral_rates(0.0, h) - lateral_rates(0.0, -h)) / (2.0 * h),
    ])
    return np.linalg.eigvals(jacobian)


def critical_speed(setup: SetupVector, v_low: float, v_high: float) -> float:
    """Return the speed [m/s] in (v_low, v_high) above which the straight line is unstable, or inf if it is not.

    The straight line must be stable at ``v_low``.
    """
    def unstable(v: float) -> bool:
        return bool(np.max(yaw_eigenvalues(v, setup).real) > 0.0)

    if unstable(v_low):
        raise ValueError(f"straight line already unstable at v={v_low} m/s for b={setup.b}")
    if not unstable(v_high):
        return math.inf
    while v_high - v_low > CRITICAL_SPEED_TOL:
        v_mid = 0.5 * (v_low + v_high)
        if unstable(v_mid):
            v_high = v_mid
        else:
            v_low = v_mid
    return 0.5 * (v_low + v_high)
