"""Minimum-time lap of the plant car on a circuit: the reference that tells how far a policy is from the limit.

An optimal-control problem over one lap, in the arc length s of the centreline (``ds`` steps, periodic):

    states     ey, e_psi, vx, vy, r, tau               (tau: the longitudinal command, as in the environment)
    controls   delta in [-DELTA_MAX, DELTA_MAX],  tau rate in [-TAU_RATE_MAX, TAU_RATE_MAX]
    algebraic  ax_LT = F_body_x / m                    (the inertial load transfer, without the one-step delay)
    dynamics   d(state)/ds = d(state)/dt / s_dot       (the equations of ``model/dynamics.py``; trapezoidal rule)
    bounds     -w_right(s) <= ey <= w_left(s)           (the centre of the car on the track, as the off-track rule;
                                                         ``edge_margin`` moves both edges inwards)
               |alpha_i| <= alpha_sat,i                 (each axle at most at its peak slip: the limit, not beyond)
               commanded longitudinal forces within mu Fz (the plant clips them; at the optimum it makes no difference)
    objective  the lap time, the integral of ds / s_dot, plus a small weight on the steering changes

Two simplifications make the problem solvable and the reference at most slightly optimistic, that is stricter for a
policy: the steering actuator lag (0.03 s) is left out, the steering angle being a control, and the slip of each axle
is bounded by its peak (beyond the peak the force falls and the solver loses its way; the limit is the peak).

The equations are the plant's for the nominal setup with a balance index b, with the drag, the power limit and the
inertial load transfer of ``PLANT``. Two kinks are smoothed for the solver, within ``SMOOTHING``: the split of tau
into throttle and brake and the minimum of the traction and power limits. ``vehicle_rates`` with ``smooth=False``
is exact and is checked against the plant in the tests. The reference is used only to measure policies; no policy
sees it in its reward, observation or weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
from contextual_bicycle_pacejka.model.dynamics import PLANT
from contextual_bicycle_pacejka.model.point_mass import speed_profile
from contextual_bicycle_pacejka.model.tires import (
    compute_lateral_scales,
    compute_saturation_threshold,
    pacejka_coefficients,
)
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    A_MAX,
    AIR_DENSITY,
    BETA_BRAKE,
    D_MAX,
    DELTA_MAX,
    G,
    HCG_NOM,
    IZ,
    LF,
    LR,
    MASS_NOM,
    MU_NOM,
    POWER_SPEED_FLOOR_MPS,
    TAU_D_NOM,
    TAU_RATE_MAX,
)

STATE_NAMES: tuple[str, ...] = ("ey", "epsi", "vx", "vy", "r", "delta", "tau")
SMOOTHING: dict[str, float] = {"tau": 1e-3, "force": 10.0}  # [-] throttle/brake split, [N] traction/power minimum
FRICTION_MARGIN: float = 0.999  # commanded longitudinal force at most this share of mu Fz (keeps the sqrt smooth)
START_FRACTION: float = 0.9  # initial guess: this fraction of the point-mass profile
SMOOTHNESS_WEIGHT: float = 1e-3  # [s/rad^2] weight of the squared steering changes between nodes


def _max0(x: Any, eps: float) -> Any:
    return 0.5 * (x + ca.sqrt(x * x + eps * eps))


def _min(a: Any, b: Any, eps: float) -> Any:
    return 0.5 * (a + b - ca.sqrt((a - b) * (a - b) + eps * eps))


def corridor(
    left: NDArray[np.float64], right: NDArray[np.float64], margin: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the edge distances ``left`` and ``right`` [m] moved inwards by ``margin`` [m].

    Raises ValueError if the margin closes the track somewhere.
    """
    left, right = left - margin, right - margin
    if margin < 0.0 or min(left.min(), right.min()) <= 0.0:
        raise ValueError(f"an edge margin of {margin} m leaves no track")
    return left, right


def vehicle_rates(
    x: Any, delta_cmd: Any, tau_rate: Any, ax_lt: Any, kappa: Any, b: float = 0.0, smooth: bool = True,
) -> dict[str, Any]:
    """Return the time derivatives of the states and the forces, for CasADi or float inputs.

    ``x`` = [ey, epsi, vx, vy, r, delta, tau]; ``ax_lt`` [m/s^2] the load-transfer acceleration; ``kappa`` [1/m] the
    centreline curvature at the car's s. With ``smooth=False`` the throttle/brake split and the power minimum are
    exact (``ca.if_else``, ``ca.fmin``).
    """
    ey, epsi, vx, vy, r, delta, tau = (x[i] for i in range(7))
    m, mu, h, wheelbase = MASS_NOM, MU_NOM, HCG_NOM, LF + LR
    alpha_f = delta - ca.atan2(vy + LF * r, vx)
    alpha_r = -ca.atan2(vy - LR * r, vx)
    fz_f = m / wheelbase * (G * LR - ax_lt * h)
    fz_r = m / wheelbase * (G * LF + ax_lt * h)
    available = m * A_MAX, PLANT.wheel_power / ca.fmax(vx, POWER_SPEED_FLOOR_MPS)
    if smooth:
        throttle, brake = _max0(tau, SMOOTHING["tau"]), _max0(-tau, SMOOTHING["tau"])
        traction = throttle * _min(*available, SMOOTHING["force"])
    else:
        throttle, brake = ca.fmax(tau, 0.0), ca.fmax(-tau, 0.0)
        traction = throttle * ca.fmin(*available)
    fx_f = -brake * m * D_MAX * BETA_BRAKE
    fx_r = traction - brake * m * D_MAX * (1.0 - BETA_BRAKE)
    lambda_f, lambda_r = compute_lateral_scales(b)
    forces = []
    axles = (("front", alpha_f, fx_f, fz_f, lambda_f), ("rear", alpha_r, fx_r, fz_r, lambda_r))
    for axle, alpha, fx, fz, scale in axles:
        cb, cc, ce = pacejka_coefficients(axle)
        ratio = fx / (mu * fz)
        capacity = scale * mu * fz * ca.sqrt(ca.fmax(1.0 - ratio * ratio, 1e-6))  # finite while the solver iterates
        ba = cb * alpha
        forces.append(capacity * ca.sin(cc * ca.atan(ba - ce * (ba - ca.atan(ba)))))
    fy_f, fy_r = forces
    drag = 0.5 * AIR_DENSITY * PLANT.drag_area * vx * vx
    f_body_x = fx_r + fx_f * ca.cos(delta) - fy_f * ca.sin(delta) - drag
    f_body_y = fy_r + fx_f * ca.sin(delta) + fy_f * ca.cos(delta)
    s_dot = (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1.0 - kappa * ey)
    rates = [
        vx * ca.sin(epsi) + vy * ca.cos(epsi),
        r - kappa * s_dot,
        f_body_x / m + vy * r,
        f_body_y / m - vx * r,
        (LF * (f_body_y - fy_r) - LR * fy_r) / IZ,
        (delta_cmd - delta) / TAU_D_NOM,
        tau_rate,
    ]
    return {"rates": rates, "s_dot": s_dot, "f_body_x": f_body_x, "fx_f": fx_f, "fx_r": fx_r, "fy_f": fy_f,
            "fy_r": fy_r, "fz_f": fz_f, "fz_r": fz_r, "alpha_f": alpha_f, "alpha_r": alpha_r}


@dataclass
class MinTimeLap:
    """Solution: positions s [m], states (len(s), 7), controls (len(s), 2), lap time [s] and solver status."""

    s: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    lap_time: float
    b: float
    status: str

    def state(self, name: str) -> NDArray[np.float64]:
        return self.states[:, STATE_NAMES.index(name)]


def solve_min_time_lap(
    track: RealTrack, b: float = 0.0, ds: float = 2.0, max_iter: int = 3000, print_level: int = 0,
    start_s: float | None = None, start_vx: float = 3.0, edge_margin: float = 0.0,
) -> MinTimeLap:
    """Solve the minimum-time lap with steps of about ``ds`` [m], with the edges ``edge_margin`` [m] inside the
    track's own.

    Without ``start_s`` the lap is flying (periodic). With it, the lap starts at ``start_s`` on the centreline,
    aligned with it, at ``start_vx`` with no yaw, steering or longitudinal command, as the protocol laps of
    ``evaluation/laps.py``, and ends after one lap length with a free state.
    """
    flying = start_s is None
    n = int(round(track.track_length / ds))
    step = track.track_length / n
    s = (np.arange(n) if flying else np.arange(n + 1)) * step + (0.0 if flying else start_s)
    nodes_count = len(s)
    kappa = track.kappa_many(s % track.track_length)
    left, right = corridor(*track.boundaries_many(s % track.track_length), edge_margin)
    setup = sample_setup(np.random.default_rng(0), (b, b))
    peak = {"f": compute_saturation_threshold(setup, "front"), "r": compute_saturation_threshold(setup, "rear")}

    opti = ca.Opti()
    x = opti.variable(6, nodes_count)  # ey, e_psi, vx, vy, r, tau
    u = opti.variable(2, nodes_count)  # delta (the actuator lag is left out), tau rate
    ax_lt = opti.variable(1, nodes_count)

    def node(k: int) -> dict[str, Any]:
        full = ca.vertcat(x[0:5, k], u[0, k], x[5, k])
        return vehicle_rates(full, u[0, k], u[1, k], ax_lt[k], float(kappa[k]), b)

    nodes = [node(k) for k in range(nodes_count)]
    time = 0
    smoothness = 0
    for k in range(nodes_count):
        for axle in ("f", "r"):
            fz = nodes[k][f"fz_{axle}"]
            opti.subject_to(fz / (MASS_NOM * G) >= 0.01)
            opti.subject_to((nodes[k][f"fx_{axle}"] / (MU_NOM * fz)) ** 2 <= FRICTION_MARGIN ** 2)
            opti.subject_to((nodes[k][f"alpha_{axle}"] / peak[axle]) ** 2 <= 1.0)
        opti.subject_to(ax_lt[k] == nodes[k]["f_body_x"] / MASS_NOM)
        if not flying and k == nodes_count - 1:
            break
        j = (k + 1) % nodes_count
        rate_k, rate_j = nodes[k]["s_dot"], nodes[j]["s_dot"]
        for i, r in enumerate((0, 1, 2, 3, 4, 6)):
            opti.subject_to(x[i, j] - x[i, k] == 0.5 * step * (nodes[k]["rates"][r] / rate_k
                                                               + nodes[j]["rates"][r] / rate_j))
        time += 0.5 * step * (1.0 / rate_k + 1.0 / rate_j)
        smoothness += (u[0, j] - u[0, k]) ** 2
    opti.subject_to(opti.bounded(-right, x[0, :].T, left))
    opti.subject_to(opti.bounded(-DELTA_MAX, u[0, :], DELTA_MAX))
    opti.subject_to(opti.bounded(-TAU_RATE_MAX, u[1, :], TAU_RATE_MAX))
    opti.subject_to(opti.bounded(-1.0, x[5, :], 1.0))
    opti.subject_to(x[2, :] >= 2.0)
    opti.subject_to(opti.bounded(-1.0, x[1, :], 1.0))  # heading error within 57 degrees: s_dot stays positive
    opti.subject_to(opti.bounded(-10.0, x[3, :], 10.0))
    if not flying:
        opti.subject_to(x[:, 0] == ca.vertcat(0.0, 0.0, start_vx, 0.0, 0.0, 0.0))
    opti.minimize(time + SMOOTHNESS_WEIGHT * smoothness)

    _, v_pm = speed_profile(track, None, lateral_ratio=1.0)
    grid = np.linspace(0.0, track.track_length, len(v_pm), endpoint=False)
    v0 = START_FRACTION * np.interp(s % track.track_length, grid, v_pm, period=track.track_length)
    if not flying:
        v0 = np.minimum(v0, np.sqrt(start_vx ** 2 + 2.0 * 5.0 * (s - start_s)))  # accelerating from the start
    opti.set_initial(x[2, :], v0)
    opti.set_initial(x[4, :], v0 * kappa)
    opti.set_initial(u[0, :], (LF + LR) * kappa)
    opti.solver("ipopt", {"print_time": False}, {"max_iter": max_iter, "print_level": print_level, "tol": 1e-6})
    try:
        solution = opti.solve()
        status = "solved"
    except RuntimeError:
        solution = opti.debug
        status = str(opti.stats().get("return_status", "failed"))
    xs, us = np.asarray(solution.value(x)), np.asarray(solution.value(u))
    states = np.vstack([xs[0:5], us[0], xs[5]]).T  # in the order of STATE_NAMES
    return MinTimeLap(s=s, states=states, controls=us.T, lap_time=float(solution.value(time)), b=b, status=status)


def slip_ratios(lap: MinTimeLap) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return |alpha| / alpha_sat of the front and rear axle along the lap (peak slip angles of ``model/tires.py``)."""
    setup = sample_setup(np.random.default_rng(0), (lap.b, lap.b))
    vx, vy, r, delta = (lap.state(name) for name in ("vx", "vy", "r", "delta"))
    alpha_f = delta - np.arctan2(vy + LF * r, vx)
    alpha_r = -np.arctan2(vy - LR * r, vx)
    return (np.abs(alpha_f) / compute_saturation_threshold(setup, "front"),
            np.abs(alpha_r) / compute_saturation_threshold(setup, "rear"))


