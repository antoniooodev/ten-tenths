"""Minimum-curvature racing line with a point-mass speed profile (Spa, Catalunya, Budapest).

Path. The racing line is the centreline displaced along its left normal n_i by offsets alpha_i,
p_i = c_i + alpha_i n_i, with -(w_right_i - margin) <= alpha_i <= w_left_i - margin (margin 1 m). The squared
curvature, linearised around the centreline with periodic cubic-spline derivative operators (Heilmeier et al.,
2020), is minimised as a bound-constrained linear least-squares problem.

Speed. v_corner = sqrt(mu g / |kappa|) clipped to [1, 90] m/s, then 4 forward (traction) and backward (braking)
passes inside the friction circle a_lat^2 + a_lon^2 <= (mu g)^2; the result is scaled by 0.85 and
ax_i = (v_{i+1}^2 - v_i^2) / (2 ds_i).

Output (``minimum_curvature_line.npz``): racing-line geometry and speed profile with the distances from the line to
the track boundaries, w_right + alpha and w_left - alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from contextual_bicycle_pacejka.spec.constants import G, VREF_REFERENCE_MU

BOUNDARY_MARGIN_M = 1.0
SPEED_UTILISATION = 0.85
V_CLIP_MPS = (1.0, 90.0)
PROFILE_PASSES = 4
LSQ_TOL = 1.0e-10
LSQ_MAX_ITER = 500
EPS = 1.0e-9


def _spline_derivative_operators(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Matrices mapping n periodic samples to the first and second derivatives of their periodic cubic spline."""
    knots = np.arange(n + 1, dtype=np.float64)
    basis = np.vstack((np.eye(n), np.eye(n)[0]))
    spline = CubicSpline(knots, basis, axis=0, bc_type="periodic")
    return np.asarray(spline(knots[:-1], 1)), np.asarray(spline(knots[:-1], 2))


def minimum_curvature_offsets(centre: NDArray[np.float64], lower: NDArray[np.float64],
                              upper: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Offsets alpha in [lower, upper] [m] minimising the linearised curvature; returns (alpha, unit left normals)."""
    d1, d2 = _spline_derivative_operators(len(centre))
    x_d, y_d = d1 @ centre[:, 0], d1 @ centre[:, 1]
    psi = np.arctan2(y_d, x_d)
    nx, ny = -np.sin(psi), np.cos(psi)
    denominator = np.maximum((x_d * x_d + y_d * y_d) ** 1.5, EPS)
    qx, qy = x_d / denominator, y_d / denominator
    kappa_ref = qx * (d2 @ centre[:, 1]) - qy * (d2 @ centre[:, 0])
    jacobian = qx[:, None] * (d2 * ny[None, :]) - qy[:, None] * (d2 * nx[None, :])
    result = lsq_linear(jacobian, -kappa_ref, bounds=(lower, upper), tol=LSQ_TOL, lsmr_tol="auto",
                        max_iter=LSQ_MAX_ITER)
    if not result.success:
        raise RuntimeError(f"minimum-curvature optimisation failed: {result.message}")
    return np.clip(result.x, lower, upper), np.column_stack((nx, ny))


def path_geometry(xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Arc length [m], unwrapped heading [rad] and curvature [1/m] of a closed polyline (periodic splines)."""
    segment = np.hypot(np.roll(xy[:, 0], -1) - xy[:, 0], np.roll(xy[:, 1], -1) - xy[:, 1])
    s = np.concatenate(([0.0], np.cumsum(segment)[:-1]))
    d1, d2 = _spline_derivative_operators(len(xy))
    x_d, y_d, x_dd, y_dd = d1 @ xy[:, 0], d1 @ xy[:, 1], d2 @ xy[:, 0], d2 @ xy[:, 1]
    kappa = (x_d * y_dd - y_d * x_dd) / np.maximum((x_d * x_d + y_d * y_d) ** 1.5, EPS)
    return s, np.unwrap(np.arctan2(y_d, x_d)), kappa


def point_mass_speed_profile(kappa: NDArray[np.float64], s: NDArray[np.float64], mu: float,
                             ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Friction-circle speed profile [m/s] scaled by 0.85 and its longitudinal acceleration [m/s^2]."""
    n = len(kappa)
    ds = np.empty(n)
    ds[:-1] = np.diff(s)
    ds[-1] = ds[-2]
    ds = np.maximum(ds, EPS)
    a_max = mu * G
    v_min, v_max = V_CLIP_MPS
    with np.errstate(divide="ignore"):
        v = np.where(np.abs(kappa) > EPS, np.sqrt(a_max / np.maximum(np.abs(kappa), 1e-12)), v_max)
    v = np.clip(v, v_min, v_max)

    def a_lon(i: int) -> float:
        a_lat = v[i] * v[i] * abs(kappa[i])
        return float(np.sqrt(max(a_max * a_max - a_lat * a_lat, 0.0)))

    for _ in range(PROFILE_PASSES):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] * v[i] + 2.0 * a_lon(i) * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[i] * v[i] + 2.0 * a_lon(i) * ds[j]))
    v_ref = np.clip(np.clip(v, v_min, v_max) * SPEED_UTILISATION, v_min, v_max)
    v_next = np.roll(v_ref, -1)
    return v_ref, (v_next * v_next - v_ref * v_ref) / (2.0 * ds)


def build_racing_line(reference: str | Path, output: str | Path, mu: float = VREF_REFERENCE_MU,
                      margin_m: float = BOUNDARY_MARGIN_M) -> Path:
    """Write the racing line (``minimum_curvature_line.npz`` format) of the centreline reference ``reference``."""
    with np.load(reference) as data:
        centre = np.column_stack((data["x_m"], data["y_m"])).astype(np.float64)
        w_right = np.asarray(data["w_tr_right_m"], dtype=np.float64)
        w_left = np.asarray(data["w_tr_left_m"], dtype=np.float64)
    if min(w_right.min(), w_left.min()) <= margin_m:
        raise ValueError("the boundary margin leaves no feasible corridor")
    alpha, normals = minimum_curvature_offsets(centre, -(w_right - margin_m), w_left - margin_m)
    xy = centre + alpha[:, None] * normals
    s, psi, kappa = path_geometry(xy)
    v_ref, ax = point_mass_speed_profile(kappa, s, mu)
    closing = float(np.hypot(xy[0, 0] - xy[-1, 0], xy[0, 1] - xy[-1, 1]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x_m=xy[:, 0], y_m=xy[:, 1], s_m=s, psi_rad=psi, kappa_radpm=kappa,
             track_length=np.float64(s[-1] + closing), v_ref_mps=v_ref, ax_mps2=ax, mu_ref=np.asarray(mu),
             w_tr_right_m=w_right + alpha, w_tr_left_m=w_left - alpha)
    return output
