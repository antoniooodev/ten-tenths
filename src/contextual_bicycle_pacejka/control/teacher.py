"""MATv0: the scripted, model-aware teacher that generates every demonstration.

The teacher reads only the 116-entry observation (it is stateless) and returns [delta_cmd, tau].
Three switchable modules extend a baseline Stanley / speed-P controller:

    baseline (all switches off)
        v_target = k * min(v_ref preview at 5..55 m),  tau = clip(Kp e_v, 0, 1) or clip(Kb e_v, -1, 0)
        delta    = -epsi - atan2(3 ey, max(vx, 0.5) + 0.5)
    T  target + feed-forward
        v_target = k v_ref(5 m), a_ff = (v1^2 - v_target^2) / (2 * 5 m) with v1 = k v_ref(10 m),
        tau = clip(a_ff / A_MAX (or / D_MAX) + K e_v, -1, 1)
    L  friction-budget limiter
        per axle, Fx_max = ETA sqrt((mu Fz)^2 - Fy_req^2) bounds tau; Fz from the load-transfer equation,
        Fy_req = max(measured |Fy|, static share of m |vx r|); (m, mu, hcg, tau_d) decoded from xi
    Y  curvature feed-forward and yaw-rate damping
        delta += L kappa_ff + (alpha_f,req - alpha_r,req) - K_R (r - kappa_0 vx),
        kappa_ff at d = vx tau_d (actuator-lag lead), kappa_0 at d = 0 (linear in the 5 m / 10 m previews)

The certified configuration is k = 1.25 with T, L and Y on (``data/teacher/matv0.json``).
With all switches off the teacher is the earlier production follower (used by the speed-profile probe).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.tires import pacejka_coefficients
from contextual_bicycle_pacejka.spec.constants import (
    A_MAX,
    BETA_BRAKE,
    D_MAX,
    DELTA_MAX,
    G,
    KAPPA_NORM_RADPM,
    LF,
    LOOKAHEAD_M,
    LR,
)
from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, ACTION_LOW, OBS, denormalise_setup

N_SPEED_PREVIEW: int = 5  # baseline target: minimum of the first five v_ref previews (5-55 m)
KP_ACCEL: float = 0.15  # [1/(m/s)] throttle gain on the speed error
KP_BRAKE: float = 0.25  # [1/(m/s)] brake gain on the speed error
K_STANLEY: float = 3.0  # [1/s] Stanley cross-track gain
K_STANLEY_SOFT_MPS: float = 0.5  # [m/s] Stanley speed softening
ETA_BUDGET: float = 0.85  # [-] usable fraction of the remaining friction budget (L)
K_R: float = 0.05  # [s] yaw-rate damping gain (Y)
ALPHA_FF_CAP_RAD: float = 0.10  # [rad] cap on each feed-forward slip term (Y)
D_FLOOR_N: float = 1.0  # [N] floor for loads and capacities used as divisors
WHEELBASE_M: float = LF + LR  # [m]

_B_F, _C_F, _ = pacejka_coefficients("front")
_B_R, _C_R, _ = pacejka_coefficients("rear")
CA_NORM_F: float = _B_F * _C_F  # [1/rad] small-slip cornering stiffness per unit capacity, front
CA_NORM_R: float = _B_R * _C_R  # [1/rad] same, rear


def _kappa_at(kappa_5: float, kappa_10: float, distance: float) -> float:
    """Curvature [1/m] at ``distance`` by linear inter/extrapolation of the 5 m and 10 m previews."""
    d0, d1 = float(LOOKAHEAD_M[0]), float(LOOKAHEAD_M[1])
    return kappa_5 + (kappa_10 - kappa_5) * (distance - d0) / (d1 - d0)


@dataclass(frozen=True)
class MATv0:
    """Model-aware scripted teacher (observation-only, stateless).

    Parameters
    ----------
    speed_safety : float
        Multiplier k on the v_ref preview.
    max_speed : float
        Upper bound of the speed target [m/s].
    T, L, Y : bool
        Target + feed-forward, friction-budget limiter, curvature feed-forward + yaw damping.
    """

    speed_safety: float = 1.25
    max_speed: float = 100.0
    T: bool = True
    L: bool = True
    Y: bool = True

    @classmethod
    def from_config(cls, path: str | Path) -> MATv0:
        """Build the teacher from a configuration file such as ``data/teacher/matv0.json``."""
        cfg = json.loads(Path(path).read_text())
        return cls(speed_safety=float(cfg["speed_safety"]), max_speed=float(cfg["max_speed_mps"]),
                   T=bool(cfg["target_feedforward"]), L=bool(cfg["friction_limiter"]),
                   Y=bool(cfg["yaw_feedforward_damping"]))

    def act(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the action [delta_cmd, tau] for an observation."""
        return self.act_with_diagnostics(obs)[0]

    def act_with_diagnostics(self, obs: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, dict[str, float]]:
        """Return (action, speed target [m/s], diagnostics of the active modules)."""
        o = np.asarray(obs, dtype=np.float64)
        vx = float(o[OBS.VX])
        k = float(self.speed_safety)
        vref = o[OBS.VREF_START:OBS.VREF_END]
        diag = {"a_ff": 0.0, "tau_unlimited": 0.0, "tau_hi": 1.0, "tau_lo": -1.0, "limiter_active": 0.0,
                "delta_ff": 0.0, "yaw_term": 0.0}

        # Longitudinal command.
        if self.T:
            v_target = min(float(vref[0]) * k, float(self.max_speed))
            v_next = min(float(vref[1]) * k, float(self.max_speed))
            a_ff = (v_next * v_next - v_target * v_target) / (2.0 * (float(LOOKAHEAD_M[1]) - float(LOOKAHEAD_M[0])))
            tau_ff = a_ff / A_MAX if a_ff >= 0.0 else a_ff / D_MAX
            speed_error = v_target - vx
            gain = KP_ACCEL if speed_error > 0.0 else KP_BRAKE
            tau = float(np.clip(tau_ff + gain * speed_error, -1.0, 1.0))
            diag["a_ff"] = a_ff
        else:
            v_target = min(float(np.min(vref[:N_SPEED_PREVIEW])) * k, float(self.max_speed))
            speed_error = v_target - vx
            if speed_error > 0.0:
                tau = float(np.clip(KP_ACCEL * speed_error, 0.0, 1.0))
            else:
                tau = float(np.clip(KP_BRAKE * speed_error, -1.0, 0.0))
        tau = float(np.clip(tau, -1.0, 1.0))
        diag["tau_unlimited"] = tau

        if self.L or self.Y:
            xi = denormalise_setup(o[OBS.XI_START:OBS.XI_END])
            m, mu, hcg, tau_d = float(xi[0]), float(xi[3]), float(xi[4]), float(xi[5])
            fz_f, fz_r = _axle_loads(o, m, mu, hcg)
            ay_kinematic = abs(vx * float(o[OBS.YAW_RATE]))
            fy_f_req = max(abs(float(o[OBS.FY_F])) * mu * fz_f, m * LR / WHEELBASE_M * ay_kinematic)
            fy_r_req = max(abs(float(o[OBS.FY_R])) * mu * fz_r, m * LF / WHEELBASE_M * ay_kinematic)

        if self.L:
            d_f, d_r = mu * fz_f, mu * fz_r
            fx_r_max = ETA_BUDGET * math.sqrt(max(d_r * d_r - fy_r_req * fy_r_req, 0.0))
            fx_f_max = ETA_BUDGET * math.sqrt(max(d_f * d_f - fy_f_req * fy_f_req, 0.0))
            tau_hi = min(1.0, fx_r_max / (m * A_MAX))
            tau_lo = -min(1.0, fx_r_max / (m * D_MAX * (1.0 - BETA_BRAKE)), fx_f_max / (m * D_MAX * BETA_BRAKE))
            limited = float(np.clip(tau, tau_lo, tau_hi))
            diag.update(tau_hi=tau_hi, tau_lo=tau_lo, limiter_active=float(limited != tau))
            tau = limited

        # Lateral command: Stanley, plus curvature feed-forward and yaw damping.
        epsi = float(o[OBS.EPSI])
        ey = float(o[OBS.EY])
        delta = float(-epsi - np.arctan2(K_STANLEY * ey, max(vx, K_STANLEY_SOFT_MPS) + K_STANLEY_SOFT_MPS))
        if self.Y:
            kappa = o[OBS.KAPPA_START:OBS.KAPPA_END] * KAPPA_NORM_RADPM
            kappa_ff = _kappa_at(float(kappa[0]), float(kappa[1]), vx * tau_d)
            kappa_0 = _kappa_at(float(kappa[0]), float(kappa[1]), 0.0)
            if tau >= 0.0:
                fx_f_planned, fx_r_planned = 0.0, tau * m * A_MAX
            else:
                fx_f_planned = tau * m * D_MAX * BETA_BRAKE
                fx_r_planned = tau * m * D_MAX * (1.0 - BETA_BRAKE)
            d_f, d_r = mu * fz_f, mu * fz_r
            cap_f = math.sqrt(max(d_f * d_f - min(fx_f_planned * fx_f_planned, d_f * d_f), 0.0))
            cap_r = math.sqrt(max(d_r * d_r - min(fx_r_planned * fx_r_planned, d_r * d_r), 0.0))
            ay_ff = kappa_ff * vx * vx
            alpha_f = (m * LR / WHEELBASE_M * ay_ff) / (CA_NORM_F * max(cap_f, D_FLOOR_N))
            alpha_r = (m * LF / WHEELBASE_M * ay_ff) / (CA_NORM_R * max(cap_r, D_FLOOR_N))
            alpha_f = float(np.clip(alpha_f, -ALPHA_FF_CAP_RAD, ALPHA_FF_CAP_RAD))
            alpha_r = float(np.clip(alpha_r, -ALPHA_FF_CAP_RAD, ALPHA_FF_CAP_RAD))
            delta_ff = WHEELBASE_M * kappa_ff + (alpha_f - alpha_r)
            yaw_term = -K_R * (float(o[OBS.YAW_RATE]) - kappa_0 * vx)
            delta = delta + delta_ff + yaw_term
            diag.update(delta_ff=delta_ff, yaw_term=yaw_term)
        delta = float(np.clip(delta, -DELTA_MAX, DELTA_MAX))
        action = np.clip(np.array([delta, tau], dtype=np.float64), ACTION_LOW, ACTION_HIGH)
        return action.astype(np.float32), v_target, diag


def _axle_loads(o: NDArray[np.float64], m: float, mu: float, hcg: float) -> tuple[float, float]:
    """Axle loads (Fz_f, Fz_r) [N] for the coming step, from the observed ax and lateral force.

    The plant uses ax_LT = ax(t-1) + Fy_f(t-1) sin(delta) / m; the observation holds ax(t-1) and Fy_f / (mu Fz_f).
    """
    ax = float(o[OBS.AX])
    delta = float(o[OBS.DELTA])
    fz_f_static = m / WHEELBASE_M * (G * LR - ax * hcg)
    fy_f_prev = float(o[OBS.FY_F]) * mu * max(fz_f_static, D_FLOOR_N)
    ax_lt = ax + fy_f_prev * math.sin(delta) / m
    fz_f = m / WHEELBASE_M * (G * LR - ax_lt * hcg)
    fz_r = m / WHEELBASE_M * (G * LF + ax_lt * hcg)
    return max(fz_f, D_FLOOR_N), max(fz_r, D_FLOOR_N)
