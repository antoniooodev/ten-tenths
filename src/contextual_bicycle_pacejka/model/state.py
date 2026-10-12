"""Plant data containers: dynamic state, one-step memory and episode setup.

All plant quantities are float64; the float32 cast happens only in the observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.spec.constants import SETUP_DIM, STATE_DIM


class AdmissibilityError(ValueError):
    """Raised when the plant reaches an inadmissible state (Fz <= 0 or 1 - kappa ey <= 0)."""


@dataclass
class VehicleState:
    """Dynamic state in Frenet coordinates.

    s [m] arc length, ey [m] lateral error (positive = left), epsi [rad] heading error,
    vx, vy [m/s] body-frame velocities, yaw_rate [rad/s], delta [rad] actual steering angle.
    """

    s: float = 0.0
    ey: float = 0.0
    epsi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    delta: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        """Return [s, ey, epsi, vx, vy, yaw_rate, delta]."""
        return np.array([self.s, self.ey, self.epsi, self.vx, self.vy, self.yaw_rate, self.delta], dtype=np.float64)

    @classmethod
    def from_array(cls, z: NDArray[np.float64]) -> VehicleState:
        """Build a state from [s, ey, epsi, vx, vy, yaw_rate, delta]."""
        if z.shape != (STATE_DIM,):
            raise ValueError(f"state shape {z.shape} != ({STATE_DIM},)")
        return cls(*(float(v) for v in z))


@dataclass
class VehicleMemory:
    """Quantities of the previous step, used only by the load-transfer model.

    Fx_r_prev, Fx_f_prev [N] longitudinal axle forces, delta_prev [rad], vy_prev [m/s], yawrate_prev [rad/s].
    """

    Fx_r_prev: float = 0.0
    Fx_f_prev: float = 0.0
    delta_prev: float = 0.0
    vy_prev: float = 0.0
    yawrate_prev: float = 0.0

    @classmethod
    def zeros(cls) -> VehicleMemory:
        """Memory at reset (all zeros)."""
        return cls()


@dataclass
class SetupVector:
    """Episode setup in physical units.

    m [kg], Cz_f, Cz_r [N/rad], mu [-], hcg [m], tau_d [s], Cpac [-].
    Cz_f, Cz_r and Cpac are sampled and observed but do not enter the plant equations.
    """

    m: float = 0.0
    Cz_f: float = 0.0
    Cz_r: float = 0.0
    mu: float = 0.0
    hcg: float = 0.0
    tau_d: float = 0.0
    Cpac: float = 0.0

    def to_array(self) -> NDArray[np.float64]:
        """Return [m, Cz_f, Cz_r, mu, hcg, tau_d, Cpac]."""
        return np.array([self.m, self.Cz_f, self.Cz_r, self.mu, self.hcg, self.tau_d, self.Cpac], dtype=np.float64)

    @classmethod
    def from_array(cls, xi: NDArray[np.float64]) -> SetupVector:
        """Build a setup from [m, Cz_f, Cz_r, mu, hcg, tau_d, Cpac]."""
        if xi.shape != (SETUP_DIM,):
            raise ValueError(f"setup shape {xi.shape} != ({SETUP_DIM},)")
        return cls(*(float(v) for v in xi))
