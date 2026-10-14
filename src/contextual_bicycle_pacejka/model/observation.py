"""Assembly of the observation of the racing policies (layout in ``spec/layout.py``).

The observation holds the observable vehicle state, the last actions, the friction and a geometric preview of the
centreline: mean curvature over N intervals ahead and the two half-widths at their ends. Every entry is divided by a
scale of the speed stage. Nominal and Randomised policies read it as is; the Oracle reads b appended at the end.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.state import SetupVector, VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    B_OBS_SCALE,
    DELTA_MAX,
    MU_NOM,
    OBS_DIM,
    SPIN_EPSI_RAD,
    TAU_RATE_MAX,
)
from contextual_bicycle_pacejka.spec.layout import OBS


class ObservationScales(NamedTuple):
    """Scales of a speed stage: speed [m/s], yaw rate [rad/s], lateral distance [m], curvature [1/m]."""

    speed: float
    yaw_rate: float
    lateral: float
    curvature: float


def vehicle_world_position(state: VehicleState, track: RealTrack) -> tuple[float, float]:
    """World position (x, y) [m] of the vehicle: offset ey along the left normal of the path at s."""
    x_ref, y_ref = track.centerline_point(state.s)
    psi_ref = track.psi_ref(state.s)
    return float(x_ref - state.ey * math.sin(psi_ref)), float(y_ref + state.ey * math.cos(psi_ref))


def build_observation(
    state: VehicleState,
    setup: SetupVector,
    delta_ref: float,
    tau: float,
    tau_rate: float,
    track: RealTrack,
    preview: NDArray[np.float64],
    scales: ObservationScales,
    oracle: bool,
) -> NDArray[np.float64]:
    """Return the observation (float64; OBS_DIM entries, one more for the Oracle).

    ``delta_ref`` [rad] and ``tau_rate`` [1/s] are the last action, ``tau`` [-] the longitudinal command in use,
    ``preview`` [m] the distances d_1..d_N ahead of the car.
    """
    obs = np.empty(OBS_DIM + int(oracle), dtype=np.float64)
    obs[OBS.VX] = state.vx / scales.speed
    obs[OBS.VY] = state.vy / scales.speed
    obs[OBS.YAW_RATE] = state.yaw_rate / scales.yaw_rate
    obs[OBS.EPSI] = state.epsi / SPIN_EPSI_RAD
    obs[OBS.EY] = state.ey / scales.lateral
    obs[OBS.DELTA] = state.delta / DELTA_MAX
    obs[OBS.DELTA_REF] = delta_ref / DELTA_MAX
    obs[OBS.TAU] = tau
    obs[OBS.TAU_RATE] = tau_rate / TAU_RATE_MAX
    obs[OBS.MU] = setup.mu / MU_NOM
    obs[OBS.KAPPA_START:OBS.KAPPA_END] = track.mean_curvature(state.s, preview) / scales.curvature
    left, right = track.boundaries_many(state.s + preview)
    obs[OBS.W_LEFT_START:OBS.W_LEFT_END] = left / scales.lateral
    obs[OBS.W_RIGHT_START:OBS.W_RIGHT_END] = right / scales.lateral
    if oracle:
        obs[OBS.B] = setup.b / B_OBS_SCALE
    return obs
