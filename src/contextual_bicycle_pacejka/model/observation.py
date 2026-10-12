"""Assembly of the 116-entry observation (layout in ``spec/layout.py``).

The observation is built from the plant state, the plant diagnostics of the last step, the track and the
episode setup. It contains the true normalised setup xi (privileged information).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.state import SetupVector, VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    ALPHA_NORM_RAD,
    AX_PREVIEW_NORM_MPS2,
    G,
    KAPPA_NORM_RADPM,
    LOOKAHEAD_M,
    OBS_DIM,
    TRAJ_LOOKAHEAD_MAX_M,
    TRAJ_LOOKAHEAD_MIN_M,
    TRAJ_N,
    TRAJ_NORMALISATION_M,
    VREF_REFERENCE_MU,
)
from contextual_bicycle_pacejka.spec.layout import OBS, normalise_setup

TRAJ_DISTANCES_M: NDArray[np.float64] = np.linspace(TRAJ_LOOKAHEAD_MIN_M, TRAJ_LOOKAHEAD_MAX_M, TRAJ_N,
                                                     dtype=np.float64)


def vehicle_world_position(state: VehicleState, track: RealTrack) -> tuple[float, float]:
    """World position (x, y) [m] of the vehicle: offset ey along the left normal of the path at s."""
    x_ref, y_ref = track.centerline_point(state.s)
    psi_ref = track.psi_ref(state.s)
    return float(x_ref - state.ey * math.sin(psi_ref)), float(y_ref + state.ey * math.cos(psi_ref))


def build_observation(
    state: VehicleState, setup: SetupVector, diag: dict[str, Any], track: RealTrack,
) -> NDArray[np.float64]:
    """Return the observation (float64, shape (116,)) after a plant step with diagnostics ``diag``."""
    obs = np.empty(OBS_DIM, dtype=np.float64)
    obs[OBS.VX] = state.vx
    obs[OBS.VY] = state.vy
    obs[OBS.YAW_RATE] = state.yaw_rate
    obs[OBS.DELTA] = state.delta
    obs[OBS.EY] = state.ey
    obs[OBS.EPSI] = state.epsi

    # Speed preview, rescaled from the profile friction to the episode friction.
    mu_scale = math.sqrt(setup.mu / VREF_REFERENCE_MU)
    obs[OBS.VREF_START:OBS.VREF_END] = np.array([track.v_ref(state.s + d) * mu_scale for d in LOOKAHEAD_M],
                                                dtype=np.float64)

    # Racing-line points in the vehicle frame.
    psi_vehicle = track.psi_ref(state.s) + state.epsi
    x_vehicle, y_vehicle = vehicle_world_position(state, track)
    cos_psi = math.cos(psi_vehicle)
    sin_psi = math.sin(psi_vehicle)
    traj = np.zeros(2 * TRAJ_N, dtype=np.float64)
    for i, distance in enumerate(TRAJ_DISTANCES_M):
        x_world, y_world = track.centerline_point(state.s + float(distance))
        dx = x_world - x_vehicle
        dy = y_world - y_vehicle
        traj[2 * i] = (cos_psi * dx + sin_psi * dy) / TRAJ_NORMALISATION_M
        traj[2 * i + 1] = (-sin_psi * dx + cos_psi * dy) / TRAJ_NORMALISATION_M
    obs[OBS.TRAJ_START:OBS.TRAJ_END] = traj

    obs[OBS.AX] = diag["ax"]
    obs[OBS.XI_START:OBS.XI_END] = normalise_setup(setup.to_array())

    mu = float(setup.mu)
    fy_f, fy_r = float(diag["Fy_f"]), float(diag["Fy_r"])
    fz_f, fz_r = max(float(diag["Fz_f"]), 1.0), max(float(diag["Fz_r"]), 1.0)
    force_scale = mu * float(setup.m) * G
    obs[OBS.RHO_LAT] = np.clip(abs(fy_f + fy_r) / force_scale, 0.0, 1.5)
    obs[OBS.RHO_LON] = np.clip(abs(float(diag["Fx_f"]) + float(diag["Fx_r"])) / force_scale, 0.0, 1.5)

    obs[OBS.KAPPA_START:OBS.KAPPA_END] = np.asarray([track.kappa(state.s + d) / KAPPA_NORM_RADPM for d in LOOKAHEAD_M],
                                                    dtype=np.float64)
    obs[OBS.AX_PREVIEW_START:OBS.AX_PREVIEW_END] = np.asarray(
        [track.ax_ref(state.s + d) / AX_PREVIEW_NORM_MPS2 for d in LOOKAHEAD_M], dtype=np.float64)

    obs[OBS.ALPHA_F] = float(diag["alpha_f"]) / ALPHA_NORM_RAD
    obs[OBS.ALPHA_R] = float(diag["alpha_r"]) / ALPHA_NORM_RAD
    obs[OBS.FY_F] = float(np.clip(fy_f / (mu * fz_f), -1.0, 1.0))
    obs[OBS.FY_R] = float(np.clip(fy_r / (mu * fz_r), -1.0, 1.0))
    return obs
