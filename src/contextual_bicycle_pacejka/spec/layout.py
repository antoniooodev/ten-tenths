"""Layouts of the state, setup, observation and action vectors, and setup validation.

Observation layout, float32 at the Gymnasium boundary; every entry is divided by a scale of
the speed stage (``env/stages.py``):

    index            content                                              scale
    0-1              vx, vy                                               top speed of the stage
    2                yaw rate r                                           yaw-rate scale of the stage
    3                heading error e_psi                                  pi / 2
    4                lateral offset e_y                                   widest half-width of the track
    5-6              steering angle delta, previous reference delta_ref   DELTA_MAX
    7-8              longitudinal command tau, its previous rate          1, TAU_RATE_MAX
    9                friction mu                                          MU_NOM
    10 .. 9+N        mean curvature over the N preview intervals          curvature of the tightest corner
    10+N .. 9+2N     left half-width at the N preview points              widest half-width
    10+2N .. 9+3N    right half-width at the N preview points             widest half-width
    10+3N            balance index b (Oracle only)                        B_OBS_SCALE

Action: [delta_ref, tau_rate] in [-1, 1]^2, scaled by DELTA_MAX and TAU_RATE_MAX.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.spec.constants import (
    ACTION_DIM,
    B_MAX,
    B_MIN,
    N_PREVIEW,
    SETUP_DIM,
    STATE_DIM,
    TAU_MIN,
    V_MIN,
)


class StateIndex(NamedTuple):
    """Indices of the dynamic state [s, ey, epsi, vx, vy, yaw_rate, delta]."""

    S: int = 0
    EY: int = 1
    EPSI: int = 2
    VX: int = 3
    VY: int = 4
    YAW_RATE: int = 5
    DELTA: int = 6


class SetupIndex(NamedTuple):
    """Indices of the setup vector [m, mu, hcg, tau_d, b]."""

    M: int = 0
    MU: int = 1
    HCG: int = 2
    TAU_D: int = 3
    B: int = 4


class ObsIndex(NamedTuple):
    """Indices and slices of the observation (see the module docstring)."""

    VX: int = 0
    VY: int = 1
    YAW_RATE: int = 2
    EPSI: int = 3
    EY: int = 4
    DELTA: int = 5
    DELTA_REF: int = 6
    TAU: int = 7
    TAU_RATE: int = 8
    MU: int = 9
    KAPPA_START: int = 10
    KAPPA_END: int = 10 + N_PREVIEW
    W_LEFT_START: int = 10 + N_PREVIEW
    W_LEFT_END: int = 10 + 2 * N_PREVIEW
    W_RIGHT_START: int = 10 + 2 * N_PREVIEW
    W_RIGHT_END: int = 10 + 3 * N_PREVIEW
    B: int = 10 + 3 * N_PREVIEW  # Oracle only


class ActionIndex(NamedTuple):
    """Indices of the action [delta_ref, tau_rate]."""

    DELTA_REF: int = 0
    TAU_RATE: int = 1


STATE = StateIndex()
SETUP = SetupIndex()
OBS = ObsIndex()
ACTION = ActionIndex()

ACTION_LOW: NDArray[np.float64] = -np.ones(ACTION_DIM, dtype=np.float64)
ACTION_HIGH: NDArray[np.float64] = np.ones(ACTION_DIM, dtype=np.float64)


def validate_setup(xi: NDArray[np.float64]) -> None:
    """Raise ``ValueError`` if a setup vector is malformed or physically inadmissible."""
    if xi.shape != (SETUP_DIM,):
        raise ValueError(f"setup shape {xi.shape} != ({SETUP_DIM},)")
    for name, index in (("m", SETUP.M), ("mu", SETUP.MU)):
        if not xi[index] > 0.0:
            raise ValueError(f"{name}={xi[index]:.6f} is not positive")
    if xi[SETUP.HCG] < 0.0:
        raise ValueError(f"hcg={xi[SETUP.HCG]:.6f} is negative")
    if xi[SETUP.TAU_D] < TAU_MIN:
        raise ValueError(f"tau_d={xi[SETUP.TAU_D]:.6f} < TAU_MIN={TAU_MIN}")
    if not B_MIN < xi[SETUP.B] < B_MAX:
        raise ValueError(f"b={xi[SETUP.B]:.6f} outside ({B_MIN}, {B_MAX:.6f})")


def validate_state(z: NDArray[np.float64], kappa_at_s: float) -> None:
    """Raise ``ValueError`` if a state violates vx >= V_MIN or Frenet regularity (1 - kappa ey > 0)."""
    if z.shape != (STATE_DIM,):
        raise ValueError(f"state shape {z.shape} != ({STATE_DIM},)")
    if z[STATE.VX] < V_MIN:
        raise ValueError(f"vx={z[STATE.VX]:.4f} < V_MIN={V_MIN}")
    if 1.0 - kappa_at_s * z[STATE.EY] <= 0.0:
        raise ValueError(f"Frenet regularity violated: kappa={kappa_at_s:.6f}, ey={z[STATE.EY]:.4f}")
