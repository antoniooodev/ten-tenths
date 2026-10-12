"""Layouts of the state, setup, observation and action vectors, and setup normalisation.

Observation layout (116 entries, float32 at the Gymnasium boundary):

    index    content                                   scaling
    0-2      vx, vy, yaw_rate                          m/s, m/s, rad/s
    3        delta (actual steering angle)             rad
    4-5      ey, epsi (Frenet errors)                  m, rad
    6-17     v_ref preview at LOOKAHEAD_M              m/s, scaled by sqrt(mu / 0.9)
    18-77    racing-line points (x, y) in body frame   m / 100
    78       ax of the last step                       m/s^2
    79-85    normalised setup xi                       [-1, 1]
    86-87    lateral / longitudinal force use          |F| / (mu m g), clipped to [0, 1.5]
    88-99    curvature preview                         (1/m) / 0.05
    100-111  reference-acceleration preview            (m/s^2) / 15
    112-113  slip angles alpha_f, alpha_r              rad / 0.40
    114-115  normalised lateral forces Fy_i / (mu Fz_i) clipped to [-1, 1]

This observation is privileged: it is read by the teacher, the safety filter and the critics. The student
observation of level S1 or S2 (``model/student.py``) has the layout

    [frame(t), frame(t - stride), ..., frame(t - (n_frames - 1) stride), preview]

with one frame per sampled step holding the dynamic channels of the level, and the preview of the current step:

    channel          content                                          scaling
    vx, yaw_rate     wheel speed, gyroscope                           m/s, rad/s
    delta            steering encoder (actual angle)                  rad
    ax_imu, ay_imu   accelerometer specific force F_body / m          m/s^2
    ey, epsi         map localisation (Frenet errors)                 m, rad
    prev_delta_cmd,  action executed at the previous step             rad, [-1, 1]
    prev_tau
    vy               lateral velocity                         (S1)    m/s
    alpha_f, alpha_r slip angles                              (S1)    rad / 0.40

    preview          v_ref at LOOKAHEAD_M at the reference friction   m/s (not scaled by sqrt(mu / 0.9))
                     racing-line points, curvature and reference-acceleration previews as in the observation

S2 has the first nine channels, S1 all twelve, so S2 is contained in S1. S0 is the privileged observation itself.

Level C is the observation of the coach (``learning/coach.py``, docs/REDESIGN.md, sections 23-24): the privileged
observation without the plan of the teacher, that is without the v_ref preview and the reference-acceleration
preview (``PLAN_SLICES``; ``COACH_INDEX`` holds the 92 entries kept). The racing line, the curvature, the tyre state,
the setup and the friction stay: the coach is privileged, but it chooses its speed itself.

A diagnostic layout may append privileged oracle entries after the preview (``STUDENT_ORACLES``; only the episode
friction ``mu``, raw). They measure how much of the gap to S0 one withheld quantity explains; a student meant to
drive never reads them, unless an estimate replaces the true value (``learning/friction.py``).

The friction estimator reads the tyre channels ``FRICTION_CHANNELS`` (vx, yaw_rate, delta, ax_imu, ay_imu), a
subset of the S2 channels: no preview and no action, so its estimate cannot come from how fast the car is driven.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.spec.constants import (
    DELTA_MAX,
    LOOKAHEAD_M,
    OBS_DIM,
    SETUP_DIM,
    SETUP_RANGES,
    STATE_DIM,
    STUDENT_HISTORY_FRAMES,
    STUDENT_HISTORY_STRIDE,
    TAU_MIN,
    TRAJ_N,
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
    """Indices of the setup vector [m, Cz_f, Cz_r, mu, hcg, tau_d, Cpac]."""

    M: int = 0
    CZ_F: int = 1
    CZ_R: int = 2
    MU: int = 3
    HCG: int = 4
    TAU_D: int = 5
    CPAC: int = 6


_N_PREVIEW = len(LOOKAHEAD_M)


class ObsIndex(NamedTuple):
    """Indices and slices of the 116-entry observation (see the module docstring)."""

    VX: int = 0
    VY: int = 1
    YAW_RATE: int = 2
    DELTA: int = 3
    EY: int = 4
    EPSI: int = 5
    VREF_START: int = 6
    VREF_END: int = 6 + _N_PREVIEW
    TRAJ_START: int = 6 + _N_PREVIEW
    TRAJ_END: int = 6 + _N_PREVIEW + 2 * TRAJ_N
    AX: int = 6 + _N_PREVIEW + 2 * TRAJ_N
    XI_START: int = 7 + _N_PREVIEW + 2 * TRAJ_N
    XI_END: int = 7 + _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    RHO_LAT: int = 7 + _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    RHO_LON: int = 8 + _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    KAPPA_START: int = 9 + _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    KAPPA_END: int = 9 + 2 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    AX_PREVIEW_START: int = 9 + 2 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    AX_PREVIEW_END: int = 9 + 3 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    ALPHA_F: int = 9 + 3 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    ALPHA_R: int = 10 + 3 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    FY_F: int = 11 + 3 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM
    FY_R: int = 12 + 3 * _N_PREVIEW + 2 * TRAJ_N + SETUP_DIM


STATE = StateIndex()
SETUP = SetupIndex()
OBS = ObsIndex()

COACH_LEVEL = "C"
STUDENT_LEVELS: tuple[str, ...] = ("S0", "S1", "S2", COACH_LEVEL)
_S2_CHANNELS = ("vx", "yaw_rate", "delta", "ax_imu", "ay_imu", "ey", "epsi", "prev_delta_cmd", "prev_tau")
STUDENT_CHANNELS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": _S2_CHANNELS + ("vy", "alpha_f", "alpha_r"),
    "S2": _S2_CHANNELS,
    COACH_LEVEL: (),
}
PLAN_SLICES: tuple[slice, ...] = (slice(OBS.VREF_START, OBS.VREF_END),
                                  slice(OBS.AX_PREVIEW_START, OBS.AX_PREVIEW_END))
COACH_INDEX: NDArray[np.int64] = np.delete(np.arange(OBS_DIM),
                                           np.concatenate([np.arange(s.start, s.stop) for s in PLAN_SLICES]))
STUDENT_PREVIEW_DIM: int = 3 * _N_PREVIEW + 2 * TRAJ_N
STUDENT_ORACLES: tuple[str, ...] = ("mu",)
FRICTION_CHANNELS: tuple[str, ...] = ("vx", "yaw_rate", "delta", "ax_imu", "ay_imu")


class StudentLayout(NamedTuple):
    """Layout of the student observation of one level (see the module docstring)."""

    level: str
    channels: tuple[str, ...]
    n_frames: int
    stride: int
    oracle: tuple[str, ...] = ()

    @property
    def privileged(self) -> bool:
        """Whether the student observation is the privileged observation itself (S0)."""
        return self.level == "S0"

    @property
    def coach(self) -> bool:
        """Whether the observation is the coach's: the privileged one without the plan (C)."""
        return self.level == COACH_LEVEL

    @property
    def frame_dim(self) -> int:
        """Dynamic channels per frame."""
        return len(self.channels)

    @property
    def dynamic_dim(self) -> int:
        """Entries of all the frames."""
        return self.frame_dim * self.n_frames

    @property
    def dim(self) -> int:
        """Size of the student observation."""
        if self.privileged:
            return OBS_DIM
        if self.coach:
            return len(COACH_INDEX)
        return self.dynamic_dim + STUDENT_PREVIEW_DIM + len(self.oracle)

    @property
    def window(self) -> int:
        """Steps spanned by the frames, current step included."""
        return (self.n_frames - 1) * self.stride + 1

    def preview_slices(self) -> dict[str, slice]:
        """Slices of the v_ref, racing-line, curvature and reference-acceleration previews."""
        start = self.dynamic_dim
        sizes = (("vref", _N_PREVIEW), ("traj", 2 * TRAJ_N), ("kappa", _N_PREVIEW), ("ax_preview", _N_PREVIEW))
        slices = {}
        for name, size in sizes:
            slices[name] = slice(start, start + size)
            start += size
        return slices


def student_layout(level: str, n_frames: int = STUDENT_HISTORY_FRAMES, stride: int = STUDENT_HISTORY_STRIDE,
                   oracle: tuple[str, ...] = ()) -> StudentLayout:
    """Layout of the student observation of ``level`` (one of ``STUDENT_LEVELS``), with optional oracle entries."""
    if level not in STUDENT_CHANNELS:
        raise ValueError(f"unknown student level {level!r}; expected one of {STUDENT_LEVELS}")
    if n_frames < 1 or stride < 1:
        raise ValueError(f"n_frames={n_frames} and stride={stride} must be positive")
    oracle = tuple(oracle)
    unknown = [name for name in oracle if name not in STUDENT_ORACLES]
    if unknown or len(set(oracle)) != len(oracle):
        raise ValueError(f"oracle {oracle} must hold distinct entries of {STUDENT_ORACLES}")
    if level in ("S0", COACH_LEVEL):
        if oracle:
            raise ValueError(f"the observation of level {level} takes no oracle entries")
        return StudentLayout(level, (), 1, 1)
    return StudentLayout(level, STUDENT_CHANNELS[level], int(n_frames), int(stride), oracle)

ACTION_LOW: NDArray[np.float64] = np.array([-DELTA_MAX, -1.0], dtype=np.float64)
ACTION_HIGH: NDArray[np.float64] = np.array([DELTA_MAX, 1.0], dtype=np.float64)


def normalise_setup(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map a setup vector from physical units to [-1, 1] (affine, per entry)."""
    xi_norm = np.empty(SETUP_DIM, dtype=np.float64)
    for i, (lo, hi) in enumerate(SETUP_RANGES):
        xi_norm[i] = 2.0 * (xi[i] - lo) / (hi - lo) - 1.0
    return xi_norm


def denormalise_setup(xi_norm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of :func:`normalise_setup`."""
    xi = np.empty(SETUP_DIM, dtype=np.float64)
    for i, (lo, hi) in enumerate(SETUP_RANGES):
        xi[i] = lo + (xi_norm[i] + 1.0) / 2.0 * (hi - lo)
    return xi


def validate_setup(xi: NDArray[np.float64]) -> None:
    """Raise ``ValueError`` if a setup vector is malformed or outside its randomisation ranges."""
    if xi.shape != (SETUP_DIM,):
        raise ValueError(f"setup shape {xi.shape} != ({SETUP_DIM},)")
    if xi[SETUP.TAU_D] < TAU_MIN:
        raise ValueError(f"tau_d={xi[SETUP.TAU_D]:.6f} < TAU_MIN={TAU_MIN}")
    names = ("m", "Cz_f", "Cz_r", "mu", "hcg", "tau_d", "Cpac")
    for name, value, (lo, hi) in zip(names, xi, SETUP_RANGES):
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value:.6f} outside its range [{lo}, {hi}]")


def validate_state(z: NDArray[np.float64], kappa_at_s: float) -> None:
    """Raise ``ValueError`` if a state violates vx >= V_MIN or Frenet regularity (1 - kappa ey > 0)."""
    if z.shape != (STATE_DIM,):
        raise ValueError(f"state shape {z.shape} != ({STATE_DIM},)")
    if z[STATE.VX] < V_MIN:
        raise ValueError(f"vx={z[STATE.VX]:.4f} < V_MIN={V_MIN}")
    if 1.0 - kappa_at_s * z[STATE.EY] <= 0.0:
        raise ValueError(f"Frenet regularity violated: kappa={kappa_at_s:.6f}, ey={z[STATE.EY]:.4f}")
