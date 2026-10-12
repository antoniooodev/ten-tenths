"""Numerical constants of the vehicle model, the environment and the observation layout.

Every physical quantity carries its unit in the trailing comment.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
DT: float = 0.01  # [s] control and integration step (100 Hz, explicit Euler)

# ---------------------------------------------------------------------------
# Vehicle (fixed, not randomised)
# ---------------------------------------------------------------------------
LF: float = 1.2  # [m] centre of mass to front axle
LR: float = 1.4  # [m] centre of mass to rear axle
IZ: float = 1800.0  # [kg m^2] yaw moment of inertia
G: float = 9.81  # [m/s^2] gravitational acceleration

# Powertrain: rear-wheel drive, braking on both axles with a fixed front bias.
A_MAX: float = 8.0  # [m/s^2] maximum commanded acceleration
D_MAX: float = 12.0  # [m/s^2] maximum commanded deceleration magnitude
BETA_BRAKE: float = 0.65  # [-] front share of the braking force

# Steering actuator.
DELTA_MAX: float = 0.5  # [rad] steering angle limit
TAU_MIN: float = 0.01  # [s] smallest actuator lag time constant

# Tyres.
EPS_D: float = 1e-4  # [N] floor of the combined-slip lateral capacity
ALPHA_SAT_XTOL: float = 1e-8  # [rad] root-finding tolerance of the peak slip angle

# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
V_MIN: float = 1.0  # [m/s] minimum admissible longitudinal speed
SLIP_TERMINATION_RATIO: float = 1.25  # [-] terminal slip, as a multiple of the peak slip angle

# ---------------------------------------------------------------------------
# Domain randomisation: nominal values and uniform ranges of the setup vector
# ---------------------------------------------------------------------------
MASS_NOM: float = 1400.0  # [kg]
MASS_LO: float = MASS_NOM * 0.70  # [kg]
MASS_HI: float = MASS_NOM * 1.30  # [kg]
CZ_F_NOM: float = 80_000.0  # [N/rad] front cornering stiffness (sampled and observed, unused by the plant)
CZ_F_LO: float = CZ_F_NOM * 0.75  # [N/rad]
CZ_F_HI: float = CZ_F_NOM * 1.25  # [N/rad]
CZ_R_NOM: float = 80_000.0  # [N/rad] rear cornering stiffness (sampled and observed, unused by the plant)
CZ_R_LO: float = CZ_R_NOM * 0.75  # [N/rad]
CZ_R_HI: float = CZ_R_NOM * 1.25  # [N/rad]
MU_NOM: float = 1.0  # [-] peak tyre-road friction
MU_LO: float = MU_NOM * 0.80  # [-]
MU_HI: float = MU_NOM * 1.20  # [-]
HCG_NOM: float = 0.50  # [m] centre-of-gravity height
HCG_LO: float = HCG_NOM - 0.10  # [m]
HCG_HI: float = HCG_NOM + 0.10  # [m]
TAU_D_LO: float = TAU_MIN  # [s] steering lag
TAU_D_HI: float = 0.050  # [s]
CPAC_NOM: float = 1.30  # [-] Pacejka shape factor (sampled and observed, unused by the plant)
CPAC_LO: float = CPAC_NOM * 0.90  # [-]
CPAC_HI: float = CPAC_NOM * 1.10  # [-]

# (lo, hi) of every setup entry, in setup-vector order.
SETUP_RANGES: tuple[tuple[float, float], ...] = (
    (MASS_LO, MASS_HI),
    (CZ_F_LO, CZ_F_HI),
    (CZ_R_LO, CZ_R_HI),
    (MU_LO, MU_HI),
    (HCG_LO, HCG_HI),
    (TAU_D_LO, TAU_D_HI),
    (CPAC_LO, CPAC_HI),
)

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
LOOKAHEAD_M: tuple[float, ...] = (5, 10, 20, 35, 55, 80, 110, 145, 185, 230, 290, 350)  # [m] preview distances
VREF_REFERENCE_MU: float = 0.9  # [-] friction at which the track v_ref profiles were computed
TRAJ_N: int = 30  # number of racing-line points in the observation
TRAJ_LOOKAHEAD_MIN_M: float = 5.0  # [m] first racing-line point
TRAJ_LOOKAHEAD_MAX_M: float = 230.0  # [m] last racing-line point
TRAJ_NORMALISATION_M: float = 100.0  # [m] scale of the racing-line coordinates
KAPPA_NORM_RADPM: float = 0.05  # [1/m] scale of the curvature preview
AX_PREVIEW_NORM_MPS2: float = 15.0  # [m/s^2] scale of the reference-acceleration preview
ALPHA_NORM_RAD: float = 0.40  # [rad] scale of the slip-angle features

# ---------------------------------------------------------------------------
# Traction cost
# ---------------------------------------------------------------------------
TRACTION_SLIP_MARGIN: float = 0.90  # [-] slip ratio |alpha| / alpha_sat above which the traction cost accrues

# ---------------------------------------------------------------------------
# Student observation
# ---------------------------------------------------------------------------
STUDENT_HISTORY_FRAMES: int = 8  # [frames] dynamic-channel frames in the student observation, current one included
STUDENT_HISTORY_STRIDE: int = 5  # [steps] spacing of the frames (8 frames x 5 steps: the last 0.35 s)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
STATE_DIM: int = 7  # [s, ey, epsi, vx, vy, yaw_rate, delta]
SETUP_DIM: int = 7  # [m, Cz_f, Cz_r, mu, hcg, tau_d, Cpac]
OBS_DIM: int = 116
IMU_DIM: int = 2  # [ax_imu, ay_imu]
