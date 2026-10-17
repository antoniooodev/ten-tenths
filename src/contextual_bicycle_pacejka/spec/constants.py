"""Numerical constants of the vehicle model, the environment, the speed stages and the observation layout.

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

# Plant admissibility.
V_MIN: float = 1.0  # [m/s] minimum admissible longitudinal speed

# ---------------------------------------------------------------------------
# Vehicle setup: nominal values (fixed) and the balance index
# ---------------------------------------------------------------------------
MASS_NOM: float = 1400.0  # [kg]
MU_NOM: float = 1.0  # [-] peak tyre-road friction
HCG_NOM: float = 0.50  # [m] centre-of-gravity height
TAU_D_NOM: float = 0.03  # [s] steering lag
B_NOM: float = 0.0  # [-] neutral balance index: the original plant
B_MIN: float = -1.0  # [-] exclusive bound: the front lateral capacity vanishes
B_MAX: float = LF / LR  # [-] exclusive bound: the rear lateral capacity vanishes

# ---------------------------------------------------------------------------
# Decisions and action
# ---------------------------------------------------------------------------
DECISION_STEPS: int = 5  # [steps] plant steps per decision: 20 Hz, the action held in between
DECISION_DT: float = DECISION_STEPS * DT  # [s]
TAU_RATE_MAX: float = 5.0  # [1/s] largest rate of the longitudinal command
SPEED_LIMIT_BAND: float = 0.05  # [-] traction fades over the last 5 % of the top speed

# ---------------------------------------------------------------------------
# Speed stages
# ---------------------------------------------------------------------------
STAGE_TOP_SPEEDS_MPS: tuple[float | None, ...] = (5.0, 10.0, 20.0, 30.0, 40.0, 60.0, None)  # [m/s] None: no bound
FREE_STAGE_SIZING_MPS: float = 92.0  # [m/s] sizes the free stage: point-mass top speed on Catalunya, rounded up
PREVIEW_TIME_S: float = 1.0  # [s] preview horizon beyond the braking distance
N_PREVIEW: int = 30  # [-] preview points, the same at every stage
START_SPEED_MIN_MPS: float = 3.0  # [m/s] slowest start; the fastest is the top speed of the previous stage

# ---------------------------------------------------------------------------
# Failures, reward and episodes
# ---------------------------------------------------------------------------
SPIN_EPSI_RAD: float = 1.5707963267948966  # [rad] spin: heading error beyond pi / 2
STALL_SPEED_MPS: float = 1.5  # [m/s] stall: vx below this speed
TRAINING_EDGE_MARGIN_M: float = 0.3  # [m] in training the car leaves the track this far inside the edge
FAILURE_PENALTY: float = 1.0  # [-] in units of the largest progress of one decision
ACTION_RATE_WEIGHT: float = 0.01  # [-] weight of the squared normalised change of the actions
REWARD_SCALE: float = 0.1  # [-] scale of the reward: discounted returns of about 10, as with progress in metres
EPISODE_DECISIONS: int = 1024  # [decisions] truncation of an episode (51.2 s)

# ---------------------------------------------------------------------------
# Observation scales
# ---------------------------------------------------------------------------
HELD_OUT_HALF_WIDTH: float = 0.01  # [-] half-width of the band of b kept out of training around a held-out value
B_OBS_SCALE: float = 0.2  # [-] largest |b| of the drivable range (scripts/check_balance_index.py)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
STATE_DIM: int = 7  # [s, ey, epsi, vx, vy, yaw_rate, delta]
SETUP_DIM: int = 5  # [m, mu, hcg, tau_d, b]
VEHICLE_OBS_DIM: int = 10  # [vx, vy, r, epsi, ey, delta, delta_ref, tau, tau_rate, mu]
OBS_DIM: int = VEHICLE_OBS_DIM + 3 * N_PREVIEW  # Nominal and Randomised; the Oracle appends b
ACTION_DIM: int = 2  # [delta_ref, tau_rate], each in [-1, 1]
