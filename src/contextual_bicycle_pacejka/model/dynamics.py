"""Single-track vehicle dynamics in Frenet coordinates and the one-step plant update.

Causal order of one step (no algebraic loop):

    1. steering actuator lag                   -> delta(t+1)
    2. slip angles                             (with the post-actuator delta)
    3. load transfer from the previous step    -> Fz_f, Fz_r
    4. powertrain (power limit) + friction limit -> Fx_f, Fx_r
    5. combined-slip capacity (scaled by b) + Pacejka -> Fy_f, Fy_r
    6. body-frame resultants, aerodynamic drag 1/2 rho cdA vx^2 against vx
    7. Newton-Euler accelerations              -> ax = dvx/dt (observed), dvy/dt, dr/dt
    8. Frenet kinematics                       -> ds/dt, dey/dt, depsi/dt
    9. explicit Euler update and new memory

Steps 2-7 are ``compute_vehicle_rates``, shared with the steady-state analysis of ``model/handling.py``.

``PlantOptions`` switches the three changes to the original plant: the inertial load transfer
(``model/load_transfer.py``), the aerodynamic drag and the power limit. The default is the current plant;
``ORIGINAL_PLANT`` switches all three off and reproduces the original plant bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from contextual_bicycle_pacejka.model.actuator import step_actuator
from contextual_bicycle_pacejka.model.load_transfer import compute_ax_lt, compute_vertical_loads
from contextual_bicycle_pacejka.model.powertrain import compute_longitudinal_commands, saturate_longitudinal
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.tires import (
    compute_combined_slip_capacity,
    compute_lateral_force,
    compute_lateral_scales,
    compute_slip_angles,
    pacejka_coefficients,
)
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import AIR_DENSITY, DELTA_MAX, DRAG_AREA_M2, IZ, LF, LR, WHEEL_POWER_W



@dataclass(frozen=True)
class PlantOptions:
    """Switches of the plant: inertial load transfer, drag area cd x A [m^2] and power at the wheels [W]."""

    inertial_load_transfer: bool = True
    drag_area: float = DRAG_AREA_M2
    wheel_power: float = WHEEL_POWER_W


PLANT = PlantOptions()
ORIGINAL_PLANT = PlantOptions(inertial_load_transfer=False, drag_area=0.0, wheel_power=math.inf)

DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "ax", "ax_lt", "alpha_f", "alpha_r", "Fx_f", "Fx_r", "Fy_f", "Fy_r", "Fz_f", "Fz_r", "F_body_x", "F_body_y",
    "s_dot",
)


def compute_body_forces(fx_f: float, fx_r: float, fy_f: float, fy_r: float, delta: float) -> tuple[float, float]:
    """Return the body-frame resultants (F_x, F_y) [N] of the axle forces."""
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    return fx_r + fx_f * cos_d - fy_f * sin_d, fy_r + fx_f * sin_d + fy_f * cos_d


def compute_accelerations(
    f_body_x: float, f_body_y: float, fy_r: float, state: VehicleState, setup: SetupVector,
) -> tuple[float, float, float]:
    """Return (dvx/dt, dvy/dt, dr/dt) from the Newton-Euler equations in the body frame."""
    m = setup.m
    ax = f_body_x / m + state.vy * state.yaw_rate
    ay = f_body_y / m - state.vx * state.yaw_rate
    yaw_acc = (LF * (f_body_y - fy_r) - LR * fy_r) / IZ
    return ax, ay, yaw_acc


def compute_vehicle_rates(
    actuated: VehicleState, tau_cmd: float, setup: SetupVector, memory: VehicleMemory, options: PlantOptions = PLANT,
) -> tuple[float, float, float, dict[str, float]]:
    """Return (dvx/dt, dvy/dt, dr/dt) and the force diagnostics for a state whose delta is already post-actuator.

    The diagnostics hold every key of ``DIAGNOSTIC_KEYS`` except ``"s_dot"``.
    """
    alpha_f, alpha_r = compute_slip_angles(actuated)

    ax_lt = compute_ax_lt(memory, setup, options.inertial_load_transfer)
    fz_f, fz_r = compute_vertical_loads(ax_lt, setup)

    fx_cmd_f, fx_cmd_r = compute_longitudinal_commands(tau_cmd, setup, actuated.vx, options.wheel_power)
    fx_f = saturate_longitudinal(fx_cmd_f, setup.mu, fz_f)
    fx_r = saturate_longitudinal(fx_cmd_r, setup.mu, fz_r)

    lambda_f, lambda_r = compute_lateral_scales(setup.b)
    b_f, c_f, e_f = pacejka_coefficients("front")
    b_r, c_r, e_r = pacejka_coefficients("rear")
    fy_f = compute_lateral_force(alpha_f, b_f, c_f, compute_combined_slip_capacity(fx_f, setup.mu, fz_f, lambda_f), e_f)
    fy_r = compute_lateral_force(alpha_r, b_r, c_r, compute_combined_slip_capacity(fx_r, setup.mu, fz_r, lambda_r), e_r)

    f_body_x, f_body_y = compute_body_forces(fx_f, fx_r, fy_f, fy_r, actuated.delta)
    if options.drag_area > 0.0:
        f_body_x -= 0.5 * AIR_DENSITY * options.drag_area * actuated.vx * abs(actuated.vx)
    ax, ay, yaw_acc = compute_accelerations(f_body_x, f_body_y, fy_r, actuated, setup)
    diag = {
        "ax": ax, "ax_lt": ax_lt, "alpha_f": alpha_f, "alpha_r": alpha_r, "Fx_f": fx_f, "Fx_r": fx_r,
        "Fy_f": fy_f, "Fy_r": fy_r, "Fz_f": fz_f, "Fz_r": fz_r, "F_body_x": f_body_x, "F_body_y": f_body_y,
    }
    return ax, ay, yaw_acc, diag


def compute_frenet_rates(state: VehicleState, track: RealTrack) -> tuple[float, float, float]:
    """Return (ds/dt, dey/dt, depsi/dt); raise AdmissibilityError if 1 - kappa ey <= 0."""
    kappa = track.kappa(state.s)
    denom = 1.0 - kappa * state.ey
    if denom <= 0.0:
        raise AdmissibilityError(f"Frenet regularity violated: 1 - kappa ey = {denom:.6f}")
    cos_epsi = math.cos(state.epsi)
    sin_epsi = math.sin(state.epsi)
    s_dot = (state.vx * cos_epsi - state.vy * sin_epsi) / denom
    ey_dot = state.vx * sin_epsi + state.vy * cos_epsi
    return s_dot, ey_dot, state.yaw_rate - kappa * s_dot


def step_plant(
    state: VehicleState,
    action: tuple[float, float],
    setup: SetupVector,
    memory: VehicleMemory,
    track: RealTrack,
    dt: float,
    options: PlantOptions = PLANT,
) -> tuple[VehicleState, VehicleMemory, dict[str, Any]]:
    """Advance the plant by one step for action (delta_cmd [rad], tau [-]).

    Returns the next state, the memory for the next step and a diagnostics dict
    with the keys of ``DIAGNOSTIC_KEYS``.
    """
    delta_cmd = max(-DELTA_MAX, min(DELTA_MAX, action[0]))
    tau_cmd = max(-1.0, min(1.0, action[1]))

    delta_next = step_actuator(state.delta, delta_cmd, setup.tau_d, dt)
    actuated = VehicleState(state.s, state.ey, state.epsi, state.vx, state.vy, state.yaw_rate, delta_next)

    ax, ay, yaw_acc, diag = compute_vehicle_rates(actuated, tau_cmd, setup, memory, options)
    s_dot, ey_dot, epsi_dot = compute_frenet_rates(actuated, track)

    next_state = VehicleState(
        s=state.s + s_dot * dt,
        ey=state.ey + ey_dot * dt,
        epsi=state.epsi + epsi_dot * dt,
        vx=state.vx + ax * dt,
        vy=state.vy + ay * dt,
        yaw_rate=state.yaw_rate + yaw_acc * dt,
        delta=delta_next,
    )
    next_memory = VehicleMemory(diag["Fx_r"], diag["Fx_f"], delta_next, state.vy, state.yaw_rate, diag["F_body_x"])
    diag["s_dot"] = s_dot
    return next_state, next_memory, diag
