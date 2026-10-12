"""Safety filter (shield): a one-step safe set over a 61 x 61 action grid and the projection onto it.

Construction of the safe set at the current true plant state (oracle: state, memory, setup and track):

1. Candidate cells lie inside the steering-rate window |delta_cmd - prev_delta_cmd| <= rate * dt and inside the
   front-slip command bounds (the delta_cmd values that keep the next front slip within +/- alpha_sat_f).
2. The centre of every candidate cell is stepped once through a vectorised copy of the plant.
3. A cell is unsafe if the next state leaves the track, if a slip angle exceeds alpha_sat (factor 1.0, stricter
   than the 1.25 of the environment), if the front slip exceeds 0.9 alpha_sat_f and no 3-step recovery branch
   brings it back, if vx < V_MIN, or if the lateral escape distance under a delayed maximal correction does not
   fit inside the boundary over the next 0.6 s.
4. If no cell is safe the set is empty: the 16 cells with the smallest boundary deficit are used as fallback.

Filtering a proposed action: it is kept if its cell is safe (and the set is not empty), otherwise it is
replaced by the centre of the nearest active cell (distance scaled by the action range).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.dynamics import step_plant
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.termination import compute_termination
from contextual_bicycle_pacejka.model.tires import pacejka_coefficients
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    A_MAX,
    BETA_BRAKE,
    D_MAX,
    DELTA_MAX,
    DT,
    EPS_D,
    G,
    IZ,
    LF,
    LR,
    TAU_MIN,
    V_MIN,
)
from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, ACTION_LOW

SAFETY: float = 0.86  # [-] fraction of the lateral capacity available for the boundary-escape manoeuvre
STEERING_RATE_LIMIT: float = 6.0  # [rad/s] steering-command rate limit
N_CELLS: int = 61  # grid cells per action dimension
FALLBACK_CELLS: int = 16  # cells activated when the safe set is empty
FRONT_SLIP_MARGIN: float = 0.90  # [-] front slip above this fraction of alpha_sat_f needs a recovery check
FRONT_SLIP_ROLLOUT_STEPS: int = 3  # recovery horizon [steps]
RECOVERY_DELTA_MAX: float = 0.20  # [rad] steering bound of the recovery branches
BOUNDARY_HORIZON_S: float = 0.6  # [s] look-ahead of the boundary-viability check
BOUNDARY_SAMPLES: int = 10  # boundary widths sampled along the look-ahead
VIABILITY_ACCEL_EPS: float = 1e-6  # [m/s^2] floor of the correction acceleration


@dataclass(frozen=True)
class SafeSet:
    """Safe cells of the action grid at one control step."""

    delta_edges: NDArray[np.float64]
    tau_edges: NDArray[np.float64]
    active: NDArray[np.bool_]  # shape (N_CELLS, N_CELLS): safe cells (or the fallback cells if empty)
    empty: bool  # True if no cell was safe and the fallback cells are used


@dataclass(frozen=True)
class FilterResult:
    """Outcome of filtering one proposed action."""

    action: NDArray[np.float32]  # executed action
    projected: bool  # the proposal was replaced
    empty: bool  # the safe set was empty at this step


def filter_action(env: Any, proposed: NDArray[np.floating], previous_action: NDArray[np.floating]) -> FilterResult:
    """Filter ``proposed`` at the current state of ``env`` (a :class:`ContextualBicyclePacejkaEnv`, possibly wrapped).

    ``previous_action`` is the action executed at the previous step (zeros after a reset); it sets the
    steering-rate window.
    """
    base = env.unwrapped
    safe = safe_action_set(base.state, base.memory, base.setup, base.track, base.alpha_sat,
                           prev_delta_cmd=float(previous_action[0]))
    if is_allowed(safe, proposed):
        return FilterResult(np.asarray(proposed, dtype=np.float32), False, safe.empty)
    return FilterResult(nearest_safe_action(safe, proposed), True, safe.empty)


def is_allowed(safe: SafeSet, action: NDArray[np.floating]) -> bool:
    """Whether ``action`` lies in an active cell of a non-empty safe set."""
    row = _cell_index(safe.delta_edges, float(action[0]))
    col = _cell_index(safe.tau_edges, float(action[1]))
    return bool(row is not None and col is not None and safe.active[row, col] and not safe.empty)


def nearest_safe_action(safe: SafeSet, proposed: NDArray[np.floating]) -> NDArray[np.float32]:
    """Centre of the active cell closest to ``proposed`` (distance scaled by the action range)."""
    rows, cols = np.nonzero(safe.active)
    if rows.size == 0:
        raise RuntimeError("safe set has no active cell")
    delta_centres = 0.5 * (safe.delta_edges[:-1] + safe.delta_edges[1:])
    tau_centres = 0.5 * (safe.tau_edges[:-1] + safe.tau_edges[1:])
    candidates = np.column_stack((delta_centres[rows], tau_centres[cols]))
    distance = np.sum(((candidates - np.asarray(proposed)) / (ACTION_HIGH - ACTION_LOW)) ** 2, axis=1)
    return candidates[int(np.argmin(distance))].astype(np.float32)


def front_slip_delta_cmd_bounds(state: VehicleState, alpha_sat_f: float, tau_d: float) -> tuple[float, float]:
    """Steering-command bounds [rad] that keep the next-step front slip within +/- alpha_sat_f.

    The front slip is delta - beta_f with beta_f = atan2(vy + LF r, vx); the next delta follows the actuator lag.
    """
    beta = math.atan2(state.vy + LF * state.yaw_rate, max(state.vx, 1.0e-9))
    alpha = abs(float(alpha_sat_f))
    decay = 1.0 - DT / max(float(tau_d), TAU_MIN)
    command_gain = 1.0 - decay
    low, high = -DELTA_MAX, DELTA_MAX
    if command_gain > 0.0:
        low = max(low, (beta - alpha - decay * state.delta) / command_gain)
        high = min(high, (beta + alpha - decay * state.delta) / command_gain)
    return max(-DELTA_MAX, low), min(DELTA_MAX, high)


def safe_action_set(
    state: VehicleState,
    memory: VehicleMemory,
    setup: SetupVector,
    track: RealTrack,
    alpha_sat: tuple[float, float],
    prev_delta_cmd: float,
    safety: float = SAFETY,
    steering_rate_limit: float = STEERING_RATE_LIMIT,
) -> SafeSet:
    """Return the safe set of the action grid at the given true plant state."""
    delta_edges = np.linspace(ACTION_LOW[0], ACTION_HIGH[0], N_CELLS + 1)
    tau_edges = np.linspace(ACTION_LOW[1], ACTION_HIGH[1], N_CELLS + 1)
    delta_centres = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    tau_centres = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    active = np.zeros((N_CELLS, N_CELLS), dtype=bool)
    deficit = np.full((N_CELLS, N_CELLS), np.inf, dtype=np.float64)

    slip_low, slip_high = front_slip_delta_cmd_bounds(state, alpha_sat[0], setup.tau_d)
    delta_min = max(ACTION_LOW[0], float(prev_delta_cmd) - steering_rate_limit * DT, slip_low)
    delta_max = min(ACTION_HIGH[0], float(prev_delta_cmd) + steering_rate_limit * DT, slip_high)
    delta_valid = (delta_edges[:-1] >= delta_min) & (delta_edges[1:] <= delta_max)
    candidates = np.argwhere(delta_valid[:, None] & np.ones((1, N_CELLS), dtype=bool))
    if candidates.size:
        actions = np.column_stack((delta_centres[candidates[:, 0]], tau_centres[candidates[:, 1]]))
        unsafe, cell_deficit = _classify_cells(state, actions, setup, memory, track, alpha_sat, safety,
                                               steering_rate_limit)
        rows, cols = candidates[:, 0], candidates[:, 1]
        deficit[rows, cols] = cell_deficit
        active[rows, cols] = ~unsafe

    empty = not bool(active.any())
    if empty:
        finite = np.flatnonzero(np.isfinite(deficit.ravel()))
        if finite.size == 0:
            finite = np.arange(deficit.size)
        active.ravel()[finite[np.argsort(deficit.ravel()[finite])[:FALLBACK_CELLS]]] = True
    return SafeSet(delta_edges, tau_edges, active, empty)


def _cell_index(edges: NDArray[np.float64], value: float) -> int | None:
    """Index of the grid cell containing ``value`` (the last edge belongs to the last cell)."""
    if value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return idx if 0 <= idx < len(edges) - 1 else None


def _classify_cells(
    state: VehicleState,
    actions: NDArray[np.float64],
    setup: SetupVector,
    memory: VehicleMemory,
    track: RealTrack,
    alpha_sat: tuple[float, float],
    safety: float,
    steering_rate_limit: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return (unsafe flag, boundary deficit [m]) of every candidate action."""
    n = actions.shape[0]
    try:
        next_states, diag = _step_plant_batch(state, actions, setup, memory, track)
    except (AdmissibilityError, ValueError):
        return np.ones(n, dtype=bool), np.full(n, np.inf, dtype=np.float64)

    alpha_sat_f, alpha_sat_r = alpha_sat
    # The excursion check uses the first candidate: the position after one step does not depend on the action.
    excursion = track.is_out_of_bounds(float(next_states[0, 1]), float(next_states[0, 0]))
    slip = (np.abs(diag["alpha_f"]) > alpha_sat_f) | (np.abs(diag["alpha_r"]) > alpha_sat_r)
    front_high = np.abs(diag["alpha_f"]) >= alpha_sat_f * FRONT_SLIP_MARGIN
    recoverable = _front_slip_recoverable_batch(state, actions, next_states, diag, setup, track, alpha_sat,
                                                steering_rate_limit)
    low_speed = next_states[:, 3] < V_MIN
    viable, h_right, h_left = _boundary_viability_batch(next_states, actions, diag, setup, track, safety,
                                                       steering_rate_limit)
    unsafe = np.full(n, bool(excursion), dtype=bool)
    unsafe |= slip | (front_high & ~recoverable) | low_speed | ~viable
    return unsafe, np.maximum(np.maximum(-h_right, -h_left), 0.0)


def _step_plant_batch(
    state: VehicleState, actions: NDArray[np.float64], setup: SetupVector, memory: VehicleMemory, track: RealTrack,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """One plant step for a batch of actions sharing state, memory and setup (vectorised ``step_plant``)."""
    if state.vx < V_MIN:
        raise AdmissibilityError(f"vx={state.vx:.4f} < V_MIN={V_MIN}")
    if setup.tau_d < TAU_MIN:
        raise ValueError(f"tau_d={setup.tau_d} < TAU_MIN={TAU_MIN}")
    delta_cmd = np.clip(actions[:, 0], -DELTA_MAX, DELTA_MAX)
    tau_cmd = np.clip(actions[:, 1], -1.0, 1.0)
    delta_next = np.clip(state.delta + (delta_cmd - state.delta) / setup.tau_d * DT, -DELTA_MAX, DELTA_MAX)

    alpha_f = delta_next - np.arctan2(state.vy + LF * state.yaw_rate, state.vx)
    alpha_r = np.full_like(alpha_f, -np.arctan2(state.vy - LR * state.yaw_rate, state.vx))

    ax_lt = (memory.Fx_r_prev + memory.Fx_f_prev * math.cos(memory.delta_prev)) / setup.m \
        + memory.vy_prev * memory.yawrate_prev
    wheelbase = LF + LR
    fz_f = (setup.m / wheelbase) * (G * LR - ax_lt * setup.hcg)
    fz_r = (setup.m / wheelbase) * (G * LF + ax_lt * setup.hcg)
    if fz_f <= 0.0 or fz_r <= 0.0:
        raise AdmissibilityError(f"non-positive axle load: Fz_f={fz_f:.4f} N, Fz_r={fz_r:.4f} N")

    fx_cmd_f = np.where(tau_cmd >= 0.0, 0.0, tau_cmd * setup.m * D_MAX * BETA_BRAKE)
    fx_cmd_r = np.where(tau_cmd >= 0.0, tau_cmd * setup.m * A_MAX, tau_cmd * setup.m * D_MAX * (1.0 - BETA_BRAKE))
    fx_f = np.copysign(np.minimum(np.abs(fx_cmd_f), setup.mu * fz_f), fx_cmd_f)
    fx_r = np.copysign(np.minimum(np.abs(fx_cmd_r), setup.mu * fz_r), fx_cmd_r)
    cap_f = _combined_slip_capacity(fx_f, setup.mu, fz_f)
    cap_r = _combined_slip_capacity(fx_r, setup.mu, fz_r)
    b_f, c_f, e_f = pacejka_coefficients("front")
    b_r, c_r, e_r = pacejka_coefficients("rear")
    fy_f = _lateral_force(alpha_f, b_f, c_f, cap_f, e_f)
    fy_r = _lateral_force(alpha_r, b_r, c_r, cap_r, e_r)

    cos_d, sin_d = np.cos(delta_next), np.sin(delta_next)
    f_body_x = fx_r + fx_f * cos_d - fy_f * sin_d
    f_body_y = fy_r + fx_f * sin_d + fy_f * cos_d
    ax = f_body_x / setup.m + state.vy * state.yaw_rate
    ay = f_body_y / setup.m - state.vx * state.yaw_rate
    yaw_acc = (LF * (f_body_y - fy_r) - LR * fy_r) / IZ

    kappa = track.kappa(state.s)
    denom = 1.0 - kappa * state.ey
    if denom <= 0.0:
        raise AdmissibilityError(f"Frenet regularity violated: 1 - kappa ey = {denom:.6f}")
    cos_e, sin_e = math.cos(state.epsi), math.sin(state.epsi)
    s_dot = (state.vx * cos_e - state.vy * sin_e) / denom
    ey_dot = state.vx * sin_e + state.vy * cos_e
    epsi_dot = state.yaw_rate - kappa * s_dot

    n = actions.shape[0]
    next_states = np.column_stack((
        np.full(n, state.s + s_dot * DT), np.full(n, state.ey + ey_dot * DT), np.full(n, state.epsi + epsi_dot * DT),
        state.vx + ax * DT, state.vy + ay * DT, state.yaw_rate + yaw_acc * DT, delta_next,
    ))
    diag = {"alpha_f": alpha_f, "alpha_r": alpha_r, "Fx_f": fx_f, "Fx_r": fx_r, "D_eps_f": cap_f, "D_eps_r": cap_r}
    return next_states, diag


def _front_slip_recoverable_batch(
    state: VehicleState,
    actions: NDArray[np.float64],
    next_states: NDArray[np.float64],
    diag: dict[str, NDArray[np.float64]],
    setup: SetupVector,
    track: RealTrack,
    alpha_sat: tuple[float, float],
    steering_rate_limit: float,
) -> NDArray[np.bool_]:
    """Whether a high (but not saturated) front slip can be brought back below the margin within the horizon."""
    alpha_sat_f, alpha_sat_r = alpha_sat
    recoverable = np.abs(diag["alpha_f"]) < alpha_sat_f * FRONT_SLIP_MARGIN
    suspect = np.flatnonzero((np.abs(diag["alpha_f"]) >= alpha_sat_f * FRONT_SLIP_MARGIN)
                             & (np.abs(diag["alpha_f"]) <= alpha_sat_f) & (np.abs(diag["alpha_r"]) <= alpha_sat_r))
    for idx in suspect:
        first = VehicleState.from_array(next_states[int(idx)])
        memory = VehicleMemory(float(diag["Fx_r"][idx]), float(diag["Fx_f"][idx]), first.delta, state.vy,
                               state.yaw_rate)
        recoverable[int(idx)] = _front_slip_recoverable(first, memory, setup, track, alpha_sat,
                                                        (float(actions[idx, 0]), float(actions[idx, 1])),
                                                        steering_rate_limit)
    return recoverable


def _front_slip_recoverable(
    state: VehicleState,
    memory: VehicleMemory,
    setup: SetupVector,
    track: RealTrack,
    alpha_sat: tuple[float, float],
    prev_action: tuple[float, float],
    steering_rate_limit: float,
) -> bool:
    """Search a tree of recovery actions for a branch whose front slip falls below the margin."""
    branches = [(state, memory, prev_action)]
    for _ in range(FRONT_SLIP_ROLLOUT_STEPS - 1):
        next_branches = []
        for branch_state, branch_memory, branch_action in branches:
            for action in _recovery_actions(branch_state, branch_action, steering_rate_limit):
                try:
                    next_state, next_memory, next_diag = step_plant(branch_state, action, setup, branch_memory,
                                                                    track, DT)
                except (AdmissibilityError, ValueError):
                    continue
                if compute_termination(next_state, next_diag, alpha_sat, track)[0]:
                    continue
                if abs(float(next_diag["alpha_f"])) < alpha_sat[0] * FRONT_SLIP_MARGIN:
                    return True
                next_branches.append((next_state, next_memory, action))
        branches = next_branches
        if not branches:
            return False
    return False


def _recovery_actions(
    state: VehicleState, prev_action: tuple[float, float], steering_rate_limit: float,
) -> tuple[tuple[float, float], ...]:
    """Three steering commands (lower bound, slip-cancelling angle, upper bound) at zero throttle."""
    rate = 0.5 * steering_rate_limit
    delta_lo = max(ACTION_LOW[0], -RECOVERY_DELTA_MAX, prev_action[0] - rate * DT)
    delta_hi = min(ACTION_HIGH[0], RECOVERY_DELTA_MAX, prev_action[0] + rate * DT)
    if delta_lo > delta_hi:
        return ()
    target = math.atan2(state.vy + LF * state.yaw_rate, state.vx)
    deltas = sorted({float(delta_lo), float(np.clip(target, delta_lo, delta_hi)), float(delta_hi)})
    return tuple((delta, 0.0) for delta in deltas)


def _viability_displacement(
    outward_speed: NDArray[np.float64], correction_accel: NDArray[np.float64], delay: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Outward travel [m] within the horizon when a maximal correction starts after ``delay``."""
    speed = np.maximum(np.asarray(outward_speed, dtype=np.float64), 0.0)
    accel = np.maximum(np.asarray(correction_accel, dtype=np.float64), VIABILITY_ACCEL_EPS)
    delay_in_horizon = np.minimum(np.maximum(np.asarray(delay, dtype=np.float64), 0.0), BOUNDARY_HORIZON_S)
    correction_time = np.minimum(speed / accel, np.maximum(BOUNDARY_HORIZON_S - delay_in_horizon, 0.0))
    return speed * delay_in_horizon + speed * correction_time - 0.5 * accel * correction_time * correction_time


def _boundary_viability_batch(
    next_states: NDArray[np.float64],
    actions: NDArray[np.float64],
    diag: dict[str, NDArray[np.float64]],
    setup: SetupVector,
    track: RealTrack,
    safety: float,
    steering_rate_limit: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Return (viable, right margin [m], left margin [m]) of the boundary-escape check for every candidate."""
    s_next, ey_next, epsi_next = next_states[:, 0], next_states[:, 1], next_states[:, 2]
    vx_next, vy_next = next_states[:, 3], next_states[:, 4]
    kappa = float(track.kappa(float(s_next[0])))
    denominator = 1.0 - kappa * ey_next
    regular = denominator > 0.0
    cos_e, sin_e = np.cos(epsi_next), np.sin(epsi_next)
    s_dot = (vx_next * cos_e - vy_next * sin_e) / np.where(regular, denominator, 1.0)
    ey_dot = vx_next * sin_e + vy_next * cos_e

    progress = max(float(np.max(s_dot)), 0.0) * BOUNDARY_HORIZON_S
    future_s = np.linspace(float(s_next[0]), float(s_next[0]) + progress, BOUNDARY_SAMPLES)
    left_width = min(float(track.boundary_left(float(s))) for s in future_s)
    right_width = min(float(track.boundary_right(float(s))) for s in future_s)

    # ey is positive to the left: ey_ddot = a_y - kappa vx^2, so the curvature helps to stop a drift to the left
    # (kappa > 0, a left-hand bend) and opposes the correction of a drift to the right.
    lateral_capacity = float(safety) * (diag["D_eps_f"] + diag["D_eps_r"]) / float(setup.m)
    curvature_accel = vx_next * vx_next * kappa
    accel_left = np.maximum(lateral_capacity + curvature_accel, VIABILITY_ACCEL_EPS)
    accel_right = np.maximum(lateral_capacity - curvature_accel, VIABILITY_ACCEL_EPS)

    # Emergency steering direction for each side, from how ey_dot responds to the extreme candidates.
    delta_cmd = np.clip(actions[:, 0], -DELTA_MAX, DELTA_MAX)
    ey_dot_low = float(np.mean(ey_dot[delta_cmd == float(np.min(delta_cmd))]))
    ey_dot_high = float(np.mean(ey_dot[delta_cmd == float(np.max(delta_cmd))]))
    if ey_dot_high >= ey_dot_low:
        emergency_left, emergency_right = -DELTA_MAX, DELTA_MAX
    else:
        emergency_left, emergency_right = DELTA_MAX, -DELTA_MAX
    delay_left = float(setup.tau_d) + np.abs(emergency_left - delta_cmd) / float(steering_rate_limit)
    delay_right = float(setup.tau_d) + np.abs(emergency_right - delta_cmd) / float(steering_rate_limit)
    h_left = left_width - ey_next - _viability_displacement(np.maximum(ey_dot, 0.0), accel_left, delay_left)
    h_right = right_width + ey_next - _viability_displacement(np.maximum(-ey_dot, 0.0), accel_right, delay_right)
    viable = np.isfinite(h_right) & np.isfinite(h_left) & regular & (h_right >= 0.0) & (h_left >= 0.0)
    return viable, h_right, h_left


def _combined_slip_capacity(fx: NDArray[np.float64], mu: float, fz: float) -> NDArray[np.float64]:
    mu_fz = mu * fz
    ratio = fx / mu_fz
    return np.maximum(mu_fz * np.sqrt(1.0 - np.minimum(ratio * ratio, 1.0)), EPS_D)


def _lateral_force(alpha: NDArray[np.float64], b: float, c: float, d: NDArray[np.float64], e: float,
                   ) -> NDArray[np.float64]:
    ba = b * alpha
    return d * np.sin(c * np.arctan(ba - e * (ba - np.arctan(ba))))
