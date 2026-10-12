"""Reward and traction cost (behaviour cloning and evaluation ignore both).

    r_t = w_progress max(ds/dt, 0) dt  -  w_lateral (|ey| / ey_max)^2 dt  -  TIME_COST_RATE dt
          - w_traction c_t dt  +  lap_bonus 1[lap completed]  -  1[failure] (CRASH_BASE + w_crash (vx^2 + vy^2))

    c_t = sum over the axles of max(|alpha_i| / alpha_sat_i - TRACTION_SLIP_MARGIN, 0)

A failure is a termination by track excursion, slip saturation, low speed or plant inadmissibility.
With w_progress = 1 and terminate-on-lap, the return of a lap is about L - TIME_COST_RATE * lap time.

The traction cost measures how far a tyre is driven past its peak: the combined-slip capacity keeps every axle
force inside the friction circle, so the force use never exceeds 1, but beyond alpha_sat the lateral force falls.
It is a cost, not a reward: rewarding the use of the friction budget made an earlier agent brake to collect it.

``RewardWeights()`` is the reward of the PPO fine-tuning. ``OFFLINE_REWARD_WEIGHTS`` is the reward logged in the
offline datasets: no lateral term (the reward must not tie the policy to the racing line) and no traction term,
because the cost is logged separately and weighted at training time.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.spec.constants import TRACTION_SLIP_MARGIN

TIME_COST_RATE: float = 12.0  # [m/s] one second of driving costs 12 m of progress
CRASH_BASE: float = 3.0  # [m] fixed part of the failure penalty

FAILURE_EVENTS: tuple[str, ...] = ("track_excursion", "slip_saturation", "low_speed", "admissibility_violation")


@dataclass(frozen=True)
class RewardWeights:
    """Weights of the reward terms."""

    w_progress: float = 1.0  # [-]
    w_lateral: float = 1.0  # [m]
    lap_bonus: float = 0.0  # [m]
    w_crash: float = 0.02  # [s^2/m]
    w_traction: float = 0.0  # [m/s] reward lost per second at unit traction cost


OFFLINE_REWARD_WEIGHTS = RewardWeights(w_lateral=0.0)


def compute_traction_cost(alpha_f: float, alpha_r: float, alpha_sat: tuple[float, float]) -> float:
    """Return the traction cost c_t [-] of a step with slip angles (alpha_f, alpha_r) [rad]."""
    cost = 0.0
    for alpha, alpha_peak in ((alpha_f, alpha_sat[0]), (alpha_r, alpha_sat[1])):
        cost += max(abs(float(alpha)) / float(alpha_peak) - TRACTION_SLIP_MARGIN, 0.0)
    return cost


def compute_reward(
    state: VehicleState,
    s_dot: float,
    ey_max: float,
    dt: float,
    weights: RewardWeights,
    terminated: bool,
    event_flags: dict[str, bool],
    lap_complete: bool,
    traction_cost: float = 0.0,
) -> float:
    """Return the reward of the transition that ended in ``state``."""
    progress = max(float(s_dot), 0.0) * dt
    lateral = (abs(float(state.ey)) / float(ey_max)) ** 2 * dt
    reward = weights.w_progress * progress - weights.w_lateral * lateral
    if lap_complete:
        reward += weights.lap_bonus
    reward -= TIME_COST_RATE * dt
    if weights.w_traction != 0.0:
        reward -= weights.w_traction * float(traction_cost) * dt
    if terminated and any(bool(event_flags.get(name, False)) for name in FAILURE_EVENTS):
        vx, vy = float(state.vx), float(state.vy)
        reward -= CRASH_BASE + weights.w_crash * (vx * vx + vy * vy)
    return reward
