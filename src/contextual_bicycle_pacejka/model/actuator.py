"""Steering actuator: first-order lag on the commanded steering angle.

    delta_next = delta + (delta_cmd - delta) / tau_d * dt,   clipped to [-DELTA_MAX, DELTA_MAX]

With tau_d = TAU_MIN = DT the commanded angle is reached in one step.
"""

from __future__ import annotations

from contextual_bicycle_pacejka.spec.constants import DELTA_MAX, TAU_MIN


def step_actuator(delta: float, delta_cmd: float, tau_d: float, dt: float) -> float:
    """Return the steering angle [rad] after one explicit-Euler step of the actuator lag."""
    if tau_d < TAU_MIN:
        raise ValueError(f"tau_d={tau_d} < TAU_MIN={TAU_MIN}")
    delta_next = delta + (delta_cmd - delta) / tau_d * dt
    return max(-DELTA_MAX, min(DELTA_MAX, delta_next))
