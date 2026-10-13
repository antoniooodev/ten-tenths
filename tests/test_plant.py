"""Vehicle plant: exact agreement with reference rollouts of the dynamics alone, on the Catalunya centreline.

Three manoeuvres drive ``step_plant`` directly (no environment, observation, reward or termination):
cruise and brake on the main straight, a tight corner near the grip limit, and trail braking into a corner. A
deterministic steering law on the state keeps the car near the centreline; the longitudinal command is scheduled. The
reference (``tests/golden/plant_catalunya.npz``) was recorded with the original plant, bit for bit equal to the
rollouts of its first implementation. Any change to the dynamics, such as the balance index at its neutral value,
must keep these rollouts unchanged.

``python tests/test_plant.py`` records the reference again (only after a deliberate change of the dynamics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import GOLDEN, REPO

from contextual_bicycle_pacejka.model.dynamics import DIAGNOSTIC_KEYS, step_plant
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import DT

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"
REFERENCE = GOLDEN / "plant_catalunya.npz"
STEPS = 400  # 4 s

# name: start (s [m], vx [m/s]), setup (m [kg], mu, hcg [m], tau_d [s]), longitudinal command at time t [s]
CASES = {
    "cruise_brake": ((0.0, 30.0), (1400.0, 1.0, 0.50, 0.03), lambda t: 0.3 if t < 2.0 else -0.8),
    "tight_corner": ((3430.0, 20.0), (1400.0, 0.9, 0.50, 0.03), lambda t: 0.3),
    "trail_brake": ((3350.0, 40.0), (1700.0, 1.1, 0.60, 0.02), lambda t: -0.6 if t < 1.2 else 0.1),
}
STEER_FEEDFORWARD_M = 2.6  # [m] wheelbase: kinematic steering for the curvature
STEER_GAIN_EY, STEER_GAIN_EPSI = 0.05, 0.5  # [rad/m], [-]


def steering(state: VehicleState, track: RealTrack) -> float:
    """Deterministic test input that keeps the car near the centreline (not a controller of the project)."""
    delta = STEER_FEEDFORWARD_M * track.kappa(state.s) - STEER_GAIN_EY * state.ey - STEER_GAIN_EPSI * state.epsi
    return float(np.clip(delta, -0.5, 0.5))


def rollout(case: str) -> tuple[np.ndarray, np.ndarray]:
    """States (steps + 1, 7) and diagnostics (steps, len(DIAGNOSTIC_KEYS)) of one manoeuvre; it stops early if the
    plant raises ``AdmissibilityError``."""
    (s0, vx0), (m, mu, hcg, tau_d), longitudinal = CASES[case]
    track = RealTrack(str(CATALUNYA))
    setup = SetupVector(m=m, mu=mu, hcg=hcg, tau_d=tau_d)
    state, memory = VehicleState(s=s0, vx=vx0), VehicleMemory.zeros()
    states, diags = [state.to_array()], []
    for k in range(STEPS):
        try:
            action = (steering(state, track), longitudinal(k * DT))
            state, memory, diag = step_plant(state, action, setup, memory, track, DT)
        except AdmissibilityError:
            break
        states.append(state.to_array())
        diags.append([diag[key] for key in DIAGNOSTIC_KEYS])
    return np.asarray(states, dtype=np.float64), np.asarray(diags, dtype=np.float64).reshape(-1, len(DIAGNOSTIC_KEYS))


@pytest.mark.parametrize("case", sorted(CASES))
def test_plant_matches_reference(case: str) -> None:
    states, diags = rollout(case)
    with np.load(REFERENCE) as reference:
        np.testing.assert_array_equal(states, reference[f"{case}_state"])
        np.testing.assert_array_equal(diags, reference[f"{case}_diag"])


def test_manoeuvres_reach_the_regimes_they_test() -> None:
    """The cruise covers the 4 s; the corner and the trail braking reach the grip limit of the front axle and load
    the rear beyond 80 % of its grip."""
    states, _ = rollout("cruise_brake")
    assert len(states) == STEPS + 1
    for case in ("tight_corner", "trail_brake"):
        _, diags = rollout(case)
        mu = CASES[case][1][1]
        for axle, least in (("f", 0.99), ("r", 0.8)):
            fy = np.abs(diags[:, DIAGNOSTIC_KEYS.index(f"Fy_{axle}")])
            fz = diags[:, DIAGNOSTIC_KEYS.index(f"Fz_{axle}")]
            assert np.max(fy / (mu * fz)) > least, (case, axle)


def record(path: Path = REFERENCE) -> None:
    """Write the reference rollouts."""
    arrays = {}
    for case in CASES:
        arrays[f"{case}_state"], arrays[f"{case}_diag"] = rollout(case)
    np.savez(path, **arrays)


if __name__ == "__main__":
    record()
