"""Vehicle plant: exact agreement with reference rollouts of the dynamics alone, on the Catalunya centreline, and the
behaviour of the inertial load transfer, the aerodynamic drag and the power limit.

Three manoeuvres drive ``step_plant`` directly (no environment, observation, reward or termination):
cruise and brake on the main straight, a tight corner near the grip limit, and trail braking into a corner. A
deterministic steering law on the state keeps the car near the centreline; the longitudinal command is scheduled.
Two references:
- ``tests/golden/plant_catalunya.npz``, recorded with the original plant, bit for bit equal to the rollouts of its
  first implementation: ``ORIGINAL_PLANT`` (the three changes switched off) must keep reproducing it;
- ``tests/golden/plant_catalunya_gt4.npz``, recorded with the current plant (inertial load transfer, drag and power
  of the GT4-class car): any later change to the dynamics must keep these rollouts unchanged.

``python tests/test_plant.py`` records the current reference again (only after a deliberate change of the dynamics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import GOLDEN, REPO

from contextual_bicycle_pacejka.env import sample_setup
from contextual_bicycle_pacejka.model.dynamics import (
    DIAGNOSTIC_KEYS,
    ORIGINAL_PLANT,
    PLANT,
    PlantOptions,
    compute_vehicle_rates,
    step_plant,
)
from contextual_bicycle_pacejka.model.load_transfer import compute_ax_lt, compute_vertical_loads
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    A_MAX,
    AIR_DENSITY,
    DRAG_AREA_M2,
    DT,
    G,
    HCG_NOM,
    LF,
    LR,
    MASS_NOM,
    REFERENCE_TOP_SPEED_MPS,
    WHEEL_POWER_W,
)

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"
ORIGINAL_REFERENCE = GOLDEN / "plant_catalunya.npz"
REFERENCE = GOLDEN / "plant_catalunya_gt4.npz"
STEPS = 400  # 4 s

# name: start (s [m], vx [m/s]), setup (m [kg], mu, hcg [m], tau_d [s]), longitudinal command at time t [s]
CASES = {
    "cruise_brake": ((0.0, 30.0), (1400.0, 1.0, 0.50, 0.03), lambda t: 0.3 if t < 2.0 else -0.8),
    "tight_corner": ((3430.0, 20.0), (1400.0, 0.9, 0.50, 0.03), lambda t: 0.3),
    "trail_brake": ((3350.0, 40.0), (1700.0, 1.1, 0.60, 0.02), lambda t: -0.6 if t < 1.2 else 0.1),
}
NOMINAL = sample_setup(np.random.default_rng(0))  # the nominal car, b = 0
STEER_FEEDFORWARD_M = 2.6  # [m] wheelbase: kinematic steering for the curvature
STEER_GAIN_EY, STEER_GAIN_EPSI = 0.05, 0.5  # [rad/m], [-]


def steering(state: VehicleState, track: RealTrack) -> float:
    """Deterministic test input that keeps the car near the centreline (not a controller of the project)."""
    delta = STEER_FEEDFORWARD_M * track.kappa(state.s) - STEER_GAIN_EY * state.ey - STEER_GAIN_EPSI * state.epsi
    return float(np.clip(delta, -0.5, 0.5))


def rollout(case: str, options: PlantOptions = PLANT) -> tuple[np.ndarray, np.ndarray]:
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
            state, memory, diag = step_plant(state, action, setup, memory, track, DT, options)
        except AdmissibilityError:
            break
        states.append(state.to_array())
        diags.append([diag[key] for key in DIAGNOSTIC_KEYS])
    return np.asarray(states, dtype=np.float64), np.asarray(diags, dtype=np.float64).reshape(-1, len(DIAGNOSTIC_KEYS))


@pytest.mark.parametrize("case", sorted(CASES))
def test_original_plant_matches_its_reference(case: str) -> None:
    """No regression: with the three changes switched off, the plant is the original one, bit for bit."""
    states, diags = rollout(case, ORIGINAL_PLANT)
    with np.load(ORIGINAL_REFERENCE) as reference:
        np.testing.assert_array_equal(states, reference[f"{case}_state"])
        np.testing.assert_array_equal(diags, reference[f"{case}_diag"])


@pytest.mark.parametrize("case", sorted(CASES))
def test_plant_matches_reference(case: str) -> None:
    states, diags = rollout(case)
    with np.load(REFERENCE) as reference:
        np.testing.assert_array_equal(states, reference[f"{case}_state"])
        np.testing.assert_array_equal(diags, reference[f"{case}_diag"])


def straight_line(options: PlantOptions, tau: float, steps: int) -> np.ndarray:
    """States of the car driven straight (zero steering, so vy = r = 0) from 20 m/s on the main straight."""
    track = RealTrack(str(CATALUNYA))
    setup, state, memory = NOMINAL, VehicleState(s=0.0, vx=20.0), VehicleMemory.zeros()
    states = [state.to_array()]
    for _ in range(steps):
        state, memory, _ = step_plant(state, (0.0, tau), setup, memory, track, DT, options)
        states.append(state.to_array())
    return np.asarray(states)


def test_inertial_load_transfer_changes_only_the_slides() -> None:
    """The inertial load transfer alone leaves a straight line (vy r = 0) bit for bit unchanged; in a slide that the
    original load transfer declared inadmissible (vy r = -25 m/s^2, no longitudinal force) both axles stay loaded;
    and the trail braking that stopped early with the original plant now runs its 4 s."""
    inertial_only = PlantOptions(inertial_load_transfer=True, drag_area=0.0, wheel_power=float("inf"))
    for tau in (0.6, -0.8):
        np.testing.assert_array_equal(straight_line(inertial_only, tau, 150), straight_line(ORIGINAL_PLANT, tau, 150))
    slide, setup = VehicleMemory(vy_prev=-5.0, yawrate_prev=5.0), NOMINAL
    with pytest.raises(AdmissibilityError):
        compute_vertical_loads(compute_ax_lt(slide, setup, inertial=False), setup)
    fz_f, fz_r = compute_vertical_loads(compute_ax_lt(slide, setup), setup)
    assert fz_f == pytest.approx(MASS_NOM * G * LR / (LF + LR)) and fz_r == pytest.approx(MASS_NOM * G * LF / (LF + LR))
    assert len(rollout("trail_brake", ORIGINAL_PLANT)[0]) < STEPS + 1 == len(rollout("trail_brake")[0])


def straight_rates(v: float, tau: float) -> float:
    """dvx/dt of the current plant at speed v, driven straight, with the longitudinal force of the previous step."""
    setup, state = NOMINAL, VehicleState(vx=v)
    memory = VehicleMemory.zeros()
    for _ in range(50):  # settle the load transfer on the force it produces
        ax, _, _, diag = compute_vehicle_rates(state, tau, setup, memory)
        memory = VehicleMemory(Fx_r_prev=diag["Fx_r"], Fx_f_prev=diag["Fx_f"], F_body_x_prev=diag["F_body_x"])
    return ax


def drag(v: float) -> float:
    return 0.5 * AIR_DENSITY * DRAG_AREA_M2 * v * v / MASS_NOM


def test_top_speed_is_the_power_drag_equilibrium() -> None:
    """Full throttle on a straight: the speed settles where the power at the wheels equals the drag power,
    (2 P / (rho cdA))^(1/3), the published top speed of the GT4-class car (304 km/h), within 1 %."""
    v_equilibrium = (2.0 * WHEEL_POWER_W / (AIR_DENSITY * DRAG_AREA_M2)) ** (1.0 / 3.0)
    assert v_equilibrium == pytest.approx(REFERENCE_TOP_SPEED_MPS, rel=1e-9)
    v = 20.0
    for _ in range(30_000):  # 300 s
        v += full_throttle_rate(v) * DT
    assert v == pytest.approx(v_equilibrium, rel=0.01)


def full_throttle_rate(v: float) -> float:
    """Full-throttle dvx/dt from one evaluation with the traction of the same step as memory (steady load)."""
    traction = min(MASS_NOM * A_MAX, WHEEL_POWER_W / v)
    setup, state = NOMINAL, VehicleState(vx=v)
    memory = VehicleMemory(Fx_r_prev=traction, F_body_x_prev=traction - MASS_NOM * drag(v))
    return compute_vehicle_rates(state, 1.0, setup, memory)[0]


@pytest.mark.parametrize("v", [10.0, 30.0, 60.0])
def test_acceleration_and_coast_down(v: float) -> None:
    """Full throttle gives min(A_MAX, P / (m v), friction) - drag; no throttle gives the drag deceleration, within 1 %.
    The rear friction limit is the fixed point of the load transfer of F_body_x = Fx - D (mu = 1, drag D):
    Fx = (m g LF - D hcg) / (L - hcg)."""
    traction = min(MASS_NOM * A_MAX, WHEEL_POWER_W / v)
    friction = (MASS_NOM * G * LF - MASS_NOM * drag(v) * HCG_NOM) / (LF + LR - HCG_NOM)
    expected = min(traction, friction) / MASS_NOM - drag(v)
    assert straight_rates(v, 1.0) == pytest.approx(expected, rel=0.01)
    assert straight_rates(v, 0.0) == pytest.approx(-drag(v), rel=0.01)


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
    """Write the reference rollouts of the current plant."""
    arrays = {}
    for case in CASES:
        arrays[f"{case}_state"], arrays[f"{case}_diag"] = rollout(case)
    np.savez(path, **arrays)


if __name__ == "__main__":
    record()
