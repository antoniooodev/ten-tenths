"""Balance index b: neutral at b = 0, constant static lateral capacity, lateral-only scaling, ordered handling.

The reference rollouts of ``test_plant.py`` pin the plant at b = 0 bit for bit; these tests pin what b changes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from contextual_bicycle_pacejka.env import sample_setup
from contextual_bicycle_pacejka.evaluation.balance import (
    CRITERION_GRID,
    RADIUS_M,
    balance_criterion,
    linear_understeer_gradient,
)
from contextual_bicycle_pacejka.model.dynamics import compute_vehicle_rates
from contextual_bicycle_pacejka.model.handling import (
    critical_speed,
    lateral_limit,
    saturating_axle,
    understeer_gradient,
)
from contextual_bicycle_pacejka.model.load_transfer import compute_vertical_loads
from contextual_bicycle_pacejka.model.state import VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.tires import compute_lateral_scales
from contextual_bicycle_pacejka.spec.constants import B_MAX, B_MIN, G, HCG_NOM, LF, LR, MASS_NOM, MU_NOM, TAU_D_NOM
from contextual_bicycle_pacejka.spec.layout import validate_setup

GRID = (-0.40, -0.20, -0.05, 0.05, 0.10, 0.20)


def setup_with(b: float):
    return sample_setup(np.random.default_rng(0), (b, b))


def test_nominal_balance_is_the_identity() -> None:
    """At b = 0 both scales are exactly 1.0, so every lateral capacity is unchanged bit for bit."""
    assert compute_lateral_scales(0.0) == (1.0, 1.0)
    assert compute_lateral_scales(-0.0) == (1.0, 1.0)


def test_sampled_setup_is_nominal_except_b() -> None:
    rng = np.random.default_rng(0)
    nominal = sample_setup(rng)
    assert (nominal.m, nominal.mu, nominal.hcg, nominal.tau_d, nominal.b) == (MASS_NOM, MU_NOM, HCG_NOM, TAU_D_NOM, 0.0)
    draws = [sample_setup(rng, (-0.2, 0.1)) for _ in range(200)]
    assert all(d.m == MASS_NOM and d.mu == MU_NOM and -0.2 <= d.b <= 0.1 for d in draws)


@pytest.mark.parametrize("b", GRID)
def test_static_lateral_capacity_is_constant(b: float) -> None:
    """lambda_f Fz_f0 + lambda_r Fz_r0 = Fz_f0 + Fz_r0 for every b (static loads, no load transfer)."""
    fz_f0, fz_r0 = compute_vertical_loads(0.0, setup_with(b))
    lambda_f, lambda_r = compute_lateral_scales(b)
    assert lambda_f * fz_f0 + lambda_r * fz_r0 == pytest.approx(fz_f0 + fz_r0, rel=1e-12)
    assert fz_f0 / fz_r0 == pytest.approx(LR / LF, rel=1e-12)


def test_balance_index_outside_its_domain_is_rejected() -> None:
    for b in (B_MIN, B_MAX, 1.0):
        with pytest.raises(ValueError):
            compute_lateral_scales(b)
        with pytest.raises(ValueError):
            validate_setup(setup_with(0.0).to_array() + np.array([0.0, 0.0, 0.0, 0.0, b]))


@pytest.mark.parametrize("b", GRID)
def test_b_scales_only_the_lateral_forces(b: float) -> None:
    """While braking in a corner, b multiplies each lateral force by its scale and leaves the longitudinal forces."""
    state = VehicleState(vx=30.0, vy=-0.4, yaw_rate=0.2, delta=0.05)
    _, _, _, nominal = compute_vehicle_rates(state, -0.5, setup_with(0.0), VehicleMemory.zeros())
    _, _, _, scaled = compute_vehicle_rates(state, -0.5, setup_with(b), VehicleMemory.zeros())
    lambda_f, lambda_r = compute_lateral_scales(b)
    assert scaled["Fy_f"] == pytest.approx(lambda_f * nominal["Fy_f"], rel=1e-12)
    assert scaled["Fy_r"] == pytest.approx(lambda_r * nominal["Fy_r"], rel=1e-12)
    assert (scaled["Fx_f"], scaled["Fx_r"]) == (nominal["Fx_f"], nominal["Fx_r"])


def test_understeer_gradient_decreases_with_b() -> None:
    """Steady state on the plant: K strictly decreasing in b; at the limit the front saturates first for clearly
    negative b, the rear for positive b."""
    ay_grid = np.linspace(0.1, 0.4, 7) * G
    gradients = [understeer_gradient(setup_with(b), RADIUS_M, ay_grid) for b in GRID]
    assert all(later < earlier for earlier, later in itertools.pairwise(gradients))
    for b in GRID:
        setup = setup_with(b)
        assert saturating_axle(lateral_limit(setup, RADIUS_M, 0.1 * G, 50), setup) == ("front" if b < 0 else "rear")


@pytest.mark.parametrize("b", (0.05, 0.10, 0.20))
def test_critical_speed_matches_the_linear_bicycle(b: float) -> None:
    """On the straight line the plant is the linear bicycle with cornering stiffness B_i C_i lambda_i mu Fz_i0."""
    expected = np.sqrt((LF + LR) / -linear_understeer_gradient(b))
    assert critical_speed(setup_with(b), 2.0, 200.0) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(("switch_at", "extra", "passed"), [
    (-0.01, None, True),  # the measured case: front up to -0.02, rear from -0.01
    (0.05, None, False),  # switch outside +-0.02
    (-0.01, 0.10, False),  # a second switch, back to the front
])
def test_saturating_axle_criterion(switch_at: float, extra: float | None, passed: bool) -> None:
    """Criterion 2: one switch of the saturating axle, from front to rear, between two values of b within +-0.02."""
    rows = [{"b": b, "K_deg_per_g": -b, "saturating_axle": "front" if b < switch_at or b == extra else "rear",
             "ay_max_g": 1.0, "critical_speed_mps": None} for b in CRITERION_GRID]
    assert balance_criterion(rows)["saturating_axle"]["passed"] is passed
