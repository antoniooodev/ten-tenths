"""Steady-state check of the balance index b, with a criterion fixed before the measurement.

Measurements on the plant (``model/handling.py``), at the nominal setup, for every b of a grid:

* understeer gradient K on a 100 m radius, fitted over ay = 0.1 ... 0.4 g;
* lateral limit ay_max on the same radius, and the axle that saturates first;
* critical speed of the straight line (none for an understeering car).

Criterion:

1. K strictly decreasing in b over the criterion grid b = -0.40 : 0.05 : +0.20, plus +-0.01 and +-0.02;
2. along the criterion grid, the axle that saturates first at the lateral limit changes exactly once, from front to
   rear, between two values of b within +-0.02 (the rear-wheel drive puts the neutral point slightly below b = 0:
   the traction that holds the speed in a steady corner takes about 3 % of the rear lateral capacity, which moves
   the neutral point to b = -0.012);
3. the plant unchanged at b = 0: ``tests/test_plant.py`` (reference rollouts) and ``tests/test_balance.py``.

Drivable interval of a speed stage: the contiguous values of the measurement grid (step 0.01) around b = 0 with

    G1  ay_max(b) >= 0.80 ay_max(0)     and     G2  critical speed >= top speed of the stage.

The linear prediction of K (cornering stiffness B_i C_i lambda_i mu Fz_i0) is reported and does not decide.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
from contextual_bicycle_pacejka.model.handling import (
    critical_speed,
    lateral_limit,
    saturating_axle,
    understeer_gradient,
)
from contextual_bicycle_pacejka.model.tires import (
    compute_lateral_scales,
    compute_saturation_threshold,
    pacejka_coefficients,
)
from contextual_bicycle_pacejka.spec.constants import G, LF, LR, MU_NOM

RADIUS_M: float = 100.0  # [m] radius of the steady-state tests
GRADIENT_AY_G: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)  # [g] fit range of K
LIMIT_AY_START_G: float = 0.10  # [g] start of the lateral-limit continuation
LIMIT_STEPS: int = 200  # [-] continuation steps of the lateral limit
STRAIGHT_SPEED_RANGE_MPS: tuple[float, float] = (2.0, 200.0)  # [m/s] search range of the critical speed
CRITERION_GRID: tuple[float, ...] = (
    -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
)
MEASUREMENT_GRID: tuple[float, ...] = tuple(round(k / 100.0, 2) for k in range(-40, 21))
SWITCH_BAND_B: float = 0.02  # [-] criterion 2: largest |b| of the two values around the switch of saturating axle
GRIP_RATIO_MIN: float = 0.80  # [-] G1: least lateral limit, as a share of the limit at b = 0
STAGE_TOP_SPEEDS_MPS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0)  # [m/s] last: no bound, about 80


def linear_understeer_gradient(b: float, mu: float = MU_NOM) -> float:
    """Return the linear prediction of K [rad / (m/s^2)] for b: (1 / (mu g)) (1 / (B_f C_f lambda_f) -
    1 / (B_r C_r lambda_r))."""
    b_f, c_f, _ = pacejka_coefficients("front")
    b_r, c_r, _ = pacejka_coefficients("rear")
    lambda_f, lambda_r = compute_lateral_scales(b)
    return (1.0 / (b_f * c_f * lambda_f) - 1.0 / (b_r * c_r * lambda_r)) / (mu * G)


def measure_balance(b: float) -> dict[str, Any]:
    """Return the steady-state measurements of the nominal setup with balance index b."""
    setup = sample_setup(np.random.default_rng(0), (b, b))
    k = understeer_gradient(setup, RADIUS_M, np.asarray(GRADIENT_AY_G) * G)
    limit = lateral_limit(setup, RADIUS_M, LIMIT_AY_START_G * G, LIMIT_STEPS)
    v_crit = critical_speed(setup, *STRAIGHT_SPEED_RANGE_MPS)
    k_linear = linear_understeer_gradient(b, setup.mu)
    return {
        "b": b,
        "K_deg_per_g": math.degrees(k * G),
        "K_linear_deg_per_g": math.degrees(k_linear * G),
        "ay_max_g": limit.ay / G,
        "saturating_axle": saturating_axle(limit, setup),
        "slip_ratio_front": abs(limit.alpha_f) / compute_saturation_threshold(setup, "front"),
        "slip_ratio_rear": abs(limit.alpha_r) / compute_saturation_threshold(setup, "rear"),
        "critical_speed_mps": None if math.isinf(v_crit) else v_crit,
        "critical_speed_linear_mps": math.sqrt((LF + LR) / -k_linear) if k_linear < 0.0 else None,
    }


def drivable_interval(rows: list[dict[str, Any]], top_speed: float) -> tuple[float, float]:
    """Return the contiguous [lo, hi] of the grid around b = 0 that satisfies G1 and G2 at ``top_speed`` [m/s]."""
    rows = sorted(rows, key=lambda row: row["b"])
    nominal = next(i for i, row in enumerate(rows) if row["b"] == 0.0)
    ay_nominal = rows[nominal]["ay_max_g"]

    def drivable(row: dict[str, Any]) -> bool:
        grip = row["ay_max_g"] >= GRIP_RATIO_MIN * ay_nominal
        stable = row["critical_speed_mps"] is None or row["critical_speed_mps"] >= top_speed
        return grip and stable

    lo = hi = nominal
    while lo > 0 and drivable(rows[lo - 1]):
        lo -= 1
    while hi < len(rows) - 1 and drivable(rows[hi + 1]):
        hi += 1
    return rows[lo]["b"], rows[hi]["b"]


def balance_criterion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the criterion of the module docstring to the measurements of ``measure_balance``."""
    by_b = {row["b"]: row for row in rows}
    grid = [by_b[b] for b in CRITERION_GRID]
    increases = [(a["b"], c["b"]) for a, c in itertools.pairwise(grid) if not c["K_deg_per_g"] < a["K_deg_per_g"]]
    switches = [(a["b"], c["b"], a["saturating_axle"], c["saturating_axle"]) for a, c in itertools.pairwise(grid)
                if a["saturating_axle"] != c["saturating_axle"]]
    single_switch = (len(switches) == 1 and switches[0][2:] == ("front", "rear")
                     and max(abs(switches[0][0]), abs(switches[0][1])) <= SWITCH_BAND_B)
    return {
        "monotonic_K": {"passed": not increases, "non_decreasing_pairs": increases},
        "saturating_axle": {"passed": single_switch, "switches": [list(sw[:2]) for sw in switches]},
        "drivable": {f"{v:g}": list(drivable_interval(rows, v)) for v in STAGE_TOP_SPEEDS_MPS},
    }


def check_balance_index() -> dict[str, Any]:
    """Measure every b of the measurement grid and apply the criterion."""
    rows = [measure_balance(b) for b in MEASUREMENT_GRID]
    return {"rows": rows, "criterion": balance_criterion(rows)}
