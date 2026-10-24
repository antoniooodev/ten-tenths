"""Minimum-time lap reference: its equations are the plant's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

ca = pytest.importorskip("casadi")

from contextual_bicycle_pacejka.env import sample_setup  # noqa: E402
from contextual_bicycle_pacejka.model.dynamics import compute_frenet_rates, compute_vehicle_rates  # noqa: E402
from contextual_bicycle_pacejka.model.state import VehicleMemory, VehicleState  # noqa: E402
from contextual_bicycle_pacejka.model.track import RealTrack  # noqa: E402
from contextual_bicycle_pacejka.reference.min_time import corridor, vehicle_rates  # noqa: E402
from contextual_bicycle_pacejka.spec.constants import MASS_NOM, TAU_D_NOM  # noqa: E402

TRACK = RealTrack(str(REPO / "data/tracks/catalunya/reference.npz"))


@pytest.mark.parametrize("b", [0.0, -0.15, 0.12])
def test_equations_match_the_plant(b: float) -> None:
    """At random states (forces within friction, away from the throttle/brake and power kinks), the exact equations
    of the reference give the plant's rates and forces, with the plant's memory holding the same F_body_x."""
    rng = np.random.default_rng(1)
    setup = sample_setup(rng, (b, b))
    for _ in range(40):
        s = rng.uniform(0.0, TRACK.track_length)
        vx = rng.uniform(8.0, 65.0)
        x = [rng.uniform(-3.0, 3.0), rng.uniform(-0.1, 0.1), vx, rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3),
             rng.uniform(-0.1, 0.1), rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.4)]
        ax_lt = rng.uniform(-6.0, 4.0)
        kappa = TRACK.kappa(s)
        ref = vehicle_rates(x, 0.07, 1.3, ax_lt, kappa, b, smooth=False)
        state = VehicleState(s=s, ey=x[0], epsi=x[1], vx=x[2], vy=x[3], yaw_rate=x[4], delta=x[5])
        ax, ay, yaw_acc, diag = compute_vehicle_rates(state, x[6], setup, VehicleMemory(F_body_x_prev=MASS_NOM * ax_lt))
        s_dot, ey_dot, epsi_dot = compute_frenet_rates(state, TRACK)
        expected = [ey_dot, epsi_dot, ax, ay, yaw_acc, (0.07 - x[5]) / TAU_D_NOM, 1.3]
        np.testing.assert_allclose([float(v) for v in ref["rates"]], expected, rtol=1e-9, atol=1e-9)
        assert float(ref["s_dot"]) == pytest.approx(s_dot, rel=1e-12)
        for key in ("Fx_f", "Fx_r", "Fy_f", "Fy_r", "Fz_f", "Fz_r", "F_body_x"):
            name = key[0].lower() + key[1:] if key != "F_body_x" else "f_body_x"
            assert float(ref[name]) == pytest.approx(diag[key], rel=1e-9, abs=1e-6), key


def test_smoothing_is_small() -> None:
    """The smoothed equations differ from the exact ones by less than 0.1 % of the forces away from the kinks."""
    x = [0.5, 0.02, 40.0, 0.3, 0.1, 0.03, 0.6]
    exact = vehicle_rates(x, 0.03, 0.0, 1.0, 0.01, smooth=False)
    smooth = vehicle_rates(x, 0.03, 0.0, 1.0, 0.01, smooth=True)
    assert float(smooth["fx_r"]) == pytest.approx(float(exact["fx_r"]), rel=1e-3)
    assert float(smooth["rates"][2]) == pytest.approx(float(exact["rates"][2]), rel=1e-3)


def test_corridor_moves_both_edges_inwards() -> None:
    """The edge margin narrows the track on both sides, and a margin that closes it somewhere is refused."""
    left, right = TRACK.boundaries_many(np.linspace(0.0, TRACK.track_length, 500, endpoint=False))
    narrow_left, narrow_right = corridor(left, right, 1.5)
    np.testing.assert_allclose(narrow_left, left - 1.5)
    np.testing.assert_allclose(narrow_right, right - 1.5)
    np.testing.assert_array_equal(corridor(left, right, 0.0)[0], left)
    with pytest.raises(ValueError):
        corridor(left, right, float(min(left.min(), right.min())))
    with pytest.raises(ValueError):
        corridor(left, right, -0.1)
