"""Episode termination: track excursion, slip saturation and low speed."""

from __future__ import annotations

from typing import Any

from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import SLIP_TERMINATION_RATIO, V_MIN


def compute_termination(
    state: VehicleState, diag: dict[str, Any], alpha_sat: tuple[float, float], track: RealTrack,
) -> tuple[bool, dict[str, bool]]:
    """Return (terminated, event flags) for the state reached after a plant step.

    track_excursion: the lateral error exceeds the boundary distance at s.
    slip_saturation: |alpha_i| > SLIP_TERMINATION_RATIO * alpha_sat_i on either axle.
    low_speed:       vx < V_MIN.
    """
    flags = {
        "track_excursion": track.is_out_of_bounds(state.ey, state.s),
        "slip_saturation": (
            abs(diag["alpha_f"]) > SLIP_TERMINATION_RATIO * alpha_sat[0]
            or abs(diag["alpha_r"]) > SLIP_TERMINATION_RATIO * alpha_sat[1]
        ),
        "low_speed": state.vx < V_MIN,
    }
    return any(flags.values()), flags
