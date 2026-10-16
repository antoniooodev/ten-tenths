"""Failures that end an episode, checked after every plant step.

    off_track:     the centre of the car beyond the track edge on its side (|e_y| > half-width at s), or beyond an
                   edge moved inwards by a margin (training only: TRAINING_EDGE_MARGIN_M)
    spin:          |e_psi| > SPIN_EPSI_RAD
    stall:         vx < STALL_SPEED_MPS
    inadmissible:  the plant raised ``AdmissibilityError`` (set by the environment)
"""

from __future__ import annotations

from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import SPIN_EPSI_RAD, STALL_SPEED_MPS

FAILURES: tuple[str, ...] = ("off_track", "spin", "stall", "inadmissible")


def check_failure(state: VehicleState, track: RealTrack, edge_margin: float = 0.0) -> str | None:
    """Return the first failure of ``FAILURES`` that ``state`` shows, or None; ``edge_margin`` [m] moves the edges
    inwards."""
    if track.is_out_of_bounds(state.ey, state.s, edge_margin):
        return "off_track"
    if abs(state.epsi) > SPIN_EPSI_RAD:
        return "spin"
    if state.vx < STALL_SPEED_MPS:
        return "stall"
    return None
