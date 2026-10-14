"""Speed stages of the curriculum: every speed-dependent parameter, from an explicit rule.

For a stage with top speed v (the free stage is sized with ``FREE_STAGE_SIZING_MPS``):

    preview horizon   H   = v^2 / (2 mu g) + v PREVIEW_TIME_S          (braking distance plus one second)
    preview points    d_i = d_1 + (i - 1) (H - d_1) / (N - 1),  d_1 = v DECISION_DT   (the first one decision ahead)
    yaw-rate scale        = min(v, sqrt(mu g R_min)) / R_min           (the largest steady yaw rate on the track)
    start speed           ~ U(START_SPEED_MIN_MPS, top speed of the previous stage)   (stage 1: its own top speed)

R_min is the radius of the tightest corner and w_max the widest half-width of the track; with them the curvature
and the widths of the preview are scaled to about [-1, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.observation import ObservationScales
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    DECISION_DT,
    FREE_STAGE_SIZING_MPS,
    G,
    MU_NOM,
    N_PREVIEW,
    PREVIEW_TIME_S,
    STAGE_TOP_SPEEDS_MPS,
    START_SPEED_MIN_MPS,
)

TRACK_SAMPLING_M: float = 0.5  # [m] grid on which R_min and w_max are measured


@dataclass(frozen=True)
class SpeedStage:
    """Parameters of one speed stage; ``top_speed`` is None for the free stage."""

    number: int
    top_speed: float | None
    sizing_speed: float
    start_speed: tuple[float, float]
    preview: NDArray[np.float64]
    scales: ObservationScales


def track_extremes(track: RealTrack) -> tuple[float, float]:
    """Return (R_min, w_max) [m]: radius of the tightest corner and widest half-width of the track."""
    s = np.arange(0.0, track.track_length, TRACK_SAMPLING_M)
    left, right = track.boundaries_many(s)
    return 1.0 / float(np.max(np.abs(track.kappa_many(s)))), float(max(np.max(left), np.max(right)))


def make_stage(number: int, track: RealTrack, mu: float = MU_NOM) -> SpeedStage:
    """Return speed stage ``number`` (1 to len(STAGE_TOP_SPEEDS_MPS)) on ``track``."""
    if not 1 <= number <= len(STAGE_TOP_SPEEDS_MPS):
        raise ValueError(f"stage {number} outside 1..{len(STAGE_TOP_SPEEDS_MPS)}")
    top_speed = STAGE_TOP_SPEEDS_MPS[number - 1]
    v = FREE_STAGE_SIZING_MPS if top_speed is None else top_speed
    previous = STAGE_TOP_SPEEDS_MPS[number - 2] if number > 1 else top_speed
    horizon = v * v / (2.0 * mu * G) + v * PREVIEW_TIME_S
    r_min, w_max = track_extremes(track)
    scales = ObservationScales(
        speed=v,
        yaw_rate=min(v, math.sqrt(mu * G * r_min)) / r_min,
        lateral=w_max,
        curvature=1.0 / r_min,
    )
    return SpeedStage(
        number=number,
        top_speed=top_speed,
        sizing_speed=v,
        start_speed=(START_SPEED_MIN_MPS, float(previous)),
        preview=np.linspace(v * DECISION_DT, horizon, N_PREVIEW),
        scales=scales,
    )
