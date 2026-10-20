"""Speed stages of the curriculum: every speed-dependent parameter, from an explicit rule.

For a stage with top speed v (the free stage is sized with ``FREE_STAGE_SIZING_MPS``):

    preview horizon   H   = v^2 / (2 mu g) + v PREVIEW_TIME_S          (braking distance plus one second)
    preview points    d_i = d_1 + (H - d_1) ((i - 1) / (N - 1))^2,  d_1 = v DECISION_DT   (dense near the car)
    yaw-rate scale        = min(v, sqrt(mu g R_min)) / R_min           (the largest steady yaw rate on the track)
    start speed at s      ~ U(START_SPEED_MIN_MPS, max(START_SPEED_MIN_MPS, START_SPEED_FRACTION v_pm(s)))

v_pm is the point-mass speed profile with the top speed of the stage (``model/point_mass.py``): a training episode
starts at a speed from which the car can still take the corners ahead. The top speed of a stage can be changed while
the observation scales and the preview stay those of its sizing speed (``with_top_speed``), for a curriculum that
raises the top speed during training; the progress reward is normalised by the current top speed (by the sizing
speed when there is none).

R_min is the radius of the tightest corner and w_max the widest half-width of the track; with them the curvature
and the widths of the preview are scaled to about [-1, 1].
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.observation import ObservationScales
from contextual_bicycle_pacejka.model.point_mass import speed_profile
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    DECISION_DT,
    FREE_STAGE_SIZING_MPS,
    G,
    MU_NOM,
    N_PREVIEW,
    PREVIEW_TIME_S,
    STAGE_TOP_SPEEDS_MPS,
    START_SPEED_FRACTION,
    START_SPEED_MIN_MPS,
)

TRACK_SAMPLING_M: float = 0.5  # [m] grid on which R_min and w_max are measured


@dataclass(frozen=True)
class SpeedStage:
    """Parameters of one speed stage; ``top_speed`` is None for the free stage."""

    number: int
    top_speed: float | None
    sizing_speed: float
    start_s: NDArray[np.float64]  # [m] positions of the start-speed bound
    start_speed_max: NDArray[np.float64]  # [m/s] fastest start at each position
    preview: NDArray[np.float64]
    scales: ObservationScales

    @property
    def reward_speed(self) -> float:
        """Speed [m/s] that normalises the progress reward: the top speed, or the sizing speed without one."""
        return self.sizing_speed if self.top_speed is None else self.top_speed

    def start_speed_bound(self, s: float, track_length: float) -> float:
        """Return the fastest start speed [m/s] at position s."""
        return float(np.interp(s % track_length, self.start_s, self.start_speed_max, period=track_length))


def start_speeds(track: RealTrack, top_speed: float | None, mu: float = MU_NOM) -> tuple[NDArray, NDArray]:
    """Return the positions and the fastest start speeds START_SPEED_FRACTION v_pm (at least the slowest start)."""
    s, v = speed_profile(track, top_speed, mu)
    return s, np.maximum(START_SPEED_MIN_MPS, START_SPEED_FRACTION * v)


def with_top_speed(stage: SpeedStage, top_speed: float | None, track: RealTrack, mu: float = MU_NOM) -> SpeedStage:
    """Return ``stage`` with another top speed: new start speeds, same sizing, preview and observation scales."""
    if top_speed is not None and top_speed > stage.sizing_speed:
        raise ValueError(f"top speed {top_speed} above the sizing speed {stage.sizing_speed} of the stage")
    start_s, start_max = start_speeds(track, top_speed, mu)
    return dataclasses.replace(stage, top_speed=top_speed, start_s=start_s, start_speed_max=start_max)


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
    horizon = v * v / (2.0 * mu * G) + v * PREVIEW_TIME_S
    r_min, w_max = track_extremes(track)
    scales = ObservationScales(
        speed=v,
        yaw_rate=min(v, math.sqrt(mu * G * r_min)) / r_min,
        lateral=w_max,
        curvature=1.0 / r_min,
    )
    start_s, start_max = start_speeds(track, top_speed, mu)
    first = v * DECISION_DT
    return SpeedStage(
        number=number,
        top_speed=top_speed,
        sizing_speed=v,
        start_s=start_s,
        start_speed_max=start_max,
        preview=first + (horizon - first) * np.linspace(0.0, 1.0, N_PREVIEW) ** 2,
        scales=scales,
    )
