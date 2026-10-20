"""Laps of a policy from a fixed set of starts, and a point-mass lap time as a yardstick.

Protocol of a speed stage: ``LAP_STARTS`` starts equally spaced along the lap, on the centreline, aligned with it,
at ``START_SPEED_MPS``, with the nominal car. The policy drives one lap (from s_0 to s_0 + L) or until the first
failure; a lap slower than ``TIMEOUT_FACTOR`` L / v_top fails as ``timeout``. The lap time includes the start and is
interpolated within the last decision.

The point-mass lap time drives the centreline at the largest speed the friction ellipse and the powertrain allow
(``model/point_mass.py``: traction and braking limits of the plant with drag and power; front brake share 0.65, so
the rear axle locks first) and the top speed of the stage. ``lateral_ratio`` scales the lateral limit, for a balance
index b with a lower cornering limit (a_y,max(b) / a_y,max(0)). No policy uses it; it only scales lap times across
stages and values of b.

An observer passed to ``drive_lap`` is called with the environment after the reset and after every decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.env.stages import SpeedStage
from contextual_bicycle_pacejka.model.dynamics import PLANT, PlantOptions
from contextual_bicycle_pacejka.model.point_mass import longitudinal_limits, profile_lap_time, speed_profile
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import DECISION_DT, MU_NOM
from contextual_bicycle_pacejka.spec.layout import OBS

LAP_STARTS: int = 10
START_SPEED_MPS: float = 3.0  # [m/s]
TIMEOUT_FACTOR: float = 4.0  # [-] a lap longer than TIMEOUT_FACTOR L / v_top fails

Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]
__all__ = ["LAP_STARTS", "LapRecord", "drive_lap", "longitudinal_limits", "point_mass_lap_time", "protocol_starts"]


@dataclass(frozen=True)
class LapRecord:
    """Outcome of one lap: lap time [s] if completed, else the failure and the arc length [m] where it happened."""

    start_s: float
    completed: bool
    lap_time: float | None
    failure: str | None
    failure_s: float | None
    decisions: int
    yaw_rate_p99: float  # [-] 99th percentile of |observed yaw rate| (in its stage scale)


def protocol_starts(track: RealTrack) -> list[float]:
    """Return the ``LAP_STARTS`` start positions [m], equally spaced from s = 0."""
    return [i * track.track_length / LAP_STARTS for i in range(LAP_STARTS)]


def drive_lap(
    policy: Policy, env: RacingEnv, start_s: float, timeout_s: float | None = None,
    observer: Callable[[RacingEnv], None] | None = None,
) -> LapRecord:
    """Drive one lap of ``env`` (an unwrapped environment, no time limit) from ``start_s`` with ``policy``."""
    length = env.track.track_length
    if timeout_s is None:
        timeout_s = TIMEOUT_FACTOR * length / env.stage.sizing_speed
    obs, _ = env.reset(options={"initial_s": start_s, "initial_vx": START_SPEED_MPS})
    if observer is not None:
        observer(env)
    finish = start_s + length
    yaw = [abs(float(obs[OBS.YAW_RATE]))]
    decisions = 0
    while decisions * DECISION_DT < timeout_s:
        s_before = env.state.s
        obs, _, terminated, _, info = env.step(policy(obs))
        decisions += 1
        if observer is not None and not terminated:
            observer(env)
        yaw.append(abs(float(obs[OBS.YAW_RATE])))
        if terminated:
            return LapRecord(start_s, False, None, info["failure"], env.state.s % length, decisions,
                             float(np.percentile(yaw, 99)))
        if env.state.s >= finish:
            fraction = (finish - s_before) / (env.state.s - s_before)
            return LapRecord(start_s, True, (decisions - 1 + fraction) * DECISION_DT, None, None, decisions,
                             float(np.percentile(yaw, 99)))
    return LapRecord(start_s, False, None, "timeout", env.state.s % length, decisions, float(np.percentile(yaw, 99)))


def point_mass_lap_time(
    track: RealTrack, stage: SpeedStage, mu: float = MU_NOM, lateral_ratio: float = 1.0, options: PlantOptions = PLANT,
) -> float:
    """Return the point-mass lap time [s] of a flying lap at the top speed of ``stage`` (none for the free stage),
    with the lateral limit mu g scaled by ``lateral_ratio`` (``model/point_mass.py``)."""
    s, v = speed_profile(track, stage.top_speed, mu, lateral_ratio, options)
    return profile_lap_time(s, v, track.track_length)
