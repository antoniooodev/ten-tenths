"""Corner and tyre measurements of a protocol lap.

The windows are the corners of Catalunya whose steady-state limit speed falls below 20 m/s for some balance index of
the evaluation grid (lateral limit 0.774 g at b = +0.20 down to 1 g at b = 0). For every window the lap drives from
its start to its end, a lap records:

    entry_speed, exit_speed   vx [m/s] at the first decision past the start and past the end of the window
    min_speed, min_speed_s    the lowest vx in the window and where [m, position along the lap]
    braking_s                 the first position, from BRAKING_LOOKBACK_M before the window to its end, where the
                              longitudinal command is negative (None: no braking)
    peak_steer                the largest |delta| [rad] in the window
    peak_slip_front/rear      the largest |alpha| / alpha_sat of each axle in the window

The recorder observes the environment once per decision (about 1 m apart at 20 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold, compute_slip_angles

CORNER_WINDOWS_M: tuple[tuple[float, float], ...] = (
    (821.0, 863.0), (1702.0, 1723.0), (2097.0, 2171.0), (2538.0, 2577.0), (3465.0, 3527.0),
    (3741.0, 3754.0), (3998.0, 4032.0), (4128.0, 4162.0), (4180.0, 4211.0),
)
BRAKING_LOOKBACK_M: float = 300.0  # [m] how far before a window the braking point is searched


@dataclass
class CornerRecorder:
    """Observer of ``drive_lap``: stores one sample per decision, then measures the windows the lap drove through."""

    windows: tuple[tuple[float, float], ...] = CORNER_WINDOWS_M
    samples: list[tuple[float, ...]] = field(default_factory=list)
    saturation: tuple[float, float] | None = None  # (front, rear) alpha_sat of the lap's setup

    def __call__(self, env: RacingEnv) -> None:
        if self.saturation is None:
            self.saturation = (compute_saturation_threshold(env.setup, "front"),
                               compute_saturation_threshold(env.setup, "rear"))
        state = env.state
        alpha_f, alpha_r = compute_slip_angles(state)
        self.samples.append((state.s, state.vx, env.tau, abs(state.delta),
                             abs(alpha_f) / self.saturation[0], abs(alpha_r) / self.saturation[1]))

    def corners(self, track_length: float) -> list[dict[str, Any]]:
        """Return the measurements of every window driven from its start to its end (positions along the lap)."""
        if not self.samples:
            return []
        s, vx, tau, steer, slip_f, slip_r = (np.array(column) for column in zip(*self.samples, strict=True))
        records = []
        for start, end in self.windows:
            for lap in range(int(s[0] // track_length) - 1, int(s[-1] // track_length) + 1):
                a, b = start + lap * track_length, end + lap * track_length
                if s[0] > a or s[-1] < b:
                    continue  # the lap did not drive through the whole window
                inside = (s >= a) & (s <= b)
                if not np.any(inside):
                    continue
                approach = (s >= a - BRAKING_LOOKBACK_M) & (s <= b) & (tau < 0.0)
                low = int(np.flatnonzero(inside)[np.argmin(vx[inside])])
                records.append({
                    "window": [start, end],
                    "entry_speed": float(vx[np.argmax(s >= a)]),
                    "exit_speed": float(vx[np.argmax(s >= b)]),
                    "min_speed": float(vx[low]),
                    "min_speed_s": float(s[low] % track_length),
                    "braking_s": float(s[np.argmax(approach)] % track_length) if np.any(approach) else None,
                    "peak_steer": float(steer[inside].max()),
                    "peak_slip_front": float(slip_f[inside].max()),
                    "peak_slip_rear": float(slip_r[inside].max()),
                })
        return records
