"""Closed race track loaded from a racing-line NPZ file.

The NPZ stores the reference path (the racing line) as arc length ``s_m``, world coordinates ``x_m, y_m``,
heading ``psi_rad``, signed curvature ``kappa_radpm``, the speed profile ``v_ref_mps`` and its
acceleration ``ax_mps2``, the boundary distances ``w_tr_right_m, w_tr_left_m`` and the lap length
``track_length``. All quantities are interpolated with periodic splines, so queries wrap over the lap.

Boundary distances are floored at ``boundary_width_floor_m`` when the file provides it, otherwise at 5 m;
in the second case the widths are splined periodically, in the first with a monotone PCHIP interpolant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator

DEFAULT_WIDTH_FLOOR_M: float = 5.0  # [m]


class RealTrack:
    """Periodic-spline representation of a closed racing line.

    Parameters
    ----------
    npz_path : str
        Racing-line NPZ file (see the module docstring for the required keys).
    """

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        s_m = np.asarray(data["s_m"], dtype=np.float64)
        psi = np.asarray(data["psi_rad"], dtype=np.float64)
        self._length = float(data["track_length"])
        has_floor = "boundary_width_floor_m" in data
        self._width_floor = float(data["boundary_width_floor_m"]) if has_floor else DEFAULT_WIDTH_FLOOR_M

        # Periodic splines need a closing knot equal to the first sample, placed at s = track_length.
        s_full = np.concatenate([s_m, [self._length]])

        def periodic(values: NDArray[np.float64]) -> CubicSpline:
            closed = np.concatenate([values, [values[0]]])
            return CubicSpline(s_full, closed, bc_type="periodic", extrapolate="periodic")

        def width(values: NDArray[np.float64]) -> CubicSpline | PchipInterpolator:
            closed = np.concatenate([values, [values[0]]])
            if has_floor:
                return PchipInterpolator(s_full, closed)
            return CubicSpline(s_full, closed, bc_type="periodic", extrapolate="periodic")

        self._x = periodic(np.asarray(data["x_m"], dtype=np.float64))
        self._y = periodic(np.asarray(data["y_m"], dtype=np.float64))
        self._kappa = periodic(np.asarray(data["kappa_radpm"], dtype=np.float64))
        self._v_ref = periodic(np.asarray(data["v_ref_mps"], dtype=np.float64))
        self._ax_ref = periodic(np.asarray(data["ax_mps2"], dtype=np.float64))
        self._w_right = width(np.asarray(data["w_tr_right_m"], dtype=np.float64))
        self._w_left = width(np.asarray(data["w_tr_left_m"], dtype=np.float64))

        # The heading grows by about +/- 2 pi per lap: spline its periodic part and add the mean trend back.
        self._psi_trend = float((psi[-1] - psi[0]) / s_m[-1])  # [rad/m]
        self._psi = periodic(psi - self._psi_trend * s_m)

    @property
    def track_length(self) -> float:
        """Lap length [m]."""
        return self._length

    def kappa(self, s: float) -> float:
        """Signed curvature [1/m] at arc length s."""
        return float(self._kappa(s % self._length))

    def centerline_point(self, s: float) -> tuple[float, float]:
        """World coordinates (x, y) [m] of the reference path at arc length s."""
        s_w = s % self._length
        return float(self._x(s_w)), float(self._y(s_w))

    def psi_ref(self, s: float) -> float:
        """Heading [rad] of the reference path, continuous across laps."""
        s_w = s % self._length
        n_laps = (s - s_w) / self._length
        within_lap = float(self._psi(s_w)) + self._psi_trend * s_w
        return within_lap + self._psi_trend * self._length * n_laps

    def boundary_right(self, s: float) -> float:
        """Distance [m] from the reference path to the right boundary (floored)."""
        return max(float(self._w_right(s % self._length)), self._width_floor)

    def boundary_left(self, s: float) -> float:
        """Distance [m] from the reference path to the left boundary (floored)."""
        return max(float(self._w_left(s % self._length)), self._width_floor)

    def ey_max(self, s: float) -> float:
        """Smaller of the two boundary distances [m]."""
        return min(self.boundary_right(s), self.boundary_left(s))

    def is_out_of_bounds(self, ey: float, s: float) -> bool:
        """Whether lateral error ey [m] (positive = left) is beyond the boundary on its own side at s."""
        if ey >= 0.0:
            return float(ey) > self.boundary_left(s)
        return abs(float(ey)) > self.boundary_right(s)

    def v_ref(self, s: float) -> float:
        """Reference speed [m/s] at arc length s (computed at mu = VREF_REFERENCE_MU)."""
        return float(self._v_ref(s % self._length))

    def ax_ref(self, s: float) -> float:
        """Reference longitudinal acceleration [m/s^2] at arc length s."""
        return float(self._ax_ref(float(s % self._length)))
