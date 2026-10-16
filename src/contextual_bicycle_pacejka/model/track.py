"""Closed race track loaded from the centreline reference NPZ of a circuit (``tracks/reference.py``).

The NPZ stores the centreline, the reference path of the Frenet frame, as arc length ``s_m``, world coordinates
``x_m, y_m``, heading ``psi_rad``, signed curvature ``kappa_radpm``, the distances to the track edges
``w_tr_right_m, w_tr_left_m`` and the lap length ``track_length``. All quantities are interpolated with periodic
splines, so queries wrap over the lap. The edge distances are the circuit's own: the car leaves the track when its
lateral offset from the centreline exceeds them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


class RealTrack:
    """Periodic-spline representation of a closed centreline.

    Parameters
    ----------
    npz_path : str
        Centreline reference NPZ file (see the module docstring for the required keys).
    """

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        s_m = np.asarray(data["s_m"], dtype=np.float64)
        psi = np.asarray(data["psi_rad"], dtype=np.float64)
        self._length = float(data["track_length"])

        # Periodic splines need a closing knot equal to the first sample, placed at s = track_length.
        s_full = np.concatenate([s_m, [self._length]])

        def periodic(values: NDArray[np.float64]) -> CubicSpline:
            closed = np.concatenate([values, [values[0]]])
            return CubicSpline(s_full, closed, bc_type="periodic", extrapolate="periodic")

        self._x = periodic(np.asarray(data["x_m"], dtype=np.float64))
        self._y = periodic(np.asarray(data["y_m"], dtype=np.float64))
        self._kappa = periodic(np.asarray(data["kappa_radpm"], dtype=np.float64))
        self._w_right = periodic(np.asarray(data["w_tr_right_m"], dtype=np.float64))
        self._w_left = periodic(np.asarray(data["w_tr_left_m"], dtype=np.float64))

        # Integral of the curvature from s = 0, within one lap; the turning of a whole lap is added per lap.
        self._kappa_integral = self._kappa.antiderivative()
        self._turning_per_lap = float(self._kappa_integral(self._length) - self._kappa_integral(0.0))  # [rad]

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
        """World coordinates (x, y) [m] of the centreline at arc length s."""
        s_w = s % self._length
        return float(self._x(s_w)), float(self._y(s_w))

    def psi_ref(self, s: float) -> float:
        """Heading [rad] of the centreline, continuous across laps."""
        s_w = s % self._length
        n_laps = (s - s_w) / self._length
        within_lap = float(self._psi(s_w)) + self._psi_trend * s_w
        return within_lap + self._psi_trend * self._length * n_laps

    def kappa_many(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vector form of :meth:`kappa` [1/m]."""
        return self._kappa(np.mod(s, self._length))

    def boundaries_many(self, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Distances [m] from the centreline to the (left, right) edges at the arc lengths s."""
        s_w = np.mod(s, self._length)
        return self._w_left(s_w), self._w_right(s_w)

    def mean_curvature(self, s: float, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean curvature [1/m] over each interval [d_{i-1}, d_i] ahead of s, with d_0 = 0.

        It is the integral of the curvature spline (the one the plant uses) over the interval divided by its length,
        so a curve shorter than the interval still shows.
        """
        ahead = s + np.concatenate([[0.0], distances])
        s_w = np.mod(ahead, self._length)
        turning = self._kappa_integral(s_w) + (ahead - s_w) / self._length * self._turning_per_lap
        return np.diff(turning) / np.diff(ahead)

    def boundary_right(self, s: float) -> float:
        """Distance [m] from the centreline to the right edge."""
        return float(self._w_right(s % self._length))

    def boundary_left(self, s: float) -> float:
        """Distance [m] from the centreline to the left edge."""
        return float(self._w_left(s % self._length))

    def ey_max(self, s: float) -> float:
        """Smaller of the two edge distances [m]."""
        return min(self.boundary_right(s), self.boundary_left(s))

    def is_out_of_bounds(self, ey: float, s: float, margin: float = 0.0) -> bool:
        """Whether lateral offset ey [m] (positive = left) is beyond the edge on its own side at s, moved inwards by
        ``margin`` [m]."""
        if ey >= 0.0:
            return float(ey) > self.boundary_left(s) - margin
        return abs(float(ey)) > self.boundary_right(s) - margin
