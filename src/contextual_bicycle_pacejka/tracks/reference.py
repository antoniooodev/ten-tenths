"""Centreline reference of a circuit from the TUM racetrack database (Betz et al., 2022).

The CSV columns are ``x_m, y_m, w_tr_right_m, w_tr_left_m``. The reference stores the closed centreline (centred
at the origin), its arc length, heading, curvature (wrapped heading change over the central segment length,
smoothed by a 10-sample periodic moving average) and the boundary distances.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

TUM_URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/{name}.csv"
TUM_NAMES = {"catalunya": "Catalunya"}
KAPPA_SMOOTH_WINDOW = 10  # [samples]
CLOSURE_TOL_M = 1.0  # a last point closer than this to the first one is a duplicate


def download_centerline(circuit: str, path: str | Path) -> Path:
    """Download the TUM centreline CSV of ``circuit`` to ``path`` (skipped if the file exists)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TUM_URL.format(name=TUM_NAMES[circuit]), timeout=60) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path


def _heading_and_curvature(x: NDArray[np.float64], y: NDArray[np.float64], segment: NDArray[np.float64],
                           ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dx, dy = np.roll(x, -1) - x, np.roll(y, -1) - y
    heading = np.arctan2(dy, dx)
    dpsi = heading - np.roll(heading, 1)
    dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))
    central = 0.5 * (np.roll(segment, 1) + segment)
    kappa = uniform_filter1d(dpsi / np.maximum(central, 1e-9), size=KAPPA_SMOOTH_WINDOW, mode="wrap")
    return np.unwrap(heading), kappa


def build_reference(csv_path: str | Path, output: str | Path) -> Path:
    """Write the reference NPZ of the centreline in ``csv_path``."""
    data = np.loadtxt(io.StringIO(Path(csv_path).read_text(encoding="utf-8")), delimiter=",", comments="#")
    x, y, w_right, w_left = (np.ascontiguousarray(data[:, i], dtype=np.float64) for i in range(4))
    if float(np.hypot(x[-1] - x[0], y[-1] - y[0])) < CLOSURE_TOL_M:
        x, y, w_right, w_left = x[:-1], y[:-1], w_right[:-1], w_left[:-1]
    segment = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    s = np.concatenate(([0.0], np.cumsum(segment)[:-1]))
    psi, kappa = _heading_and_curvature(x, y, segment)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x_m=x - float(np.mean(x)), y_m=y - float(np.mean(y)), s_m=s, psi_rad=psi, kappa_radpm=kappa,
             w_tr_right_m=w_right, w_tr_left_m=w_left, track_length=np.float64(np.sum(segment)))
    return output
