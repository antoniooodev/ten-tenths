"""Shared pieces of the thesis figures and animations: theme, colours, track geometry and recorded laps.

The laps are those of scripts/figures/record_laps.py (outputs/trajectories/). A policy run starts at s = 0 at 3 m/s; its
second lap is a flying lap, compared with the flying minimum-time lap of the same b.

Theme. The colours are those of a light or a dark theme, chosen at import by the environment variable
``TEN_TENTHS_THEME`` (``light``, the default, or ``dark``), so that every script that imports them draws in the same
theme; ``scripts/figures/make_readme_figures.py`` renders both.

Geometry. A Frenet point (s, e_y, e_psi) is drawn at C(s) + e_y n(s) with heading theta(s) + e_psi, where C is the
centreline spline, theta the heading of its tangent and n the left normal. The plant's own reference heading
``RealTrack.psi_ref`` is the heading of each 5 m segment of the TUM polyline: it lags the tangent of the drawn
centreline by half a segment and differs from it by up to 14 degrees in the tight corners, so offsets along it put
the car and the edges off the drawn track. The heading of the tangent keeps every drawn point on its own normal.
Between the samples of a run (one per decision, 20 Hz) s, e_y and e_psi are interpolated with cubic Hermite
polynomials whose slopes are the Frenet rates of the recorded state, so a path drawn from them follows the car through
a corner instead of cutting it with 3 m chords.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from contextual_bicycle_pacejka.model.track import RealTrack

CATALUNYA = "data/tracks/catalunya/reference.npz"
TRAJECTORIES = Path("outputs/trajectories")
G = 9.81
THEME = os.environ.get("TEN_TENTHS_THEME", "light")
if THEME not in ("light", "dark"):
    raise ValueError(f"TEN_TENTHS_THEME must be 'light' or 'dark', got {THEME!r}")
DARK = THEME == "dark"

# Surfaces and ink. The categorical, sequential and diverging colours are the validated default palette of the
# figures (blue, orange, red; one-hue blue ramp; blue-red diverging with a grey midpoint), stepped for each surface.
INK = "#f0efec" if DARK else "#0b0b0b"
INK_2 = "#c3c2b7" if DARK else "#52514e"
INK_3 = "#8e8d86" if DARK else "#8a8984"
GRID = "#2c2c2a" if DARK else "#e6e5e0"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
TRACK_FILL = "#2a2a28" if DARK else "#ecebe6"
EDGE = "#7d7c76" if DARK else "#3a3a38"
OPTIMUM = INK
NOMINAL = "#a3a29c" if DARK else "#6b6a65"
NOMINAL_LIGHT = "#6b6a65" if DARK else "#bdbcb6"
RANDOMISED = "#3987e5" if DARK else "#2a78d6"
ORACLE = "#d95926" if DARK else "#eb6834"
FAILURE = "#e66767" if DARK else "#e34948"
WIDTH_STEPS = {"randomised": {"w13": "#86b6ef", "w23": "#3987e5", "w1": "#5598e7" if DARK else "#184f95"},
               "oracle": {"w13": "#f4b08e", "w23": "#eb6834", "w1": "#ec835a" if DARK else "#a8401a"}}
_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SPEED_MAP = LinearSegmentedColormap.from_list("speed", _BLUES[::-1] if DARK else _BLUES)
DIVERGING = LinearSegmentedColormap.from_list(
    "gap", ["#184f95", "#6da7ec", "#383835" if DARK else "#f0efec", "#ee8a7a", "#b3302f"])

LABELS = {
    "nominal_limit": "Nominal (600 M)",
    "nominal_limit_cont": "Nominal continued",
    "randomised_limit_w13": "Randomised W1/3",
    "randomised_limit_w23": "Randomised W2/3",
    "randomised_limit_w1": "Randomised W1",
    "oracle_limit_w13": "Oracle W1/3",
    "oracle_limit_w23": "Oracle W2/3",
    "oracle_limit_w1": "Oracle W1",
}
COLOURS = {
    "nominal_limit": NOMINAL, "nominal_limit_cont": NOMINAL_LIGHT,
    **{f"randomised_limit_{w}": c for w, c in WIDTH_STEPS["randomised"].items()},
    **{f"oracle_limit_{w}": c for w, c in WIDTH_STEPS["oracle"].items()},
}
TAGS = tuple(LABELS)
SEEDS = (10, 11, 12, 13, 14)
WINDOWS = ((821.0, 863.0), (1702.0, 1723.0), (2097.0, 2171.0), (2538.0, 2577.0), (3465.0, 3527.0),
           (3741.0, 3754.0), (3998.0, 4032.0), (4128.0, 4162.0), (4180.0, 4211.0))


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold", "mathtext.sf": "Inter",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.titleweight": "medium",
        "axes.titlelocation": "left", "axes.titlepad": 10, "figure.titlesize": 13, "figure.titleweight": "semibold",
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.labelcolor": INK_2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "patch.linewidth": 0,
        "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
        "image.cmap": "Blues",
    })


@lru_cache(maxsize=1)
def real_track() -> RealTrack:
    return RealTrack(CATALUNYA)


def tangent_heading(real: RealTrack, s: np.ndarray) -> np.ndarray:
    """Heading [rad] of the tangent of the centreline spline at s, continuous across laps like ``psi_ref``."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    s_w = np.mod(s, real.track_length)
    geometric = np.arctan2(real._y(s_w, 1), real._x(s_w, 1))
    base = np.array([real.psi_ref(v) for v in s])
    return base + np.angle(np.exp(1j * (geometric - base)))


def frenet_to_world(s: np.ndarray, ey: np.ndarray, epsi: np.ndarray | None = None,
                    real: RealTrack | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World coordinates (x, y) [m] and heading [rad] of Frenet points (e_y positive to the left)."""
    real = real or real_track()
    s = np.atleast_1d(np.asarray(s, dtype=float))
    ey = np.broadcast_to(np.asarray(ey, dtype=float), s.shape)
    s_w = np.mod(s, real.track_length)
    theta = tangent_heading(real, s)
    x = real._x(s_w) - ey * np.sin(theta)
    y = real._y(s_w) + ey * np.cos(theta)
    return x, y, theta + (0.0 if epsi is None else np.asarray(epsi, dtype=float))


def frenet_rates(run: dict[str, np.ndarray], real: RealTrack | None = None) -> tuple[np.ndarray, ...]:
    """(ds/dt, de_y/dt, de_psi/dt) of every sample of a run, from the plant's kinematics."""
    real = real or real_track()
    kappa = real.kappa_many(run["s"])
    cos_e, sin_e = np.cos(run["epsi"]), np.sin(run["epsi"])
    s_dot = (run["vx"] * cos_e - run["vy"] * sin_e) / (1.0 - kappa * run["ey"])
    return s_dot, run["vx"] * sin_e + run["vy"] * cos_e, run["r"] - kappa * s_dot


def hermite(t: np.ndarray, values: np.ndarray, slopes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation of samples (t, values) with the given slopes at the times ``query``."""
    query = np.clip(np.asarray(query, dtype=float), t[0], t[-1])
    i = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 2)
    h = t[i + 1] - t[i]
    u = (query - t[i]) / h
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * values[i] + h10 * h * slopes[i] + h01 * values[i + 1] + h11 * h * slopes[i + 1]


def resample(run: dict[str, np.ndarray], times: np.ndarray) -> dict[str, np.ndarray]:
    """Every channel of a run at ``times``: s, e_y and e_psi by Hermite interpolation of the Frenet state, the other
    channels linearly, and x, y, psi from the geometry."""
    t = run["t"]
    times = np.atleast_1d(np.asarray(times, dtype=float))
    out = {"t": times}
    for key, values in run.items():
        if isinstance(values, np.ndarray) and values.shape == t.shape and key not in ("t", "x", "y", "psi"):
            out[key] = np.interp(times, t, values)
    if len(t) > 1 and all(key in run for key in ("vx", "vy", "r")):
        for key, slope in zip(("s", "ey", "epsi"), frenet_rates(run), strict=True):
            out[key] = hermite(t, run[key], slope, times)
    out["x"], out["y"], out["psi"] = frenet_to_world(out["s"], out["ey"], out["epsi"])
    return out


def sample_at(run: dict[str, np.ndarray], t: float) -> dict[str, float]:
    """Channels of a run at time t (see ``resample``)."""
    return {key: float(value[0]) for key, value in resample(run, np.array([t])).items()}


def dense(run: dict[str, np.ndarray], keep: np.ndarray | None = None, dt: float = 0.01) -> dict[str, np.ndarray]:
    """A run resampled every ``dt`` seconds (100 Hz by default), optionally over the samples ``keep`` only."""
    t = run["t"] if keep is None else run["t"][keep]
    if len(t) < 2:
        return {key: value[keep] if keep is not None else value for key, value in run.items()
                if isinstance(value, np.ndarray) and value.shape == run["t"].shape}
    return resample(run, np.arange(t[0], t[-1] + 0.5 * dt, dt))


@dataclass
class Track:
    """Centreline and edges of the circuit sampled every ``step`` metres."""

    real: RealTrack
    s: np.ndarray
    centre: np.ndarray
    left: np.ndarray
    right: np.ndarray
    psi: np.ndarray  # heading of the tangent [rad]

    @property
    def length(self) -> float:
        return self.real.track_length

    def point(self, s: float, ey: float = 0.0) -> np.ndarray:
        x, y, _ = frenet_to_world(np.array([s]), np.array([ey]), real=self.real)
        return np.array([x[0], y[0]])

    def outline(self) -> np.ndarray:
        """Closed polygon of the track surface (left edge forward, right edge back)."""
        return np.vstack([self.left, self.right[::-1]])


def load_track(step: float = 1.0) -> Track:
    real = real_track()
    s = np.arange(0.0, real.track_length + step, step)
    w_left, w_right = real.boundaries_many(s)
    x, y, psi = frenet_to_world(s, np.zeros_like(s), real=real)
    centre = np.stack([x, y], axis=1)
    normal = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    return Track(real, s, centre, centre + w_left[:, None] * normal, centre - w_right[:, None] * normal, psi)


def load(name: str) -> dict[str, np.ndarray]:
    """A recorded run, with x, y and psi recomputed from its Frenet state (see the module docstring)."""
    with np.load(TRAJECTORIES / f"{name}.npz") as data:
        run = {key: data[key] for key in data.files}
    run["x"], run["y"], run["psi"] = frenet_to_world(run["s"], run["ey"], run["epsi"])
    return run


def reference(b: float = 0.0) -> dict[str, np.ndarray]:
    return load(f"reference_b{b:+.2f}")


def flying(run: dict[str, np.ndarray], length: float) -> dict[str, np.ndarray] | None:
    """The second lap of a run, with t from the crossing of s = L and s within [0, L]; None if not driven.

    The samples just outside the lap are kept, so that an interpolation reaches both ends of the lap."""
    s, t = run["s"], run["t"]
    if s[-1] < 2.0 * length:
        return None
    t0, t1 = np.interp([length, 2.0 * length], s, t)
    inside = np.flatnonzero((s >= length) & (s <= 2.0 * length))
    keep = np.arange(max(inside[0] - 1, 0), min(inside[-1] + 2, len(s)))
    lap = {key: value[keep] for key, value in run.items() if isinstance(value, np.ndarray) and value.shape == s.shape}
    lap["t"] = lap["t"] - t0
    lap["s"] = lap["s"] - length
    lap["lap_time"] = np.array(t1 - t0)
    return lap


def time_at(lap: dict[str, np.ndarray], s: np.ndarray) -> np.ndarray:
    """Time [s] at which a lap reaches the positions s (monotone interpolation)."""
    order = np.maximum.accumulate(lap["s"])
    return np.interp(s, order, lap["t"])


def reference_time_at(ref: dict[str, np.ndarray], s: np.ndarray) -> np.ndarray:
    return np.interp(s, ref["s_grid"], ref["t_grid"])


def window_bbox(track: Track, start: float, end: float, pad: float = 25.0) -> tuple[float, float, float, float]:
    """Square world box around the track between start and end [m]."""
    keep = (track.s >= start) & (track.s <= end)
    pts = np.vstack([track.left[keep], track.right[keep]])
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    half = 0.5 * max(x1 - x0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return cx - half, cx + half, cy - half, cy + half


def draw_track(ax, track: Track, edge_width: float = 0.9, fill: bool = True) -> None:
    if fill:
        ax.fill(*track.outline().T, color=TRACK_FILL, zorder=0, lw=0)
    ax.plot(*track.left.T, color=EDGE, lw=edge_width, zorder=1)
    ax.plot(*track.right.T, color=EDGE, lw=edge_width, zorder=1)
    ax.set_aspect("equal")


TITLES = True  # False: figures without their title (the suptitle, and the title of a single-panel figure)


def save(fig, out: Path, name: str) -> None:
    if not TITLES:
        if fig._suptitle is not None:
            fig._suptitle.set_text("")
        panels = [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is None]
        if len(panels) == 1:
            panels[0].set_title("")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
