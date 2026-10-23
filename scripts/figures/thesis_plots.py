"""Shared pieces of the thesis figures and animations: style, colours, track geometry and recorded laps.

The laps are those of scripts/figures/record_laps.py (outputs/trajectories/). A policy run starts at s = 0 at 3 m/s; its
second lap is a flying lap, compared with the flying minimum-time lap of the same b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from contextual_bicycle_pacejka.model.track import RealTrack

CATALUNYA = "data/tracks/catalunya/reference.npz"
TRAJECTORIES = Path("outputs/trajectories")
G = 9.81

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
TRACK_FILL = "#ecebe6"
EDGE = "#3a3a38"
OPTIMUM = INK
NOMINAL = "#6b6a65"
RANDOMISED = "#2a78d6"
ORACLE = "#eb6834"
FAILURE = "#e34948"
WIDTH_STEPS = {"randomised": {"w13": "#86b6ef", "w23": "#3987e5", "w1": "#184f95"},
               "oracle": {"w13": "#f4b08e", "w23": "#eb6834", "w1": "#a8401a"}}
SPEED_MAP = LinearSegmentedColormap.from_list(
    "speed", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("gap", ["#184f95", "#6da7ec", "#f0efec", "#ee8a7a", "#b3302f"])

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
    "nominal_limit": NOMINAL, "nominal_limit_cont": "#a3a29c",
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
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.titleweight": "regular",
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.6, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


@dataclass
class Track:
    """Centreline and edges of the circuit sampled every ``step`` metres."""

    real: RealTrack
    s: np.ndarray
    centre: np.ndarray
    left: np.ndarray
    right: np.ndarray
    psi: np.ndarray

    @property
    def length(self) -> float:
        return self.real.track_length

    def point(self, s: float, ey: float = 0.0) -> np.ndarray:
        x, y = self.real.centerline_point(s)
        psi = self.real.psi_ref(s)
        return np.array([x - ey * np.sin(psi), y + ey * np.cos(psi)])

    def outline(self) -> np.ndarray:
        """Closed polygon of the track surface (left edge forward, right edge back)."""
        return np.vstack([self.left, self.right[::-1]])


def load_track(step: float = 1.0) -> Track:
    real = RealTrack(CATALUNYA)
    s = np.arange(0.0, real.track_length + step, step)
    centre = np.array([real.centerline_point(v) for v in s])
    psi = np.array([real.psi_ref(v) for v in s])
    w_left, w_right = real.boundaries_many(s)
    normal = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    return Track(real, s, centre, centre + w_left[:, None] * normal, centre - w_right[:, None] * normal, psi)


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(TRAJECTORIES / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def reference(b: float = 0.0) -> dict[str, np.ndarray]:
    return load(f"reference_b{b:+.2f}")


def flying(run: dict[str, np.ndarray], length: float) -> dict[str, np.ndarray] | None:
    """The second lap of a run, with t from the crossing of s = L and s within [0, L]; None if not driven."""
    s, t = run["s"], run["t"]
    if s[-1] < 2.0 * length:
        return None
    t0, t1 = np.interp([length, 2.0 * length], s, t)
    keep = (s >= length) & (s <= 2.0 * length)
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
