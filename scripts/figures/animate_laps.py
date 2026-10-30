"""Animations of the recorded laps (scripts/figures/record_laps.py), as GIF and MP4, for the slides.

    python scripts/figures/animate_laps.py
    python scripts/figures/animate_laps.py --only chase --workers 8
    python scripts/figures/animate_laps.py --no-titles --output outputs/figures/thesis_no_titles/animations

Animations (``--only`` picks some):
- chase:        one flying lap of a policy against the minimum-time lap: a camera that follows the car with the
                track ahead pointing up, the ghost of the minimum-time lap at the same lap time and the preview points
                the policy observes, the speed and the time gap along the lap, the g-g diagram, the commands and tyre
                slips, a map; for the Nominal at the limit, the Randomised W1 and the Oracle W1, seed 12;
- race:         the same three policies and the minimum-time lap on the whole circuit, flying lap, with the gaps;
- balance:      the three policies with an oversteering (b = +0.05) and an understeering (b = -0.21) car, from
                the standing start at s = 0;
- stochastic:   Nominal continued, seed 14: the mean action leaves the track in corner C3, five runs of the
                stochastic policy pass.

The car is drawn to scale from above (body, cabin, four wheels; the front wheels turn with the steering angle), with
a trail that fades over the last seconds. ``--no-titles`` leaves out the heading lines of every frame (the live
read-out and the panel labels stay). Frames are rendered in parallel with matplotlib and assembled with ffmpeg;
<output>/<name>.gif and .mp4.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from thesis_plots import (
    FAILURE,
    GRID,
    INK,
    INK_2,
    INK_3,
    LABELS,
    NOMINAL,
    OPTIMUM,
    ORACLE,
    RANDOMISED,
    SPEED_MAP,
    SURFACE,
    TRACK_FILL,
    WINDOWS,
    G,
    Track,
    apply_style,
    dense,
    flying,
    load,
    load_track,
    reference,
    reference_time_at,
    resample,
    sample_at,
    time_at,
    window_bbox,
)

from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.spec.constants import LF, LR

POLICY_COLOURS = {"nominal_limit": NOMINAL, "nominal_limit_cont": NOMINAL, "randomised_limit_w1": RANDOMISED,
                  "oracle_limit_w1": ORACLE}
FIGSIZE, DPI = (16, 9), 100
GRASS = "#e9eee4"
ASPHALT = "#d3d2cc"
LINE_WHITE = "#ffffff"
KERB_RED = "#d8403f"
GLASS = "#2b3440"
TYRE = "#1d1d1b"
TRAIL_S = 6.0  # [s] length of the fading trail

# Top view of a Porsche 718 Cayman GT4 (4.45 m x 1.85 m) in car coordinates, x forward, y to the left [m]
BODY = np.array([[2.22, 0.50], [2.12, 0.78], [1.75, 0.90], [-1.85, 0.925], [-2.18, 0.80], [-2.225, 0.55],
                 [-2.225, -0.55], [-2.18, -0.80], [-1.85, -0.925], [1.75, -0.90], [2.12, -0.78], [2.22, -0.50]])
CABIN = np.array([[1.00, 0.0], [0.80, 0.64], [-0.70, 0.70], [-1.30, 0.56], [-1.30, -0.56], [-0.70, -0.70],
                  [0.80, -0.64]])
WHEEL = np.array([[0.33, 0.13], [-0.33, 0.13], [-0.33, -0.13], [0.33, -0.13]])
WHEEL_CENTRES = ((LF, 0.86), (LF, -0.86), (-LR, 0.86), (-LR, -0.86))


class CarSprite:
    """The car from above; ``ghost`` draws only a dashed outline (the minimum-time lap)."""

    def __init__(self, ax, colour: str, ghost: bool = False, zorder: float = 6.0, scale: float = 1.0) -> None:
        self.ax, self.scale, self.parts, self.wheels = ax, scale, [], []
        if ghost:
            self.parts.append(ax.add_patch(Polygon(BODY * scale, closed=True, fill=False, ec=OPTIMUM, lw=1.3,
                                                   ls=(0, (3, 2)), alpha=0.85, zorder=zorder)))
            return
        for k, (cx, cy) in enumerate(WHEEL_CENTRES):
            wheel = ax.add_patch(Polygon((WHEEL + [cx, cy]) * scale, closed=True, fc=TYRE, ec="none",
                                         zorder=zorder - 0.1))
            self.wheels.append((wheel, k < 2, cx * scale, cy * scale))
        self.parts.append(ax.add_patch(Polygon(BODY * scale, closed=True, fc=colour, ec=INK, lw=0.8, zorder=zorder)))
        self.parts.append(ax.add_patch(Polygon(CABIN * scale, closed=True, fc=GLASS, ec="none", alpha=0.75,
                                               zorder=zorder + 0.1)))

    def place(self, x: float, y: float, psi: float, delta: float, view) -> None:
        base = Affine2D().rotate(psi).translate(x, y) + view
        for part in self.parts:
            part.set_transform(base)
        for wheel, front, cx, cy in self.wheels:
            wheel.set_transform((Affine2D().rotate_around(cx, cy, delta) if front else Affine2D()) + base)

    def hide(self) -> None:
        for part in self.parts + [w[0] for w in self.wheels]:
            part.set_visible(False)


def pixels_per_metre(ax, limits: tuple[float, float, float, float]) -> float:
    """Screen scale of a view with equal aspect showing at least ``limits`` (x0, x1, y0, y1) [m]."""
    box = ax.get_position()
    width, height = box.width * FIGSIZE[0] * DPI, box.height * FIGSIZE[1] * DPI
    x0, x1, y0, y1 = limits
    return min(width / (x1 - x0), height / (y1 - y0))


def metres_to_points(metres: float, px_per_m: float) -> float:
    return metres * px_per_m * 72.0 / DPI


class TrackLayer:
    """Asphalt, edge lines and, with a screen scale ``px_per_m``, the kerbs of the corner windows and the
    start/finish line, sized in metres (kerbs 0.9 m wide, edge lines 0.25 m)."""

    def __init__(self, ax, track: Track, px_per_m: float | None = None) -> None:
        self.artists = []
        detail = px_per_m is not None
        ax.set_facecolor(GRASS if detail else SURFACE)
        self.artists.append(ax.add_patch(Polygon(track.outline(), closed=True, fc=ASPHALT if detail else TRACK_FILL,
                                                 ec="none", zorder=0)))
        if detail:
            kerb = metres_to_points(0.9, px_per_m)
            edge = {"color": LINE_WHITE, "lw": max(1.0, metres_to_points(0.25, px_per_m))}
        else:
            edge = {"color": "#8f8e88", "lw": 0.7}
        for side in (track.left, track.right):
            self.artists.append(ax.plot(*side.T, solid_capstyle="butt", zorder=1, **edge)[0])
        if detail:
            for a, b in WINDOWS:
                keep = (track.s >= a - 15) & (track.s <= b + 15)
                for side in (track.left, track.right):
                    self.artists.append(ax.plot(*side[keep].T, color=LINE_WHITE, lw=kerb, solid_capstyle="butt",
                                                zorder=1.1)[0])
                    self.artists.append(ax.plot(*side[keep].T, color=KERB_RED, lw=kerb, ls=(0, (1.6, 1.6)),
                                                dash_capstyle="butt", zorder=1.2)[0])
            start = np.array([track.left[0], track.right[0]])
            self.artists.append(ax.plot(*start.T, color=INK, lw=kerb, solid_capstyle="butt", zorder=1.3)[0])
            self.artists.append(ax.plot(*start.T, color=LINE_WHITE, lw=kerb, ls=(0, (1, 1)), zorder=1.4)[0])

    def set_view(self, view) -> None:
        for artist in self.artists:
            artist.set_transform(view)


class Trail:
    """The last ``TRAIL_S`` seconds of a run, fading towards the past."""

    def __init__(self, ax, colour: str, width: float = 2.6, seconds: float = TRAIL_S, zorder: float = 4.0) -> None:
        self.colour, self.seconds = to_rgba(colour), seconds
        self.lines = LineCollection([], linewidths=width, capstyle="round", joinstyle="round", zorder=zorder)
        ax.add_collection(self.lines, autolim=False)

    def update(self, run: dict[str, np.ndarray], t: float, view) -> None:
        start = max(float(run["t"][0]), t - self.seconds)
        if t - start < 0.02:
            self.lines.set_segments([])
            return
        path = resample(run, np.append(np.arange(start, t, 0.02), t))  # the Frenet path, not 20 Hz chords
        x, y, age = path["x"], path["y"], t - path["t"]
        if len(x) < 2:
            self.lines.set_segments([])
            return
        points = np.stack([x, y], axis=1)
        self.lines.set_segments(np.stack([points[:-1], points[1:]], axis=1))
        alpha = np.clip(1.0 - age[1:] / self.seconds, 0.0, 1.0) ** 1.5
        colours = np.tile(self.colour, (len(alpha), 1))
        colours[:, 3] = 0.9 * alpha
        self.lines.set_color(colours)
        self.lines.set_transform(view)


def follow_view(ax, track: Track, x: float, y: float, s: float) -> Any:
    """Transform from world to a camera on (x, y) with the centreline ahead pointing up (smoothed heading)."""
    ahead = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    index = np.mod(np.round(s + ahead).astype(int), len(track.s) - 1)
    heading = np.arctan2(np.sin(track.psi[index]).mean(), np.cos(track.psi[index]).mean())
    return Affine2D().translate(-x, -y).rotate(np.pi / 2 - heading) + ax.transData


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def scale_bar(ax, metres: float = 20.0) -> None:
    """A bar of ``metres`` in the lower left of a view whose data unit is the metre."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    left, bottom = x0 + 0.05 * (x1 - x0), y0 + 0.04 * (y1 - y0)
    ax.plot([left, left + metres], [bottom, bottom], color=INK, lw=2.5, solid_capstyle="butt", zorder=9)
    ax.text(left + metres / 2, bottom + 0.012 * (y1 - y0), f"{metres:.0f} m", ha="center", va="bottom", fontsize=9,
            color=INK, zorder=9)


def header(fig, titles: bool, title: str, subtitle: str, colour: str = INK) -> tuple[Any, float]:
    """Title lines (with ``titles``) and the live read-out; return the read-out and the top of the panels."""
    if titles:
        fig.text(0.03, 0.955, title, fontsize=19, color=INK, weight="medium")
        fig.text(0.03, 0.918, subtitle, fontsize=11.5, color=INK_2)
        return fig.text(0.03, 0.875, "", fontsize=13, color=colour, family="monospace"), 0.86
    return fig.text(0.03, 0.955, "", fontsize=13, color=colour, family="monospace"), 0.93


def view_legend(ax, handles: list) -> None:
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID,
              framealpha=0.9, borderpad=0.6).set_zorder(10)


def car_handle(colour: str, label: str) -> Line2D:
    return Line2D([], [], marker="s", ls="", ms=9, mfc=colour, mec=INK, mew=0.8, label=label)


class Chase:
    """One flying lap of a policy, with a camera that follows the car."""

    HALF_WIDTH_M = 24.0

    def __init__(self, tag: str, seed: int, frame_dt: float, titles: bool = True) -> None:
        self.tag, self.seed, self.frame_dt, self.titles = tag, seed, frame_dt, titles
        self.track = load_track(1.0)
        self.lap = flying(load(f"flying_{tag}_s{seed}"), self.track.length)
        self.ref = reference(0.0)
        self.ghost_run = self.ref
        self.colour = POLICY_COLOURS[tag]
        self.preview = make_stage(7, self.track.real).preview
        self.frames = int(np.ceil(float(self.lap["lap_time"]) / frame_dt)) + 1
        self.s_grid = np.arange(0.0, self.track.length, 2.0)
        self.gap = time_at(self.lap, self.s_grid) - reference_time_at(self.ref, self.s_grid)

    def setup(self) -> None:
        apply_style()
        plt.rcParams["axes.titlelocation"] = "center"  # the live read-out sits above the left of the panels
        plt.rcParams["savefig.bbox"] = "standard"  # every frame the same size
        fig = self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        lap_time = float(self.lap["lap_time"])
        self.status, top = header(
            fig, self.titles, f"Pure RL at the limit: {LABELS[self.tag]}, seed {self.seed}",
            f"PPO from interaction only, no demonstrations · Catalunya · b = 0 · flying lap {lap_time:.2f} s vs "
            f"minimum-time lap {float(self.ref['lap_time']):.2f} s "
            f"(+{100 * (lap_time / float(self.ref['lap_time']) - 1):.1f} %)", self.colour)
        grid = fig.add_gridspec(3, 4, width_ratios=[2.1, 1, 1, 1], height_ratios=[1, 1, 1.25], left=0.03,
                                right=0.985, top=top, bottom=0.07, wspace=0.32, hspace=0.42)

        ax = self.view = fig.add_subplot(grid[:, 0])
        bare(ax)
        ax.set_aspect("equal", adjustable="datalim")
        half = self.HALF_WIDTH_M
        limits = (-half, half, -0.4 * 2.3 * half, 0.6 * 2.3 * half)
        self.layer = TrackLayer(ax, self.track, pixels_per_metre(ax, limits))
        self.racing_line, = ax.plot(self.ref["x"], self.ref["y"], color=OPTIMUM, lw=1.0, ls=(0, (5, 4)), alpha=0.55,
                                    zorder=2)
        self.preview_dots, = ax.plot([], [], "o", ms=2.6, color=INK_2, alpha=0.5, zorder=3)
        self.trail = Trail(ax, self.colour)
        self.ghost = CarSprite(ax, OPTIMUM, ghost=True, zorder=5)
        self.car = CarSprite(ax, self.colour, zorder=7)
        ax.set_xlim(*limits[:2])
        ax.set_ylim(*limits[2:])
        ax.set_autoscale_on(False)
        scale_bar(ax, 10.0)
        view_legend(ax, [car_handle(self.colour, "policy (to scale, 4.45 m)"),
                         Line2D([], [], marker="s", ls="", ms=9, mfc="none", mec=OPTIMUM, label="minimum-time lap"),
                         Line2D([], [], color=OPTIMUM, lw=1.0, ls=(0, (5, 4)), alpha=0.7,
                                label="minimum-time racing line"),
                         Line2D([], [], marker="o", ls="", ms=3, color=INK_2, alpha=0.6,
                                label="track preview the policy observes")])

        ax = self.speed_ax = fig.add_subplot(grid[0, 1:])
        ax.plot(self.ref["s"], 3.6 * self.ref["vx"], color=OPTIMUM, lw=1.0, ls="--", alpha=0.7, label="minimum time")
        self.speed_line, = ax.plot([], [], color=self.colour, lw=1.6, label="policy")
        self.speed_dot, = ax.plot([], [], "o", color=self.colour, ms=6)
        ax.set_xlim(0, self.track.length)
        ax.set_ylim(40, 260)
        ax.set_ylabel("speed [km/h]")
        ax.set_title("speed along the lap", fontsize=11)
        ax.legend(loc="upper right", fontsize=8.5, ncol=2)

        ax = self.gap_ax = fig.add_subplot(grid[1, 1:], sharex=self.speed_ax)
        ax.axhline(0, color=OPTIMUM, lw=1.0, ls="--", alpha=0.7)
        self.gap_line, = ax.plot([], [], color=self.colour, lw=1.6)
        self.gap_dot, = ax.plot([], [], "o", color=self.colour, ms=6)
        ax.set_ylim(min(-0.3, 1.1 * self.gap.min()), 1.15 * max(self.gap.max(), 0.5))
        ax.set_xlabel("position along the lap s [m]")
        ax.set_ylabel("time behind [s]")
        ax.set_title("time behind the minimum-time lap at the same position", fontsize=11)

        ax = self.gg = fig.add_subplot(grid[2, 1])
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle), np.sin(circle), color=INK_3, lw=1, ls="--")
        self.gg_points = ax.scatter([], [], c=[], cmap=SPEED_MAP, vmin=60, vmax=250, s=7, lw=0)
        self.gg_dot, = ax.plot([], [], "o", ms=9, mfc=self.colour, mec=INK, mew=1.0)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("a_y / g")
        ax.set_ylabel("a_x / g")
        ax.set_title("accelerations (g-g)", fontsize=11)
        ax.tick_params(labelsize=8)

        ax = self.bars = fig.add_subplot(grid[2, 2])
        names = ["steering\nδ / δ_max", "command τ\n(+ gas, − brake)", "front slip", "rear slip"]
        self.bar_artists = ax.barh(range(4), [0, 0, 0, 0], color=[self.colour] * 4, height=0.55)
        ax.axvline(0, color=INK_3, lw=0.8)
        ax.axvline(1.0, color=FAILURE, lw=1.0, ls="--")
        ax.text(1.03, -0.45, "peak", color=FAILURE, fontsize=8.5)
        ax.set_yticks(range(4), names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-1.05, 1.6)
        ax.set_title("commands and tyres", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)

        ax = self.mini = fig.add_subplot(grid[2, 3])
        TrackLayer(ax, self.track)
        ax.set_aspect("equal")
        bare(ax)
        ax.autoscale_view()
        ax.set_title("position", fontsize=11)
        self.mini_trail = Trail(ax, self.colour, width=2.0, seconds=12.0)
        self.mini_ghost, = ax.plot([], [], "o", ms=6, mfc="none", mec=OPTIMUM, mew=1.3, zorder=5)
        self.mini_car, = ax.plot([], [], "o", ms=7, color=self.colour, mec=SURFACE, mew=1.0, zorder=6)

    def render(self, k: int, path: Path) -> None:
        lap, ref, track = self.lap, self.ref, self.track
        t = min(k * self.frame_dt, float(lap["lap_time"]))
        now = sample_at(lap, t)
        ghost = sample_at(self.ghost_run, t)
        view = follow_view(self.view, track, now["x"], now["y"], now["s"])
        self.layer.set_view(view)
        self.racing_line.set_transform(view)
        self.trail.update(lap, t, view)
        self.car.place(now["x"], now["y"], now["psi"], now["delta"], view)
        self.ghost.place(ghost["x"], ghost["y"], ghost["psi"], ghost["delta"], view)
        ahead = np.array([track.point(now["s"] + d) for d in self.preview])
        self.preview_dots.set_data(ahead[:, 0], ahead[:, 1])
        self.preview_dots.set_transform(view)
        past = lap["t"] <= t
        s_now = now["s"]
        self.speed_line.set_data(lap["s"][past], 3.6 * lap["vx"][past])
        self.speed_dot.set_data([s_now], [3.6 * now["vx"]])
        upto = self.s_grid <= s_now
        gap_now = float(time_at(lap, np.array([s_now]))[0] - reference_time_at(ref, np.array([s_now]))[0])
        self.gap_line.set_data(self.s_grid[upto], self.gap[upto])
        self.gap_dot.set_data([s_now], [gap_now])
        self.gg_points.set_offsets(np.stack([lap["ay"][past] / G, lap["ax"][past] / G], axis=1))
        self.gg_points.set_array(3.6 * lap["vx"][past])
        self.gg_dot.set_data([now["ay"] / G], [now["ax"] / G])
        values = [now["delta"] / 0.5, now["tau"], now["slip_f"], now["slip_r"]]
        for bar, value, index in zip(self.bar_artists, values, range(4), strict=True):
            bar.set_width(value)
            bar.set_color(FAILURE if index >= 2 and value > 1.0 else self.colour)
        self.mini_trail.update(lap, t, self.mini.transData)
        self.mini_car.set_data([now["x"]], [now["y"]])
        self.mini_ghost.set_data([ghost["x"]], [ghost["y"]])
        self.status.set_text(f"t {t:6.2f} s   speed {3.6 * now['vx']:5.1f} km/h   steering "
                             f"{np.degrees(now['delta']):+5.1f}°   gap to the optimum {gap_now:+5.2f} s")
        self.fig.savefig(path, dpi=DPI)


class Race:
    """Flying laps of several policies and the minimum-time lap on the whole circuit."""

    TAGS = ("nominal_limit", "randomised_limit_w1", "oracle_limit_w1")

    def __init__(self, seed: int, frame_dt: float, titles: bool = True) -> None:
        self.frame_dt, self.titles, self.seed = frame_dt, titles, seed
        self.track = load_track(1.0)
        self.ref = reference(0.0)
        self.ghost_run = self.ref
        self.laps = {tag: flying(load(f"flying_{tag}_s{seed}"), self.track.length) for tag in self.TAGS}
        end = max(float(lap["lap_time"]) for lap in self.laps.values()) + 2.0
        self.frames = int(np.ceil(end / frame_dt)) + 1
        self.s_grid = np.arange(0.0, self.track.length, 2.0)
        self.gaps = {tag: time_at(lap, self.s_grid) - reference_time_at(self.ref, self.s_grid)
                     for tag, lap in self.laps.items()}

    def setup(self) -> None:
        apply_style()
        plt.rcParams["axes.titlelocation"] = "center"  # the live read-out sits above the left of the panels
        plt.rcParams["savefig.bbox"] = "standard"
        fig = self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        self.clock, top = header(fig, self.titles, "Flying lap at b = 0: three PPO policies against the minimum-time "
                                 "lap", f"seed {self.seed}; the Randomised and the Oracle continue from the Nominal on "
                                 "cars with b over [−0.21, +0.03]; the Oracle also observes b")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.7, 1], left=0.02, right=0.98, top=top,
                                bottom=0.08, wspace=0.12, hspace=0.3)
        ax = self.map = fig.add_subplot(grid[:, 0])
        TrackLayer(ax, self.track)
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        bare(ax)
        self.trails = {tag: Trail(ax, POLICY_COLOURS[tag], width=3.0, seconds=4.0) for tag in self.TAGS}
        self.markers = {"optimum": ax.plot([], [], "o", ms=11, mfc="none", mec=OPTIMUM, mew=2.0, zorder=6)[0]}
        for tag in self.TAGS:
            self.markers[tag] = ax.plot([], [], "o", ms=10, color=POLICY_COLOURS[tag], mec=SURFACE, mew=1.4,
                                        zorder=5)[0]
        handles = [Line2D([], [], marker="o", ls="", mfc="none", mec=OPTIMUM, mew=2, ms=10,
                          label=f"minimum-time lap ({float(self.ref['lap_time']):.2f} s)")]
        handles += [Line2D([], [], marker="o", ls="", color=POLICY_COLOURS[tag], ms=10,
                           label=f"{LABELS[tag]} ({float(lap['lap_time']):.2f} s)") for tag, lap in self.laps.items()]
        ax.legend(handles=handles, loc="lower right", fontsize=10.5)
        self.board = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11.5, family="monospace")

        ax = self.zoom = fig.add_subplot(grid[0, 1])
        bare(ax)
        ax.set_aspect("equal", adjustable="datalim")
        limits = (-30.0, 30.0, -24.0, 30.0)
        self.layer = TrackLayer(ax, self.track, pixels_per_metre(ax, limits))
        self.racing_line, = ax.plot(self.ref["x"], self.ref["y"], color=OPTIMUM, lw=0.9, ls=(0, (5, 4)), alpha=0.5,
                                    zorder=2)
        self.zoom_trails = {tag: Trail(ax, POLICY_COLOURS[tag], width=2.2, seconds=3.0) for tag in self.TAGS}
        self.zoom_ghost = CarSprite(ax, OPTIMUM, ghost=True, zorder=5)
        self.zoom_cars = {tag: CarSprite(ax, POLICY_COLOURS[tag], zorder=7 + i) for i, tag in enumerate(self.TAGS)}
        ax.set_xlim(*limits[:2])
        ax.set_ylim(*limits[2:])
        ax.set_autoscale_on(False)
        scale_bar(ax, 10.0)
        ax.set_title("camera on the Nominal", fontsize=11)

        ax = self.gap_ax = fig.add_subplot(grid[1, 1])
        ax.axhline(0, color=OPTIMUM, lw=1.0, ls="--")
        self.gap_lines = {tag: ax.plot([], [], color=POLICY_COLOURS[tag], lw=1.8)[0] for tag in self.TAGS}
        ax.set_xlim(0, self.track.length)
        ax.set_ylim(-0.3, 1.1 * max(g.max() for g in self.gaps.values()))
        ax.set_xlabel("position along the lap s [m]")
        ax.set_ylabel("time behind the optimum [s]")

    def render(self, k: int, path: Path) -> None:
        t = k * self.frame_dt
        ghost = sample_at(self.ghost_run, min(t, float(self.ref["t"][-1])))
        self.markers["optimum"].set_data([ghost["x"]], [ghost["y"]])
        centre = sample_at(self.laps["nominal_limit"], min(t, float(self.laps["nominal_limit"]["lap_time"])))
        view = follow_view(self.zoom, self.track, centre["x"], centre["y"], centre["s"])
        self.layer.set_view(view)
        self.racing_line.set_transform(view)
        self.zoom_ghost.place(ghost["x"], ghost["y"], ghost["psi"], ghost["delta"], view)
        lines = []
        for tag, lap in self.laps.items():
            end = float(lap["lap_time"])
            now = sample_at(lap, min(t, end))
            self.markers[tag].set_data([now["x"]], [now["y"]])
            self.trails[tag].update(lap, min(t, end), self.map.transData)
            self.zoom_trails[tag].update(lap, min(t, end), view)
            self.zoom_cars[tag].place(now["x"], now["y"], now["psi"], now["delta"], view)
            upto = self.s_grid <= now["s"]
            self.gap_lines[tag].set_data(self.s_grid[upto], self.gaps[tag][upto])
            behind = float(time_at(lap, np.array([now["s"]]))[0] - reference_time_at(self.ref, np.array([now["s"]]))[0])
            text = f"finished {end:.2f} s" if t >= end else f"{3.6 * now['vx']:5.1f} km/h"
            lines.append(f"{LABELS[tag]:<17} {behind:+5.2f} s   {text}")
        self.board.set_text("time behind the optimum at the same position\n" + "\n".join(lines))
        self.clock.set_text(f"t {t:6.2f} s   (minimum-time lap {'finished' if t >= self.ref['t'][-1] else 'running'})")
        self.fig.savefig(path, dpi=DPI)


class Balance:
    """Three policies with the same car of balance index b, from the standing start at s = 0."""

    TAGS = ("nominal_limit_cont", "randomised_limit_w1", "oracle_limit_w1")

    def __init__(self, b: float, frame_dt: float, titles: bool = True) -> None:
        self.b, self.frame_dt, self.titles = b, frame_dt, titles
        self.track = load_track(1.0)
        self.runs = {tag: load(f"balance_{tag}_s12_b{b:+.2f}") for tag in self.TAGS}
        self.ends = {}
        for tag, run in self.runs.items():
            done = run["lap_times"]
            self.ends[tag] = float(done[0]) if len(done) else float(run["t"][-1])
        self.frames = int(np.ceil((max(self.ends.values()) + 2.0) / frame_dt)) + 1
        failed = [run for run in self.runs.values() if run["failure"] != "None"]
        self.focus = float(failed[0]["s"][-1]) if failed else 850.0

    def setup(self) -> None:
        apply_style()
        plt.rcParams["axes.titlelocation"] = "center"  # the live read-out sits above the left of the panels
        plt.rcParams["savefig.bbox"] = "standard"
        fig = self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        kind = "oversteering" if self.b > 0 else "understeering"
        self.clock, top = header(fig, self.titles, f"An {kind} car, b = {self.b:+.2f}: who can drive it?",
                                 "seed 12, standing start at s = 0, mean action; Nominal trained only on b = 0, "
                                 "Randomised and Oracle on b in [−0.21, +0.03]")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], left=0.02, right=0.98, top=top, bottom=0.08,
                                wspace=0.12, hspace=0.3)
        ax = self.map = fig.add_subplot(grid[:, 0])
        TrackLayer(ax, self.track)
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        bare(ax)
        self.trails = {tag: Trail(ax, POLICY_COLOURS[tag], width=3.0, seconds=4.0) for tag in self.TAGS}
        self.markers = {tag: ax.plot([], [], "o", ms=10, color=POLICY_COLOURS[tag], mec=SURFACE, mew=1.4, zorder=5)[0]
                        for tag in self.TAGS}
        self.crashes = {tag: ax.plot([], [], "X", ms=16, color=FAILURE, mec=SURFACE, zorder=7)[0] for tag in self.TAGS}
        handles = [Line2D([], [], marker="o", ls="", color=POLICY_COLOURS[tag], ms=10, label=LABELS[tag])
                   for tag in self.TAGS]
        handles.append(Line2D([], [], marker="X", ls="", color=FAILURE, ms=12, label="off track"))
        ax.legend(handles=handles, loc="lower right", fontsize=10.5)
        self.board = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11.5, family="monospace")

        ax = self.zoom = fig.add_subplot(grid[0, 1])
        bare(ax)
        ax.set_aspect("equal", adjustable="datalim")
        x0, x1, y0, y1 = window_bbox(self.track, self.focus - 160, self.focus + 40, 8)
        TrackLayer(ax, self.track, pixels_per_metre(ax, (x0, x1, y0, y1)))
        self.zoom_trails = {tag: Trail(ax, POLICY_COLOURS[tag], width=2.2, seconds=8.0) for tag in self.TAGS}
        self.zoom_cars = {tag: CarSprite(ax, POLICY_COLOURS[tag], zorder=7 + i, scale=1.3)
                          for i, tag in enumerate(self.TAGS)}
        self.zoom_crashes = {tag: ax.plot([], [], "X", ms=18, color=FAILURE, mec=SURFACE, zorder=12)[0]
                             for tag in self.TAGS}
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_autoscale_on(False)
        scale_bar(ax, 20.0)
        ax.set_title(f"where the first car leaves the track (s ≈ {self.focus:.0f} m; cars drawn 1.3× larger)",
                     fontsize=11)

        ax = self.speed_ax = fig.add_subplot(grid[1, 1])
        self.speed_lines = {tag: ax.plot([], [], color=POLICY_COLOURS[tag], lw=1.6)[0] for tag in self.TAGS}
        ax.set_xlim(0, self.track.length)
        ax.set_ylim(0, 260)
        ax.set_xlabel("position along the lap s [m]")
        ax.set_ylabel("speed [km/h]")

    def render(self, k: int, path: Path) -> None:
        t = k * self.frame_dt
        lines = []
        for tag, run in self.runs.items():
            end = self.ends[tag]
            now = sample_at(run, min(t, end))
            past = run["t"] <= min(t, end)
            failed = run["failure"] != "None"
            self.markers[tag].set_data([now["x"]], [now["y"]])
            self.trails[tag].update(run, min(t, end), self.map.transData)
            self.zoom_trails[tag].update(run, min(t, end), self.zoom.transData)
            self.zoom_cars[tag].place(now["x"], now["y"], now["psi"], now["delta"], self.zoom.transData)
            self.speed_lines[tag].set_data(run["s"][past], 3.6 * run["vx"][past])
            if t >= end:
                if failed:
                    self.crashes[tag].set_data([now["x"]], [now["y"]])
                    self.zoom_crashes[tag].set_data([now["x"]], [now["y"]])
                    lines.append(f"{LABELS[tag]:<18} off track at {now['s']:6.0f} m")
                else:
                    lines.append(f"{LABELS[tag]:<18} lap completed in {end:.2f} s")
            else:
                lines.append(f"{LABELS[tag]:<18} {now['s']:6.0f} m  {3.6 * now['vx']:5.1f} km/h")
        self.board.set_text("\n".join(lines))
        self.clock.set_text(f"t {t:6.2f} s")
        self.fig.savefig(path, dpi=DPI)


class Stochastic:
    """Mean action against the stochastic policy in corner C3, Nominal continued seed 14."""

    def __init__(self, frame_dt: float, titles: bool = True) -> None:
        self.frame_dt, self.titles = frame_dt, titles
        self.track = load_track(1.0)
        self.mean = load("stochastic_nominal_limit_cont_s14_mean")
        self.noisy = [load(f"stochastic_nominal_limit_cont_s14_noise{k}") for k in range(5)]
        self.t0 = float(np.interp(1990.0, self.mean["s"], self.mean["t"]))
        self.t_fail = float(self.mean["t"][-1])
        self.frames = int(np.ceil((self.t_fail + 3.0 - self.t0) / frame_dt)) + 1

    def setup(self) -> None:
        apply_style()
        plt.rcParams["axes.titlelocation"] = "center"  # the live read-out sits above the left of the panels
        plt.rcParams["savefig.bbox"] = "standard"
        fig = self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        sigma = self.noisy[0]["std_delta"][1]
        self.clock, top = header(fig, self.titles, "Mean action against the stochastic policy (Nominal continued, "
                                 "seed 14, b = 0)", "the same network: the mean action leaves the track in C3 from "
                                 f"every start; with the training noise (σ = {sigma:.3f} of the steering range) all "
                                 "five runs pass")
        grid = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], left=0.02, right=0.98, top=top, bottom=0.08,
                                wspace=0.15, hspace=0.35)
        ax = self.zoom = fig.add_subplot(grid[:, 0])
        bare(ax)
        ax.set_aspect("equal", adjustable="datalim")
        x0, x1, y0, y1 = window_bbox(self.track, 2040, 2215, 6)
        TrackLayer(ax, self.track, pixels_per_metre(ax, (x0, x1, y0, y1)))
        ref = reference(0.0)
        ax.plot(ref["x"], ref["y"], color=OPTIMUM, lw=1.0, ls=(0, (5, 4)), alpha=0.55, zorder=2)
        self.trails = [Trail(ax, RANDOMISED, width=1.6, seconds=4.0) for _ in self.noisy]
        self.cars = [CarSprite(ax, RANDOMISED, zorder=6 + 0.3 * i) for i in range(len(self.noisy))]
        self.mean_trail = Trail(ax, FAILURE, width=2.6, seconds=4.0, zorder=4.5)
        self.mean_path, = ax.plot([], [], color=FAILURE, lw=1.2, alpha=0.6, zorder=4.4)
        self.mean_car = CarSprite(ax, FAILURE, zorder=9)
        self.crash, = ax.plot([], [], "X", ms=20, color=FAILURE, mec=SURFACE, zorder=12)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_autoscale_on(False)
        scale_bar(ax, 10.0)
        view_legend(ax, [car_handle(FAILURE, "mean action"), car_handle(RANDOMISED, "stochastic policy (5 runs)"),
                         Line2D([], [], color=OPTIMUM, lw=1.0, ls=(0, (5, 4)), alpha=0.7,
                                label="minimum-time racing line")])
        self.axes = []
        for row, (key, label, scale, limits) in enumerate((("vx", "speed [km/h]", 3.6, (50, 160)),
                                                           ("delta", "steering δ [deg]", 180 / np.pi, (-25, 10)),
                                                           ("ey", "lateral offset e_y [m]", 1.0, (-6.5, 6.5)))):
            axis = fig.add_subplot(grid[row, 1])
            axis.axvspan(2097, 2171, color=GRID, alpha=0.7, lw=0)
            axis.set_xlim(1990, 2230)
            axis.set_ylim(*limits)
            axis.set_ylabel(label)
            if key == "ey":
                w_left, w_right = self.track.real.boundaries_many(np.arange(1990, 2230.0))
                axis.plot(np.arange(1990, 2230.0), w_left, color=INK_3, lw=0.8)
                axis.plot(np.arange(1990, 2230.0), -w_right, color=INK_3, lw=0.8)
                axis.set_xlabel("position along the lap s [m]")
            else:
                axis.tick_params(labelbottom=False)
            lines = [axis.plot([], [], color=RANDOMISED, lw=1.0, alpha=0.8)[0] for _ in self.noisy]
            mean_line = axis.plot([], [], color=FAILURE, lw=1.8)[0]
            self.axes.append((key, scale, lines, mean_line))

    def render(self, k: int, path: Path) -> None:
        t = self.t0 + k * self.frame_dt
        for run, trail, car in zip(self.noisy, self.trails, self.cars, strict=True):
            now = sample_at(run, t)
            trail.update(run, t, self.zoom.transData)
            car.place(now["x"], now["y"], now["psi"], now["delta"], self.zoom.transData)
        past = (self.mean["t"] <= t) & (self.mean["s"] >= 1990)
        now = sample_at(self.mean, min(t, self.t_fail))
        self.mean_trail.update(self.mean, min(t, self.t_fail), self.zoom.transData)
        path = dense(self.mean, past)
        self.mean_path.set_data(path["x"], path["y"])
        self.mean_car.place(now["x"], now["y"], now["psi"], now["delta"], self.zoom.transData)
        if t >= self.t_fail:
            self.crash.set_data([now["x"]], [now["y"]])
        for key, scale, lines, mean_line in self.axes:
            for run, line in zip(self.noisy, lines, strict=True):
                keep = (run["t"] <= t) & (run["s"] >= 1990)
                line.set_data(run["s"][keep], scale * run[key][keep])
            mean_line.set_data(self.mean["s"][past], scale * self.mean[key][past])
        state = "OFF TRACK" if t >= self.t_fail else f"{3.6 * now['vx']:5.1f} km/h"
        self.clock.set_text(f"t {t:6.2f} s   mean action at s = {now['s']:7.1f} m   {state}")
        self.fig.savefig(path, dpi=DPI)


SCENE: Any = None


def make_scene(spec: tuple) -> Any:
    kind, *args = spec
    return {"chase": Chase, "race": Race, "balance": Balance, "stochastic": Stochastic}[kind](*args)


def init_worker(spec: tuple) -> None:
    global SCENE
    SCENE = make_scene(spec)
    SCENE.setup()


def render_chunk(job: tuple[list[int], str]) -> int:
    frames, folder = job
    for k in frames:
        SCENE.render(k, Path(folder) / f"{k:05d}.png")
    return len(frames)


def animate(spec: tuple, name: str, output: Path, workers: int, mp4_fps: float, gif_fps: float,
            gif_width: int) -> None:
    frames = make_scene(spec).frames
    folder = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=output))
    chunks = [(list(range(i, frames, workers)), str(folder)) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(spec,)) as pool:
        sum(pool.map(render_chunk, chunks))
    pattern = str(folder / "%05d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(mp4_fps), "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "scale=1600:-2",
                    str(output / f"{name}.mp4")], check=True)
    palette = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:"
               "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(mp4_fps), "-i", pattern,
                    "-vf", palette, "-loop", "0", str(output / f"{name}.gif")], check=True)
    shutil.rmtree(folder)
    sizes = ", ".join(f"{(output / f'{name}.{ext}').stat().st_size / 1e6:.1f} MB" for ext in ("gif", "mp4"))
    print(f"{name}: {frames} frames ({sizes})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=("chase", "race", "balance", "stochastic"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-titles", action="store_true", help="leave out the heading lines of every frame")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/thesis/animations"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    titles = not args.no_titles
    groups = args.only or ("chase", "race", "balance", "stochastic")
    # frames every 0.125 s of lap time: the MP4 at 24 fps plays at 3x, the GIF keeps one frame in three at 8 fps
    if "chase" in groups:
        for tag in ("nominal_limit", "randomised_limit_w1", "oracle_limit_w1"):
            animate(("chase", tag, 12, 0.125, titles), f"chase_{tag}_s12", args.output, args.workers, 24, 8, 1280)
    if "race" in groups:
        animate(("race", 12, 0.125, titles), "race_flying_lap_s12", args.output, args.workers, 24, 8, 1280)
    if "balance" in groups:
        for b in (0.05, -0.21):
            animate(("balance", b, 0.125, titles), f"balance_b{b:+.2f}_s12", args.output, args.workers, 24, 8, 1280)
    if "stochastic" in groups:
        # one frame per decision, played at real time
        animate(("stochastic", 0.05, titles), "mean_vs_stochastic_s14", args.output, args.workers, 20, 10, 1280)


if __name__ == "__main__":
    main()
