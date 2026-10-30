"""Static figures of the pure-RL results at the limit, for the thesis and the slides (PNG and PDF).

    python scripts/figures/make_thesis_figures.py
    python scripts/figures/make_thesis_figures.py --only gg_diagram lap_profiles
    python scripts/figures/make_thesis_figures.py --no-titles --output outputs/figures/thesis_no_titles
    TEN_TENTHS_THEME=dark python scripts/figures/make_thesis_figures.py --output outputs/figures/thesis_dark

Reads the recorded laps (scripts/figures/record_laps.py), the evaluations of outputs/evaluation/limit and
outputs/evaluation/balance_limit, the training curves of outputs/logs and the steady-state analysis of the balance
index, and writes <output>/<number>_<name>.png and .pdf. The policy shown alone is seed 12, the best seed of the
Nominal at the limit; the Randomised and Oracle policies of seed 12 continue from it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import thesis_plots
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from thesis_plots import (
    COLOURS,
    DIVERGING,
    EDGE,
    FAILURE,
    GRID,
    INK,
    INK_2,
    INK_3,
    LABELS,
    NOMINAL,
    NOMINAL_LIGHT,
    OPTIMUM,
    ORACLE,
    RANDOMISED,
    SEEDS,
    SPEED_MAP,
    SURFACE,
    TAGS,
    TRACK_FILL,
    WINDOWS,
    G,
    Track,
    apply_style,
    dense,
    draw_track,
    flying,
    load,
    load_track,
    reference,
    reference_time_at,
    save,
    time_at,
    window_bbox,
)

EVALUATION = Path("outputs/evaluation")
LOGS = Path("outputs/logs")
SEED = 12
BALANCE_TAGS = ("nominal_limit_cont", "randomised_limit_w13", "oracle_limit_w13", "randomised_limit_w23",
                "oracle_limit_w23", "randomised_limit_w1", "oracle_limit_w1")
SHORT = {"nominal_limit_cont": "Nominal", "randomised_limit_w13": "R W1/3", "oracle_limit_w13": "O W1/3",
         "randomised_limit_w23": "R W2/3", "oracle_limit_w23": "O W2/3", "randomised_limit_w1": "R W1",
         "oracle_limit_w1": "O W1"}
RANGES = {"w13": (-0.07, 0.01), "w23": (-0.14, 0.02), "w1": (-0.21, 0.03)}
HELD_OUT = {"w13": -0.04, "w23": -0.07, "w1": -0.11}
FIGURES: dict[str, Callable[[Track, Path], None]] = {}


def figure(name: str) -> Callable:
    def register(function: Callable[[Track, Path], None]) -> Callable[[Track, Path], None]:
        FIGURES[name] = function
        return function
    return register


def coloured_line(ax, x, y, values, cmap, vmin, vmax, width=2.2):
    points = np.stack([x, y], axis=1)[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=width, capstyle="round")
    lines.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection(lines)
    return lines


def flying_laps(tag: str, track: Track, seeds=SEEDS) -> dict[int, dict]:
    laps = {}
    for seed in seeds:
        lap = flying(load(f"flying_{tag}_s{seed}"), track.length)
        if lap is not None:
            laps[seed] = lap
    return laps


def shade_windows(ax, label: bool = False) -> None:
    for i, (a, b) in enumerate(WINDOWS):
        ax.axvspan(a, b, color=GRID, alpha=0.7, lw=0, zorder=0)
        if label:
            nudge = {7: -14.0, 8: 14.0}.get(i, 0.0)  # C8 and C9 are 20 m apart
            ax.text(0.5 * (a + b) + nudge, 1.0, f"C{i + 1}", transform=ax.get_xaxis_transform(),
                    ha="right" if i == 7 else "left" if i == 8 else "center", va="bottom", fontsize=8, color=INK_2)


def no_frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def scale_bar(ax, metres: float = 200.0, where: tuple[float, float] = (0.04, 0.04)) -> None:
    """A bar of ``metres`` in a view whose data unit is the metre, at the axes fraction ``where``."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    left, bottom = x0 + where[0] * (x1 - x0), y0 + where[1] * (y1 - y0)
    ax.plot([left, left + metres], [bottom, bottom], color=INK, lw=2.2, solid_capstyle="butt", zorder=9)
    ax.text(left + metres / 2, bottom + 0.012 * (y1 - y0), f"{metres:.0f} m", ha="center", va="bottom", fontsize=8.5,
            color=INK_2, zorder=9)


def outward(track: Track, s: float, distance: float) -> np.ndarray:
    """The point ``distance`` metres from the centreline at s, on the side away from the centre of the circuit."""
    p = track.point(s)
    n = track.point(s, 1.0) - p
    centre = track.centre.mean(axis=0)
    side = 1.0 if np.dot(n, p - centre) > 0 else -1.0
    return p + side * distance * n


def arrow(ax, start, vector, colour, label=None, offset=(0.0, 0.0), lw=1.8, fontsize=11, ha="center"):
    """An arrow from ``start`` along ``vector`` (data units), with a label near its tip."""
    start, vector = np.asarray(start, dtype=float), np.asarray(vector, dtype=float)
    ax.annotate("", start + vector, start, arrowprops={"arrowstyle": "-|>", "color": colour, "lw": lw,
                                                       "mutation_scale": 13, "shrinkA": 0, "shrinkB": 0}, zorder=6)
    if label:
        ax.text(*(start + vector + np.asarray(offset)), label, color=colour, fontsize=fontsize, ha=ha, va="center",
                zorder=7)


def rotate(points, angle):
    c, s_ = np.cos(angle), np.sin(angle)
    return np.asarray(points) @ np.array([[c, s_], [-s_, c]])


@figure("vehicle_model")
def vehicle_model(track: Track, out: Path) -> None:
    from contextual_bicycle_pacejka.spec.constants import LF, LR

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    # a left-hand bend of radius R around the origin, the centreline at radius R
    radius, half = 40.0, 6.0
    phi = np.linspace(np.radians(-118), np.radians(-62), 200)
    for r, kw in ((radius + half, {"color": EDGE, "lw": 1.1}), (radius - half, {"color": EDGE, "lw": 1.1})):
        ax.plot(r * np.cos(phi), r * np.sin(phi) + radius, **kw)
    ax.fill(np.r_[(radius + half) * np.cos(phi), (radius - half) * np.cos(phi[::-1])],
            np.r_[(radius + half) * np.sin(phi), (radius - half) * np.sin(phi[::-1])] + radius, color=TRACK_FILL, lw=0)
    ax.plot(radius * np.cos(phi), radius * np.sin(phi) + radius, color=INK_3, lw=1.2, ls=(0, (6, 4)))
    # the closest point of the centreline, its tangent and normal
    phi0 = np.radians(-90)
    foot = np.array([radius * np.cos(phi0), radius * np.sin(phi0) + radius])
    tangent = np.array([-np.sin(phi0), np.cos(phi0)])
    normal = np.array([-tangent[1], tangent[0]])
    ey, epsi, delta = 3.0, np.radians(13), np.radians(16)
    cg = foot + ey * normal
    heading = np.arctan2(tangent[1], tangent[0]) + epsi
    body = rotate(np.array([[2.25, 0.95], [-2.25, 0.95], [-2.25, -0.95], [2.25, -0.95]]) * [1, 1], heading) + cg
    ax.add_patch(plt.Polygon(body, closed=True, fc=RANDOMISED, alpha=0.12, ec=RANDOMISED, lw=1.2, zorder=3))
    axis = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-axis[1], axis[0]])
    front, rear = cg + LF * axis, cg - LR * axis
    ax.plot(*np.array([rear, front]).T, color=INK, lw=1.4, zorder=4)
    for centre, angle in ((front, heading + delta), (rear, heading)):
        wheel = rotate(np.array([[0.45, 0.13], [-0.45, 0.13], [-0.45, -0.13], [0.45, -0.13]]), angle) + centre
        ax.add_patch(plt.Polygon(wheel, closed=True, fc=INK, ec="none", zorder=5))
    ax.plot(*cg, "o", ms=7, mfc=SURFACE, mec=INK, mew=1.4, zorder=8)
    ax.text(*(cg + 0.45 * axis - 0.55 * side), "CG", fontsize=9, color=INK, ha="left", va="top", zorder=8)
    # velocities in the body frame
    arrow(ax, cg, 3.4 * axis, RANDOMISED, "$v_x$", 0.55 * axis)
    arrow(ax, cg, 1.4 * side, RANDOMISED, "$v_y$", 0.45 * side)
    # tyre forces
    front_normal = np.array([-np.sin(heading + delta), np.cos(heading + delta)])
    arrow(ax, front, 2.4 * front_normal, ORACLE, "$F_{y,f}$", 0.5 * front_normal)
    arrow(ax, rear, 2.0 * side, ORACLE, "$F_{y,r}$", 0.5 * side)
    arrow(ax, rear, -1.8 * axis, ORACLE, "$F_{x,r}$", -0.6 * axis)
    # steering angle and heading error
    ax.plot(*np.array([front, front + 1.9 * axis]).T, color=INK_3, lw=0.8, ls=":")
    ax.plot(*np.array([front, front + 1.9 * np.array([np.cos(heading + delta), np.sin(heading + delta)])]).T,
            color=INK_3, lw=0.8)
    a = np.linspace(heading, heading + delta, 20)
    ax.plot(front[0] + 1.6 * np.cos(a), front[1] + 1.6 * np.sin(a), color=INK_2, lw=1.0)
    ax.text(*(front + 2.05 * np.array([np.cos(heading + delta / 2), np.sin(heading + delta / 2)])), "$\\delta$",
            fontsize=11, color=INK_2, ha="center", va="center")
    ax.plot(*np.array([cg, cg + 5.6 * tangent]).T, color=INK_3, lw=0.8, ls=":")
    a = np.linspace(np.arctan2(tangent[1], tangent[0]), heading, 20)
    ax.plot(cg[0] + 4.9 * np.cos(a), cg[1] + 4.9 * np.sin(a), color=INK_2, lw=1.0)
    ax.text(*(cg + 5.4 * np.array([np.cos(a[10]), np.sin(a[10])])), "$e_\\psi$", fontsize=11, color=INK_2,
            ha="left", va="center")
    # yaw rate
    a = np.linspace(np.radians(110), np.radians(230), 30) + heading
    ax.plot(cg[0] + 0.75 * np.cos(a), cg[1] + 0.75 * np.sin(a), color=INK_2, lw=1.0)
    ax.annotate("", cg + 0.75 * np.array([np.cos(a[-1]), np.sin(a[-1])]),
                cg + 0.75 * np.array([np.cos(a[-3]), np.sin(a[-3])]),
                arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.0, "mutation_scale": 9})
    ax.text(*(cg + 1.2 * np.array([np.cos(a[12]), np.sin(a[12])])), "$r$", fontsize=11, color=INK_2,
            ha="center", va="center")
    # lateral offset from the closest point of the centreline, arc length s
    ax.annotate("", cg, foot, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.1, "shrinkA": 0, "shrinkB": 4})
    ax.text(*(0.42 * foot + 0.58 * cg - 0.35 * tangent), "$e_y$", fontsize=11, color=INK, ha="right", va="center")
    ax.plot(*foot, "o", ms=5, color=INK, zorder=6)
    ax.text(*(foot - 0.55 * normal), "closest point of the centreline: arc length $s$", fontsize=9,
            color=INK_2, ha="center", va="top")
    s_arc = np.linspace(np.radians(-110), np.radians(-100), 20)
    ax.annotate("", (radius + 1.2) * np.array([np.cos(s_arc[-1]), np.sin(s_arc[-1])]) + [0, radius],
                (radius + 1.2) * np.array([np.cos(s_arc[0]), np.sin(s_arc[0])]) + [0, radius],
                arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.0, "connectionstyle": "arc3,rad=0.04"})
    ax.text(*((radius + 1.9) * np.array([np.cos(s_arc[8]), np.sin(s_arc[8])]) + [0, radius]),
            "$\\dot s$: progress along the centreline", fontsize=9, color=INK_2, ha="center", va="top")
    ax.text(7.0, 2.3, "centreline $x(s),\\ y(s)$, curvature $\\kappa(s)$\n"
            "edges at $w_\\mathrm{left}(s),\\ w_\\mathrm{right}(s)$", fontsize=9, color=INK_2, ha="left", va="bottom")
    rows = [("state", "$z = [s,\\ e_y,\\ e_\\psi,\\ v_x,\\ v_y,\\ r,\\ \\delta]$ in the Frenet frame of the "
                      "centreline"),
            ("action", "$a = [\\delta_\\mathrm{ref},\\ \\dot\\tau] \\in [-1, 1]^2$ at 20 Hz: steering reference and "
                       "rate of the command $\\tau$ (+ throttle, − brake)"),
            ("setup", "$\\xi = [m,\\ \\mu,\\ h_\\mathrm{cg},\\ \\tau_d,\\ b]$, with only the balance index $b$ varied"),
            ("fixed", f"$l_f$ = {LF} m, $l_r$ = {LR} m, $I_z$ = 1800 kg m², rear-wheel drive, brakes 65 % front, "
                      "drag and 249 kW at the wheels")]
    for k, (name, text) in enumerate(rows):
        y = 0.135 - 0.043 * k
        fig.text(0.07, y, name, fontsize=9.5, color=INK_2, weight="semibold", va="center")
        fig.text(0.15, y, text, fontsize=9.5, color=INK, va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-13.5, 13.5)
    ax.set_ylim(-2.6, 6.9)
    no_frame(ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.2)
    ax.set_title("Single-track model in the Frenet frame of the centreline")
    save(fig, out, "00_vehicle_model")
    plt.close(fig)


@figure("tyre_model")
def tyre_model(track: Track, out: Path) -> None:
    from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
    from contextual_bicycle_pacejka.model.tires import (
        compute_combined_slip_capacity,
        compute_lateral_force,
        compute_saturation_threshold,
        pacejka_coefficients,
    )
    from contextual_bicycle_pacejka.spec.constants import LF, LR, MASS_NOM, G

    setup = sample_setup(np.random.default_rng(0), (0.0, 0.0))
    alpha = np.radians(np.linspace(-20, 20, 801))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    ax = axes[0]
    for axle, colour in (("front", RANDOMISED), ("rear", ORACLE)):
        b, c, e = pacejka_coefficients(axle)
        peak = compute_saturation_threshold(setup, axle)
        ax.plot(np.degrees(alpha), [compute_lateral_force(a, b, c, 1.0, e) for a in alpha], color=colour,
                label=f"{axle}: B = {b:.2f}, C = {c:.2f}, E = {e:.2f}")
        ax.axvline(np.degrees(peak), color=colour, lw=0.9, ls=(0, (3, 2)))
        ax.text(np.degrees(peak) + 0.4, -0.18 - 0.16 * (axle == "rear"), f"{np.degrees(peak):.1f}°", color=colour,
                fontsize=8.5)
    ax.set_xlabel("slip angle $\\alpha$ [deg]")
    ax.set_ylabel("$F_y\\,/\\,D$")
    ax.set_title("Pacejka lateral force per unit capacity")
    ax.text(0.02, 0.97, "dashed: peak slip $\\alpha_\\mathrm{peak}$,\nmeasured, never imposed", transform=ax.transAxes,
            fontsize=8.5, color=INK_2, va="top")
    ax.legend(loc="lower right", fontsize=8.5)
    ax = axes[1]
    fz = MASS_NOM / (LF + LR) * G * LR
    b, c, e = pacejka_coefficients("front")
    shades = ("#2a78d6", "#6da7ec", "#b7d3f6") if not thesis_plots.DARK else ("#9ec5f4", "#5598e7", "#256abf")
    for ratio, colour in zip((0.0, 0.6, 0.9), shades, strict=True):
        d = compute_combined_slip_capacity(ratio * fz, 1.0, fz)
        ax.plot(np.degrees(alpha), [compute_lateral_force(a, b, c, d, e) / 1000 for a in alpha], color=colour,
                label=f"$F_x\\,/\\,\\mu F_z$ = {ratio:.1f}")
    ax.set_xlabel("front slip angle $\\alpha_f$ [deg]")
    ax.set_ylabel("$F_{y,f}$ [kN]")
    ax.set_title("Combined slip: braking takes lateral capacity")
    ax.legend(loc="lower right", fontsize=8.5)
    ax = axes[2]
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(circle), np.sin(circle), color=TRACK_FILL, lw=0)
    ax.plot(np.cos(circle), np.sin(circle), color=EDGE, lw=1.0)
    for ratio, colour in zip((0.0, 0.6, 0.9), shades, strict=True):
        d = np.sqrt(1 - ratio**2)
        ax.plot([ratio, ratio], [-d, d], color=colour, lw=2.4, solid_capstyle="round")
        ax.plot(ratio, d, "o", color=colour, ms=6, mec=SURFACE, mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("$F_x\\,/\\,\\mu F_z$ (the longitudinal force has priority)")
    ax.set_ylabel("$D\\,/\\,\\mu F_z$")
    ax.set_title("Friction circle of an axle")
    fig.tight_layout()
    save(fig, out, "00_tyre_model")
    plt.close(fig)


@figure("track_layout")
def track_layout(track: Track, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    draw_track(ax, track, 0.8)
    for i, (a, b) in enumerate(WINDOWS):
        keep = (track.s >= a) & (track.s <= b)
        ax.plot(*track.centre[keep].T, color=ORACLE, lw=5.5, solid_capstyle="round", zorder=2)
        mid = 0.5 * (a + b)
        p, q = track.point(mid), outward(track, mid, 46 if i not in (7, 8) else 58)
        if i == 7:
            q = q + np.array([-34.0, -6.0])
        ax.annotate(f"C{i + 1}", p, q, ha="center", va="center", fontsize=9.5, color=INK, weight="medium",
                    arrowprops={"arrowstyle": "-", "color": INK_3, "lw": 0.6, "shrinkA": 6, "shrinkB": 4})
    for k in range(10):
        s0 = k * track.length / 10
        p = track.point(s0)
        ax.plot(*p, "o", ms=12, mfc=SURFACE, mec=RANDOMISED, mew=1.6, zorder=3)
        ax.text(*p, f"{k + 1}", fontsize=7, color=RANDOMISED, ha="center", va="center", weight="semibold", zorder=4)
    left, right = track.left[0], track.right[0]
    ax.plot([left[0], right[0]], [left[1], right[1]], color=INK, lw=3, zorder=2.5)
    p0, p1 = track.point(0.0), track.point(70.0)
    ax.annotate("", p1 + (p1 - p0) * 0.0 + np.array([22.0, -4.0]), p0 + np.array([22.0, -4.0]),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3, "mutation_scale": 12}, zorder=5)
    ax.text(*(p0 + np.array([34, 10])), "start / finish", fontsize=9, color=INK, va="center")
    handles = [Line2D([], [], color=ORACLE, lw=5, label="grip-limited corners C1–C9"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=RANDOMISED, mew=1.6, ms=9,
                      label="the 10 protocol starts (3 m/s), every 465 m")]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(f"Circuit de Barcelona-Catalunya: {track.length:.0f} m")
    no_frame(ax)
    ax.margins(0.04)
    ax.autoscale_view()
    scale_bar(ax, 200.0, (0.72, 0.04))
    save(fig, out, "01_track_layout")
    plt.close(fig)


@figure("optimal_line")
def optimal_line(track: Track, out: Path) -> None:
    lap_time = float(reference(0.0)["lap_time"])
    ref = dense(reference(0.0), dt=0.02)
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    draw_track(ax, track, 0.6)
    lines = coloured_line(ax, ref["x"], ref["y"], 3.6 * ref["vx"], SPEED_MAP, 60, 250, 2.8)
    bar = fig.colorbar(lines, ax=ax, shrink=0.5, pad=0.0, aspect=24)
    bar.set_label("speed [km/h]")
    bar.outline.set_visible(False)
    bar.ax.tick_params(size=0)
    for index, offset in ((np.argmin(ref["vx"]), (-80, -18)), (np.argmax(ref["vx"]), (30, -14))):
        ax.annotate(f"{3.6 * ref['vx'][index]:.0f} km/h", (ref["x"][index], ref["y"][index]), offset,
                    textcoords="offset points", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK_3, "lw": 0.6})
    ax.set_title(f"Minimum-time lap, b = 0: {lap_time:.2f} s flying")
    no_frame(ax)
    ax.autoscale_view()
    scale_bar(ax, 200.0, (0.62, 0.04))
    save(fig, out, "02_optimal_line")
    plt.close(fig)


@figure("speed_difference_map")
def speed_difference_map(track: Track, out: Path) -> None:
    ref = reference(0.0)
    lap = flying_laps("nominal_limit", track, (SEED,))[SEED]
    s = np.arange(0.0, track.length, 1.0)
    dv = 3.6 * (np.interp(s, lap["s"], lap["vx"]) - np.interp(s, ref["s_grid"], np.interp(
        ref["s_grid"], ref["s"], ref["vx"])))
    gap = time_at(lap, s) - reference_time_at(ref, s)
    xy = np.array([track.point(v) for v in s])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    for ax, values, cmap, limits, label, title in (
        (axes[0], dv, DIVERGING.reversed(), (-30, 30), "policy − optimum speed [km/h]",
         "Speed difference at the same position"),
        (axes[1], gap, plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "loss", [GRID, "#ee8a7a", "#b3302f", "#5c1413"]), (0, float(gap[-1])),
         "cumulative time lost [s]", "Time lost to the optimum, accumulated along the lap"),
    ):
        draw_track(ax, track, 0.6)
        lines = coloured_line(ax, xy[:, 0], xy[:, 1], values, cmap, *limits, 3.2)
        bar = fig.colorbar(lines, ax=ax, shrink=0.55, pad=0.01, orientation="horizontal")
        bar.set_label(label)
        bar.outline.set_visible(False)
        ax.set_title(title)
        no_frame(ax)
    fig.suptitle(f"Nominal at the limit (seed {SEED}), flying lap {float(lap['lap_time']):.2f} s vs minimum-time "
                 f"lap {float(ref['lap_time']):.2f} s", y=0.98)
    save(fig, out, "03_speed_difference_map")
    plt.close(fig)


@figure("lap_profiles")
def lap_profiles(track: Track, out: Path) -> None:
    ref = reference(0.0)
    laps = flying_laps("nominal_limit", track)
    s = np.arange(0.0, track.length, 1.0)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.6, 2]})
    for ax in axes:
        shade_windows(ax)
    shade_windows(axes[0], label=True)
    w_left, w_right = track.real.boundaries_many(s)
    axes[1].fill_between(s, -w_right, w_left, color=TRACK_FILL, lw=0)
    axes[1].plot(s, w_left, color=EDGE, lw=0.8)
    axes[1].plot(s, -w_right, color=EDGE, lw=0.8)
    for seed, lap in laps.items():
        highlight = seed == SEED
        style = {"color": NOMINAL if highlight else NOMINAL_LIGHT, "lw": 1.5 if highlight else 0.8,
                 "zorder": 3 if highlight else 2}
        axes[0].plot(lap["s"], 3.6 * lap["vx"], **style)
        axes[1].plot(lap["s"], lap["ey"], **style)
        axes[2].plot(lap["s"], lap["tau"], **style)
        axes[3].plot(s, time_at(lap, s) - reference_time_at(ref, s), **style,
                     label=f"seed {seed}: {float(lap['lap_time']):.2f} s")
    axes[0].plot(ref["s"], 3.6 * ref["vx"], color=OPTIMUM, lw=1.3, ls="--", zorder=4)
    axes[1].plot(ref["s"], ref["ey"], color=OPTIMUM, lw=1.1, ls="--", zorder=4)
    axes[2].plot(ref["s"], ref["tau"], color=OPTIMUM, lw=1.1, ls="--", zorder=4)
    axes[3].axhline(0, color=OPTIMUM, lw=1.1, ls="--")
    axes[0].set_ylabel("speed [km/h]")
    axes[1].set_ylabel("lateral offset $e_y$ [m]\n(+ left)")
    axes[2].set_ylabel("command $\\tau$\n(+ throttle, − brake)")
    axes[3].set_ylabel("time behind\nthe optimum [s]")
    axes[3].set_xlabel("position along the lap s [m]")
    axes[3].legend(ncol=5, fontsize=8, loc="upper left")
    axes[0].legend(handles=[Line2D([], [], color=OPTIMUM, ls="--", label=f"minimum-time lap "
                                   f"({float(ref['lap_time']):.2f} s)"),
                            Line2D([], [], color=NOMINAL, label=f"Nominal at the limit, seed {SEED}"),
                            Line2D([], [], color=NOMINAL_LIGHT, lw=0.8, label="other seeds")],
                   loc="lower left", bbox_to_anchor=(0.0, 1.06), fontsize=8.5, ncol=3)
    axes[0].set_xlim(0, track.length)
    fig.suptitle("Flying lap of the Nominal at the limit against the minimum-time lap (b = 0)", y=0.995)
    fig.tight_layout()
    save(fig, out, "04_lap_profiles")
    plt.close(fig)


@figure("corner_zooms")
def corner_zooms(track: Track, out: Path) -> None:
    ref = reference(0.0)
    runs = {tag: flying_laps(tag, track, (SEED,))[SEED]
            for tag in ("nominal_limit", "randomised_limit_w1", "oracle_limit_w1")}
    colours = {"nominal_limit": NOMINAL, "randomised_limit_w1": RANDOMISED, "oracle_limit_w1": ORACLE}
    chosen = (0, 2, 3, 4, 6, 7)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, k in zip(axes.flat, chosen, strict=True):
        a, b = WINDOWS[k]
        start, end = a - 110, b + 80
        draw_track(ax, track, 1.0)
        path = dense(ref, (ref["s"] >= start) & (ref["s"] <= end))
        ax.plot(path["x"], path["y"], color=OPTIMUM, ls="--", lw=1.4, zorder=5)
        for tag, lap in runs.items():
            path = dense(lap, (lap["s"] >= start) & (lap["s"] <= end))
            ax.plot(path["x"], path["y"], color=colours[tag], lw=1.8, zorder=4)
        x0, x1, y0, y1 = window_bbox(track, start, end, 12)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        slow = ref["vx"][(ref["s"] >= a) & (ref["s"] <= b)].min()
        mins = "  ".join(f"{3.6 * lap['vx'][(lap['s'] >= a) & (lap['s'] <= b)].min():.0f}" for lap in runs.values())
        ax.set_title(f"C{k + 1}  ·  {a:.0f}–{b:.0f} m\nminimum speed [km/h]: optimum {3.6 * slow:.0f} · "
                     + " · ".join(f"{n} {v}" for n, v in zip("NRO", mins.split(), strict=True)), fontsize=9.5)
        no_frame(ax)
    handles = [Line2D([], [], color=OPTIMUM, ls="--", label="minimum-time lap"),
               Line2D([], [], color=NOMINAL, label=f"Nominal at the limit (s{SEED})"),
               Line2D([], [], color=RANDOMISED, label=f"Randomised W1 (s{SEED})"),
               Line2D([], [], color=ORACLE, label=f"Oracle W1 (s{SEED})")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Racing lines in the grip-limited corners, b = 0 (flying lap)", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out, "05_corner_zooms")
    plt.close(fig)


def reference_nodes(b: float = 0.0) -> dict[str, np.ndarray]:
    """Accelerations and speed of the minimum-time lap at the nodes of its optimal-control problem.

    The recorded reference is resampled in time, and between two nodes its accelerations are interpolated across the
    jumps of the controls, which scatters points inside the friction circle; the nodes are the solution itself."""
    from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
    from contextual_bicycle_pacejka.model.dynamics import compute_vehicle_rates
    from contextual_bicycle_pacejka.model.state import VehicleMemory, VehicleState

    data = json.loads(Path(f"outputs/reference/min_time_b{b:+.2f}.json").read_text())
    st = {k: np.asarray(v) for k, v in data["states"].items()}
    setup = sample_setup(np.random.default_rng(0), (b, b))
    ax, ay = np.zeros(len(data["s"])), np.zeros(len(data["s"]))
    for i in range(len(data["s"])):
        state = VehicleState(data["s"][i], st["ey"][i], st["epsi"][i], st["vx"][i], st["vy"][i], st["r"][i],
                             st["delta"][i])
        memory = VehicleMemory.zeros()
        for _ in range(8):  # no one-step delay in the reference: iterate the load transfer to its fixed point
            _, _, _, diag = compute_vehicle_rates(state, float(st["tau"][i]), setup, memory)
            memory = VehicleMemory(F_body_x_prev=diag["F_body_x"])
        ax[i], ay[i] = diag["F_body_x"] / setup.m, diag["F_body_y"] / setup.m
    return {"ax": ax, "ay": ay, "vx": st["vx"], "lap_time": np.array(data["lap_time"])}


@figure("gg_diagram")
def gg_diagram(track: Track, out: Path) -> None:
    ref = reference_nodes(0.0)
    lap = flying_laps("nominal_limit", track, (SEED,))[SEED]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True, sharey=True, layout="constrained")
    circle = np.linspace(0, 2 * np.pi, 200)
    for ax, data, title in ((axes[0], ref, f"Minimum-time lap ({float(ref['lap_time']):.2f} s)"),
                            (axes[1], lap, f"Nominal at the limit, seed {SEED} ({float(lap['lap_time']):.2f} s)")):
        ax.plot(np.cos(circle), np.sin(circle), color=INK_3, lw=1, ls="--", zorder=1)
        points = ax.scatter(data["ay"] / G, data["ax"] / G, c=3.6 * data["vx"], cmap=SPEED_MAP, vmin=60, vmax=250,
                            s=6, lw=0, zorder=2)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("lateral acceleration $a_y\\,/\\,g$")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.0)
    axes[0].set_ylabel("longitudinal acceleration $a_x\\,/\\,g$")
    axes[0].text(0.72, 0.72, "$\\mu g$", color=INK_3, fontsize=9)
    bar = fig.colorbar(points, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label("speed [km/h]")
    bar.outline.set_visible(False)
    fig.suptitle("g-g diagram: how much of the friction circle the lap uses (b = 0)")
    save(fig, out, "06_gg_diagram")
    plt.close(fig)


@figure("tyre_usage")
def tyre_usage(track: Track, out: Path) -> None:
    ref = reference(0.0)
    laps = flying_laps("nominal_limit", track)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bins = np.linspace(0, 1.6, 33)

    def cornering(data):
        return np.abs(data["ay"]) > 0.5 * G

    pooled = {key: np.concatenate([lap[key][cornering(lap)] for lap in laps.values()]) for key in ("slip_f", "slip_r")}
    for ax, key, axle in ((axes[0], "slip_f", "front axle"), (axes[1], "slip_r", "rear axle")):
        ax.hist(ref[key][cornering(ref)], bins=bins, density=True, histtype="step", color=OPTIMUM, lw=1.6, ls="--",
                label="minimum-time lap")
        ax.hist(pooled[key], bins=bins, density=True, color=NOMINAL, alpha=0.55, lw=0,
                label="Nominal at the limit (5 seeds)")
        ax.axvline(1.0, color=FAILURE, lw=1.2)
        ax.text(1.02, 0.95, "peak slip", transform=ax.get_xaxis_transform(), color=FAILURE, fontsize=8.5, va="top")
        beyond = np.mean(pooled[key] > 1.0)
        ax.set_title(f"{axle}: policy beyond the peak {100 * beyond:.0f} % of the time")
        ax.set_xlabel("slip $|\\alpha|\\,/\\,\\alpha_{\\mathrm{peak}}$")
    axes[0].set_ylabel("density (samples with $|a_y| > 0.5\\,g$)")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Tyre use in corners: the optimum holds both axles at their peak slip; the policy spreads below and "
                 "beyond it, with the rear mostly at 0.7–0.9 of its peak")
    fig.tight_layout()
    save(fig, out, "07_tyre_usage")
    plt.close(fig)


def read_log(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    failures = np.array([sum(r["failures_per_1000"].values()) for r in rows])
    return {"decisions": np.array([r["timesteps"] for r in rows]) / 1e6,
            "speed": np.array([r["mean_speed"] for r in rows]),
            "progress": np.array([r["progress_per_decision"] for r in rows]),
            "failures": failures,
            "slip_f": np.array([r["slip_beyond_peak"]["front"] for r in rows]),
            "slip_r": np.array([r["slip_beyond_peak"]["rear"] for r in rows])}


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, values[0]), values, np.full(window - 1 - window // 2, values[-1])])
    return np.convolve(padded, kernel, mode="valid")


@figure("training_nominal")
def training_nominal(track: Track, out: Path) -> None:
    logs = {seed: read_log(LOGS / "nominal_limit" / f"s{seed}_stage7.jsonl") for seed in SEEDS}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    panels = (("speed", "mean speed in training [m/s]", False), ("failures", "failures per 1000 decisions", True),
              ("slip_f", "share of decisions with the front\nbeyond its peak slip", False),
              ("slip_r", "share of decisions with the rear\nbeyond its peak slip", False))
    n = min(len(data["decisions"]) for data in logs.values())
    x = logs[SEEDS[0]]["decisions"][:n]
    for ax, (key, label, log) in zip(axes.flat, panels, strict=True):
        runs = np.array([smooth(data[key][:n], 41) for data in logs.values()])
        floor = 1e-3 if log else -np.inf
        for values in runs:
            ax.plot(x, np.maximum(values, floor), color=NOMINAL_LIGHT, lw=0.9, zorder=2)
        ax.plot(x, np.maximum(runs.mean(axis=0), floor), color=NOMINAL, lw=2.0, zorder=3)
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0) if key != "speed"
                                         else plt.matplotlib.ticker.ScalarFormatter())
    for ax in axes[1]:
        ax.set_xlabel("decisions [millions]")
    axes[0, 0].legend(handles=[Line2D([], [], color=NOMINAL, lw=2, label="mean of 5 seeds"),
                               Line2D([], [], color=NOMINAL_LIGHT, lw=1, label="each seed")], loc="lower right")
    fig.suptitle("Training of the Nominal at the limit from random weights (free stage, PPO, 600 M decisions)")
    fig.tight_layout()
    save(fig, out, "08_training_nominal")
    plt.close(fig)


@figure("training_continued")
def training_continued(track: Track, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for tag in BALANCE_TAGS:
        runs = [read_log(LOGS / tag / f"s{seed}_stage7.jsonl") for seed in SEEDS]
        n = min(len(r["decisions"]) for r in runs)
        x = runs[0]["decisions"][:n]
        style = {"color": COLOURS[tag], "label": LABELS[tag], "lw": 1.5,
                 "ls": "-" if not tag.startswith("oracle") else (0, (5, 2))}
        for ax, key in zip(axes, ("speed", "failures", "slip_r"), strict=True):
            ax.plot(x, smooth(np.mean([r[key][:n] for r in runs], axis=0)), **style)
    axes[0].set_ylabel("mean speed in training [m/s]")
    axes[1].set_ylabel("failures per 1000 decisions")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("share with the rear beyond its peak")
    for ax in axes:
        ax.set_xlabel("decisions [millions]")
    axes[0].legend(fontsize=7.5, ncol=1)
    fig.suptitle("Continuation from the Nominal at the limit (150 M decisions, mean of 5 seeds); "
                 "training cars drawn over the range of b of each width")
    fig.tight_layout()
    save(fig, out, "09_training_continued")
    plt.close(fig)


def protocol_gaps(tag: str) -> tuple[list[list[float]], int]:
    """Distance from the limit [%] of every completed protocol lap at b = 0, per seed, and the laps completed."""
    if tag == "nominal_limit":
        data = json.loads((EVALUATION / "limit" / "nominal_limit_stage7_vs_reference.json").read_text())
        gaps = [[100 * g for g in seed if g is not None and np.isfinite(g)] for seed in data["gaps"]]
        return gaps, sum(len(g) for g in gaps)
    data = json.loads((EVALUATION / "balance_limit" / f"{tag}.json").read_text())
    refs = {round(lap["start_s"], 3): lap["lap_time"]
            for lap in json.loads(Path("outputs/reference/min_time_standing_b+0.00.json").read_text())["laps"]}
    gaps = [[100 * (lap["lap_time"] - refs[round(lap["start_s"], 3)]) / refs[round(lap["start_s"], 3)]
             for lap in seed if lap["completed"]] for seed in data["deterministic"]["+0.00"]]
    return gaps, sum(len(g) for g in gaps)


@figure("protocol_gaps")
def protocol_gap_figure(track: Track, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)
    for i, tag in enumerate(TAGS):
        gaps, done = protocol_gaps(tag)
        for k, seed_gaps in enumerate(gaps):
            x = i + 0.55 * (k - 2) / 5 + rng.uniform(-0.03, 0.03, len(seed_gaps))
            ax.scatter(x, seed_gaps, s=14, color=COLOURS[tag], alpha=0.85, lw=0, zorder=3)
        pooled = [g for seed in gaps for g in seed]
        ax.plot([i - 0.38, i + 0.38], [np.mean(pooled)] * 2, color=INK, lw=2, zorder=4)
        ax.text(i, 6.55, f"{done}/50", ha="center", fontsize=9, color=INK_2)
        ax.text(i + 0.4, np.mean(pooled), f"{np.mean(pooled):.2f}", fontsize=8, color=INK, va="center")
    ax.axhline(5.0, color=FAILURE, lw=1, ls="--")
    ax.text(-0.45, 5.05, "5 % criterion", color=FAILURE, fontsize=8.5, ha="left", va="bottom")
    ax.set_xticks(range(len(TAGS)), [LABELS[t].replace(" ", "\n", 1) for t in TAGS], fontsize=9)
    ax.set_ylabel("distance from the minimum-time lap\nfrom the same start [%]")
    ax.set_ylim(2.3, 6.9)
    ax.text(-0.5, 6.55, "laps", fontsize=9, color=INK_2, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_title("Protocol laps at b = 0 (10 standing starts × 5 seeds, mean action): each dot is a lap, "
                 "seeds side by side; bar: mean")
    save(fig, out, "10_protocol_gaps")
    plt.close(fig)


@figure("nominal_seeds")
def nominal_seeds(track: Track, out: Path) -> None:
    gaps, _ = protocol_gaps("nominal_limit")
    medians = [np.median(g) for g in gaps]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([f"seed {s}" for s in SEEDS], medians, color=[NOMINAL if m <= 5 else NOMINAL_LIGHT for m in medians],
                  width=0.6, zorder=3)
    for bar, m, g in zip(bars, medians, gaps, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.08, f"{m:.2f} %\n{len(g)}/10 laps", ha="center",
                fontsize=8.5, color=INK)
    ax.axhline(5.0, color=FAILURE, lw=1.1, ls="--", zorder=4)
    ax.text(4.45, 5.05, "5 %", color=FAILURE, fontsize=9, va="bottom", ha="right")
    ax.set_ylabel("median distance from the limit [%]")
    ax.set_ylim(0, 6.6)
    ax.grid(axis="x", visible=False)
    ax.set_title("Nominal at the limit, 600 M decisions: 3 of 5 seeds within 5 %\n(criterion: 4 of 5)")
    save(fig, out, "11_nominal_seeds")
    plt.close(fig)


def balance_summary() -> dict:
    return json.loads((EVALUATION / "balance_limit" / "summary.json").read_text())


@figure("balance_heatmaps")
def balance_heatmaps(track: Track, out: Path) -> None:
    summary = balance_summary()
    grid = summary["grid"]
    keys = [f"{b:+.2f}" for b in grid]
    completion = np.array([[summary["policies"][t][k]["completion"]["mean"] for t in BALANCE_TAGS] for k in keys])
    distance = np.array([[(summary["policies"][t][k]["distance_from_limit"]["mean"] or np.nan) * 100
                          if summary["policies"][t][k]["distance_from_limit"]["n"] else np.nan
                          for t in BALANCE_TAGS] for k in keys])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    blues = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "c", [GRID, "#184f95", "#3987e5", "#9ec5f4"] if thesis_plots.DARK else [GRID, "#86b6ef", "#256abf", "#0d366b"])
    oranges = plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#fde6d8", "#f4a27a", "#d95926",
                                                                            "#7a2b0d"])
    oranges.set_bad(SURFACE)
    for ax, values, cmap, limits, fmt, title in (
        (axes[0], completion * 50, blues, (0, 50), "{:.0f}", "Laps completed (of 50)"),
        (axes[1], distance, oranges, (3.3, 6.5), "{:.2f}", "Distance from the minimum-time lap of each b [%]"),
    ):
        image = ax.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
        for i in range(len(keys)):
            for j in range(len(BALANCE_TAGS)):
                v = values[i, j]
                if np.isfinite(v):
                    strong = (v - limits[0]) / (limits[1] - limits[0]) > 0.55
                    light_text = strong != (thesis_plots.DARK and cmap is blues)
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                            color="white" if light_text else "#0b0b0b")
        for j, tag in enumerate(BALANCE_TAGS):
            width = tag.rsplit("_", 1)[-1]
            if width in RANGES:
                lo, hi = RANGES[width]
                rows = [i for i, b in enumerate(grid) if lo - 1e-9 <= b <= hi + 1e-9]
                ax.add_patch(Rectangle((j - 0.5, rows[0] - 0.5), 1, len(rows), fill=False, ec=INK, lw=1.4))
                held = grid.index(HELD_OUT[width])
                ax.plot(j + 0.36, held, "D", ms=4, color=INK)
        ax.set_xticks(range(len(BALANCE_TAGS)), [SHORT[t] for t in BALANCE_TAGS])
        ax.set_yticks(range(len(keys)), keys)
        ax.set_xticks(np.arange(-0.5, len(BALANCE_TAGS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", size=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)
        bar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
        bar.outline.set_visible(False)
    axes[0].set_ylabel("balance index b  (− understeer, + oversteer)")
    fig.suptitle("Every car of the grid, 5 seeds × 10 starts, mean action; box: training range of the width, "
                 "diamond: held-out value")
    fig.tight_layout()
    save(fig, out, "12_balance_heatmaps")
    plt.close(fig)


@figure("balance_handling")
def balance_handling(track: Track, out: Path) -> None:
    rows = json.loads(Path("outputs/analysis/balance_index_gt4.json").read_text())["rows"]
    b = np.array([r["b"] for r in rows])
    summary = balance_summary()
    grid = np.array(summary["grid"])
    t_opt = np.array([float(reference(v)["lap_time"]) for v in grid])
    pm = np.array([summary["point_mass_lap_time"][f"{v:+.2f}"] for v in grid])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    from contextual_bicycle_pacejka.model.tires import compute_lateral_scales
    lam = np.array([compute_lateral_scales(v) for v in b])
    axes[0].plot(b, lam[:, 0], color=RANDOMISED, label="front λ_f = 1 + b")
    axes[0].plot(b, lam[:, 1], color=ORACLE, label="rear λ_r = 1 − b l_r / l_f")
    axes[0].set_ylabel("lateral capacity scale")
    axes[0].legend(fontsize=8)
    axes[1].plot(b, [r["K_deg_per_g"] for r in rows], color=INK)
    axes[1].axhline(0, color=INK_3, lw=0.8)
    axes[1].set_ylabel("understeer gradient K [deg/g]")
    axes[2].plot(b, [r["ay_max_g"] for r in rows], color=INK)
    axes[2].set_ylabel("steady lateral limit a_y,max [g]")
    axes[3].plot(grid, t_opt, "o-", color=INK, ms=4, label="minimum-time lap (flying)")
    axes[3].plot(grid, pm, "s-", color=INK_3, ms=4, label="point-mass lap")
    axes[3].set_ylabel("lap time [s]")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("balance index b")
        ax.axvspan(-0.21, 0.03, color=GRID, alpha=0.6, lw=0, zorder=0)
    axes[0].text(-0.205, 0.98, "drivable range\nat the limit", transform=axes[0].get_xaxis_transform(), fontsize=7.5,
                 va="top", color=INK_2)
    fig.suptitle("The family of cars: one scalar b moves lateral grip between the axles at a constant total")
    fig.tight_layout()
    save(fig, out, "13_balance_handling")
    plt.close(fig)


@figure("flying_laps")
def flying_lap_figure(track: Track, out: Path) -> None:
    ref = float(reference(0.0)["lap_time"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, tag in enumerate(TAGS):
        laps = flying_laps(tag, track)
        times = [float(lap["lap_time"]) for lap in laps.values()]
        ax.scatter([i] * len(times), times, s=36, color=COLOURS[tag], zorder=3, ec=SURFACE, lw=1.5)
        if times:
            ax.text(i + 0.12, min(times), f"best {min(times):.2f}\n(+{100 * (min(times) - ref) / ref:.1f} %)",
                    fontsize=7.5, color=INK_2, va="center")
        if len(times) < len(SEEDS):
            ax.text(i, max(times) + 0.35, f"{len(SEEDS) - len(times)} seed off\ntrack in lap 1", ha="center",
                    fontsize=7.5,
                    color=FAILURE)
    ax.axhline(ref, color=OPTIMUM, ls="--", lw=1.2)
    ax.text(len(TAGS) - 0.5, ref + 0.15, f"minimum-time lap {ref:.2f} s", ha="right", fontsize=8.5)
    ax.set_xticks(range(len(TAGS)), [LABELS[t].replace(" ", "\n", 1) for t in TAGS], fontsize=9)
    ax.set_ylabel("flying lap time [s]")
    ax.grid(axis="x", visible=False)
    ax.set_title("Flying lap (second lap from s = 0) at b = 0, mean action, one dot per seed")
    save(fig, out, "14_flying_laps")
    plt.close(fig)


@figure("failure_map")
def failure_map(track: Track, out: Path) -> None:
    summary = balance_summary()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=-0.24, vcenter=0.0, vmax=0.05)
    for ax, tag in zip(axes, ("nominal_limit_cont", "randomised_limit_w1", "oracle_limit_w1"), strict=True):
        draw_track(ax, track, 0.6)
        total = 0
        for key, cell in summary["policies"][tag].items():
            for place, count in cell["failure_places_m"].items():
                p = track.point(float(place))
                ax.scatter(*p, s=10 + 8 * count, color=DIVERGING(norm(float(key))), alpha=0.85, lw=0.4,
                           ec=SURFACE, zorder=3)
                total += count
        ax.set_title(f"{LABELS[tag]}: {total} failed laps of 650", fontsize=10)
        no_frame(ax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    bar = fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.01, orientation="horizontal")
    bar.set_label("balance index b of the car (blue: understeering, red: oversteering); dot size: failures")
    bar.outline.set_visible(False)
    fig.suptitle("Where the policies fail (off track, spin or stall; grouped every 10 m), all cars of the grid")
    save(fig, out, "15_failure_map")
    plt.close(fig)


@figure("mean_vs_stochastic")
def mean_vs_stochastic(track: Track, out: Path) -> None:
    mean = load("stochastic_nominal_limit_cont_s14_mean")
    noisy = [load(f"stochastic_nominal_limit_cont_s14_noise{k}") for k in range(5)]
    ref = reference(0.0)
    start, end = 1950.0, 2230.0
    fig = plt.figure(figsize=(13, 6.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.3])
    ax = fig.add_subplot(grid[:, 0])
    draw_track(ax, track, 1.0)

    def segment(run, lap=0):
        s = run["s"] - lap * track.length
        keep = (s >= start) & (s <= end)
        return keep, s

    keep, _ = segment(ref)
    path = dense(ref, keep)
    ax.plot(path["x"], path["y"], color=OPTIMUM, ls="--", lw=1.3, zorder=3)
    for run in noisy:
        keep, _ = segment(run)
        path = dense(run, keep)
        ax.plot(path["x"], path["y"], color=RANDOMISED, lw=1.0, alpha=0.8, zorder=4)
    keep, _ = segment(mean)
    path = dense(mean, keep)
    ax.plot(path["x"], path["y"], color=FAILURE, lw=2.0, zorder=5)
    ax.plot(mean["x"][-1], mean["y"][-1], "X", color=FAILURE, ms=11, mec=SURFACE, zorder=6)
    x0, x1, y0, y1 = window_bbox(track, 2060, end - 20, 10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    no_frame(ax)
    ax.set_title("Corner C3 (2097–2171 m), first lap from s = 0")
    keys = (("vx", "speed [km/h]", 3.6), ("delta", "steering $\\delta$ [deg]", 180 / np.pi),
            ("ey", "lateral offset $e_y$ [m]", 1.0))
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    for axis, (key, label, scale) in zip(axes, keys, strict=True):
        axis.axvspan(2097, 2171, color=GRID, alpha=0.7, lw=0)
        if key in ref and key != "delta":  # the steering of the reference chatters between its nodes
            keep, s = segment(ref)
            axis.plot(s[keep], scale * ref[key][keep], color=OPTIMUM, ls="--", lw=1.2)
        for run in noisy:
            keep, s = segment(run)
            axis.plot(s[keep], scale * run[key][keep], color=RANDOMISED, lw=0.9, alpha=0.8)
        keep, s = segment(mean)
        axis.plot(s[keep], scale * mean[key][keep], color=FAILURE, lw=1.8)
        axis.set_ylabel(label)
        axis.set_xlim(start, end)
    axes[-1].set_xlabel("position along the lap s [m]")
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    handles = [Line2D([], [], color=FAILURE, lw=2, label=f"mean action: off track at {mean['s'][-1]:.1f} m"),
               Line2D([], [], color=RANDOMISED, label="stochastic policy, 5 runs: all complete the lap"),
               Line2D([], [], color=OPTIMUM, ls="--", label="minimum-time lap")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Nominal continued, seed 14, b = 0: the mean action fails where the noisy policy passes "
                 f"(σ = {noisy[0]['std_delta'][1]:.3f} of the steering range, {noisy[0]['std_tau'][1]:.3f} of the "
                 "command rate)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, out, "16_mean_vs_stochastic")
    plt.close(fig)


@figure("balance_corner")
def balance_corner(track: Track, out: Path) -> None:
    values = (-0.21, 0.0, 0.03)
    shades = {-0.21: RANDOMISED if thesis_plots.DARK else "#184f95", 0.0: INK_3, 0.03: FAILURE}
    a, b = WINDOWS[4]
    start, end = a - 150, b + 120
    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
    for col, tag in enumerate(("randomised_limit_w1", "oracle_limit_w1")):
        ax = fig.add_subplot(grid[0, col])
        draw_track(ax, track, 1.0)
        for v in values:
            run = load(f"balance_{tag}_s12_b{v:+.2f}")
            lap = flying(run, track.length)
            path = dense(lap, (lap["s"] >= start) & (lap["s"] <= end))
            ax.plot(path["x"], path["y"], color=shades[v], lw=1.6, label=f"b = {v:+.2f}")
        x0, x1, y0, y1 = window_bbox(track, a - 60, b + 40, 6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        no_frame(ax)
        ax.set_title(f"{LABELS[tag]} (seed 12)")
    ax = fig.add_subplot(grid[0, 2])
    draw_track(ax, track, 1.0)
    for v in values:
        ref = reference(v)
        path = dense(ref, (ref["s"] >= start) & (ref["s"] <= end))
        ax.plot(path["x"], path["y"], color=shades[v], lw=1.6, ls="--", label=f"b = {v:+.2f}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    no_frame(ax)
    ax.set_title("Minimum-time laps")
    for col, tag in enumerate(("randomised_limit_w1", "oracle_limit_w1", None)):
        axis = fig.add_subplot(grid[1, col])
        axis.axvspan(a, b, color=GRID, alpha=0.7, lw=0)
        for v in values:
            if tag is None:
                data, ls = reference(v), "--"
            else:
                data, ls = flying(load(f"balance_{tag}_s12_b{v:+.2f}"), track.length), "-"
            keep = (data["s"] >= start) & (data["s"] <= end)
            axis.plot(data["s"][keep], 3.6 * data["vx"][keep], color=shades[v], lw=1.4, ls=ls)
        axis.set_xlim(start, end)
        axis.set_ylim(40, 220)
        axis.set_xlabel("s [m]")
        if col == 0:
            axis.set_ylabel("speed [km/h]")
        if tag is None:
            axis.legend(handles=[Line2D([], [], color=shades[v], label=f"b = {v:+.2f}") for v in values],
                        fontsize=8.5, loc="upper center")
    fig.suptitle(f"Slowest corner C5 ({a:.0f}–{b:.0f} m) for three cars: the Oracle changes line and speed with b, "
                 "the Randomised much less")
    fig.tight_layout()
    save(fig, out, "17_balance_corner")
    plt.close(fig)


@figure("adaptation_slopes")
def adaptation_slopes(track: Track, out: Path) -> None:
    adaptation = balance_summary()["adaptation"]
    order = ["minimum_time", *BALANCE_TAGS]
    names = {"minimum_time": "optimum", **SHORT}
    colours = {"minimum_time": OPTIMUM, **COLOURS, "nominal_limit_cont": NOMINAL}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, key, title in ((axes[0], "entry_speed", "entry speed"), (axes[1], "min_speed", "minimum speed")):
        values = [adaptation[t][key] for t in order]
        ax.bar(range(len(order)), values, color=[colours[t] for t in order], width=0.65, zorder=3)
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.axhline(adaptation["minimum_time"][key], color=OPTIMUM, ls="--", lw=1)
        ax.set_xticks(range(len(order)), [names[t] for t in order], fontsize=8.5)
        ax.set_title(f"slope of the {title} in C5 against b")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("m/s per unit of b")
    fig.suptitle("How each policy adapts the slowest corner to the car (dashed: the minimum-time lap)")
    fig.tight_layout()
    save(fig, out, "18_adaptation_slopes")
    plt.close(fig)


@figure("cost_and_regret")
def cost_and_regret(track: Track, out: Path) -> None:
    summary = balance_summary()
    widths = ("w13", "w23", "w1")
    names = {"w13": "W1/3", "w23": "W2/3", "w1": "W1"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for i, w in enumerate(widths):
        cost = summary["question1"][w]["nominal"]["time"]
        regret = summary["question3"][w]["net_gap"]["regret"]
        for ax, stat, colour in ((axes[0], cost, RANDOMISED), (axes[1], regret, ORACLE)):
            lo, hi = stat["interval"]
            ax.plot([lo, hi], [i, i], color=colour, lw=2.4, solid_capstyle="round")
            ax.plot(stat["mean"], i, "o", color=colour, ms=8, mec=SURFACE, mew=1.5)
            ax.text(hi + 0.06, i, f"{stat['mean']:+.2f} s [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=8.5)
    for ax in axes:
        ax.set_yticks(range(3), [names[w] for w in widths])
        ax.axvline(0, color=INK_3, lw=0.8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Randomised − Nominal lap time at b = 0 [s]")
    axes[0].set_title("Cost of randomising b, on the nominal car")
    axes[0].set_xlim(-0.6, 3.4)
    axes[1].axvline(0.5, color=FAILURE, lw=1, ls="--")
    axes[1].text(0.53, 2.42, "rule fixed in advance: 0.5 s", color=FAILURE, fontsize=8.5, va="bottom")
    axes[1].set_xlabel("net regret Randomised − Oracle in the range [s]")
    axes[1].set_title("Value of knowing b (net of the gap at b = 0)")
    axes[1].set_xlim(-0.6, 1.4)
    fig.suptitle("Paired differences with 95 % intervals (seed and start matched)")
    fig.tight_layout()
    save(fig, out, "19_cost_and_regret")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    parser.add_argument("--no-titles", action="store_true",
                        help="leave out the figure titles (the suptitle and the title of a single-panel figure)")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/thesis"))
    parser.add_argument("--dpi", type=int, help="resolution of the PNG files (default: that of the style)")
    args = parser.parse_args()
    thesis_plots.TITLES = not args.no_titles
    apply_style()
    if args.dpi:
        plt.rcParams["savefig.dpi"] = args.dpi
    track = load_track()
    for name in args.only or FIGURES:
        FIGURES[name](track, args.output)
        print(f"saved {name}")


if __name__ == "__main__":
    main()
