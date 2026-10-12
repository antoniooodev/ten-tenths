"""Explanatory figures of the README (written to docs/figures/).

    python scripts/make_readme_figures.py            # every figure (about 2 minutes)
    python scripts/make_readme_figures.py --only tyre_model safety_filter

Every figure is computed from the stored data and the package code itself (plant, teacher, safety filter,
demonstration generator, policies), so it shows what the code does, not a sketch of it:

    vehicle_model.png        single-track model in the Frenet frame: state, inputs and forces (schematic)
    circuits.png             the four racing lines coloured by v_ref, with the evaluation start points
    racing_line_repair.png   the La Source curvature kink of Spa R0 and its repair in R1
    tyre_model.png           Pacejka lateral force, saturation / termination thresholds, combined slip
    observation.png          the 116-entry observation: layout and one real snapshot before a corner
    teacher.png              MATv0 on Spa R1: speed target, friction-budget limiter, steering terms
    safety_filter.png        the 61 x 61 action grid at one real step: candidate, unsafe and safe cells
    demonstrations.png       perturbed demonstrations: OU noise, executed vs label actions
    policy_vs_teacher.png    the ALL4 policy and the teacher on the same observations (unfiltered)
    reward.png               the PPO reward terms along a teacher lap and the failure penalty
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from contextual_bicycle_pacejka.control.safety_filter import (  # noqa: E402
    STEERING_RATE_LIMIT,
    filter_action,
    front_slip_delta_cmd_bounds,
    is_allowed,
    nearest_safe_action,
    safe_action_set,
)
from contextual_bicycle_pacejka.control.teacher import MATv0  # noqa: E402
from contextual_bicycle_pacejka.env.factory import make_env  # noqa: E402
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup  # noqa: E402
from contextual_bicycle_pacejka.evaluation.episode import progress_increment, run_episode  # noqa: E402
from contextual_bicycle_pacejka.learning.demonstrations import episode_specs, run_lap  # noqa: E402
from contextual_bicycle_pacejka.learning.policy import Policy  # noqa: E402
from contextual_bicycle_pacejka.model.reward import CRASH_BASE, TIME_COST_RATE, RewardWeights  # noqa: E402
from contextual_bicycle_pacejka.model.state import SetupVector  # noqa: E402
from contextual_bicycle_pacejka.model.tires import (  # noqa: E402
    compute_combined_slip_capacity,
    compute_lateral_force,
    compute_saturation_threshold,
    pacejka_coefficients,
)
from contextual_bicycle_pacejka.model.track import RealTrack  # noqa: E402
from contextual_bicycle_pacejka.spec.constants import (  # noqa: E402
    DT,
    G,
    LF,
    LOOKAHEAD_M,
    LR,
    SLIP_TERMINATION_RATIO,
    TRAJ_NORMALISATION_M,
    V_MIN,
)
from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, ACTION_LOW, OBS  # noqa: E402

OUT = REPO / "docs/figures"
TRACKS = REPO / "data/tracks"
PROTOCOLS = REPO / "data/protocols"

# Palette: categorical slots in fixed order, recessive ink and grid.
INK, INK2, MUTED, GRID, PANEL = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def racing_line(circuit: str) -> Path:
    r1 = TRACKS / circuit / "racing_line_R1.npz"
    return r1 if r1.exists() else TRACKS / circuit / "racing_line_R0.npz"


def corridor(track: RealTrack, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points of the right (ey = -w_right) and left (ey = +w_left) track edges of the termination rule."""
    edges = []
    for sign, width in ((-1.0, track.boundary_right), (1.0, track.boundary_left)):
        pts = []
        for si in s:
            x, y = track.centerline_point(si)
            psi = track.psi_ref(si)
            ey = sign * width(si)
            pts.append((x - ey * math.sin(psi), y + ey * math.cos(psi)))
        edges.append(np.asarray(pts))
    return edges[0], edges[1]


def teacher_run(raceline: Path, start_s: float, start_vx: float, mu: float, seed: int, steps: int,
                filtered: bool = False, teacher: MATv0 | None = None) -> dict[str, np.ndarray]:
    """Teacher episode (unfiltered by default); per-step observation, state, action and teacher diagnostics."""
    teacher = teacher or MATv0()
    env = make_env(raceline)
    obs, _ = env.reset(seed=seed, options={"setup": randomized_setup(mu, np.random.default_rng(seed)),
                                           "initial_s": start_s, "initial_vx": max(V_MIN, start_vx)})
    base = env.unwrapped
    rec: dict[str, list] = {k: [] for k in ("obs", "state", "action", "v_target", "tau_hi", "tau_lo", "tau_unlimited",
                                            "delta_ff", "yaw_term", "reward", "progress_term", "lateral_term",
                                            "ey_max", "s_dot")}
    previous = np.zeros(2)
    for _ in range(steps):
        action, v_target, diag = teacher.act_with_diagnostics(obs)
        if filtered:
            action = filter_action(env, action, previous).action
        rec["obs"].append(np.asarray(obs, dtype=np.float64))
        rec["state"].append(base.state.to_array())
        rec["action"].append(np.asarray(action, dtype=np.float64))
        rec["v_target"].append(v_target)
        for key in ("tau_hi", "tau_lo", "tau_unlimited", "delta_ff", "yaw_term"):
            rec[key].append(diag[key])
        obs, reward, terminated, truncated, info = env.step(action)
        previous = np.asarray(action, dtype=np.float64)
        state = np.asarray(info["state"])
        s_dot = float(info["diag"]["s_dot"])
        ey_max = base.track.ey_max(state[0])
        rec["reward"].append(reward)
        rec["s_dot"].append(s_dot)
        rec["ey_max"].append(ey_max)
        rec["progress_term"].append(max(s_dot, 0.0) * DT)
        rec["lateral_term"].append((abs(state[1]) / ey_max) ** 2 * DT)
        if terminated or truncated:
            break
    env.close()
    return {k: np.asarray(v) for k, v in rec.items()}


def protocol_starts(circuit: str, line: str) -> list[tuple[str, float]]:
    protocol = json.loads((PROTOCOLS / f"{circuit}_{line}.json").read_text())
    return sorted({(e["start_label"], float(e["start_s"])) for e in protocol["episodes"]}, key=lambda t: t[1])


# ---------------------------------------------------------------------------------------------------------------------
# Schematic of the plant
# ---------------------------------------------------------------------------------------------------------------------
def fig_vehicle_model() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    radius, half_width = 45.0, 4.0

    def path(theta: np.ndarray) -> np.ndarray:  # reference path: arc turning left, s = radius * theta
        return np.column_stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta))))

    theta = np.linspace(-0.2, 0.3, 200)
    centre = path(theta)
    normal = np.column_stack((-np.sin(theta), np.cos(theta)))
    ax.fill(np.r_[(centre + half_width * normal)[:, 0], (centre - half_width * normal)[::-1, 0]],
            np.r_[(centre + half_width * normal)[:, 1], (centre - half_width * normal)[::-1, 1]], color=PANEL, lw=0)
    ax.plot(*(centre + half_width * normal).T, color=MUTED, lw=1.0)
    ax.plot(*(centre - half_width * normal).T, color=MUTED, lw=1.0)
    ax.plot(*centre.T, color=BLUE, lw=1.6, ls=(0, (6, 3)))
    ax.text(6.3, -0.3, "racing line = reference path\n" r"$x(s),\ y(s),\ \psi_{ref}(s),\ \kappa(s),\ v_{ref}(s)$",
            color=BLUE, fontsize=8.5, va="top")
    ax.text(-7.3, -4.55, r"track edge (distance $w_{left}(s)$, $w_{right}(s)$ from the line)", color=INK2, fontsize=8)

    def arrow(a: np.ndarray, b: np.ndarray, color: str, lw: float = 1.7, style: str = "-|>") -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, zorder=8,
                                     shrinkA=0, shrinkB=0))

    # Projection point and vehicle pose (psi_ref = 0 at the projection point).
    p0 = np.array([0.0, 0.0])
    ey, epsi, delta = 1.9, math.radians(12.0), math.radians(14.0)
    cg = p0 + np.array([0.0, ey])
    u = np.array([math.cos(epsi), math.sin(epsi)])
    n = np.array([-math.sin(epsi), math.cos(epsi)])
    front, rear = cg + LF * u, cg - LR * u
    body = np.array([cg + 2.3 * u + 0.9 * n, cg + 2.3 * u - 0.9 * n, cg - 2.2 * u - 0.9 * n, cg - 2.2 * u + 0.9 * n])
    ax.add_patch(Polygon(body, closed=True, facecolor="#cde2fb", edgecolor=BLUE, lw=1.2, zorder=3))

    def wheel(centre_pt: np.ndarray, angle: float) -> None:
        w_u = np.array([math.cos(angle), math.sin(angle)])
        w_n = np.array([-math.sin(angle), math.cos(angle)])
        pts = np.array([centre_pt + 0.38 * w_u + 0.14 * w_n, centre_pt + 0.38 * w_u - 0.14 * w_n,
                        centre_pt - 0.38 * w_u - 0.14 * w_n, centre_pt - 0.38 * w_u + 0.14 * w_n])
        ax.add_patch(Polygon(pts, closed=True, facecolor=INK, edgecolor=INK, zorder=5))

    wheel(front, epsi + delta)
    wheel(rear, epsi)
    ax.plot(*np.array([rear, front]).T, color=INK, lw=1.0, zorder=4)
    ax.plot(*cg, "o", ms=6, color="white", mec=INK, mew=1.2, zorder=9)
    ax.text(*(cg + np.array([-0.15, -0.55])), "CG", fontsize=8, ha="right", zorder=9)

    # Frenet errors.
    ax.plot(*p0, "o", ms=5, color=BLUE, zorder=6)
    arrow(p0, cg, ORANGE, style="<|-|>")
    ax.text(0.15, 0.55, r"$e_y$", color=ORANGE, fontsize=12, zorder=9)
    ax.text(0.2, -0.75, r"closest point, arc length $s$", color=BLUE, fontsize=8)
    ax.plot([cg[0], cg[0] + 5.6], [cg[1], cg[1]], color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(cg[0] + 5.7, cg[1] - 0.1, r"$\psi_{ref}$ direction", color=INK2, fontsize=7.5, va="center")
    arc = np.linspace(0, epsi, 30)
    ax.plot(cg[0] + 4.6 * np.cos(arc), cg[1] + 4.6 * np.sin(arc), color=ORANGE, lw=1.3)
    ax.text(cg[0] + 4.75, cg[1] + 0.33, r"$e_\psi$", color=ORANGE, fontsize=12)
    arrow(np.array([-6.5, -2.0]), np.array([-2.5, -2.0]), BLUE, lw=1.3)
    ax.text(-6.5, -2.55, r"$s$ grows along the line; $\dot s$ = progress", color=BLUE, fontsize=8, va="top")

    # Velocities and yaw rate.
    arrow(cg, cg + 3.4 * u, AQUA)
    ax.text(*(cg + 3.45 * u + np.array([0.1, -0.5])), r"$v_x$", color=AQUA, fontsize=12)
    arrow(cg, cg + 1.25 * n, AQUA)
    ax.text(*(cg + 1.3 * n + np.array([-0.55, -0.1])), r"$v_y$", color=AQUA, fontsize=12)
    yaw = np.radians(np.linspace(215, 250, 30))
    ax.plot(cg[0] + 2.6 * np.cos(yaw), cg[1] + 2.6 * np.sin(yaw), color=AQUA, lw=1.4)
    arrow(cg + 2.6 * np.array([np.cos(yaw[-3]), np.sin(yaw[-3])]),
          cg + 2.6 * np.array([np.cos(yaw[-1]), np.sin(yaw[-1])]), AQUA, lw=1.4)
    ax.text(cg[0] - 2.75, cg[1] - 2.2, r"$r$ (yaw rate)", color=AQUA, fontsize=9.5, ha="right")

    # Steering angle.
    w_dir = np.array([math.cos(epsi + delta), math.sin(epsi + delta)])
    ax.plot(*np.array([front, front + 2.0 * w_dir]).T, color=INK, lw=0.9, zorder=4)
    arc = np.linspace(epsi, epsi + delta, 20)
    ax.plot(front[0] + 1.6 * np.cos(arc), front[1] + 1.6 * np.sin(arc), color=VIOLET, lw=1.3, zorder=6)
    ax.text(*(front + 2.0 * np.array([math.cos(epsi + delta + 0.12), math.sin(epsi + delta + 0.12)])),
            r"$\delta$", color=VIOLET, fontsize=12)

    # Tyre forces.
    w_n = np.array([-math.sin(epsi + delta), math.cos(epsi + delta)])
    arrow(front, front + 2.2 * w_n, RED)
    ax.text(*(front + 2.3 * w_n + np.array([0.05, 0.0])), r"$F_{y,f}$", color=RED, fontsize=11)
    arrow(rear, rear + 2.0 * n, RED)
    ax.text(*(rear + 2.1 * n + np.array([-0.9, 0.0])), r"$F_{y,r}$", color=RED, fontsize=11)
    arrow(rear, rear - 1.9 * u, MAGENTA)
    ax.text(*(rear - 1.9 * u + np.array([-0.2, 0.25])), r"$F_{x,r}$", color=MAGENTA, fontsize=11, ha="right")

    fig.text(0.02, 0.02,
             "state   z = [s, e_y, e_psi, vx, vy, r, delta]            (Frenet errors w.r.t. the racing line)\n"
             "action  a = [delta_cmd (rad, |.| <= 0.5), tau (-1 brake ... +1 throttle)]\n"
             "setup   xi = [m, Cz_f, Cz_r, mu, hcg, tau_d, Cpac]       (sampled per episode)\n"
             "fixed   l_f = 1.2 m, l_r = 1.4 m, I_z = 1800 kg m^2, rear-wheel drive, brake bias 65 % front",
             fontsize=8, family="monospace", color=INK, va="bottom",
             bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL, "edgecolor": GRID})
    ax.set_xlim(-7.5, 10.5)
    ax.set_ylim(-4.8, 6.6)
    ax.set_title("Single-track (bicycle) model in the Frenet frame of the racing line", pad=4)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "vehicle_model.png")


# ---------------------------------------------------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------------------------------------------------
def fig_circuits() -> None:
    names = {"monza": "Monza", "spa": "Spa-Francorchamps", "catalunya": "Catalunya", "budapest": "Budapest"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.4))
    vmin, vmax = 5.0, 60.0
    for ax, circuit in zip(axes.ravel(), names):
        path = racing_line(circuit)
        line = path.stem.split("_")[-1]
        track = RealTrack(str(path))
        with np.load(path) as data:
            x, y, v = data["x_m"], data["y_m"], data["v_ref_mps"]
        s_dense = np.linspace(0.0, track.track_length, 1500)
        right, left = corridor(track, s_dense)
        ax.fill(np.r_[right[:, 0], left[::-1, 0]], np.r_[right[:, 1], left[::-1, 1]], color=PANEL, lw=0)
        ax.plot(*right.T, color=MUTED, lw=0.5)
        ax.plot(*left.T, color=MUTED, lw=0.5)
        pts = np.column_stack((np.r_[x, x[0]], np.r_[y, y[0]]))
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=SPEED_CMAP, norm=plt.Normalize(vmin, vmax), lw=2.4, capstyle="round")
        lc.set_array(np.asarray(v))
        ax.add_collection(lc)
        for label, s0 in protocol_starts(circuit, line):
            px, py = track.centerline_point(s0)
            if label == "orig":
                ax.plot(px, py, "o", ms=9, color=ORANGE, mec="white", mew=1.5, zorder=5)
                ax.annotate("start", (px, py), xytext=(8, 8), textcoords="offset points", color=INK, fontsize=8,
                            fontweight="bold")
            else:
                ax.plot(px, py, "s", ms=5, color=INK2, mec="white", mew=1.0, zorder=5)
                ax.annotate(label, (px, py), xytext=(5, -10), textcoords="offset points", color=INK2, fontsize=7)
        sx, sy = track.centerline_point(0.0)
        ex, ey = track.centerline_point(60.0)
        ax.annotate("", (ex, ey), (sx, sy), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2})
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{names[circuit]}  ·  racing line {line}  ·  {track.track_length / 1000:.2f} km  ·  "
                     f"v_ref {v.min():.0f}-{v.max():.0f} m/s", fontsize=9.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=SPEED_CMAP), ax=axes,
                        orientation="horizontal", fraction=0.035, pad=0.03, aspect=50)
    cbar.set_label(r"reference speed $v_{ref}$ at $\mu$ = 0.9 [m/s]   (orange dot: protocol start = slowest point;"
                   r" squares h1-h8: held-out starts; arrow: driving direction at s = 0)", color=INK2)
    cbar.outline.set_visible(False)
    save(fig, "circuits.png")


def fig_racing_line_repair() -> None:
    r0, r1 = TRACKS / "spa/racing_line_R0.npz", TRACKS / "spa/racing_line_R1.npz"
    s_kink = 398.7
    window = 90.0
    fig, (ax_map, ax_k) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    track0 = RealTrack(str(r0))
    s_dense = np.linspace(s_kink - window, s_kink + window, 300)
    right, left = corridor(track0, s_dense)
    ax_map.fill(np.r_[right[:, 0], left[::-1, 0]], np.r_[right[:, 1], left[::-1, 1]], color=PANEL, lw=0)
    ax_map.plot(*right.T, color=MUTED, lw=0.8)
    ax_map.plot(*left.T, color=MUTED, lw=0.8)
    for path, color, label in ((r0, ORANGE, "R0 (kink)"), (r1, BLUE, "R1 (repaired)")):
        with np.load(path) as d:
            sel = np.abs(d["s_m"] - s_kink) < window
            ax_map.plot(d["x_m"][sel], d["y_m"][sel], "-o", ms=2.5, color=color, lw=1.6, label=label)
        with np.load(path) as d:
            sel = np.abs(d["s_m"] - s_kink) < window
            ax_k.plot(d["s_m"][sel], d["kappa_radpm"][sel], "-o", ms=2.5, color=color, lw=1.6, label=label)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.grid(False)
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_title("Spa, La Source hairpin (track corridor in grey)")
    ax_map.legend(loc="best")
    for sign in (1, -1):
        ax_k.axhline(sign * 0.06, color=RED, lw=1.0, ls=(0, (4, 3)))
    ax_k.text(s_kink - window + 2, -0.066, "spike threshold |κ| = 0.06 1/m", color=RED, fontsize=8, va="top")
    ax_k.set_xlabel("arc length s [m]")
    ax_k.set_ylabel(r"curvature $\kappa$ [1/m]")
    ax_k.set_title("Curvature: a 2-sample spike on R0, smooth on R1")
    ax_k.legend(loc="upper right")
    save(fig, "racing_line_repair.png")


# ---------------------------------------------------------------------------------------------------------------------
# Tyres
# ---------------------------------------------------------------------------------------------------------------------
def fig_tyre_model() -> None:
    setup = SetupVector(1400.0, 80_000.0, 80_000.0, 1.0, 0.5, 0.02, 1.3)
    sat = {"front": compute_saturation_threshold(setup, "front"), "rear": compute_saturation_threshold(setup, "rear")}
    alpha = np.linspace(-0.35, 0.35, 701)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    for axle, color in (("front", BLUE), ("rear", ORANGE)):
        b, c, e = pacejka_coefficients(axle)
        fy = np.array([compute_lateral_force(a, b, c, 1.0, e) for a in alpha])
        ax_a.plot(np.degrees(alpha), fy, color=color, label=f"{axle}  B={b:.2f} C={c} E={e}")
        ax_a.axvline(np.degrees(sat[axle]), color=color, lw=0.9, ls=(0, (4, 3)))
        ax_a.axvline(np.degrees(SLIP_TERMINATION_RATIO * sat[axle]), color=color, lw=0.9, ls=(0, (1, 2)))
    ax_a.text(np.degrees(sat["front"]) + 0.3, 0.12, r"$\alpha_{sat}$" + "\n(safety filter)", fontsize=7.5, color=INK2)
    ax_a.text(np.degrees(SLIP_TERMINATION_RATIO * sat["rear"]) + 0.3, -0.35,
              r"1.25 $\alpha_{sat}$" + "\n(termination)", fontsize=7.5, color=INK2)
    ax_a.set_xlabel(r"slip angle $\alpha$ [deg]")
    ax_a.set_ylabel(r"$F_y / D$")
    ax_a.set_title("Pacejka lateral force per unit capacity")
    ax_a.legend(loc="center left", fontsize=7.5)

    fz_f = setup.m / (LF + LR) * G * LR
    b, c, e = pacejka_coefficients("front")
    for ratio, color in ((0.0, BLUE), (0.6, AQUA), (0.9, YELLOW)):
        cap = compute_combined_slip_capacity(ratio * setup.mu * fz_f, setup.mu, fz_f)
        fy = np.array([compute_lateral_force(a, b, c, cap, e) for a in alpha])
        ax_b.plot(np.degrees(alpha), fy / 1000.0, color=color, label=rf"$F_x / \mu F_z$ = {ratio}")
    ax_b.set_xlabel(r"front slip angle $\alpha_f$ [deg]")
    ax_b.set_ylabel(r"$F_{y,f}$ [kN]")
    ax_b.set_title("Combined slip: braking shrinks the lateral force")
    ax_b.legend(loc="lower right")

    t = np.linspace(0, 2 * np.pi, 300)
    ax_c.fill(np.cos(t), np.sin(t), color=PANEL)
    ax_c.plot(np.cos(t), np.sin(t), color=INK2, lw=1.2)
    for ratio, color in ((0.0, BLUE), (0.6, AQUA), (0.9, YELLOW)):
        cap = math.sqrt(1 - ratio ** 2)
        ax_c.plot([ratio, ratio], [-cap, cap], color=color, lw=2.2)
        ax_c.plot(ratio, cap, "o", color=color, ms=5)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel(r"$F_x / (\mu F_z)$  (longitudinal has priority)")
    ax_c.set_ylabel(r"$D / (\mu F_z)$")
    ax_c.set_title("Friction circle per axle")
    ax_c.set_xlim(-1.15, 1.15)
    ax_c.set_ylim(-1.15, 1.15)
    save(fig, "tyre_model.png")


# ---------------------------------------------------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------------------------------------------------
def fig_observation() -> None:
    path = TRACKS / "spa/racing_line_R1.npz"
    start_s = 400.8
    run = teacher_run(path, start_s, 4.6, 1.0, 42, 9000)
    obs, state = run["obs"], run["state"]
    kappa = obs[:, OBS.KAPPA_START:OBS.KAPPA_END] * 0.05
    # A step on a straight at speed with a tight corner 80-145 m ahead.
    score = np.abs(kappa[:, 5:8]).max(axis=1) * (np.abs(kappa[:, 0]) < 0.003) * state[:, 3]
    k = int(np.argmax(score[1000:]) + 1000)
    o = obs[k]

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.6], hspace=0.45, wspace=0.3)
    ax_l = fig.add_subplot(gs[0, :])
    groups = [  # (first, end, colour, label inside the bar, callout above the bar)
        (0, 6, BLUE, "", "vx vy r δ ey eψ"), (6, 18, AQUA, "v_ref", "v_ref preview (12)"),
        (18, 78, AQUA, "racing-line points (x, y) in the vehicle frame: 30 points, 5-230 m ahead", ""),
        (78, 79, BLUE, "", "ax"), (79, 86, ORANGE, "", "setup ξ"), (86, 88, YELLOW, "", "ρ"),
        (88, 100, AQUA, "κ", "curvature preview (12)"), (100, 112, AQUA, "a_ref", "ref.-accel. preview (12)"),
        (112, 114, YELLOW, "", "α"), (114, 116, YELLOW, "", "Fy"),
    ]
    callout_y = {0: 1.3, 6: 2.2, 78: 1.3, 79: 2.2, 86: 1.3, 88: 2.2, 100: 1.3, 112: 2.2, 114: 1.3}
    for a, b, color, inside, callout in groups:
        ax_l.barh(0, b - a, left=a, height=1.0, color=color, edgecolor="white", lw=1.5)
        ax_l.text(0.5 * (a + b), -0.7, f"{a}" if b - a == 1 else f"{a}-{b - 1}", ha="center", va="top",
                  fontsize=6.5 if b - a < 4 else 7, color=INK2, rotation=0 if b - a >= 4 else 90)
        if inside:
            ax_l.text(0.5 * (a + b), 0, inside, ha="center", va="center", fontsize=7.5, color="white",
                      fontweight="bold")
        if callout:
            y = callout_y[a]
            ax_l.annotate(callout, (0.5 * (a + b), 0.5), xytext=(0.5 * (a + b), y), ha="center", fontsize=7.5,
                          color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA, ORANGE, YELLOW)]
    ax_l.legend(handles, ["vehicle state", "track preview", "setup ξ (privileged information)", "tyre state"],
                ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    ax_l.set_xlim(-1, 117)
    ax_l.set_ylim(-1.9, 2.8)
    ax_l.axis("off")
    ax_l.set_title("Observation vector (116 float32 entries, layout in spec/layout.py)")

    ax_b = fig.add_subplot(gs[1, 0])
    traj = o[OBS.TRAJ_START:OBS.TRAJ_END].reshape(-1, 2) * TRAJ_NORMALISATION_M
    ax_b.plot(-traj[:, 1], traj[:, 0], "o", ms=4, color=AQUA, label="30 racing-line points")
    ax_b.plot(0, 0, marker=(3, 0, 0), ms=12, color=BLUE, ls="none", label="vehicle (heading up)")
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("lateral [m]  (left -, right +)")
    ax_b.set_ylabel("longitudinal [m]")
    ax_b.set_title("Racing line in the vehicle frame")
    ax_b.legend(loc="lower left", fontsize=7)

    ax_v = fig.add_subplot(gs[1, 1])
    d = np.asarray(LOOKAHEAD_M, dtype=float)
    ax_v.plot(d, o[OBS.VREF_START:OBS.VREF_END], "-o", ms=4, color=AQUA, label=r"$v_{ref}\sqrt{\mu/0.9}$ preview")
    ax_v.axhline(o[OBS.VX], color=BLUE, lw=1.2, ls=(0, (4, 3)), label=f"current vx = {o[OBS.VX]:.1f} m/s")
    ax_v.set_xlabel("look-ahead distance [m]")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.set_title("Speed preview: braking is needed")
    ax_v.legend(loc="center right", fontsize=7)

    ax_k = fig.add_subplot(gs[1, 2])
    ax_k.plot(d, o[OBS.KAPPA_START:OBS.KAPPA_END], "-o", ms=4, color=AQUA, label=r"$\kappa / 0.05$")
    ax_k.plot(d, o[OBS.AX_PREVIEW_START:OBS.AX_PREVIEW_END], "-o", ms=4, color=VIOLET,
              label=r"$a_{x,ref} / 15$")
    ax_k.axhline(0, color=MUTED, lw=0.8)
    ax_k.set_xlabel("look-ahead distance [m]")
    ax_k.set_ylabel("normalised value [-]")
    ax_k.set_title("Curvature and ref.-acceleration previews")
    ax_k.legend(loc="lower left", fontsize=7)
    fig.text(0.5, -0.01, f"Snapshot: MATv0 on Spa R1, μ = 1.0, setup seed 42, s = {state[k, 0]:.0f} m, "
             f"t = {k * DT:.1f} s after the start.", ha="center", color=INK2, fontsize=8)
    save(fig, "observation.png")


# ---------------------------------------------------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------------------------------------------------
def fig_teacher() -> None:
    path = TRACKS / "spa/racing_line_R1.npz"
    run = teacher_run(path, 400.8, 4.6, 1.0, 42, 7000)
    t = np.arange(len(run["reward"])) * DT
    sel = t <= 60.0
    t = t[sel]
    state, action = run["state"][sel], run["action"][sel]
    k = MATv0().speed_safety
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True, gridspec_kw={"hspace": 0.32})
    fig.subplots_adjust(top=0.92)
    ax = axes[0]
    ax.plot(t, run["v_target"][sel], color=ORANGE, lw=1.4,
            label=rf"speed target $v_{{target}}$ = {k} · $v_{{ref}}$(5 m) · $\sqrt{{\mu/0.9}}$")
    ax.plot(t, state[:, 3], color=BLUE, label=r"$v_x$")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Longitudinal: target speed with feed-forward (module T)")
    ax.legend(loc="lower right", ncol=2)

    ax = axes[1]
    ax.fill_between(t, run["tau_lo"][sel], run["tau_hi"][sel], color=PANEL, label="friction-budget window (module L)")
    ax.plot(t, run["tau_unlimited"][sel], color=MUTED, lw=1.0, label=r"$\tau$ before the limiter")
    ax.plot(t, action[:, 1], color=BLUE, label=r"$\tau$ applied")
    ax.set_ylabel(r"$\tau$ [-]  (+ throttle, - brake)")
    ax.set_ylim(-1.08, 1.08)
    ax.set_title("Throttle/brake bounded per axle by the remaining friction budget")
    ax.legend(loc="lower right", ncol=3)

    ax = axes[2]
    stanley = action[:, 0] - run["delta_ff"][sel] - run["yaw_term"][sel]
    ax.plot(t, np.degrees(stanley), color=MUTED, lw=1.0, label="Stanley (heading + cross-track)")
    ax.plot(t, np.degrees(run["delta_ff"][sel]), color=AQUA, lw=1.2, label="curvature feed-forward (Y)")
    ax.plot(t, np.degrees(run["yaw_term"][sel]), color=VIOLET, lw=1.2, label="yaw-rate damping (Y)")
    ax.plot(t, np.degrees(action[:, 0]), color=BLUE, label=r"$\delta_{cmd}$")
    ax.set_ylabel("steering [deg]")
    ax.set_xlabel("time [s]")
    ax.set_title("Lateral: Stanley law + curvature feed-forward + yaw-rate damping")
    ax.legend(loc="lower right", ncol=4)
    fig.suptitle("MATv0 teacher on Spa R1 (μ = 1.0, setup seed 42, unfiltered), first 60 s from the protocol start",
                 x=0.07, ha="left", y=0.98, fontsize=10, color=INK2)
    save(fig, "teacher.png")


# ---------------------------------------------------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------------------------------------------------
def fig_safety_filter() -> None:
    # Treatment A (trained on Monza only) overspeeds into the corners of Spa, so the filter has work to do. The policy
    # drives through the filter (condition F); the first step whose proposal is projected onto a non-empty safe set is
    # shown, with the number of safe cells per step around it.
    path = TRACKS / "spa/racing_line_R1.npz"
    env = make_env(path)
    base = env.unwrapped
    policy = Policy(REPO / "models/treatment_a", env)
    obs, _ = env.reset(seed=42, options={"setup": randomized_setup(1.0, np.random.default_rng(42)),
                                         "initial_s": 400.8, "initial_vx": 4.6})
    previous = np.zeros(2)
    shown = None
    history = []  # (step, safe cells, projected, empty)
    for step in range(20_000):
        proposal = np.clip(policy(obs), ACTION_LOW, ACTION_HIGH)
        safe = safe_action_set(base.state, base.memory, base.setup, base.track, base.alpha_sat, float(previous[0]))
        allowed = is_allowed(safe, proposal)
        executed = proposal if allowed else nearest_safe_action(safe, proposal)
        history.append((step, int(safe.active.sum()), not allowed, safe.empty))
        if shown is None and not allowed and not safe.empty:
            slip_low, slip_high = front_slip_delta_cmd_bounds(base.state, base.alpha_sat[0], base.setup.tau_d)
            lo = max(ACTION_LOW[0], previous[0] - STEERING_RATE_LIMIT * DT, slip_low)
            hi = min(ACTION_HIGH[0], previous[0] + STEERING_RATE_LIMIT * DT, slip_high)
            shown = (step, safe, proposal.copy(), np.asarray(executed), previous.copy(), base.state.to_array(), lo, hi)
        obs, _, terminated, truncated, _ = env.step(executed)
        previous = np.asarray(executed, dtype=np.float64)
        if terminated or truncated or (shown is not None and step > shown[0] + 300):
            break
    policy.close()
    env.close()
    step0, safe, proposal, projected, prev, state, lo, hi = shown
    cand = (safe.delta_edges[:-1] >= lo) & (safe.delta_edges[1:] <= hi)
    status = np.zeros_like(safe.active, dtype=int)  # 0 not a candidate, 1 unsafe, 2 safe
    status[cand, :] = 1
    status[safe.active] = 2
    unsafe_colour = "#f2b3b2"
    cmap = ListedColormap([PANEL, unsafe_colour, AQUA])

    fig, (ax_full, ax_zoom, ax_t) = plt.subplots(1, 3, figsize=(13.5, 4.6),
                                                 gridspec_kw={"width_ratios": [1, 0.8, 1.15], "wspace": 0.3})
    fig.subplots_adjust(top=0.84, bottom=0.27)
    for ax in (ax_full, ax_zoom):
        ax.imshow(status.T, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest",
                  extent=(safe.delta_edges[0], safe.delta_edges[-1], safe.tau_edges[0], safe.tau_edges[-1]))
        ax.grid(False)
        ax.set_xlabel(r"steering command $\delta_{cmd}$ [rad]")
        ax.plot(*proposal, "D", ms=8, color=RED, mec="white", mew=1.5, label="policy proposal", zorder=5)
        ax.annotate("", projected, proposal, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3}, zorder=4)
        ax.plot(*projected, "s", ms=8, color=INK, mec="white", mew=1.5, label="executed: nearest safe cell",
                zorder=5)
    ax_full.set_ylabel(r"$\tau$ [-]   (+ throttle, - brake)")
    ax_full.axvspan(lo - 0.05, hi + 0.05, fill=False, edgecolor=INK, lw=1.0, ls=(0, (3, 2)))
    ax_full.set_title(f"61 × 61 action grid, step {step0}")
    for edge in safe.delta_edges:
        ax_zoom.axvline(edge, color="white", lw=0.6)
    ax_zoom.axvline(prev[0], color=INK2, lw=1.0, ls=(0, (3, 2)))
    ax_zoom.text(prev[0], -0.97, " previous\n command", fontsize=7.5, color=INK2, va="bottom")
    ax_zoom.set_xlim(min(lo, proposal[0]) - 0.05, max(hi, proposal[0]) + 0.05)
    ax_zoom.set_title("Zoom on the candidate band")

    hist = np.asarray(history)
    sel = (hist[:, 0] >= step0 - 150) & (hist[:, 0] <= step0 + 150)
    h = hist[sel]
    t = (h[:, 0] - step0) * DT
    ax_t.step(t, h[:, 1], where="mid", color=BLUE, label="active cells")
    ax_t.plot(t[h[:, 3] == 1], h[h[:, 3] == 1, 1], "o", ms=4, color=RED, label="empty set: 16 fallback cells")
    ax_t.plot(t[(h[:, 2] == 1) & (h[:, 3] == 0)], h[(h[:, 2] == 1) & (h[:, 3] == 0), 1], "s", ms=5, color=INK,
              label="projection onto a non-empty set")
    ax_t.set_xlabel(f"time from step {step0} [s]")
    ax_t.set_ylabel("active cells")
    ax_t.set_ylim(0, 480)
    ax_t.set_title("Size of the safe set around the event")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PANEL, unsafe_colour, AQUA)]
    labels = ["not a candidate (steering-rate window / front-slip bound)", "candidate, unsafe", "candidate, safe"]
    for ax in (ax_zoom, ax_t):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Safety filter at real states: Treatment A (Monza-only policy) driving Spa R1 through the filter, "
                 f"μ = 1.0, setup seed 42 (s = {state[0]:.0f} m, vx = {state[3]:.1f} m/s at step {step0})",
                 x=0.07, ha="left", fontsize=9, color=INK2)
    save(fig, "safety_filter.png")


# ---------------------------------------------------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------------------------------------------------
def fig_demonstrations() -> None:
    path = TRACKS / "monza/racing_line_R0.npz"
    spec = episode_specs(path)[0]
    steps = 4500
    runs = {kind: run_lap(path, spec, kind, 0, "monza", max_steps=steps)[1] for kind in ("none", "speed", "lateral")}
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.6), sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.2})
    clean = runs["none"]

    def time(arrays: dict) -> np.ndarray:
        return np.arange(len(arrays["label"])) * DT

    sp, lat = runs["speed"], runs["lateral"]
    axes[0, 0].plot(time(sp), 1.0 + sp["speed_noise"], color=ORANGE)
    axes[0, 0].axhline(1.0, color=MUTED, lw=0.8)
    axes[0, 0].set_ylabel(r"factor $1 + d_t$")
    axes[0, 0].set_title(r"speed kind: v_ref preview × (1 + $d_t$), OU σ = 0.08")
    axes[0, 1].plot(time(lat), lat["lateral_noise"], color=ORANGE)
    axes[0, 1].axhline(0.0, color=MUTED, lw=0.8)
    axes[0, 1].set_ylabel(r"offset $n_t$ [m]")
    axes[0, 1].set_title(r"lateral kind: observed $e_y - n_t$, OU σ = 0.25 m")

    vx = {k: np.array([o[OBS.VX] for o in r["obs"]]) for k, r in runs.items()}
    ey = {k: np.array([o[OBS.EY] for o in r["obs"]]) for k, r in runs.items()}
    axes[1, 0].plot(time(clean), vx["none"], color=MUTED, lw=1.2, label="clean teacher lap")
    axes[1, 0].plot(time(sp), vx["speed"], color=BLUE, label="perturbed lap")
    axes[1, 0].set_ylabel(r"$v_x$ [m/s]")
    axes[1, 0].set_title("the car is driven off the nominal speed ...")
    axes[1, 0].legend(loc="best")
    axes[1, 1].plot(time(clean), ey["none"], color=MUTED, lw=1.2, label="clean teacher lap")
    axes[1, 1].plot(time(lat), ey["lateral"], color=BLUE, label="perturbed lap")
    axes[1, 1].set_ylabel(r"$e_y$ [m]")
    axes[1, 1].set_title("... and off the racing line")
    axes[1, 1].legend(loc="best")

    axes[2, 0].plot(time(sp), sp["exec"][:, 1], color=ORANGE, lw=1.2, label=r"executed: filter(MATv0($o'_t$))")
    axes[2, 0].plot(time(sp), sp["label"][:, 1], color=BLUE, lw=1.2, label=r"stored label: filter(MATv0($o_t$))")
    axes[2, 0].set_ylabel(r"$\tau$ [-]")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_title("the label is the correction from the visited state")
    axes[2, 0].legend(loc="best")
    axes[2, 1].plot(time(lat), np.degrees(lat["exec"][:, 0]), color=ORANGE, lw=1.2, label="executed")
    axes[2, 1].plot(time(lat), np.degrees(lat["label"][:, 0]), color=BLUE, lw=1.2, label="stored label")
    axes[2, 1].set_ylabel(r"$\delta_{cmd}$ [deg]")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_title("the label is the correction from the visited state")
    axes[2, 1].legend(loc="best")
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"Perturbed demonstrations on Monza R0, lap 0 (μ = {spec.mu}), first {steps * DT:.0f} s",
                 x=0.07, ha="left", y=0.98, fontsize=10, color=INK2)
    save(fig, "demonstrations.png")


# ---------------------------------------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------------------------------------
def fig_policy_vs_teacher() -> None:
    protocol = json.loads((PROTOCOLS / "spa_R1.json").read_text())
    spec = next(e for e in protocol["episodes"] if e["block"] == "in_sample_gate" and e["mu"] == 1.0
                and e["seed"] == 42)
    rows: list[dict] = []
    run_episode(spec, str(REPO / protocol["racing_line_file"]), "policy", False, model_dir=REPO / "models/all4",
                max_steps=6000, on_step=lambda r: rows.append(
                    {"vx": float(r["state"][3]), "v_target": float(r["v_target"]), "action": np.array(r["action"]),
                     "teacher": np.array(r["teacher_action"])}))
    t = np.arange(len(rows)) * DT
    action = np.array([r["action"] for r in rows])
    teacher = np.array([r["teacher"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), sharex=True, gridspec_kw={"hspace": 0.3})
    axes[0].plot(t, [r["v_target"] for r in rows], color=ORANGE, lw=1.2, label="teacher speed target on this state")
    axes[0].plot(t, [r["vx"] for r in rows], color=BLUE, label=r"$v_x$ driven by the policy")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].set_ylim(0, 95)
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].set_title("ALL4 policy driving Spa R1 unfiltered; the teacher is only evaluated, not applied")
    axes[1].plot(t, teacher[:, 1], color=ORANGE, lw=1.2, label="teacher")
    axes[1].plot(t, action[:, 1], color=BLUE, lw=1.2, label="policy")
    axes[1].set_ylabel(r"$\tau$ [-]")
    axes[1].legend(loc="lower right")
    axes[2].plot(t, np.degrees(teacher[:, 0]), color=ORANGE, lw=1.2, label="teacher")
    axes[2].plot(t, np.degrees(action[:, 0]), color=BLUE, lw=1.2, label="policy")
    axes[2].set_ylabel(r"$\delta_{cmd}$ [deg]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="lower right")
    mae = np.abs(action - teacher).mean(axis=0)
    axes[2].set_title(f"mean |policy - teacher|: steering {np.degrees(mae[0]):.2f} deg, τ {mae[1]:.3f}")
    save(fig, "policy_vs_teacher.png")


# ---------------------------------------------------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------------------------------------------------
def fig_reward() -> None:
    path = TRACKS / "monza/racing_line_R1.npz"
    track = RealTrack(str(path))
    run = teacher_run(path, 0.0, 15.0, 1.0, 42, 20_000)
    s = run["state"][:, 0]
    progress = np.cumsum([progress_increment(a, b, track.track_length) for a, b in zip(s[:-1], s[1:])])
    end = int(np.searchsorted(progress, track.track_length)) + 1
    weights = RewardWeights()
    prog = weights.w_progress * run["progress_term"][:end]
    lat = -weights.w_lateral * run["lateral_term"][:end]
    time_cost = np.full(end, -TIME_COST_RATE * DT)
    t = np.arange(end) * DT
    per_second = int(round(1.0 / DT))
    n_sec = end // per_second

    def rate(x: np.ndarray) -> np.ndarray:
        return x[: n_sec * per_second].reshape(n_sec, per_second).sum(axis=1)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1.2, 1]})
    ts = np.arange(n_sec) + 0.5
    ax_a.plot(ts, rate(prog), color=BLUE, label=r"progress  $w_p\,\max(\dot s, 0)\,dt$")
    ax_a.plot(ts, rate(time_cost), color=ORANGE, label=r"time cost  $-12\,dt$")
    ax_a.plot(ts, rate(lat), color=VIOLET, label=r"lateral  $-w_l (|e_y|/e_{y,max})^2 dt$")
    ax_a.axhline(0, color=MUTED, lw=0.8)
    ax_a.set_xlabel("time [s]")
    ax_a.set_ylabel("reward per second")
    ax_a.set_ylim(-20, 95)
    ax_a.set_title("Reward terms along a teacher lap")
    ax_a.legend(loc="upper left", fontsize=7.5, ncol=1)
    ax_a.text(ts[-1], -16, f"lateral term over the lap: {lat.sum():.2f}", ha="right", fontsize=7.5, color=INK2)

    ret = np.cumsum(prog + lat + time_cost)
    ax_b.plot(t, ret, color=BLUE, label="return so far")
    ax_b.plot(t, np.cumsum(prog), color=MUTED, lw=1.0, ls=(0, (4, 3)), label="progress only (= distance)")
    lap_time = end * DT
    ax_b.text(0.97, 0.04, f"lap time T = {lap_time:.1f} s\nreturn {ret[-1]:.0f} ≈ L - 12 T\n"
              f"= {track.track_length:.0f} - 12 · {lap_time:.1f}", transform=ax_b.transAxes, ha="right",
              va="bottom", fontsize=8, color=INK2)
    ax_b.set_xlabel("time [s]")
    ax_b.set_ylabel("cumulative reward [m]")
    ax_b.set_title("Return: faster laps score higher")
    ax_b.legend(loc="upper left", fontsize=7.5)

    v = np.linspace(0, 60, 200)
    ax_c.plot(v, -(CRASH_BASE + weights.w_crash * v * v), color=RED)
    ax_c.set_xlabel(r"speed at failure $\sqrt{v_x^2+v_y^2}$ [m/s]")
    ax_c.set_ylabel("one-off penalty [m]")
    ax_c.set_title("Failure penalty  -(3 + 0.02 v²)")
    save(fig, "reward.png")


FIGURES = {
    "vehicle_model": fig_vehicle_model, "circuits": fig_circuits, "racing_line_repair": fig_racing_line_repair,
    "tyre_model": fig_tyre_model, "observation": fig_observation, "teacher": fig_teacher,
    "safety_filter": fig_safety_filter, "demonstrations": fig_demonstrations,
    "policy_vs_teacher": fig_policy_vs_teacher, "reward": fig_reward,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="figures to generate (default: all)")
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
