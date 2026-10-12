"""Racing line R1: R0 with its curvature kinks replaced by local minimum-curvature re-optimisations.

Kink detection. Sample i is a spike if |kappa_i| > 0.06 1/m and |kappa_i| > 2 max |kappa_{i+j}|, j in {-4,-3,3,4};
spikes closer than 10 m form one kink located at their mean arc length.

Repair of a kink at s_k. For half-windows W = 25, 40, 60 m (first that passes), the samples strictly inside
(s_k - W, s_k + W) are replaced. Stations every 2.5 m of centreline arc length carry an offset along the centreline
normal, bounded 1 m inside the boundaries; the offsets minimise the discrete (Menger) squared curvature of the
chain formed with two fixed samples on each side (L-BFGS-B). The chain is splined and resampled at the nominal
spacing. A window passes if its peak curvature is at most 1.2 x the centreline peak curvature within 25 m and its
boundary margin is at least 1 m; if no window passes, the first window with margin >= 1 m and no spike in the
repaired samples (with 4 samples of context on each side) is used.

Everything outside the windows is copied bit-for-bit. The speed profile is carried over through the old-to-new
arc-length map and ax is recomputed. Checks: every kink repaired (G1), no spike left (G2), kept samples identical
and lap length changed by less than 5 m (G3), the result loads as a track with finite geometry (G4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from contextual_bicycle_pacejka.model.track import RealTrack

SPIKE_ABS = 0.06  # [1/m]
SPIKE_RATIO = 2.0
SPIKE_NEIGHBOURS = (-4, -3, 3, 4)
KINK_MERGE_M = 10.0
WINDOWS_M = (25.0, 40.0, 60.0)
KAPPA_RATIO_MAX = 1.2
CENTRE_RADIUS_M = 25.0
MARGIN_MIN_M = 1.0
WIDTH_VALIDATION_TOL_M = 0.05
LENGTH_CHANGE_MAX_M = 5.0
STATION_M = 2.5
N_DENSE = 4000
# Duplicate key names of older racing-line files, kept equal to the canonical ones.
DUPLICATE_KEYS = {"s": "s_m", "x": "x_m", "y": "y_m", "psi": "psi_rad", "kappa": "kappa_radpm", "v_ref": "v_ref_mps",
                  "ax": "ax_mps2"}
STALE_KEYS = ("alpha_m", "centreline_s_m")  # per-sample metadata that no longer matches the repaired samples


def spikes(kappa: NDArray[np.float64]) -> list[int]:
    """Indices of curvature spikes."""
    n = len(kappa)
    out = []
    for i in range(n):
        neighbours = np.abs(kappa[[(i + j) % n for j in SPIKE_NEIGHBOURS]]).max()
        if abs(kappa[i]) > SPIKE_ABS and abs(kappa[i]) > SPIKE_RATIO * neighbours:
            out.append(i)
    return out


def kinks(s: NDArray[np.float64], spike_indices: list[int]) -> list[float]:
    """Arc lengths [m] of the kinks (spikes grouped within 10 m)."""
    groups: list[list[int]] = []
    for i in spike_indices:
        if groups and s[i] - s[groups[-1][-1]] <= KINK_MERGE_M:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [float(np.mean(s[g])) for g in groups]


def boundaries(reference: dict[str, NDArray]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Closed right and left boundary polylines of the centreline reference [m]."""
    x, y = reference["x_m"], reference["y_m"]
    dx = np.gradient(np.r_[x[-1], x, x[0]])[1:-1]
    dy = np.gradient(np.r_[y[-1], y, y[0]])[1:-1]
    h = np.hypot(dx, dy)
    tx, ty = dx / h, dy / h
    right = np.c_[x + ty * reference["w_tr_right_m"], y - tx * reference["w_tr_right_m"]]
    left = np.c_[x - ty * reference["w_tr_left_m"], y + tx * reference["w_tr_left_m"]]
    return right, left


def distance_to_polyline(points: NDArray[np.float64], polyline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance [m] of each point to a closed polyline."""
    a, b = polyline, np.roll(polyline, -1, axis=0)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    out = np.empty(len(points))
    for n, q in enumerate(points):
        t = np.clip(np.sum((q - a) * ab, axis=1) / ab2, 0.0, 1.0)
        out[n] = np.min(np.hypot(*(a + t[:, None] * ab - q).T))
    return out


class Centreline:
    """Periodic splines of the centreline reference: position, left normal and boundary distances [m]."""

    def __init__(self, reference: dict[str, NDArray]) -> None:
        s = np.asarray(reference["s_m"], float)
        self.length = float(reference["track_length"])
        knots = np.r_[s, self.length]

        def spline(values: NDArray) -> CubicSpline:
            return CubicSpline(knots, np.r_[values, values[0]], bc_type="periodic")

        self.x, self.y = spline(reference["x_m"]), spline(reference["y_m"])
        self.w_right, self.w_left = spline(reference["w_tr_right_m"]), spline(reference["w_tr_left_m"])
        self.dense = np.arange(0.0, self.length, 0.1)
        self.dense_xy = np.c_[self.x(self.dense), self.y(self.dense)]

    def at(self, sigma: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Points and unit left normals at centreline arc lengths ``sigma``."""
        sigma = np.mod(sigma, self.length)
        tangent = np.c_[self.x(sigma, 1), self.y(sigma, 1)]
        tangent /= np.hypot(tangent[:, 0], tangent[:, 1])[:, None]
        return np.c_[self.x(sigma), self.y(sigma)], np.c_[-tangent[:, 1], tangent[:, 0]]

    def project(self, point: NDArray[np.float64]) -> tuple[float, float]:
        """(centreline arc length, signed lateral offset) of a point (nearest of the 0.1 m samples)."""
        sigma = float(self.dense[int(np.argmin(np.hypot(*(self.dense_xy - point).T)))])
        centre, normal = self.at(np.array([sigma]))
        return sigma, float((point - centre[0]) @ normal[0])


def _menger_curvature(points: NDArray[np.float64]) -> NDArray[np.float64]:
    a, b, c = points[:-2], points[1:-1], points[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    return 2.0 * cross / (np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T))


def repair_window(line: dict[str, NDArray], ia: int, ib: int, ds_nominal: float, psi_offset: float,
                  centreline: Centreline) -> dict[str, Any]:
    """Re-optimised samples strictly between the kept samples ``ia`` and ``ib``."""
    x, y, s = line["x_m"], line["y_m"], line["s_m"]
    n = len(s)

    def point(i: int) -> NDArray[np.float64]:
        return np.array([x[i % n], y[i % n]])

    sa, offset_a = centreline.project(point(ia))
    sb, offset_b = centreline.project(point(ib))
    sigma = np.linspace(sa, sb, max(2, int(round((sb - sa) / STATION_M))) + 1)
    centre, normal = centreline.at(sigma)
    lower = -(centreline.w_right(np.mod(sigma, centreline.length)) - MARGIN_MIN_M)
    upper = centreline.w_left(np.mod(sigma, centreline.length)) - MARGIN_MIN_M
    projected = [centreline.project(point(i)) for i in range(ia, ib + 1)]
    initial = np.clip(np.interp(sigma, [p[0] for p in projected], [p[1] for p in projected]), lower, upper)
    before, after = np.array([point(ia - 2), point(ia - 1)]), np.array([point(ib + 1), point(ib + 2)])

    def chain(inner: NDArray[np.float64]) -> NDArray[np.float64]:
        stations = centre + np.r_[offset_a, inner, offset_b][:, None] * normal
        stations[0], stations[-1] = point(ia), point(ib)
        return np.vstack([before, stations, after])

    def objective(inner: NDArray[np.float64]) -> float:
        points = chain(inner)
        segment = np.hypot(*np.diff(points, axis=0).T)
        return float(np.sum(_menger_curvature(points) ** 2 * 0.5 * (segment[:-1] + segment[1:])))

    result = minimize(objective, initial[1:-1], method="L-BFGS-B", bounds=list(zip(lower[1:-1], upper[1:-1])),
                      options={"maxiter": 5000})
    points = chain(result.x)
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    sx, sy = CubicSpline(t, points[:, 0]), CubicSpline(t, points[:, 1])
    u = np.linspace(t[2], t[-3], N_DENSE)
    xd, yd, xdd, ydd = sx(u, 1), sy(u, 1), sx(u, 2), sy(u, 2)
    speed = np.hypot(xd, yd)
    arc = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))]
    new_length = float(arc[-1])
    targets = np.linspace(0.0, new_length, max(1, int(round(new_length / ds_nominal))) + 1)[1:-1]
    uu = np.interp(targets, arc, u)
    xd_t, yd_t = sx(uu, 1), sy(uu, 1)
    heading = np.unwrap(np.r_[line["psi_rad"][ia] - psi_offset, np.arctan2(yd_t, xd_t)])[1:]
    return {"x": sx(uu), "y": sy(uu), "psi": heading + psi_offset,
            "kappa": (xd_t * sy(uu, 2) - yd_t * sx(uu, 2)) / np.hypot(xd_t, yd_t) ** 3, "s_rel": targets,
            "new_len": new_length, "old_len": float(s[ib] - s[ia]), "dense_xy": np.c_[sx(u), sy(u)],
            "dense_kappa": (xd * ydd - yd * xdd) / speed ** 3, "opt_success": bool(result.success)}


def _choose_window(line: dict[str, NDArray], reference: dict[str, NDArray], s_kink: float, ds_nominal: float,
                   psi_offset: float, centreline: Centreline, right: NDArray, left: NDArray,
                   ) -> tuple[dict[str, Any], tuple | None]:
    s, kappa = line["s_m"], line["kappa_radpm"]
    record: dict[str, Any] = {"s_kink_m": s_kink, "tried": [], "fallback": False}
    candidates = []
    for half_width in WINDOWS_M:
        inside = np.where((s > s_kink - half_width) & (s < s_kink + half_width))[0]
        ia, ib = int(inside[0]) - 1, int(inside[-1]) + 1
        window = repair_window(line, ia, ib, ds_nominal, psi_offset, centreline)
        near = np.hypot(reference["x_m"][:, None] - window["dense_xy"][None, ::50, 0],
                        reference["y_m"][:, None] - window["dense_xy"][None, ::50, 1]).min(axis=1) < CENTRE_RADIUS_M
        kappa_centre = float(np.abs(reference["kappa_radpm"][near]).max())
        kappa_new = float(np.abs(window["dense_kappa"]).max())
        dense = window["dense_xy"][::20]
        margin = float(min(distance_to_polyline(dense, right).min(), distance_to_polyline(dense, left).min()))
        passed = kappa_new <= KAPPA_RATIO_MAX * kappa_centre and margin >= MARGIN_MIN_M
        record["tried"].append({"W_m": half_width, "kappa_ratio": kappa_new / kappa_centre, "min_margin_m": margin,
                                "len_change_m": window["new_len"] - window["old_len"], "pass": passed})
        candidates.append((half_width, ia, ib, window, margin))
        if passed:
            return record, (half_width, ia, ib, window)
    for half_width, ia, ib, window, margin in candidates:
        context = np.r_[kappa[np.arange(ia - 4, ia + 1) % len(kappa)], window["kappa"],
                        kappa[np.arange(ib, ib + 5) % len(kappa)]]
        local = [j for j in spikes(context) if 5 <= j < 5 + len(window["kappa"])]
        if margin >= MARGIN_MIN_M and not local:
            record["fallback"] = True
            return record, (half_width, ia, ib, window)
    return record, None


def repair_racing_line(source: str | Path, reference: str | Path, output: str | Path) -> dict[str, Any]:
    """Write the R1 repair of racing line ``source`` (centreline ``reference``) to ``output``; return the log."""
    with np.load(source, allow_pickle=False) as data:
        line = {k: np.asarray(data[k]) for k in data.files}
    with np.load(reference, allow_pickle=False) as data:
        ref = {k: np.asarray(data[k]) for k in data.files}
    s, x, y, kappa = line["s_m"], line["x_m"], line["y_m"], line["kappa_radpm"]
    n, length = len(s), float(line["track_length"])
    psi_offset = float(np.median(np.angle(np.exp(1j * (line["psi_rad"][:-1] - np.arctan2(np.diff(y), np.diff(x)))))))
    ds_nominal = float(np.median(np.hypot(np.diff(x), np.diff(y))))
    right, left = boundaries(ref)
    centreline = Centreline(ref)
    probe = np.arange(0, n, 25)
    points = np.c_[x[probe], y[probe]]
    width_error = float(np.median(np.r_[np.abs(distance_to_polyline(points, right) - line["w_tr_right_m"][probe]),
                                        np.abs(distance_to_polyline(points, left) - line["w_tr_left_m"][probe])]))
    if width_error > WIDTH_VALIDATION_TOL_M:
        raise ValueError(f"boundary distances of {source} do not match the reference (median {width_error:.3f} m)")
    log: dict[str, Any] = {"psi_offset_rad": psi_offset, "ds_nominal_m": ds_nominal,
                           "width_validation_median_abs_m": width_error, "kinks": []}
    plan = []
    for s_kink in kinks(s, spikes(kappa)):
        record, chosen = _choose_window(line, ref, s_kink, ds_nominal, psi_offset, centreline, right, left)
        record["repaired"] = chosen is not None
        if chosen:
            record["W_m"] = chosen[0]
            plan.append(chosen)
        log["kinks"].append(record)

    keep = np.ones(n, bool)
    for _, ia, ib, _ in plan:
        keep[ia + 1:ib] = False
    windows = {ia: (ib, window) for _, ia, ib, window in plan}
    xs, ys, psis, kappas, s_new, s_old = [], [], [], [], [], []
    shift = 0.0
    for i in range(n):
        if not keep[i]:
            continue
        xs.append(x[i])
        ys.append(y[i])
        psis.append(line["psi_rad"][i])
        kappas.append(kappa[i])
        s_new.append(s[i] + shift)
        s_old.append(s[i])
        if i in windows:
            _, w = windows[i]
            xs += list(w["x"])
            ys += list(w["y"])
            psis += list(w["psi"])
            kappas += list(w["kappa"])
            s_new += list(s[i] + shift + w["s_rel"])
            s_old += list(s[i] + w["s_rel"] * (w["old_len"] / w["new_len"]))
            shift += w["new_len"] - w["old_len"]
    s_new, s_old = np.asarray(s_new), np.asarray(s_old)
    new_length = length + shift
    speed = np.interp(s_old, s, line["v_ref_mps"], period=length)
    ax = np.diff(np.r_[speed * speed, speed[0] * speed[0]]) / (2.0 * np.diff(np.r_[s_new, new_length]))
    xs, ys = np.asarray(xs), np.asarray(ys)
    added = ~np.isin(np.round(s_old, 9), np.round(s[keep], 9))
    w_right, w_left = np.empty(len(xs)), np.empty(len(xs))
    kept_index = iter(np.where(keep)[0])
    for j in range(len(xs)):
        if added[j]:
            w_right[j] = distance_to_polyline(np.c_[xs[j:j + 1], ys[j:j + 1]], right)[0]
            w_left[j] = distance_to_polyline(np.c_[xs[j:j + 1], ys[j:j + 1]], left)[0]
        else:
            i = next(kept_index)
            w_right[j], w_left[j] = line["w_tr_right_m"][i], line["w_tr_left_m"][i]
    out = dict(line)
    out.update({"x_m": xs, "y_m": ys, "s_m": s_new, "psi_rad": np.asarray(psis), "kappa_radpm": np.asarray(kappas),
                "track_length": np.float64(new_length), "v_ref_mps": speed, "v_ref": speed, "ax_mps2": ax, "ax": ax,
                "w_tr_right_m": w_right, "w_tr_left_m": w_left,
                "repair": np.asarray(json.dumps({"source": Path(source).name, "method": "kink repair R1"}))})
    for duplicate, canonical in DUPLICATE_KEYS.items():
        if duplicate in out:
            out[duplicate] = out[canonical]
    for key in STALE_KEYS:
        out.pop(key, None)
    output = Path(output)
    np.savez(output, **out)

    kept_identical = all(np.array_equal(line[f][keep], out[f][~added])
                         for f in ("x_m", "y_m", "psi_rad", "kappa_radpm", "w_tr_right_m", "w_tr_left_m"))
    track = RealTrack(str(output))
    queries = np.arange(0.0, new_length, 0.5)
    values = np.array([[track.kappa(q), *track.centerline_point(q), track.boundary_left(q), track.boundary_right(q)]
                       for q in queries])
    log.update({"track_length_old_m": length, "track_length_new_m": new_length, "n_samples_old": n,
                "n_samples_new": len(xs), "max_abs_kappa": float(np.abs(kappas).max()),
                "G1_all_repaired": all(r["repaired"] for r in log["kinks"]),
                "G2_no_spike_left": spikes(np.asarray(kappas)) == [],
                "G3_kept_identical_length_ok": bool(kept_identical and abs(new_length - length) < LENGTH_CHANGE_MAX_M),
                "G4_finite_geometry": bool(np.isfinite(values).all())})
    return log
