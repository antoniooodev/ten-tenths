"""Is an agent a copy of the teacher? Speed traces of protocol episodes and the measures that tell a copy apart.

An agent that copies the teacher's plan drives the teacher's speed profile, possibly scaled, and leaves the track
where the teacher does (docs/REDESIGN.md, section 23.1). The measures compare an agent with the teacher on the same
protocol episodes (same setup, start and friction), on the episodes robust for both:

    shape          the speed profile on a grid of progress (``GRID_M``) against the best scaled teacher profile,
                   k = sum(v_a v_t) / sum(v_t^2): the residual rms(v_a - k v_t) / mean(v_a). A copy, scaled or not,
                   has a small residual; a plan of its own a large one
    sectors        the time gain over the teacher in each sector of progress (``SECTOR_M``), 1 - t_agent / t_teacher:
                   a scaled copy gains about the same everywhere, a plan of its own gains in some sectors and loses in
                   others (dispersion across the sectors, share of sectors slower than the teacher)
    friction law   the spread across the frictions of the lap time over the teacher's: a copy keeps the teacher's
                   friction scaling of the plan, so the ratio is the same at every friction
    failures       the share of the agent's failures within ``FAILURE_RADIUS_M`` of a teacher failure on the same
                   protocol, and the teacher failures the agent survives
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.spec.constants import DT

GRID_M = 10.0  # [m] progress grid of the speed profiles
SECTOR_M = 500.0  # [m]
FAILURE_RADIUS_M = 100.0  # [m]


def speed_trace(spec: dict[str, Any], racing_line: str, model_dir: str | None = None, plan_scale: float = 1.0,
                ) -> dict[str, Any]:
    """Progress [m] and speed [m/s] of every step of one protocol episode (the teacher when ``model_dir`` is
    None), with the episode's outcome."""
    progress: list[float] = []
    speed: list[float] = []

    def record(step: dict[str, Any]) -> None:
        progress.append(float(step["progress"]))
        speed.append(float(step["state"][3]))

    record_ = run_episode(spec, racing_line, "teacher" if model_dir is None else "policy", False,
                          model_dir=model_dir, on_step=record, plan_scale=plan_scale)
    return {"progress": np.asarray(progress, dtype=np.float32), "speed": np.asarray(speed, dtype=np.float32),
            "robust": bool(record_["robust"]), "mu": float(spec["mu"]), "terminal_s": float(record_["terminal_s_m"]),
            "lap_time_s": record_["lap_time_s"]}


def profile(progress: NDArray, speed: NDArray, length: float, grid_m: float = GRID_M) -> NDArray[np.float64]:
    """Mean speed in each cell of a progress grid over ``length`` (NaN where the episode has no step)."""
    cells = np.minimum((np.asarray(progress) // grid_m).astype(int), int(length // grid_m) - 1)
    keep = cells >= 0
    sums = np.bincount(cells[keep], weights=np.asarray(speed, dtype=np.float64)[keep], minlength=int(length // grid_m))
    counts = np.bincount(cells[keep], minlength=int(length // grid_m))
    with np.errstate(invalid="ignore"):
        return sums / counts


def shape_residual(agent: NDArray, teacher: NDArray) -> tuple[float, float]:
    """Scale k of the teacher profile closest to the agent's, and the residual rms(agent - k teacher) / mean(agent),
    on the cells both profiles have."""
    both = np.isfinite(agent) & np.isfinite(teacher)
    a, t = agent[both], teacher[both]
    k = float(np.dot(a, t) / np.dot(t, t))
    return k, float(np.sqrt(np.mean((a - k * t) ** 2)) / np.mean(a))


def sector_times(progress: NDArray, length: float, sector_m: float = SECTOR_M) -> NDArray[np.float64]:
    """Time [s] spent in each sector of progress."""
    sectors = np.minimum((np.asarray(progress) // sector_m).astype(int), int(np.ceil(length / sector_m)) - 1)
    return np.bincount(sectors[sectors >= 0], minlength=int(np.ceil(length / sector_m))) * DT


def colocated(sites: list[float], references: list[float], length: float, radius_m: float = FAILURE_RADIUS_M,
              ) -> list[bool]:
    """Whether each failure site lies within ``radius_m`` of a reference site (distance along the closed lap)."""
    out = []
    for s in sites:
        gaps = [abs((s - r + 0.5 * length) % length - 0.5 * length) for r in references]
        out.append(bool(gaps) and min(gaps) <= radius_m)
    return out


def independence(agent: list[dict[str, Any]], teacher: list[dict[str, Any]], length: float) -> dict[str, Any]:
    """The measures of the module docstring for one agent against the teacher, the traces paired by episode."""
    shapes, gains, by_mu = [], [], {}
    for a, t in zip(agent, teacher, strict=True):
        if not (a["robust"] and t["robust"]):
            continue
        shapes.append(shape_residual(profile(a["progress"], a["speed"], length),
                                     profile(t["progress"], t["speed"], length)))
        gains.append(1.0 - sector_times(a["progress"], length) / sector_times(t["progress"], length))
        times = by_mu.setdefault(a["mu"], [0.0, 0.0])
        times[0] += a["lap_time_s"]
        times[1] += t["lap_time_s"]
    gains_ = np.asarray(gains)
    ratios = {mu: a / t for mu, (a, t) in sorted(by_mu.items())}
    teacher_sites = [t["terminal_s"] for t in teacher if not t["robust"]]
    agent_sites = [a["terminal_s"] for a in agent if not a["robust"]]
    inherited = colocated(agent_sites, teacher_sites, length)
    return {
        "episodes": len(shapes), "robust": sum(a["robust"] for a in agent), "n": len(agent),
        "shape_k": float(np.mean([k for k, _ in shapes])) if shapes else None,
        "shape_residual": float(np.mean([r for _, r in shapes])) if shapes else None,
        "sector_gain_mean": float(gains_.mean()) if gains else None,
        "sector_gain_std": float(gains_.std(axis=1).mean()) if gains else None,
        "sector_slower_share": float((gains_ < 0.0).mean()) if gains else None,
        "ratio_by_mu": ratios, "ratio_spread": max(ratios.values()) - min(ratios.values()) if ratios else None,
        "failures": len(agent_sites), "failures_at_teacher_sites": int(sum(inherited)),
        "teacher_failures": len(teacher_sites),
        "teacher_failures_survived": sum(a["robust"] for a, t in zip(agent, teacher, strict=True) if not t["robust"]),
    }
