"""Assembly of behaviour-cloning datasets and of offline (MDP) datasets from demonstration laps.

A behaviour-cloning dataset NPZ holds ``observations`` (N x 116, float32), ``actions`` (N x 2, the labels),
``episode_ends`` (cumulative lap ends) and, per lap, ``start_s_per_lap``, ``start_vx_per_lap``, ``mu_per_lap`` and
``setup_per_lap``. The behaviour-cloning gate replays the start of the first lap, so the first circuit of a dataset
is Monza.

The final datasets combine, for every training circuit in the order Monza, Spa, Catalunya, Budapest, the kept
speed-perturbed laps followed by the kept lateral-perturbed laps:

    treatment_a     Monza only
    all4            Monza, Spa, Catalunya, Budapest
    loco_<circuit>  all circuits except <circuit> (leave-one-circuit-out)

An offline (MDP) dataset is a behaviour-cloning dataset with the transitions of the laps (``assemble_mdp``):

    observations            N x 116   privileged observation o_t (float32)
    imu                     N x 2     accelerometer channels of o_t
    actions                 N x 2     teacher label (so that behaviour cloning reads an MDP dataset unchanged)
    actions_exec            N x 2     executed action a_t, the one that produced o_{t+1}
    rewards                 N         reward of the transition (OFFLINE_REWARD_WEIGHTS)
    traction_costs          N         traction cost of the transition
    terminals               N         1 at the last transition of a lap that ended by a failure
    timeouts                N         1 at the last transition of a lap that ended by completion or step limit
    final_observations      E x 116   o_T after the last transition of each lap (E laps)
    final_imu               E x 2     accelerometer channels of o_T

plus the per-lap metadata above and ``circuit_per_lap``, ``kind_per_lap``, ``attempt_per_lap``,
``healthy_per_lap``, ``speed_safety_per_lap`` (k of the labelling teacher) and ``exec_speed_safety_per_lap`` (k of the
teacher that drove the lap). The next observation of transition t is o_{t+1} inside a lap and
the lap's final observation at its last transition: with the E final observations appended after the N transitions,
it is row ``next_index(episode_ends)[t]``, so the next states need no second copy of the rows. Student observations
of any level are built from the same arrays (``student_dataset``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.learning.demonstrations import LapSpec
from contextual_bicycle_pacejka.model.student import previous_actions, student_observations
from contextual_bicycle_pacejka.spec.layout import StudentLayout

CIRCUITS = ("monza", "spa", "catalunya", "budapest")
DATASETS: dict[str, tuple[str, ...]] = {
    "treatment_a": ("monza",),
    "all4": CIRCUITS,
    "loco_spa": ("monza", "catalunya", "budapest"),
    "loco_catalunya": ("monza", "spa", "budapest"),
    "loco_budapest": ("monza", "spa", "catalunya"),
}
KINDS = ("speed", "lateral")
MDP_LAP_KEYS = ("obs", "label", "exec", "imu", "reward", "traction_cost", "final_obs", "final_imu")


def kept_laps(lap_dir: str | Path) -> list[dict]:
    """Kept laps of a generation directory, in lap order."""
    lap_dir = Path(lap_dir)
    report = json.loads((lap_dir / "generation.json").read_text())
    laps = []
    for entry in report:
        if entry["kept_attempt"] is None:
            continue
        with np.load(lap_dir / f"lap{entry['lap']:02d}.npz") as data:
            laps.append({"lap": entry["lap"], "obs": data["obs"].astype(np.float32),
                         "label": data["label"].astype(np.float32)})
    return laps


def assemble(laps_by_circuit: dict[str, dict[str, list[dict]]], specs_by_circuit: dict[str, list[LapSpec]],
             circuits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Concatenate the kept laps of ``circuits`` (speed then lateral, per circuit) into dataset arrays."""
    laps, meta = [], {"start_s_per_lap": [], "start_vx_per_lap": [], "mu_per_lap": [], "setup_per_lap": []}
    for circuit in circuits:
        for kind in KINDS:
            for lap in laps_by_circuit[circuit][kind]:
                spec = specs_by_circuit[circuit][lap["lap"]]
                laps.append(lap)
                meta["start_s_per_lap"].append(spec.start_s)
                meta["start_vx_per_lap"].append(spec.start_vx)
                meta["mu_per_lap"].append(spec.mu)
                meta["setup_per_lap"].append(spec.setup)
    observations = np.concatenate([lap["obs"] for lap in laps])
    return {
        "observations": observations,
        "actions": np.concatenate([lap["label"] for lap in laps]),
        "episode_ends": np.cumsum([len(lap["obs"]) for lap in laps]).astype(np.int64),
        **{k: np.asarray(v) for k, v in meta.items()},
    }


# --------------------------------------------------------------------------------------------- offline datasets
def mdp_laps(lap_dir: str | Path, include_failed: bool = False, max_failed: int | None = None,
             ) -> list[dict[str, Any]]:
    """Laps of a generation directory with their transitions: the kept laps and, with ``include_failed``, the
    saved attempts that were not kept (the first ``max_failed`` of each lap, all by default), in lap and attempt
    order. Each lap holds ``MDP_LAP_KEYS`` and ``summary``."""
    lap_dir = Path(lap_dir)
    report = json.loads((lap_dir / "generation.json").read_text())
    laps = []
    for entry in report:
        files = list(entry.get("failed_files", []))[:max_failed] if include_failed else []
        if entry["kept_attempt"] is not None:
            files.append(f"lap{entry['lap']:02d}.npz")
        for name in files:
            with np.load(lap_dir / name, allow_pickle=False) as data:
                missing = [k for k in MDP_LAP_KEYS if k not in data.files]
                if missing:
                    raise ValueError(f"{lap_dir / name} lacks {missing}: regenerate it with the current generator")
                lap = {k: np.asarray(data[k]) for k in MDP_LAP_KEYS}
                lap["summary"] = json.loads(str(data["meta"]))
            laps.append(lap)
    return laps


def assemble_mdp(laps: list[dict[str, Any]], specs_by_circuit: dict[str, list[LapSpec]]) -> dict[str, np.ndarray]:
    """Concatenate laps of ``mdp_laps`` (in the given order) into an offline dataset (see the module docstring)."""
    if not laps:
        raise ValueError("no laps to assemble")
    meta: dict[str, list] = {k: [] for k in ("start_s_per_lap", "start_vx_per_lap", "mu_per_lap", "setup_per_lap",
                                              "circuit_per_lap", "kind_per_lap", "attempt_per_lap",
                                              "healthy_per_lap", "speed_safety_per_lap",
                                              "exec_speed_safety_per_lap")}
    terminals, timeouts = [], []
    for lap in laps:
        summary = lap["summary"]
        spec = specs_by_circuit[summary["circuit"]][summary["lap"]]
        if float(spec.mu) != float(summary["mu"]):
            raise ValueError(f"lap {summary['lap']} of {summary['circuit']}: friction differs from its specification")
        meta["start_s_per_lap"].append(spec.start_s)
        meta["start_vx_per_lap"].append(spec.start_vx)
        meta["mu_per_lap"].append(spec.mu)
        meta["setup_per_lap"].append(spec.setup)
        meta["circuit_per_lap"].append(summary["circuit"])
        meta["kind_per_lap"].append(summary["kind"])
        meta["attempt_per_lap"].append(summary["attempt"])
        meta["healthy_per_lap"].append(summary["healthy"])
        meta["speed_safety_per_lap"].append(summary["speed_safety"])
        meta["exec_speed_safety_per_lap"].append(summary.get("exec_speed_safety", summary["speed_safety"]))
        n = len(lap["obs"])
        end = np.zeros(n, dtype=np.float32)
        end[-1] = 1.0
        terminals.append(end * float(summary["terminated"]))
        timeouts.append(end * float(not summary["terminated"]))
    return {
        "observations": np.concatenate([lap["obs"] for lap in laps]).astype(np.float32),
        "imu": np.concatenate([lap["imu"] for lap in laps]).astype(np.float32),
        "actions": np.concatenate([lap["label"] for lap in laps]).astype(np.float32),
        "actions_exec": np.concatenate([lap["exec"] for lap in laps]).astype(np.float32),
        "rewards": np.concatenate([lap["reward"] for lap in laps]).astype(np.float32),
        "traction_costs": np.concatenate([lap["traction_cost"] for lap in laps]).astype(np.float32),
        "terminals": np.concatenate(terminals),
        "timeouts": np.concatenate(timeouts),
        "episode_ends": np.cumsum([len(lap["obs"]) for lap in laps]).astype(np.int64),
        "final_observations": np.stack([lap["final_obs"] for lap in laps]).astype(np.float32),
        "final_imu": np.stack([lap["final_imu"] for lap in laps]).astype(np.float32),
        **{k: np.asarray(v) for k, v in meta.items()},
    }


def episode_bounds(episode_ends: NDArray[np.integer]) -> list[tuple[int, int]]:
    """(start, end) row ranges of the laps."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def next_index(episode_ends: NDArray[np.integer]) -> NDArray[np.int64]:
    """Row of the next state of each transition, (N,), among the N transition rows followed by the E final rows.

    The row is t + 1 inside a lap and N + lap at the lap's last transition.
    """
    ends = np.asarray(episode_ends, dtype=np.int64)
    index = np.arange(1, int(ends[-1]) + 1, dtype=np.int64)
    index[ends - 1] = int(ends[-1]) + np.arange(len(ends))
    return index


def student_dataset(data: dict[str, NDArray] | np.lib.npyio.NpzFile, layout: StudentLayout,
                    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Student observations of an offline dataset with those of the next states: (N + E, layout.dim) rows, the N
    transitions followed by the final observation of each of the E laps, and the row of the next state of each
    transition (``next_index``).

    The previous action of the next state of transition t is the executed action a_t, as in the environment.
    """
    return _student_rows(data, layout, with_next=True), next_index(data["episode_ends"])


def student_dataset_observations(data: dict[str, NDArray] | np.lib.npyio.NpzFile, layout: StudentLayout,
                                 ) -> NDArray[np.float32]:
    """Student observations of every transition of an offline dataset, (N, layout.dim) (no next states)."""
    return _student_rows(data, layout, with_next=False)


def _student_rows(data: dict[str, NDArray] | np.lib.npyio.NpzFile, layout: StudentLayout, with_next: bool,
                  ) -> NDArray[np.float32]:
    observations, imu = np.asarray(data["observations"]), np.asarray(data["imu"])
    executed = np.asarray(data["actions_exec"])
    final_obs, final_imu = np.asarray(data["final_observations"]), np.asarray(data["final_imu"])
    mus = np.asarray(data["mu_per_lap"])
    n = len(observations)
    rows = np.empty((n + len(final_obs) if with_next else n, layout.dim), dtype=np.float32)
    for lap, (start, end) in enumerate(episode_bounds(data["episode_ends"])):
        obs = np.concatenate([observations[start:end], final_obs[lap][None]])
        lap_imu = np.concatenate([imu[start:end], final_imu[lap][None]])
        prev = previous_actions(np.concatenate([executed[start:end], executed[end - 1:end]]))
        student = student_observations(obs, lap_imu, prev, float(mus[lap]), layout)
        rows[start:end] = student[:-1]
        if with_next:
            rows[n + lap] = student[-1]
    return rows
