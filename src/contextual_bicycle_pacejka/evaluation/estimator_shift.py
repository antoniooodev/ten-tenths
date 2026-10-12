"""Why a friction estimate leaves its calibration (docs/REDESIGN.md §25.5, diagnostic b).

``record_inputs`` replays a protocol episode and keeps, for the reset and every step, what the estimator reads
(privileged observation, IMU, previous executed action) and the arc length s. ``stages`` recomputes the estimate
offline (``FrictionEstimator.estimate_episode``), which equals the online one to rounding, and returns every stage:
the network measurement, the filtered mean and the lower bound, as ratios to the true mu.

Coverage: ``training_windows`` builds the estimator's input windows of one circuit from the datasets of its report
(one window every ``every`` steps, normalised as the estimator normalises them); ``nearest_distances`` gives the
Euclidean distance of each query window to its nearest training window. The datasets are uncompressed NPZ files, so
their arrays are memory-mapped (``stored_array``) instead of loaded.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning.friction import ESTIMATOR_FILE, FrictionEstimator
from contextual_bicycle_pacejka.model.student import channel_frames, previous_actions, stack_window
from contextual_bicycle_pacejka.spec.layout import FRICTION_CHANNELS


def record_inputs(spec: dict[str, Any], racing_line: str | Path, model_dir: str | Path,
                  mu_estimator_dir: str | Path) -> dict[str, Any]:
    """Replay one episode on the estimate; return its record and the estimator inputs of the reset and every step."""
    rows: dict[str, list] = {"obs": [], "imu": [], "executed": [], "s": []}

    def keep(obs: NDArray, imu: NDArray, executed: NDArray, s: float) -> None:
        rows["obs"].append(np.asarray(obs, dtype=np.float32))
        rows["imu"].append(np.asarray(imu, dtype=np.float32))
        rows["executed"].append(np.asarray(executed, dtype=np.float32))
        rows["s"].append(float(s))

    def on_reset(step: dict[str, Any]) -> None:
        keep(step["obs"], step["imu"], np.zeros(2, dtype=np.float32), spec["start_s"])

    def on_step(step: dict[str, Any]) -> None:
        keep(step["obs"], step["imu"], step["action"], step["state"][0])

    record = run_episode(spec, racing_line, "policy", False, model_dir=model_dir, mu_estimator_dir=mu_estimator_dir,
                         on_step=on_step, on_reset=on_reset)
    executed = np.asarray(rows["executed"])
    # the estimator reads, with each observation, the action executed just before it (zeros at the reset)
    return {"record": record, "obs": np.asarray(rows["obs"]), "imu": np.asarray(rows["imu"]),
            "prev_actions": executed, "s": np.asarray(rows["s"])}


def stages(estimator: FrictionEstimator, inputs: dict[str, Any], mu: float) -> dict[str, NDArray[np.float64]]:
    """Offline estimate of one replayed episode: every stage as a ratio to the true mu, per step, and the steps at
    which the network measures (every ``update_steps``)."""
    out = estimator.estimate_episode(inputs["obs"], inputs["imu"], inputs["prev_actions"])
    steps = np.arange(0, len(inputs["obs"]), estimator.config.update_steps)
    return {"measured_ratio": out["measured"] / mu, "measured_steps": steps, "mean_ratio": out["mean"] / mu,
            "estimate_ratio": out["estimate"] / mu}


def stored_array(path: str | Path, key: str) -> NDArray:
    """Memory-mapped view of the array ``key`` of an uncompressed NPZ file (``np.savez``)."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{key} is compressed and cannot be memory-mapped")
    with path.open("rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        name_length, extra_length = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        f.seek(info.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def training_windows(estimator_dir: str | Path, circuit: str, every: int) -> NDArray[np.float32]:
    """Normalised input windows of the estimator's training datasets on one circuit, one every ``every`` steps."""
    estimator_dir = Path(estimator_dir)
    spec = json.loads((estimator_dir / ESTIMATOR_FILE).read_text())
    config = spec["config"]
    mean, std = np.asarray(spec["input_mean"]), np.asarray(spec["input_std"])
    windows = []
    for dataset in spec["report"]["datasets"]:
        with np.load(dataset) as data:
            circuits = [str(c) for c in data["circuit_per_lap"]]
            ends = np.asarray(data["episode_ends"], dtype=np.int64)
        obs, imu, executed = (stored_array(dataset, k) for k in ("observations", "imu", "actions_exec"))
        for lap, (start, end) in enumerate(zip(np.concatenate(([0], ends[:-1])), ends, strict=True)):
            if circuits[lap] != circuit:
                continue
            frames = channel_frames(np.asarray(obs[start:end]), np.asarray(imu[start:end]),
                                    previous_actions(np.asarray(executed[start:end])), FRICTION_CHANNELS)
            lap_windows = stack_window(frames, config["n_frames"], config["stride"])[::every]
            windows.append(((lap_windows - mean) / std).astype(np.float32))
    return np.concatenate(windows)


def nearest_distances(queries: NDArray[np.floating], reference: NDArray[np.floating],
                      chunk: int = 4096) -> NDArray[np.float64]:
    """Euclidean distance of every query to its nearest reference row (exact, in chunks of the reference)."""
    queries = np.asarray(queries, dtype=np.float64)
    q2 = np.sum(queries ** 2, axis=1)[:, None]
    best = np.full(len(queries), np.inf)
    for start in range(0, len(reference), chunk):
        block = np.asarray(reference[start:start + chunk], dtype=np.float64)
        d2 = q2 - 2.0 * queries @ block.T + np.sum(block ** 2, axis=1)[None, :]
        best = np.minimum(best, d2.min(axis=1))
    return np.sqrt(np.maximum(best, 0.0))


def measured_windows(estimator: FrictionEstimator, inputs: dict[str, Any]) -> NDArray[np.float64]:
    """Normalised input windows of the measuring steps of one replayed episode (as the network reads them)."""
    windows = estimator.windows(inputs["obs"], inputs["imu"], inputs["prev_actions"])[::estimator.config.update_steps]
    return (np.asarray(windows, dtype=np.float64) - estimator.input_mean) / estimator.input_std
