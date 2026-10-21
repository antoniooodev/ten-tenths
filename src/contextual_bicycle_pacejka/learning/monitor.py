"""Curves of a training run, one JSON line per rollout.

Every line holds the decisions so far, the decisions of the rollout, the mean normalised progress per decision, the
mean speed, the share of decisions above SLIP_MIN_SPEED_MPS that end with the slip of each axle beyond its peak
(|alpha| > alpha_sat; at a few m/s any steering puts the front beyond it without lateral acceleration), the
failures per 1000 decisions by type, the laps completed in the training episodes, the decisions per second of the
run, and the losses of the last update.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from contextual_bicycle_pacejka.model.termination import FAILURES

SLIP_MIN_SPEED_MPS: float = 5.0  # [m/s] the slip beyond the peak is counted above this speed


class Snapshots(BaseCallback):
    """Save a copy of the model the first time the decisions reach each of ``timesteps``, at ``path_of(timesteps)``."""

    def __init__(self, path_of: Any, timesteps: list[int]) -> None:
        super().__init__()
        self.path_of = path_of
        self.pending = sorted(timesteps)

    def _on_step(self) -> bool:
        while self.pending and self.num_timesteps >= self.pending[0]:
            path = Path(self.path_of(self.pending.pop(0)))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
        return True


class TrainingCurves(BaseCallback):
    """Write the curves of a training run to ``path`` (JSON lines), one line per rollout."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._start = 0.0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._start = time.perf_counter()
        self._laps = np.zeros(self.training_env.num_envs, dtype=np.int64)
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        self._decisions = 0
        self._progress = 0.0
        self._speed = 0.0
        self._beyond_peak = np.zeros(2)
        self._moving = 0
        self._failures: Counter[str] = Counter()
        self._laps_completed = 0

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"], strict=True)):
            self._decisions += 1
            self._progress += info["reward_terms"]["progress"]
            self._speed += info["state"][3]
            if info["state"][3] >= SLIP_MIN_SPEED_MPS:
                self._moving += 1
                self._beyond_peak += np.asarray(info.get("slip_ratio", (0.0, 0.0))) > 1.0
            if info["failure"] is not None:
                self._failures[info["failure"]] += 1
            self._laps_completed += max(info["laps"] - self._laps[i], 0)
            self._laps[i] = 0 if done else info["laps"]
        return True

    def _on_rollout_end(self) -> None:
        line: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "decisions": self._decisions,
            "progress_per_decision": self._progress / max(self._decisions, 1),
            "mean_speed": self._speed / max(self._decisions, 1),
            "slip_beyond_peak": dict(zip(("front", "rear"), (self._beyond_peak / max(self._moving, 1)).tolist(),
                                         strict=True)),
            "failures_per_1000": {f: 1000.0 * self._failures[f] / max(self._decisions, 1) for f in FAILURES},
            "laps": int(self._laps_completed),
            "decisions_per_second": self.num_timesteps / (time.perf_counter() - self._start),
            "losses": {k: float(v) for k, v in self.model.logger.name_to_value.items() if k.startswith("train/")},
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        self._reset_rollout()
