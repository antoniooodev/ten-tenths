"""Automatic speed curriculum: the top speed of the environments rises when the training failures become rare.

Training starts with top speed ``start``. After every rollout the failures and the decisions of the last ``window``
rollouts at the current top speed are pooled; when they make fewer than ``threshold`` failures per equivalent lap
(one lap length L driven at the top speed, L / (v_top DECISION_DT) decisions), the top speed rises by ``step``.
Once it would reach the sizing speed of the stage, the bound is removed. The rule is fixed and the same for every
seed; it only reads the training episodes, never a reference trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from contextual_bicycle_pacejka.spec.constants import DECISION_DT, FREE_STAGE_SIZING_MPS

CURRICULUM_START_MPS: float = 10.0  # [m/s] first top speed
CURRICULUM_STEP_MPS: float = 2.5  # [m/s] rise of the top speed
CURRICULUM_THRESHOLD: float = 0.2  # [failures per equivalent lap] below which the top speed rises
CURRICULUM_WINDOW: int = 5  # [rollouts] pooled at the current top speed


def failures_per_lap(failures: int, decisions: int, top_speed: float, track_length: float) -> float:
    """Return the failures per equivalent lap: per L / (v_top DECISION_DT) decisions."""
    return failures / max(decisions, 1) * track_length / (top_speed * DECISION_DT)


def next_top_speed(
    top_speed: float, step: float = CURRICULUM_STEP_MPS, sizing: float = FREE_STAGE_SIZING_MPS,
) -> float | None:
    """Return the top speed after a rise: ``top_speed + step``, or None (no bound) once it reaches ``sizing``."""
    raised = top_speed + step
    return None if raised >= sizing - 1e-9 else raised


class SpeedCurriculum(BaseCallback):
    """Raise the top speed of the training environments (``RacingEnv.set_top_speed``) by the rule of the module.

    Every change is appended to ``log`` (JSON lines: decisions so far, the new top speed, the failures per equivalent
    lap that triggered it).
    """

    def __init__(
        self, track_length: float, log: str | Path, start: float = CURRICULUM_START_MPS,
        step: float = CURRICULUM_STEP_MPS, threshold: float = CURRICULUM_THRESHOLD, window: int = CURRICULUM_WINDOW,
        sizing: float = FREE_STAGE_SIZING_MPS,
    ) -> None:
        super().__init__()
        self.track_length = track_length
        self.log = Path(log)
        self.top_speed: float | None = start
        self.step, self.threshold, self.window, self.sizing = step, threshold, window, sizing
        self._rollouts: list[tuple[int, int]] = []  # (failures, decisions) of the rollouts at the current top speed
        self._failures = self._decisions = 0

    def _on_training_start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("")
        self._set(self.top_speed, None)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._decisions += 1
            self._failures += info["failure"] is not None
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts.append((self._failures, self._decisions))
        self._failures = self._decisions = 0
        if self.top_speed is None or len(self._rollouts) < self.window:
            return
        recent = self._rollouts[-self.window:]
        rate = failures_per_lap(sum(f for f, _ in recent), sum(d for _, d in recent), self.top_speed,
                                self.track_length)
        if rate < self.threshold:
            self._set(next_top_speed(self.top_speed, self.step, self.sizing), rate)

    def _set(self, top_speed: float | None, rate: float | None) -> None:
        self.top_speed = top_speed
        self._rollouts = []
        self.training_env.env_method("set_top_speed", top_speed)
        with self.log.open("a") as handle:
            handle.write(json.dumps({"timesteps": int(self.num_timesteps), "top_speed": top_speed,
                                     "failures_per_lap": rate}) + "\n")
