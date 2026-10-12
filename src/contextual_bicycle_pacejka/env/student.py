"""Environment seen by a student: its observation is the student observation of one level (``model/student.py``).

The wrapper keeps the frames of the last ``layout.window`` steps and builds each observation with the same
functions used on the datasets, so online and offline student observations are identical. The privileged
observation stays available to the teacher, the safety filter and the diagnostics:

    info["privileged_obs"]     privileged observation of the current step
    wrapper.privileged_obs     same, between steps

For the coach (level C) the observation is the privileged one without the plan (``model/student.py``).
The executed action of a step (the one passed to ``step``, clipped to the action box) becomes the previous action
of the next one. The safety filter works on ``env.unwrapped`` and is unaffected. ``oracle_offset`` adds a constant
error to the oracle entries of a diagnostic layout (sensitivity of a policy to an estimated mu); with
``oracle_clip`` the entries are then clipped to the range an estimate can take (``model/student.py``:
``ORACLE_RANGES``). ``mu_estimator`` replaces the true mu of the oracle entry by an estimate built from the same
logged channels (``learning/friction.py``); the entry of the current step is in ``info["oracle"]``.
``plan_scale`` multiplies the v_ref preview of the student observation (``model/student.py``); a plan coach changes
it during an episode with ``set_plan_scale``, which also returns the current observation with the new scale.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.student import (
    ACTION_DIM,
    coach_observation,
    stack_frames,
    student_frames,
    student_oracle,
    student_previews,
)
from contextual_bicycle_pacejka.spec.layout import StudentLayout


class OnlineEstimator(Protocol):
    """An estimator updated once per step from the logged channels (observation, IMU, previous action)."""

    def reset(self) -> None: ...

    def update(self, obs: NDArray[np.floating], imu: NDArray[np.floating], previous: NDArray[np.floating],
               ) -> float: ...


class StudentObservation(gym.Wrapper):
    """Replace the privileged observation by the student observation of ``layout``.

    Parameters
    ----------
    env : gym.Env
        Environment built by ``make_env``.
    layout : StudentLayout
        Level and history of the student observation.
    oracle_offset : dict, optional
        Constant error added to named oracle entries of ``layout``.
    mu_estimator : OnlineEstimator, optional
        Source of the mu oracle entry in place of the true friction.
    oracle_clip : bool
        Clip the oracle entries to the range an estimate can take, after the offset.
    plan_scale : float
        Scale of the v_ref preview the student reads (not for S0 and C, whose observations have no student preview).
    """

    def __init__(self, env: gym.Env, layout: StudentLayout, oracle_offset: dict[str, float] | None = None,
                 mu_estimator: OnlineEstimator | None = None, oracle_clip: bool = False, plan_scale: float = 1.0,
                 ) -> None:
        super().__init__(env)
        self.layout = layout
        self.oracle_offset = dict(oracle_offset or {})
        self.oracle_clip = bool(oracle_clip)
        self.mu_estimator = mu_estimator
        if plan_scale != 1.0 and (layout.privileged or layout.coach):
            raise ValueError(f"level {layout.level} reads no student preview to scale")
        self.plan_scale = float(plan_scale)
        student_oracle(1, 1.0, layout, self.oracle_offset)  # reject offsets of entries the layout does not have
        if mu_estimator is not None and "mu" not in layout.oracle:
            raise ValueError(f"a mu estimator needs the mu oracle entry; the layout's oracle is {layout.oracle}")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(layout.dim,), dtype=np.float32)
        self.privileged_obs: NDArray[np.float32] = np.zeros(env.observation_space.shape, dtype=np.float32)
        self._frames: list[NDArray[np.float32]] = []
        self._parts: tuple[NDArray[np.float32], NDArray[np.float32], float, NDArray[np.float32]] | None = None

    def set_plan_scale(self, plan_scale: float) -> NDArray[np.float32]:
        """Change the plan scale from the current step on; return the current observation with the new scale."""
        if self._parts is None:
            raise RuntimeError("set_plan_scale needs the observation of a reset or a step of a student level")
        self.plan_scale = float(plan_scale)
        stack, obs, mu, oracle = self._parts
        return np.concatenate([stack, student_previews(obs, mu, self.plan_scale)[0], oracle]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames = []
        if self.mu_estimator is not None:
            self.mu_estimator.reset()
        return self._observe(obs, info, np.zeros(ACTION_DIM, dtype=np.float32))

    def step(self, action: NDArray[np.floating]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        executed = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        student, info = self._observe(obs, info, executed)
        return student, reward, terminated, truncated, info

    def _observe(self, obs: NDArray[np.float32], info: dict[str, Any], previous: NDArray[np.float32],
                 ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Student observation of the step that returned ``obs`` and ``info``."""
        self.privileged_obs = np.asarray(obs, dtype=np.float32)
        info = {**info, "privileged_obs": self.privileged_obs}
        if self.layout.privileged:
            return self.privileged_obs.copy(), info
        if self.layout.coach:
            return coach_observation(self.privileged_obs), info
        frame = student_frames(obs, info["imu"], previous, self.layout)[0]
        self._frames = (self._frames + [frame])[-self.layout.window:]
        stack = stack_frames(np.asarray(self._frames), self.layout)[-1]
        mu = float(self.env.unwrapped.setup.mu)
        preview = student_previews(obs, mu, self.plan_scale)[0]
        oracle_mu = mu if self.mu_estimator is None else self.mu_estimator.update(obs, info["imu"], previous)
        oracle = student_oracle(1, oracle_mu, self.layout, self.oracle_offset, self.oracle_clip)[0]
        if self.layout.oracle:
            info["oracle"] = dict(zip(self.layout.oracle, map(float, oracle)))
        self._parts = (stack, np.asarray(obs, dtype=np.float32), mu, oracle)
        return np.concatenate([stack, preview, oracle]).astype(np.float32), info
