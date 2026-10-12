"""Construction of the environment used by demonstrations, behaviour cloning, evaluation and PPO, and of the
environment seen by a student."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from contextual_bicycle_pacejka.env.environment import ContextualBicyclePacejkaEnv
from contextual_bicycle_pacejka.env.student import OnlineEstimator, StudentObservation
from contextual_bicycle_pacejka.model.reward import RewardWeights
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.layout import StudentLayout

MAX_EPISODE_STEPS: int = 100_000  # [steps] 1000 s, far longer than any lap


def make_env(
    raceline: str | Path | RealTrack,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    reward_weights: RewardWeights | None = None,
) -> gym.Env:
    """Return the environment on a racing line, wrapped in a time limit.

    ``raceline`` is a racing-line NPZ path or an already loaded :class:`RealTrack`.
    """
    track = raceline if isinstance(raceline, RealTrack) else RealTrack(str(raceline))
    env = ContextualBicyclePacejkaEnv(track=track, reward_weights=reward_weights)
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))


def make_student_env(
    raceline: str | Path | RealTrack,
    layout: StudentLayout,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    reward_weights: RewardWeights | None = None,
    oracle_offset: dict[str, float] | None = None,
    mu_estimator: OnlineEstimator | None = None,
    oracle_clip: bool = False,
    plan_scale: float = 1.0,
) -> gym.Env:
    """Return ``make_env(...)`` with the student observation of ``layout`` (see ``env/student.py``)."""
    return StudentObservation(make_env(raceline, max_episode_steps, reward_weights), layout, oracle_offset,
                              mu_estimator, oracle_clip, plan_scale)
