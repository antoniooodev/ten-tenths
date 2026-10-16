"""Construction of the environment and of its vectorised form for PPO."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.env.vec_env import BatchedSubprocVecEnv
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import B_NOM, EPISODE_DECISIONS


def make_env(
    reference: str | Path | RealTrack,
    stage: int = 1,
    b_range: tuple[float, float] = (B_NOM, B_NOM),
    oracle: bool = False,
    edge_margin: float = 0.0,
    max_episode_steps: int = EPISODE_DECISIONS,
) -> gym.Env:
    """Return the environment of speed stage ``stage`` on a circuit, wrapped in a time limit of decisions.

    ``reference`` is the centreline reference NPZ of the circuit or an already loaded :class:`RealTrack`.
    """
    track = reference if isinstance(reference, RealTrack) else RealTrack(str(reference))
    env = RacingEnv(track, make_stage(stage, track), b_range=b_range, oracle=oracle, edge_margin=edge_margin)
    return gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))


def make_vec_env(
    reference: str | Path,
    n_envs: int,
    seed: int,
    stage: int = 1,
    b_range: tuple[float, float] = (B_NOM, B_NOM),
    oracle: bool = False,
    edge_margin: float = 0.0,
    workers: int = 0,
) -> VecEnv:
    """Return ``n_envs`` copies of ``make_env``, copy i seeded with ``seed + i``.

    With ``workers`` > 0 the copies run in that many subprocesses, several per process (``BatchedSubprocVecEnv``);
    with 0 they run in this process (``DummyVecEnv``). A failure or a truncation resets that copy at once, so the
    collection of a rollout never stops.
    """
    def factory() -> Callable[[], gym.Env]:
        return lambda: make_env(str(reference), stage=stage, b_range=b_range, oracle=oracle, edge_margin=edge_margin)

    env_fns = [factory() for _ in range(n_envs)]
    vec_env = BatchedSubprocVecEnv(env_fns, workers) if workers > 0 else DummyVecEnv(env_fns)
    vec_env.seed(seed)
    return vec_env
