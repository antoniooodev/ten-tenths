"""The learned policy: a Stable-Baselines3 actor-critic MLP plus a frozen observation normaliser.

A policy directory contains ``policy.zip`` (SB3 archive) and ``vecnormalize.pkl`` (``VecNormalize`` statistics
fitted once on the training observations). The controller is the deterministic actor mean:

    obs (116) -> VecNormalize -> MLP 116-512-512 (tanh) -> linear -> [delta_cmd, tau]

Behaviour cloning trains only the actor; the critic exists because the SB3 actor-critic class is reused
(PPO fine-tuning trains both).

A student policy reads the student observation of its level instead (``model/student.py``); its directory also
holds ``student.json`` with the level, the frame history, the oracle entries (diagnostic layouts only) and the
training target (``label`` or ``exec``). A
directory without it is a privileged policy trained on the labels (level S0). ``policy_env`` builds the
environment a policy expects.

A plan coach directory (``learning/plan_coach.py``) holds ``plan_coach.pt`` instead: an actor on the privileged
observation that chooses, every ``decision_steps`` steps, the scale of the v_ref preview read by a student policy,
its executor. ``PlanCoach`` loads it for inference. A sensor plan student (``learning/plan_student.py``) uses the
same file with ``"input": "sensors"``: its actor reads the student observation of the executor (preview at scale 1,
``plan_student_input``) instead of the privileged one, and ``"mu_estimator"`` names the friction estimator its
executor reads, which belongs to the student.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from contextual_bicycle_pacejka.env.factory import MAX_EPISODE_STEPS, make_env, make_student_env
from contextual_bicycle_pacejka.env.student import OnlineEstimator
from contextual_bicycle_pacejka.model.reward import RewardWeights
from contextual_bicycle_pacejka.model.student import student_previews
from contextual_bicycle_pacejka.spec.layout import STUDENT_PREVIEW_DIM, StudentLayout, student_layout

NET_ARCH: list[int] = [512, 512]
POLICY_FILE = "policy.zip"
NORMALIZER_FILE = "vecnormalize.pkl"
STUDENT_FILE = "student.json"
TARGETS = ("label", "exec")
REWARD_TARGET = "reward"
PLAN_COACH_FILE = "plan_coach.pt"
PLAN_SCALE_RANGE: tuple[float, float] = (0.90, 1.20)  # [-] range of a plan coach's scale of the v_ref preview
PLAN_SCALE_MID = 0.5 * (PLAN_SCALE_RANGE[0] + PLAN_SCALE_RANGE[1])
PLAN_SCALE_HALF = 0.5 * (PLAN_SCALE_RANGE[1] - PLAN_SCALE_RANGE[0])


def write_student_spec(model_dir: str | Path, layout: StudentLayout, target: str) -> None:
    """Write ``student.json``: the level, frame history and oracle entries of the observation and the target (the
    regression target, or ``reward`` for a policy trained on the reward alone, the coach)."""
    if target not in TARGETS + (REWARD_TARGET,):
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS + (REWARD_TARGET,)}")
    spec = {"level": layout.level, "n_frames": layout.n_frames, "stride": layout.stride, "target": target}
    if layout.oracle:
        spec["oracle"] = list(layout.oracle)
    (Path(model_dir) / STUDENT_FILE).write_text(json.dumps(spec, indent=2))


def policy_layout(model_dir: str | Path) -> StudentLayout:
    """Layout of the observation the policy in ``model_dir`` reads (S0 without ``student.json``)."""
    path = Path(model_dir) / STUDENT_FILE
    if not path.exists():
        return student_layout("S0")
    spec = json.loads(path.read_text())
    return student_layout(spec["level"], spec["n_frames"], spec["stride"], tuple(spec.get("oracle", ())))


def policy_env(
    raceline: Any, layout: StudentLayout, max_episode_steps: int = MAX_EPISODE_STEPS,
    reward_weights: RewardWeights | None = None, oracle_offset: dict[str, float] | None = None,
    mu_estimator: OnlineEstimator | None = None, oracle_clip: bool = False, plan_scale: float = 1.0,
) -> gym.Env:
    """Environment whose observation is the one of ``layout`` (``make_env`` for S0, so S0 runs are unchanged).

    ``oracle_offset`` adds a constant error to the oracle entries of a diagnostic layout, and ``oracle_clip`` clips
    them to the range an estimate can take; ``mu_estimator`` replaces the true mu of the oracle entry by an estimate
    (``learning/friction.py``); ``plan_scale`` multiplies the v_ref preview the student reads.
    """
    if layout.privileged:
        if oracle_offset or mu_estimator is not None or oracle_clip or plan_scale != 1.0:
            raise ValueError("the privileged observation (S0) has no oracle entries to offset, clip or estimate "
                             "and no student preview to scale")
        return make_env(raceline, max_episode_steps, reward_weights)
    return make_student_env(raceline, layout, max_episode_steps, reward_weights, oracle_offset, mu_estimator,
                            oracle_clip, plan_scale)

# Archives written before the code reorganisation reference classes that no longer exist under their old module
# paths; they are standard SB3 objects with no extra parameters and load into the standard classes.
LEGACY_OBJECTS: dict[str, Any] = {
    "policy_class": ActorCriticPolicy,
    "rollout_buffer_class": RolloutBuffer,
    "learning_rate": 0.0,
    "lr_schedule": lambda _progress: 0.0,
}


def new_bc_model(vec_env: VecNormalize, seed: int) -> PPO:
    """A randomly initialised actor-critic (seeded) used as the starting point of behaviour cloning."""
    return PPO(ActorCriticPolicy, env=vec_env, learning_rate=3.0e-4, n_steps=64, batch_size=64, n_epochs=1,
               policy_kwargs={"net_arch": list(NET_ARCH)}, seed=seed, device="cpu", verbose=0)


def load_model(path: str | Path, env: Any = None) -> PPO:
    """Load an SB3 archive (current or legacy format) on the CPU."""
    return PPO.load(str(path), env=env, device="cpu", custom_objects=dict(LEGACY_OBJECTS))


def load_normalizer(path: str | Path, env: gym.Env) -> VecNormalize:
    """Load frozen observation statistics around ``env`` (no further updates, no reward normalisation)."""
    vec = VecNormalize.load(str(path), DummyVecEnv([lambda: env]))
    vec.training = False
    vec.norm_reward = False
    return vec


def actor_mean(policy: ActorCriticPolicy, normalized_obs: NDArray[np.floating]) -> th.Tensor:
    """Actor mean for a batch of normalised observations (differentiable)."""
    obs_tensor = obs_as_tensor(normalized_obs, policy.device)
    features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


class Policy:
    """A trained policy ready for inference: ``policy(obs)`` returns the deterministic action mean.

    Parameters
    ----------
    model_dir : str or Path
        Directory with ``policy.zip`` and ``vecnormalize.pkl``.
    env : gym.Env
        Environment the policy is attached to (only its spaces are used).
    """

    def __init__(self, model_dir: str | Path, env: gym.Env) -> None:
        model_dir = Path(model_dir)
        self.normalizer = load_normalizer(model_dir / NORMALIZER_FILE, env)
        self.model = load_model(model_dir / POLICY_FILE, env=self.normalizer)
        self.model.policy.set_training_mode(False)

    def __call__(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the action mean [delta_cmd, tau] for one raw observation (not clipped)."""
        normalized = self.normalizer.normalize_obs(np.asarray(obs)[None, :])
        with th.no_grad():
            mean = actor_mean(self.model.policy, normalized)
        return mean.cpu().numpy()[0]

    def close(self) -> None:
        self.normalizer.close()


def plan_actor(n_inputs: int, hidden: tuple[int, ...]) -> th.nn.Sequential:
    """Actor of a plan coach: MLP (ReLU) -> tanh, the plan scale in [-1, 1] units (``plan_scale_of``). The last
    layer starts at zero with the bias of the scale 1, the plan of the teacher."""
    layers: list[th.nn.Module] = []
    width = n_inputs
    for size in hidden:
        layers += [th.nn.Linear(width, size), th.nn.ReLU()]
        width = size
    last = th.nn.Linear(width, 1)
    with th.no_grad():
        last.weight.zero_()
        last.bias.fill_(float(np.arctanh(plan_unit_of(1.0))))
    return th.nn.Sequential(*layers, last, th.nn.Tanh())


def plan_scale_of(unit: Any) -> Any:
    """Plan scale of a unit value in [-1, 1] (the actor output and the critic action)."""
    return PLAN_SCALE_MID + PLAN_SCALE_HALF * unit


def plan_unit_of(scale: Any) -> Any:
    """Unit value in [-1, 1] of a plan scale."""
    return (scale - PLAN_SCALE_MID) / PLAN_SCALE_HALF


def is_plan_coach(model_dir: str | Path | None) -> bool:
    """Whether ``model_dir`` holds a plan coach (``learning/plan_coach.py``) rather than a policy."""
    return model_dir is not None and (Path(model_dir) / PLAN_COACH_FILE).exists()


class PlanCoach:
    """A trained plan coach (or sensor plan student) ready for inference: ``coach(privileged_obs, scale)`` returns the
    plan scale of the next decision of a privileged coach, given the scale in use, and ``decide`` that of either kind.
    ``executor`` is the directory of the student policy that drives with it, ``decision_steps`` the steps between two
    decisions, ``sensors`` whether the actor reads the student observation and ``mu_estimator`` the friction estimator
    of a sensor plan student's executor (None for a coach)."""

    def __init__(self, model_dir: str | Path) -> None:
        blob = th.load(Path(model_dir) / PLAN_COACH_FILE, weights_only=False)
        self.executor = Path(blob["executor"])
        self.decision_steps = int(blob["decision_steps"])
        self.sensors = blob.get("input", "privileged") == "sensors"
        self.mu_estimator = Path(blob["mu_estimator"]) if blob.get("mu_estimator") else None
        self.layout = policy_layout(self.executor) if self.sensors else None
        self.mean = th.as_tensor(blob["obs_mean"])
        self.std = th.as_tensor(blob["obs_std"])
        self.actor = plan_actor(len(blob["obs_mean"]), tuple(blob["actor_hidden"]))
        self.actor.load_state_dict(blob["actor"])
        self.actor.eval()

    def __call__(self, privileged_obs: NDArray[np.floating], scale: float) -> float:
        """Plan scale of the next decision of a privileged coach."""
        if self.sensors:
            raise ValueError("a sensor plan student decides on the student observation (use decide)")
        return self._scale(plan_coach_input(privileged_obs, scale))

    def decide(self, privileged_obs: NDArray[np.floating], student_obs: NDArray[np.floating], mu: float,
               scale: float) -> float:
        """Plan scale of the next decision; a sensor plan student reads only ``student_obs`` (``mu`` and the
        privileged observation give the preview at the reference friction, which does not depend on mu)."""
        if not self.sensors:
            return self(privileged_obs, scale)
        return self._scale(plan_student_input(student_obs, privileged_obs, mu, scale, self.layout))

    def _scale(self, inputs: NDArray[np.float32]) -> float:
        row = th.as_tensor(inputs[None, :])
        with th.no_grad():
            unit = self.actor((row - self.mean) / self.std)
        return float(plan_scale_of(unit.item()))


def plan_coach_input(privileged_obs: NDArray[np.floating], scale: float) -> NDArray[np.float32]:
    """Raw input of a plan coach: the privileged observation and the plan scale in use (unit value)."""
    return np.append(np.asarray(privileged_obs, dtype=np.float32), np.float32(plan_unit_of(float(scale))))


def teacher_plan_observation(student_obs: NDArray[np.floating], privileged_obs: NDArray[np.floating], mu: float,
                             layout: StudentLayout) -> NDArray[np.float32]:
    """The student observation with its v_ref preview at scale 1 (the teacher's plan) instead of the scale in use.
    The preview is at the reference friction, so the true ``mu`` only undoes the scaling of the privileged v_ref."""
    x = np.array(student_obs, dtype=np.float32)
    x[layout.dynamic_dim:layout.dynamic_dim + STUDENT_PREVIEW_DIM] = student_previews(privileged_obs, mu, 1.0)[0]
    return x


def plan_student_input(student_obs: NDArray[np.floating], privileged_obs: NDArray[np.floating], mu: float,
                       scale: float, layout: StudentLayout) -> NDArray[np.float32]:
    """Raw input of a sensor plan student: ``teacher_plan_observation`` and the plan scale in use (unit value)."""
    return np.append(teacher_plan_observation(student_obs, privileged_obs, mu, layout),
                     np.float32(plan_unit_of(float(scale))))
