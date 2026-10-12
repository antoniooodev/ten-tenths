"""Sensor plan student: the plan coach's scale chosen from onboard sensors only, learned online from the reward.

The plan coach (``learning/plan_coach.py``) reads the privileged observation; the student is the same learner with
the inputs a car has (docs/REDESIGN.md, sections 23.6 and 26.1). Everything not listed here is the plan coach's:
decisions every ``decision_steps`` steps, TD3 with ``updates_per_decision`` updates, discount per decision, CAPS
smoothness, the bound before the tanh, the warm-up and the exploration noise, the episodes and the starts before
failures.

    actor         the student observation of the executor with its v_ref preview at scale 1, the teacher's plan
                  (S2 channels with their history, previews, mu_hat), and the scale in use
                  (``learning/policy.py``: ``plan_student_input``); the inputs from which a regression recovers most
                  of the coach's scale (section 25.7, measure 0c)
    critic        the privileged observation and the actor's input: a value of the state alone is a biased
                  estimate of the value of a history-dependent policy (Baisero and Amato 2022, theorems 4.2 and
                  5.1); training-time privileged information enters only the critic (Pinto et al. 2018; Scaffolder,
                  Hu et al. 2024)
    executor      the student policy ``executor`` reading the friction estimate of ``mu_estimator``, which belongs
                  to the student and is saved with it
    start         c = 1, the teacher's plan; nothing is imitated
    guide         at every reset, with probability ``guide_share``, the privileged plan coach ``guide`` chooses the
                  scale of the episode's decisions (no noise); its transitions enter the same replay with the
                  student's inputs. The coach is an exploration policy (Scaffolder), the replay is shared between the
                  two policies (TGRL, Shenfeld et al. 2023, section 4.4). Failures of guided episodes do not enter
                  the failure starts, which follow the student

A plan student directory holds ``plan_coach.pt`` with ``"input": "sensors"`` and ``"mu_estimator"`` (loaded by
``learning.policy.PlanCoach`` and evaluated by ``evaluation/`` like a plan coach, with its estimator), ``critic.pt``
and ``plan_student.json`` (configuration, training curve and validations).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from numpy.typing import NDArray

from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning.coach import CRITIC_FILE, episode_options
from contextual_bicycle_pacejka.learning.friction import FrictionEstimator
from contextual_bicycle_pacejka.learning.offline_rl import Critic, polyak_update
from contextual_bicycle_pacejka.learning.plan_coach import (
    PlanCoachConfig,
    failure_start,
    lap_summary,
    validation_specs,
)
from contextual_bicycle_pacejka.learning.policy import (
    PLAN_COACH_FILE,
    PLAN_SCALE_HALF,
    PLAN_SCALE_RANGE,
    PlanCoach,
    Policy,
    plan_actor,
    plan_scale_of,
    plan_student_input,
    plan_unit_of,
    policy_env,
    policy_layout,
)
from contextual_bicycle_pacejka.learning.statistics import column_moments
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.spec.constants import OBS_DIM
from contextual_bicycle_pacejka.spec.layout import SETUP

LOGGER = logging.getLogger(__name__)
REPORT_FILE = "plan_student.json"


@dataclass
class PlanStudentConfig(PlanCoachConfig):
    """Hyperparameters of a sensor plan student: those of the plan coach and the share of guided episodes."""

    guide_share: float = 0.3


class _StudentDecisions:
    """Replay of the student's decisions: actor input, privileged observation, action (unit value), reward, terminal,
    bootstrap discount, and the next actor input and privileged observation."""

    def __init__(self, capacity: int, n_inputs: int) -> None:
        self.inputs = np.zeros((capacity, n_inputs), dtype=np.float32)
        self.next_inputs = np.zeros((capacity, n_inputs), dtype=np.float32)
        self.privileged = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.next_privileged = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, inputs: NDArray, privileged: NDArray, unit: float, reward: float, done: bool, discount: float,
            next_inputs: NDArray, next_privileged: NDArray) -> None:
        i = self.cursor
        self.inputs[i], self.privileged[i], self.actions[i, 0], self.rewards[i] = inputs, privileged, unit, reward
        self.dones[i], self.discounts[i] = float(done), discount
        self.next_inputs[i], self.next_privileged[i] = next_inputs, next_privileged
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, n: int) -> tuple[NDArray, ...]:
        idx = rng.integers(0, self.size, n)
        return (self.inputs[idx], self.privileged[idx], self.actions[idx], self.rewards[idx], self.dones[idx],
                self.discounts[idx], self.next_inputs[idx], self.next_privileged[idx])


def train_plan_student(executor: str | Path, mu_estimator: str | Path, guide: str | Path | None,
                       racing_line: str | Path, output: str | Path,
                       config: PlanStudentConfig | None = None) -> dict[str, Any]:
    """Train a sensor plan student for the student policy ``executor`` reading the estimate of ``mu_estimator``, with
    the privileged plan coach ``guide`` as exploration policy (None: no guided episodes); write it to ``output``."""
    config = config or PlanStudentConfig()
    th.set_num_threads(int(config.threads))
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layout = policy_layout(executor)
    if layout.privileged or layout.coach or "mu" not in layout.oracle:
        raise ValueError(f"the executor {executor} must read the student preview and the mu entry")
    coach = PlanCoach(guide) if guide is not None else None
    if coach is not None and coach.sensors:
        raise ValueError(f"the guide {guide} must be a privileged plan coach")
    env = policy_env(racing_line, layout, config.episode_steps, OFFLINE_REWARD_WEIGHTS,
                     mu_estimator=FrictionEstimator(mu_estimator))
    driver = Policy(executor, env)
    low, high = env.action_space.low, env.action_space.high

    n_inputs = layout.dim + 1
    n_critic = OBS_DIM + n_inputs
    actor = plan_actor(n_inputs, config.hidden)
    actor_target = copy.deepcopy(actor)
    actor_opt = th.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic = Critic(n_critic, config.hidden, layer_norm=True, action_dim=1)
    critic_target = copy.deepcopy(critic)
    critic_opt = th.optim.Adam(critic.parameters(), lr=config.critic_lr)
    replay = _StudentDecisions(min(config.buffer_size, config.env_steps // config.decision_steps + 1), n_inputs)
    mean_t, std_t = th.zeros(n_inputs), th.ones(n_inputs)  # actor input, frozen after learning_starts decisions
    cmean_t, cstd_t = th.zeros(n_critic), th.ones(n_critic)  # critic input, frozen at the same time
    policy_sigma, policy_clip = config.policy_noise / PLAN_SCALE_HALF, config.noise_clip / PLAN_SCALE_HALF

    def student_scale(inputs: NDArray) -> float:
        with th.no_grad():
            return float(plan_scale_of(actor((th.as_tensor(inputs[None, :]) - mean_t) / std_t).item()))

    def update(updates: int) -> dict[str, float]:
        inputs, privileged, unit, reward, done, discount, next_inputs, next_privileged = replay.sample(
            rng, config.batch_size)
        s, s_next = (th.as_tensor(inputs) - mean_t) / std_t, (th.as_tensor(next_inputs) - mean_t) / std_t
        c = (th.as_tensor(np.concatenate([privileged, inputs], axis=1)) - cmean_t) / cstd_t
        c_next = (th.as_tensor(np.concatenate([next_privileged, next_inputs], axis=1)) - cmean_t) / cstd_t
        unit = th.as_tensor(unit)
        with th.no_grad():
            noise = th.clamp(th.randn_like(unit) * policy_sigma, -policy_clip, policy_clip)
            unit_next = th.clamp(actor_target(s_next) + noise, -1.0, 1.0)
            bootstrap = th.as_tensor(discount) * (1.0 - th.as_tensor(done))
            backup = th.as_tensor(reward) + bootstrap * th.min(*critic_target(c_next, unit_next))
        q1, q2 = critic(c, unit)
        critic_loss = th.mean((q1 - backup) ** 2) + th.mean((q2 - backup) ** 2)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        logs = {"critic_loss": critic_loss.item(), "q": q1.mean().item()}
        if updates % config.policy_delay == 0:
            preactivation = actor[:-1](s)
            unit_pi = th.tanh(preactivation)
            bound = th.mean(th.relu(preactivation.abs() - config.bound_preactivation) ** 2)
            smooth_time = th.mean(th.abs(unit_pi - actor(s_next)).squeeze(1) * (1.0 - th.as_tensor(done)))
            smooth_space = th.mean(th.abs(unit_pi - actor(s + config.smooth_space_sigma * th.randn_like(s))))
            actor_loss = (-critic.q1(th.cat([c, unit_pi], dim=1)).mean() + config.smooth_time_weight * smooth_time
                          + config.smooth_space_weight * smooth_space + config.bound_weight * bound)
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()
            polyak_update(critic_target, critic, config.polyak)
            polyak_update(actor_target, actor, config.polyak)
            logs.update({"actor_loss": actor_loss.item(), "smooth_time": smooth_time.item(),
                         "smooth_space": smooth_space.item(), "bound": bound.item()})
        return logs

    def save() -> None:
        th.save({"actor": actor.state_dict(), "actor_hidden": list(config.hidden), "obs_mean": mean_t.numpy(),
                 "obs_std": std_t.numpy(), "executor": str(executor), "decision_steps": config.decision_steps,
                 "input": "sensors", "mu_estimator": str(mu_estimator), "guide": None if guide is None else str(guide)},
                output / PLAN_COACH_FILE)
        th.save({"critic": critic.state_dict(), "obs_mean": cmean_t.numpy(), "obs_std": cstd_t.numpy()},
                output / CRITIC_FILE)

    specs = validation_specs(racing_line, config.validation_steps)
    teacher_laps = [run_episode(spec, racing_line, "teacher", False) for spec in specs]
    validations: list[dict[str, Any]] = []

    def validate(step: int) -> None:
        save()
        entry = {"step": step, "teacher_completed": sum(t["completed"] for t in teacher_laps),
                 **lap_summary([run_episode(spec, racing_line, "policy", False, model_dir=output) for spec in specs],
                               teacher_laps)}
        validations.append(entry)
        LOGGER.info("plan student validation at %d: %d/%d laps (teacher %d); lap time / teacher %s, mean scale %.4f",
                    step, entry["completed"], len(specs), entry["teacher_completed"], entry["lap_time_ratio"],
                    entry["plan_scale"])

    failures: deque[tuple[float, float]] = deque(maxlen=config.failure_memory)
    length = float(env.unwrapped.track.track_length)
    episode_mu, guided = 0.0, False

    def reset() -> tuple[NDArray, NDArray]:
        nonlocal episode_mu, guided
        if failures and rng.random() < config.failure_start_share:
            failure = failures[int(rng.integers(len(failures)))]
            options, seed = failure_start(racing_line, failure, rng, config.failure_lead_m)
            stats["failure_starts"] += 1
        else:
            options, seed = episode_options(racing_line, rng)
        guided = coach is not None and rng.random() < config.guide_share
        stats["guided"] += int(guided)
        episode_mu = float(np.asarray(options["setup"])[SETUP.MU])
        student_obs, info = env.reset(seed=seed, options=options)
        return student_obs, info["privileged_obs"]

    validate(0)
    curve: list[dict[str, float]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    stats = {"episodes": 0, "failures": 0, "failure_starts": 0, "guided": 0, "decisions": 0, "scale": 0.0,
             "steps": 0, "vx": 0.0}
    student_obs, privileged = reset()
    scale, step, decisions, updates, logged, validated = 1.0, 0, 0, 0, 0, 0
    wall = time.perf_counter()
    while step < config.env_steps:
        inputs = plan_student_input(student_obs, privileged, episode_mu, scale, layout)
        decision_privileged = privileged
        if decisions < config.learning_starts:
            scale = float(rng.uniform(*config.warmup_range))
        elif guided:
            scale = coach(privileged, scale)
        else:
            scale = float(np.clip(student_scale(inputs) + rng.normal(0.0, config.exploration_noise),
                                  *PLAN_SCALE_RANGE))
        student_obs = env.set_plan_scale(scale)
        reward, k, terminated, truncated, info = 0.0, 0, False, False, {}
        while k < config.decision_steps and not (terminated or truncated):
            action = np.clip(driver(student_obs), low, high).astype(np.float32)
            student_obs, r, terminated, truncated, info = env.step(action)
            reward += config.step_gamma ** k * r
            k += 1
            stats["vx"] += float(info["state"][3])
        privileged = info["privileged_obs"]
        replay.add(inputs, decision_privileged, plan_unit_of(scale), reward, terminated, config.step_gamma ** k,
                   plan_student_input(student_obs, privileged, episode_mu, scale, layout), privileged)
        step += k
        decisions += 1
        stats["decisions"] += 1
        stats["scale"] += scale
        stats["steps"] += k
        if terminated or truncated:
            stats["episodes"] += 1
            stats["failures"] += int(terminated)
            if terminated and not guided:
                failures.append((float(env.unwrapped.state.s) % length, episode_mu))
            student_obs, privileged = reset()
            scale = 1.0
        if decisions == config.learning_starts:
            mean, var = column_moments(replay.inputs[:replay.size])
            mean_t = th.as_tensor(mean.astype(np.float32))
            std_t = th.as_tensor(np.maximum(np.sqrt(var), 1.0e-6).astype(np.float32))
            cmean, cvar = column_moments(np.concatenate([replay.privileged[:replay.size],
                                                         replay.inputs[:replay.size]], axis=1))
            cmean_t = th.as_tensor(cmean.astype(np.float32))
            cstd_t = th.as_tensor(np.maximum(np.sqrt(cvar), 1.0e-6).astype(np.float32))
        if decisions >= config.learning_starts:
            for _ in range(config.updates_per_decision):
                for key, val in update(updates).items():
                    sums[key] = sums.get(key, 0.0) + val
                    counts[key] = counts.get(key, 0) + 1
                updates += 1
        if step // config.log_every > logged:
            logged = step // config.log_every
            entry = {"step": step, "updates": updates, "episodes": stats["episodes"],
                     "failure_share": stats["failures"] / max(stats["episodes"], 1),
                     "failure_start_share": stats["failure_starts"] / max(stats["episodes"], 1),
                     "guided_share": stats["guided"] / max(stats["episodes"], 1),
                     "plan_scale": stats["scale"] / max(stats["decisions"], 1),
                     "vx": stats["vx"] / max(stats["steps"], 1), "wall_s": time.perf_counter() - wall,
                     **{key: val / counts[key] for key, val in sums.items()}}
            curve.append(entry)
            LOGGER.info("plan student step %d: %s", step,
                        ", ".join(f"{key} {val:.4g}" for key, val in entry.items() if key != "step"))
            sums, counts = {}, {}
            stats = {"episodes": 0, "failures": 0, "failure_starts": 0, "guided": 0, "decisions": 0, "scale": 0.0,
                     "steps": 0, "vx": 0.0}
        if step // config.validate_every > validated or step >= config.env_steps:
            validated = step // config.validate_every
            validate(step)
    driver.close()
    env.close()
    report = {"config": asdict(config), "executor": str(executor), "mu_estimator": str(mu_estimator),
              "guide": None if guide is None else str(guide), "racing_line": str(racing_line), "curve": curve,
              "teacher_laps": lap_summary(teacher_laps, teacher_laps)["laps"], "validations": validations}
    (output / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report
