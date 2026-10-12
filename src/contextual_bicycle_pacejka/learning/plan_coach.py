"""Plan coach: a privileged policy that rewrites the speed plan of a student, learned online from the reward.

The student is a tracking controller of the plan it reads, and the plan, not the control, limits its lap time
(docs/REDESIGN.md, section 23.1). A coach that corrects the actions every 50 ms cannot learn: one decision changes the
return less than the error of its critic (section 24.12). The plan coach decides instead how fast to drive over the
next stretch of track, and a student policy, its executor, drives:

    decisions     every ``decision_steps`` simulator steps (1 s) the coach chooses a plan scale c in
                  ``PLAN_SCALE_RANGE``; the executor (a student of level S1/S2, ``executor``) reads its v_ref preview
                  multiplied by c (``env/student.py``: ``set_plan_scale``) and decides every step. A scale held for
                  1 s changes the return 20 times more than one 50 ms action (section 24.13), as the action gap
                  grows with the duration of a decision (Tallec et al. 2019). The coach chooses decision variables of
                  a fixed controller, as the policy search for MPC of Song and Scaramuzza (T-RO 2022)
    observation   the privileged observation of the decision step and the scale in use (unit value,
                  ``learning/policy.py``: ``plan_coach_input``), standardised with statistics frozen after the first
                  ``learning_starts`` decisions
    start         the actor starts at c = 1, the plan of the teacher (last layer zero, bias of the scale 1); nothing
                  is imitated: the scale is learned from the reward alone
    algorithm     TD3 (Fujimoto et al. 2018) on decisions, ``updates_per_decision`` updates after each, critics with
                  layer normalisation (Ball et al. 2023); the reward of a decision is the discounted sum over its
                  steps and the discount of its bootstrap step_gamma^k for its k steps, step_gamma =
                  gamma^(1 / decision_steps): gamma 0.99 per decision, a horizon of about 100 s, a lap. A failure ends
                  the episode and loses the rest of the lap: with the 5 s horizon of the step coach it cost too
                  little, and the coach learned a plan that leaves the track in almost every lap (section 24.14). The
                  target policy smoothing (``policy_noise``, clipped to
                  ``noise_clip``, in scale units) is small: the critic learns the value of the policy with that noise
                  (section 24.12), so it must be a noise the executor tolerates
    exploration   the first ``learning_starts`` decisions draw c uniformly in ``warmup_range``; then Gaussian noise
                  of ``exploration_noise`` (scale units) on the actor's scale, held for the decision. A scale up to
                  1.08 held for 2 s loses under 1 % of the stretches, 1.12 3-5 % (section 24.13)
    smoothness    the actor also pays the temporal and spatial smoothness terms of CAPS (Mysore et al. 2021):
                  ``smooth_time_weight`` |pi(s_t) - pi(s_t+1)| between consecutive decisions (not after a failure)
                  and ``smooth_space_weight`` |pi(s) - pi(s + n)|, n Gaussian of ``smooth_space_sigma`` on the
                  standardised input, in unit values. Without them the scale jumps between the bounds from one
                  decision to the next, and the laps fail where a high value falls on a braking (section 24.16).
                  The weight is the median |dQ/du| of the critic at the decisions of the coach without them
                  (13-16, section 24.17): a change is kept only where the critic's gain per unit exceeds it
    bound         the actor's output before the tanh pays ``bound_weight`` * mean squared excess beyond
                  +/- ``bound_preactivation``: a coach whose output ran to -7 had a tanh derivative of 1e-6 and stayed
                  at the lower bound for 9 M steps (section 24.23), as the bound term of the step coach prevents. At 2
                  the scale still reaches 0.905-1.195 and the derivative is 0.07; the weight is that of CAPS
    episodes      as the step coach (``coach.episode_options``): start uniform on the lap, friction uniform in
                  [MU_LO, MU_HI], setup from the coach's own seed stream; truncated after ``episode_steps`` steps
                  (bootstrapped). A share ``failure_start_share`` of the episodes starts instead ``failure_lead_m``
                  metres (uniform) before one of the last ``failure_memory`` failures of the training, at its
                  friction (``failure_start``): the critic misses the risk where the outcome changes irregularly with
                  c, near the upper friction bound, and with uniform starts fewer than 3 % of the episodes reach
                  those places (section 24.19). Starting from the states that matter is a curriculum from the
                  failures (reverse curriculum, Florensa et al. 2017; prioritized level replay, Jiang et al. 2021)
    reward        the environment reward of the offline datasets (``OFFLINE_REWARD_WEIGHTS``)
    validation    every ``validate_every`` steps, the fixed laps of the step coach (``coach.validation_laps``)
                  driven through ``evaluation.episode.run_episode`` exactly as a protocol episode, against the
                  teacher on the same laps

A plan coach directory holds ``plan_coach.pt`` (actor, observation statistics, executor, decision steps; loaded by
``learning.policy.PlanCoach`` and evaluated by ``evaluation/`` like a policy), ``critic.pt`` and ``plan_coach.json``
(configuration, training curve and validations). The coach files are rewritten at every validation, so an
interrupted training keeps its last validated state.
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

from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import run_episode
from contextual_bicycle_pacejka.learning.closed_loop import START_FRACTION_LO, start_speed
from contextual_bicycle_pacejka.learning.coach import CRITIC_FILE, episode_options, validation_laps
from contextual_bicycle_pacejka.learning.offline_rl import Critic, polyak_update
from contextual_bicycle_pacejka.learning.policy import (
    PLAN_COACH_FILE,
    PLAN_SCALE_HALF,
    PLAN_SCALE_RANGE,
    Policy,
    plan_actor,
    plan_coach_input,
    plan_scale_of,
    plan_unit_of,
    policy_env,
    policy_layout,
)
from contextual_bicycle_pacejka.learning.statistics import column_moments
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.spec.constants import OBS_DIM
from contextual_bicycle_pacejka.spec.layout import SETUP

LOGGER = logging.getLogger(__name__)
REPORT_FILE = "plan_coach.json"


@dataclass
class PlanCoachConfig:
    """Hyperparameters of a plan coach (see the module docstring)."""

    seed: int = 0
    env_steps: int = 10_000_000
    decision_steps: int = 100  # 1 s
    gamma: float = 0.99  # per decision: horizon about 100 s
    updates_per_decision: int = 4
    learning_starts: int = 2_000  # decisions
    warmup_range: tuple[float, float] = (0.92, 1.12)
    batch_size: int = 256
    buffer_size: int = 1_000_000
    episode_steps: int = 3_000
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    hidden: tuple[int, ...] = (256, 256)
    polyak: float = 0.005
    policy_noise: float = 0.01  # scale units
    noise_clip: float = 0.02
    policy_delay: int = 2
    exploration_noise: float = 0.02  # scale units
    smooth_time_weight: float = 15.0  # per unit value
    smooth_space_weight: float = 15.0
    smooth_space_sigma: float = 0.05  # standardised input units
    failure_start_share: float = 0.3
    failure_lead_m: tuple[float, float] = (400.0, 800.0)  # 10-20 s at about 40 m/s
    failure_memory: int = 100
    bound_preactivation: float = 2.0
    bound_weight: float = 15.0
    log_every: int = 100_000
    validate_every: int = 1_000_000
    validation_steps: int = 30_000
    threads: int = 2

    @property
    def step_gamma(self) -> float:
        """Discount per simulator step."""
        return self.gamma ** (1.0 / self.decision_steps)


class _Decisions:
    """Replay of the coach's decisions: input, action (unit value), reward, terminal, bootstrap discount, next input."""

    def __init__(self, capacity: int, n_inputs: int) -> None:
        self.inputs = np.zeros((capacity, n_inputs), dtype=np.float32)
        self.next_inputs = np.zeros((capacity, n_inputs), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, inputs: NDArray, unit: float, reward: float, done: bool, discount: float, next_inputs: NDArray,
            ) -> None:
        i = self.cursor
        self.inputs[i], self.actions[i, 0], self.rewards[i] = inputs, unit, reward
        self.dones[i], self.discounts[i], self.next_inputs[i] = float(done), discount, next_inputs
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, n: int) -> tuple[NDArray, ...]:
        idx = rng.integers(0, self.size, n)
        return (self.inputs[idx], self.actions[idx], self.rewards[idx], self.dones[idx], self.discounts[idx],
                self.next_inputs[idx])


def validation_specs(racing_line: str | Path, max_steps: int) -> list[dict[str, Any]]:
    """The validation laps of the step coach as protocol episodes (``run_episode``)."""
    return [{"block": "validation", "start_label": f"v{i}", "start_s": lap["start_s"], "start_vx": lap["start_vx"],
             "mu": lap["mu"], "seed": lap["seed"], "max_steps": max_steps, "stall_rule": False}
            for i, lap in enumerate(validation_laps(racing_line))]


def failure_start(racing_line: str | Path, failure: tuple[float, float], rng: np.random.Generator,
                  lead_m: tuple[float, float]) -> tuple[dict[str, Any], int]:
    """Reset options and seed of an episode that starts ``lead_m`` metres (uniform) before a failure (arc length,
    friction), at its friction, with a new setup and the start speed rule of ``closed_loop.lap_start``."""
    failure_s, mu = failure
    with np.load(racing_line, allow_pickle=False) as data:
        length = float(data["track_length"])
    start_s = float((failure_s - rng.uniform(*lead_m)) % length)
    seed = int(rng.integers(1_000_000, 2_000_000))
    fraction = float(rng.uniform(START_FRACTION_LO, 1.0))
    return {"setup": randomized_setup(mu, np.random.default_rng(seed)), "initial_s": start_s,
            "initial_vx": start_speed(racing_line, start_s, mu, fraction)}, seed


def lap_summary(records: list[dict[str, Any]], teacher: list[dict[str, Any]]) -> dict[str, Any]:
    """Completed laps, lap time over the teacher's on the laps both complete, and mean plan scale of a validation."""
    both = [(r["lap_time_s"], t["lap_time_s"]) for r, t in zip(records, teacher, strict=True)
            if r["completed"] and t["completed"]]
    return {"completed": sum(r["completed"] for r in records),
            "lap_time_ratio": sum(c for c, _ in both) / sum(t for _, t in both) if both else None,
            "plan_scale": float(np.mean([r["plan_scale"]["mean"] for r in records])),
            "laps": [{"mu": r["mu"], "completed": r["completed"], "lap_time_s": r["lap_time_s"],
                      "progress_m": r["progress_m"], "plan_scale": r["plan_scale"]} for r in records]}


def train_plan_coach(executor: str | Path, racing_line: str | Path, output: str | Path,
                     config: PlanCoachConfig | None = None) -> dict[str, Any]:
    """Train a plan coach for the student policy ``executor`` on ``racing_line``; write it to ``output``."""
    config = config or PlanCoachConfig()
    th.set_num_threads(int(config.threads))
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layout = policy_layout(executor)
    if layout.privileged or layout.coach:
        raise ValueError(f"the executor {executor} reads no student preview (level {layout.level})")
    env = policy_env(racing_line, layout, config.episode_steps, OFFLINE_REWARD_WEIGHTS)
    driver = Policy(executor, env)
    low, high = env.action_space.low, env.action_space.high

    n_inputs = OBS_DIM + 1
    actor = plan_actor(n_inputs, config.hidden)
    actor_target = copy.deepcopy(actor)
    actor_opt = th.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic = Critic(n_inputs, config.hidden, layer_norm=True, action_dim=1)
    critic_target = copy.deepcopy(critic)
    critic_opt = th.optim.Adam(critic.parameters(), lr=config.critic_lr)
    replay = _Decisions(min(config.buffer_size, config.env_steps // config.decision_steps + 1), n_inputs)
    mean_t, std_t = th.zeros(n_inputs), th.ones(n_inputs)  # frozen after the first learning_starts decisions
    policy_sigma, policy_clip = config.policy_noise / PLAN_SCALE_HALF, config.noise_clip / PLAN_SCALE_HALF

    def coach_scale(inputs: NDArray) -> float:
        with th.no_grad():
            return float(plan_scale_of(actor((th.as_tensor(inputs[None, :]) - mean_t) / std_t).item()))

    def update(updates: int) -> dict[str, float]:
        inputs, unit, reward, done, discount, next_inputs = replay.sample(rng, config.batch_size)
        s, s_next = (th.as_tensor(inputs) - mean_t) / std_t, (th.as_tensor(next_inputs) - mean_t) / std_t
        unit = th.as_tensor(unit)
        with th.no_grad():
            noise = th.clamp(th.randn_like(unit) * policy_sigma, -policy_clip, policy_clip)
            unit_next = th.clamp(actor_target(s_next) + noise, -1.0, 1.0)
            bootstrap = th.as_tensor(discount) * (1.0 - th.as_tensor(done))
            backup = th.as_tensor(reward) + bootstrap * th.min(*critic_target(s_next, unit_next))
        q1, q2 = critic(s, unit)
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
            actor_loss = (-critic.q1(th.cat([s, unit_pi], dim=1)).mean() + config.smooth_time_weight * smooth_time
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
                 "obs_std": std_t.numpy(), "executor": str(executor), "decision_steps": config.decision_steps},
                output / PLAN_COACH_FILE)
        th.save({"critic": critic.state_dict(), "obs_mean": mean_t.numpy(), "obs_std": std_t.numpy()},
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
        LOGGER.info("plan coach validation at %d: %d/%d laps (teacher %d); lap time / teacher %s, mean scale %.4f",
                    step, entry["completed"], len(specs), entry["teacher_completed"], entry["lap_time_ratio"],
                    entry["plan_scale"])

    failures: deque[tuple[float, float]] = deque(maxlen=config.failure_memory)
    length = float(env.unwrapped.track.track_length)
    episode_mu = 0.0

    def reset() -> tuple[NDArray, NDArray]:
        nonlocal episode_mu
        if failures and rng.random() < config.failure_start_share:
            failure = failures[int(rng.integers(len(failures)))]
            options, seed = failure_start(racing_line, failure, rng, config.failure_lead_m)
            stats["failure_starts"] += 1
        else:
            options, seed = episode_options(racing_line, rng)
        episode_mu = float(np.asarray(options["setup"])[SETUP.MU])
        student_obs, info = env.reset(seed=seed, options=options)
        return student_obs, info["privileged_obs"]

    validate(0)
    curve: list[dict[str, float]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    stats = {"episodes": 0, "failures": 0, "failure_starts": 0, "decisions": 0, "scale": 0.0, "steps": 0, "vx": 0.0}
    student_obs, privileged = reset()
    scale, step, decisions, updates, logged, validated = 1.0, 0, 0, 0, 0, 0
    wall = time.perf_counter()
    while step < config.env_steps:
        inputs = plan_coach_input(privileged, scale)
        if decisions < config.learning_starts:
            scale = float(rng.uniform(*config.warmup_range))
        else:
            scale = float(np.clip(coach_scale(inputs) + rng.normal(0.0, config.exploration_noise), *PLAN_SCALE_RANGE))
        student_obs = env.set_plan_scale(scale)
        reward, k, terminated, truncated, info = 0.0, 0, False, False, {}
        while k < config.decision_steps and not (terminated or truncated):
            action = np.clip(driver(student_obs), low, high).astype(np.float32)
            student_obs, r, terminated, truncated, info = env.step(action)
            reward += config.step_gamma ** k * r
            k += 1
            stats["vx"] += float(info["state"][3])
        privileged = info["privileged_obs"]
        replay.add(inputs, plan_unit_of(scale), reward, terminated, config.step_gamma ** k,
                   plan_coach_input(privileged, scale))
        step += k
        decisions += 1
        stats["decisions"] += 1
        stats["scale"] += scale
        stats["steps"] += k
        if terminated or truncated:
            stats["episodes"] += 1
            stats["failures"] += int(terminated)
            if terminated:
                failures.append((float(env.unwrapped.state.s) % length, episode_mu))
            student_obs, privileged = reset()
            scale = 1.0
        if decisions == config.learning_starts:
            mean, var = column_moments(replay.inputs[:replay.size])
            mean_t = th.as_tensor(mean.astype(np.float32))
            std_t = th.as_tensor(np.maximum(np.sqrt(var), 1.0e-6).astype(np.float32))
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
                     "plan_scale": stats["scale"] / max(stats["decisions"], 1),
                     "vx": stats["vx"] / max(stats["steps"], 1), "wall_s": time.perf_counter() - wall,
                     **{key: val / counts[key] for key, val in sums.items()}}
            curve.append(entry)
            LOGGER.info("plan coach step %d: %s", step,
                        ", ".join(f"{key} {val:.4g}" for key, val in entry.items() if key != "step"))
            sums, counts = {}, {}
            stats = {"episodes": 0, "failures": 0, "failure_starts": 0, "decisions": 0, "scale": 0.0, "steps": 0,
                     "vx": 0.0}
        if step // config.validate_every > validated or step >= config.env_steps:
            validated = step // config.validate_every
            validate(step)
    driver.close()
    env.close()
    report = {"config": asdict(config), "executor": str(executor), "racing_line": str(racing_line), "curve": curve,
              "teacher_laps": lap_summary(teacher_laps, teacher_laps)["laps"], "validations": validations}
    (output / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report
