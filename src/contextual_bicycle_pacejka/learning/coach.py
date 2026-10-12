"""Coach: a privileged policy trained online by reinforcement learning to drive fast without the teacher's plan.

The student is a tracking controller of the plan it reads (the v_ref and reference-acceleration previews): it drives
at the teacher's pace and inherits the teacher's errors (docs/REDESIGN.md, section 23). The coach reads the
privileged observation without the plan (level C, ``spec/layout.py``) and learns its speed from the reward, in the
simulator. A coach that beats the teacher replaces it as the source of the student's data (section 23.5).

    decisions     the coach decides every ``action_repeat`` simulator steps (20 Hz) and holds its action: at 100 Hz
                  one action barely changes the return and the critic cannot tell actions apart (Tallec et al. 2019;
                  section 24.3), and 20 Hz is the lowest rate at which the teacher loses nothing (section 24.4). A
                  transition spans one decision: the reward is the discounted sum over its steps and the discount
                  of the bootstrap is gamma_step^k for its k steps, gamma_step = gamma^(1 / action_repeat); gamma is
                  per decision (0.99 at 20 Hz, about 5 s, the horizon of Fuchs et al. 2021)
    start         from scratch the coach fails (section 24.4): it starts instead from a policy of level C trained
                  offline (``init_policy``: behaviour cloning or TD3+BC), with a critic first trained at the decision
                  rate for ``pretrain_updates`` updates on the prior data while the actor stays fixed (the critic
                  learns the value of the starting policy, which the actor updates would otherwise change before
                  any interaction); then the training goes online (offline-to-online: AWAC, Nair et al. 2020; Beeson
                  and Montana 2022)
    algorithm     TD3 (Fujimoto et al. 2018) online, ``updates_per_decision`` updates after every decision; actor
                  loss - lambda Q_1(s, pi(o)) + w(t) BC(pi(o), y) on the prior half of the batch (TD3+BC,
                  lambda = alpha / mean |Q_1|), with the BC weight w falling linearly from 1 to 0 over the first
                  ``bc_decay_fraction`` of the training (the relaxed constraint of Beeson and Montana 2022): the
                  teacher's labels anchor the start, then only the reward counts. Against the unlearning at the
                  start of the fine-tuning (section 24.10; Cal-QL, Nakamoto et al. 2023; PROTO, Li et al. 2023): the
                  actor stays fixed for the first ``actor_warmup_steps`` online steps while the critic calibrates on
                  the coach's own transitions; lambda is fixed at alpha / mean |Q_1| of the starting policy at the
                  end of the pretraining (``fixed_lambda``; alpha / mean |Q_1| explodes when Q crosses zero); and the
                  actor pays ``trust_weight`` times the BC error to the Polyak-averaged actor (a trust region that
                  moves with the actor, as the iterative regularisation of PROTO)
    prior data    an offline (MDP) dataset (``learning/datasets.py``) in the replay, cut into windows of
                  ``action_repeat`` steps (one from every step, shorter at the end of a lap) whose action is the mean
                  of the executed ones (a pseudo-action, Hashimoto and Tsuruoka 2021): half of every batch comes from
                  it, half from the coach's transitions, and the critics have layer normalisation (RLPD, Ball et al.
                  2023)
    guide         jump start (JSRL, Uchendu et al. 2023): the teacher drives the first h decisions of an episode at
                  its own 100 Hz, h uniform in [0, H], H falling linearly from ``guide_steps`` / ``action_repeat``
                  to 0 over the first ``guide_fraction`` of the training; its windows enter the replay with the
                  mean of its actions, like the prior data
    episodes      start uniform on the lap at 50-100 % of the teacher's speed there (``closed_loop.lap_start``),
                  friction uniform in [MU_LO, MU_HI], setup ``randomized_setup`` from the coach's own seed stream
                  (never the setups of the protocols); truncated after ``episode_steps`` steps (bootstrapped)
    reward        the environment reward of the offline datasets: progress, time cost, failure penalty, no lateral
                  term (``OFFLINE_REWARD_WEIGHTS``), as the progress reward with a contact penalty of Fuchs et al.
                  2021; the traction cost is logged, not weighted (a saturation already ends the episode)
    critic        two Q networks on the full privileged observation (asymmetric, ``learning/offline_rl.py``)
    actor         the network of behaviour cloning on the coach observation, so that a coach is stored, loaded and
                  evaluated like a student; loss - Q_1(s, clip(pi(o))) + ``bound_weight`` * mean squared excess of
                  pi(o) beyond the action box (the clipped action carries no gradient there), and the BC term above
    exploration   on the friction the coach reads, not on its actions: at the limit of adhesion action noise of
                  0.005 already loses laps (section 24.8). The mu entry of the setup is multiplied by (1 + d_t),
                  d_t Ornstein-Uhlenbeck with standard deviation ``friction_noise`` and correlation time
                  ``friction_noise_time`` (as the perturbations of the demonstrations), clipped to +/- 2 sigma, the
                  result clipped to [MU_LO, MU_HI]: the coach drives for a few seconds faster or slower, which
                  explores its choice of speed. Optional Gaussian noise of ``exploration_noise`` on the actions (off
                  by default); target policy smoothing as in TD3
    validation    every ``validate_every`` steps, fixed laps that are not those of the protocols (``VALIDATION_MUS``
                  x ``VALIDATION_STARTS``; one lap or a failure, deterministic actor, no guide), against the teacher
                  on the same laps, with the coach deciding at 100 Hz (as it is evaluated) and at its training rate

A coach evaluated by ``evaluation/`` decides at every simulator step, like Fuchs et al. 2021 (trained at 10 Hz,
evaluated at 60 Hz). A coach directory holds the files of a student (``policy.zip``, ``vecnormalize.pkl``,
``student.json`` with level C and target ``reward``), ``critic.pt`` and ``coach.json`` (configuration, training curve
and validations). The policy files are rewritten at every validation, so an interrupted training keeps its last
validated state.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch as th
from numpy.typing import NDArray

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env.factory import make_env
from contextual_bicycle_pacejka.env.setup_sampling import randomized_setup
from contextual_bicycle_pacejka.evaluation.episode import progress_increment
from contextual_bicycle_pacejka.learning.behaviour_cloning import BCConfig, load_or_fit_normalizer
from contextual_bicycle_pacejka.learning.closed_loop import lap_start
from contextual_bicycle_pacejka.learning.offline_rl import Critic, bc_error, polyak_update
from contextual_bicycle_pacejka.learning.policy import (
    NORMALIZER_FILE,
    POLICY_FILE,
    REWARD_TARGET,
    actor_mean,
    load_model,
    new_bc_model,
    policy_layout,
    write_student_spec,
)
from contextual_bicycle_pacejka.learning.statistics import column_moments
from contextual_bicycle_pacejka.model.reward import OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.model.student import coach_observation
from contextual_bicycle_pacejka.spec.constants import DT, MU_HI, MU_LO
from contextual_bicycle_pacejka.spec.layout import (
    ACTION_HIGH,
    ACTION_LOW,
    COACH_LEVEL,
    OBS,
    SETUP,
    SETUP_RANGES,
    student_layout,
)

LOGGER = logging.getLogger(__name__)
CRITIC_FILE = "critic.pt"
REPORT_FILE = "coach.json"
VALIDATION_MUS = (0.8, 1.0, 1.2)
VALIDATION_STARTS = 2  # starts per friction
VALIDATION_SEED = 90_000  # seeds of the validation laps (setups and starts), disjoint from the protocols'


@dataclass
class CoachConfig:
    """Hyper-parameters of the coach (TD3 defaults of Fujimoto et al. 2018 unless stated)."""

    seed: int = 0
    env_steps: int = 10_000_000  # training budget in simulator steps, fixed in advance (docs/REDESIGN.md, 24.1)
    action_repeat: int = 5  # simulator steps per decision: 20 Hz (docs/REDESIGN.md, 24.4)
    gamma: float = 0.99  # per decision: about 5 s at 20 Hz, the horizon of Fuchs et al. 2021
    updates_per_decision: int = 1
    init_policy: str | None = None  # policy of level C trained offline to start from; None: from scratch
    pretrain_updates: int = 100_000  # offline updates at the decision rate before going online (with init_policy)
    alpha: float = 2.5  # TD3+BC weight of the Q term
    bc_decay_fraction: float = 0.3  # share of the training over which the BC weight falls from 1 to 0
    actor_warmup_steps: int = 1_000_000  # online steps with the actor fixed (with init_policy)
    fixed_lambda: bool = True  # lambda fixed after the pretraining instead of alpha / mean |Q_1| (with init_policy)
    trust_weight: float = 1.0  # weight of the BC error to the Polyak-averaged actor (with init_policy)
    learning_starts: int = 20_000  # simulator steps before the first update
    batch_size: int = 256  # half from the prior data, half from the coach's transitions
    buffer_size: int = 1_000_000  # transitions (decisions) kept, first in first out
    episode_steps: int = 3_000  # 30 s
    guide_steps: int = 0  # largest teacher prefix of an episode at the start of the training (steps; JSRL)
    guide_fraction: float = 0.3  # share of the training over which the prefix falls to zero
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    critic_hidden: tuple[int, ...] = (256, 256)
    polyak: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.0  # Gaussian noise on the scaled actions (off: section 24.8)
    friction_noise: float = 0.05  # std of the relative error on the friction the coach reads
    friction_noise_time: float = 2.0  # [s] correlation time of that error
    bound_weight: float = 1.0
    log_every: int = 50_000  # simulator steps
    validate_every: int = 1_000_000  # simulator steps
    validation_steps: int = 30_000  # step limit of a validation lap
    threads: int = 2

    @property
    def step_gamma(self) -> float:
        """Discount of one simulator step."""
        return float(self.gamma ** (1.0 / self.action_repeat))


class _Replay:
    """Transitions of one decision, first in first out: privileged observations, scaled (mean) actions, discounted
    rewards, terminals and bootstrap discounts."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, obs: NDArray, action: NDArray, reward: float, done: bool, discount: float,
            next_obs: NDArray) -> None:
        i = self.cursor
        self.obs[i], self.actions[i], self.next_obs[i] = obs, action, next_obs
        self.rewards[i], self.dones[i], self.discounts[i] = reward, done, discount
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, n: int) -> tuple[NDArray, ...]:
        index = rng.integers(0, self.size, size=n)
        return (self.obs[index], self.actions[index], self.rewards[index], self.dones[index],
                self.discounts[index], self.next_obs[index])


class PriorWindows:
    """Windows of ``repeat`` steps of an offline (MDP) dataset (``learning/datasets.py``), one from every step.

    A window from step t of a lap ending at e covers k = min(repeat, e - t) steps: its action is the mean of the
    executed actions, scaled to [-1, 1]; its reward is sum_j step_gamma^j r_(t+j); it is terminal if it reaches the
    end of a lap that ended by a failure; its bootstrap discount is step_gamma^k; its next state is row t + k, or the
    lap's final observation. Its label is the teacher's label at t (physical units), the target of the BC term.
    """

    def __init__(self, dataset: str | Path, repeat: int, step_gamma: float) -> None:
        with np.load(dataset, allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            self.rows = np.concatenate([observations, np.asarray(data["final_observations"], dtype=np.float32)])
            ends = np.asarray(data["episode_ends"], dtype=np.int64)
            executed = np.asarray(data["actions_exec"], dtype=np.float64) / ACTION_HIGH
            self.labels = np.asarray(data["actions"], dtype=np.float32)
            rewards = np.asarray(data["rewards"], dtype=np.float64)
            terminals = np.asarray(data["terminals"], dtype=bool)
        self.n = int(ends[-1])
        t = np.arange(self.n)
        lap = np.searchsorted(ends, t, side="right")
        end = ends[lap]
        k = np.minimum(repeat, end - t)
        reward, action = np.zeros(self.n), np.zeros((self.n, 2))
        for j in range(repeat):
            inside = j < k
            index = np.minimum(t + j, self.n - 1)
            reward += inside * step_gamma ** j * rewards[index]
            action += inside[:, None] * executed[index]
        self.actions = (action / k[:, None]).astype(np.float32)
        self.rewards = reward.astype(np.float32)
        self.dones = ((t + k == end) & terminals[end - 1]).astype(np.float32)
        self.discounts = (step_gamma ** k).astype(np.float32)
        self.next = np.where(t + k < end, t + k, self.n + lap)

    def sample(self, rng: np.random.Generator, n: int) -> tuple[tuple[NDArray, ...], NDArray]:
        """A batch (rows, actions, rewards, terminals, discounts, next rows) and its labels."""
        index = rng.integers(0, self.n, size=n)
        return (self.rows[index], self.actions[index], self.rewards[index], self.dones[index],
                self.discounts[index], self.rows[self.next[index]]), self.labels[index]


def perceived_friction(obs: NDArray, drift: float) -> NDArray:
    """A privileged observation whose normalised friction entry reads mu (1 + drift), clipped to [MU_LO, MU_HI]."""
    index = OBS.XI_START + SETUP.MU
    lo, hi = SETUP_RANGES[SETUP.MU]
    mu = lo + (float(obs[index]) + 1.0) / 2.0 * (hi - lo)
    perceived = np.array(obs, copy=True)
    perceived[index] = 2.0 * (min(max(mu * (1.0 + drift), MU_LO), MU_HI) - lo) / (hi - lo) - 1.0
    return perceived


def guide_horizon(step: int, config: CoachConfig) -> int:
    """Largest teacher prefix H of an episode started at simulator step ``step``, in decisions (linear to 0)."""
    end = config.guide_fraction * config.env_steps
    largest = config.guide_steps // config.action_repeat
    return int(round(largest * max(0.0, 1.0 - step / end))) if end > 0 else 0


def episode_options(racing_line: str | Path, rng: np.random.Generator) -> tuple[dict[str, Any], int]:
    """Reset options (setup, start) and reset seed of a training episode."""
    mu = float(rng.uniform(MU_LO, MU_HI))
    seed = int(rng.integers(1_000_000, 2_000_000))
    start_s, start_vx = lap_start(racing_line, mu, seed)
    return {"setup": randomized_setup(mu, np.random.default_rng(seed)), "initial_s": start_s,
            "initial_vx": start_vx}, seed


def validation_laps(racing_line: str | Path) -> list[dict[str, Any]]:
    """The fixed validation laps (friction, setup, start, reset seed)."""
    laps = []
    for i, mu in enumerate(VALIDATION_MUS):
        for j in range(VALIDATION_STARTS):
            seed = VALIDATION_SEED + VALIDATION_STARTS * i + j
            start_s, start_vx = lap_start(racing_line, mu, seed)
            laps.append({"mu": mu, "seed": seed, "start_s": start_s, "start_vx": start_vx,
                         "setup": randomized_setup(mu, np.random.default_rng(seed))})
    return laps


def drive_lap(racing_line: str | Path, lap: dict[str, Any], act: Callable[[NDArray], NDArray], max_steps: int,
              hold: int = 1) -> dict[str, Any]:
    """Drive one validation lap with ``act(privileged_obs) -> action``, decided every ``hold`` steps; return its
    outcome."""
    env = make_env(racing_line, max_episode_steps=max_steps, reward_weights=OFFLINE_REWARD_WEIGHTS)
    obs, _ = env.reset(seed=lap["seed"], options={"setup": lap["setup"], "initial_s": lap["start_s"],
                                                  "initial_vx": lap["start_vx"]})
    length = float(env.unwrapped.track.track_length)
    progress, previous_s, steps, terminated, truncated, info = 0.0, lap["start_s"], 0, False, False, {}
    action = np.zeros(2, dtype=np.float32)
    while progress < length and not (terminated or truncated):
        if steps % hold == 0:
            action = np.clip(act(obs), ACTION_LOW, ACTION_HIGH).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        current_s = float(info["state"][0])
        progress += progress_increment(previous_s, current_s, length)
        previous_s = current_s
    env.close()
    completed = progress >= length
    return {"mu": lap["mu"], "completed": bool(completed), "lap_time_s": steps * DT if completed else None,
            "progress_m": progress, "events": [k for k, v in info.get("event_flags", {}).items() if v]}


def _lap_summary(coach_laps: list[dict[str, Any]], teacher_laps: list[dict[str, Any]]) -> dict[str, Any]:
    both = [(c["lap_time_s"], t["lap_time_s"]) for c, t in zip(coach_laps, teacher_laps, strict=True)
            if c["completed"] and t["completed"]]
    return {"completed": sum(c["completed"] for c in coach_laps),
            "lap_time_ratio": sum(c for c, _ in both) / sum(t for _, t in both) if both else None,
            "laps": coach_laps}


def train_coach(dataset: str | Path, racing_line: str | Path, output: str | Path, config: CoachConfig | None = None,
                ) -> dict[str, Any]:
    """Train a coach on ``racing_line`` with the prior data ``dataset``; write it to ``output``."""
    config = config or CoachConfig()
    th.set_num_threads(int(config.threads))
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output = Path(output)
    layout = student_layout(COACH_LEVEL)
    repeat, step_gamma = config.action_repeat, config.step_gamma
    prior = PriorWindows(dataset, repeat, step_gamma)
    state_mean, state_var = column_moments(prior.rows[:prior.n])
    state_mean_t = th.as_tensor(state_mean.astype(np.float32))
    state_std_t = th.as_tensor(np.maximum(np.sqrt(state_var), 1.0e-6).astype(np.float32))
    if config.init_policy is not None:
        if not policy_layout(config.init_policy).coach:
            raise ValueError(f"init_policy {config.init_policy} is not a policy of level {COACH_LEVEL}")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(config.init_policy) / NORMALIZER_FILE, output / NORMALIZER_FILE)
    vec = load_or_fit_normalizer(output, racing_line, coach_observation(prior.rows[:prior.n]), layout)
    if config.init_policy is not None:
        model = load_model(Path(config.init_policy) / POLICY_FILE, env=vec)
    else:
        model = new_bc_model(vec, config.seed)
    policy = model.policy
    policy.log_std.data.fill_(-4.0)  # as every stored policy; the coach acts on the mean
    actor_target = copy.deepcopy(policy)
    actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    actor_opt = th.optim.Adam(actor_params, lr=config.actor_lr)
    critic = Critic(prior.rows.shape[1], config.critic_hidden, layer_norm=True)
    critic_target = copy.deepcopy(critic)
    critic_opt = th.optim.Adam(critic.parameters(), lr=config.critic_lr)
    high = th.as_tensor(ACTION_HIGH.astype(np.float32))
    bc_weights = th.as_tensor([float(BCConfig.steering_weight), 1.0])
    replay = _Replay(min(config.buffer_size, config.env_steps // repeat + 1), prior.rows.shape[1])
    teacher = MATv0()

    def actor_input(rows: NDArray) -> NDArray:
        return vec.normalize_obs(coach_observation(rows))

    def states(rows: NDArray) -> th.Tensor:
        return (th.as_tensor(rows) - state_mean_t) / state_std_t

    def act(obs: NDArray) -> NDArray:
        with th.no_grad():
            return actor_mean(policy, actor_input(obs[None, :])).numpy()[0]

    laps = validation_laps(racing_line)
    teacher_laps = [drive_lap(racing_line, lap, teacher.act, config.validation_steps) for lap in laps]
    env = make_env(racing_line, max_episode_steps=config.episode_steps, reward_weights=OFFLINE_REWARD_WEIGHTS)
    curve: list[dict[str, float]] = []
    validations: list[dict[str, Any]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    stats = {"steps": 0, "episodes": 0, "failures": 0, "coach_steps": 0, "coach_vx": 0.0, "traction_cost": 0.0}
    updates, wall = 0, time.perf_counter()

    sigma = config.friction_noise
    decay = float(np.exp(-repeat * DT / config.friction_noise_time))
    drift = 0.0

    def reset(step: int) -> tuple[NDArray, int]:
        nonlocal drift
        options, seed = episode_options(racing_line, rng)
        obs, _ = env.reset(seed=seed, options=options)
        drift = float(np.clip(rng.normal(0.0, sigma), -2.0 * sigma, 2.0 * sigma))  # the stationary distribution
        return obs, int(rng.integers(0, guide_horizon(step, config) + 1))

    def bc_weight(step: int) -> float:
        end = config.bc_decay_fraction * config.env_steps
        return max(0.0, 1.0 - step / end) if end > 0 else 0.0

    def update(weight: float, online: bool = True, train_actor: bool = True) -> dict[str, float]:
        """One TD3 update; ``weight`` is the BC weight, ``online`` draws half of the batch from the replay and
        ``train_actor`` False updates the critic only (pretraining)."""
        n_prior = config.batch_size // 2 if online else config.batch_size
        prior_batch, labels = prior.sample(rng, n_prior)
        batches = [prior_batch] + ([replay.sample(rng, config.batch_size - n_prior)] if online else [])
        rows, a, r, d, discount, rows_next = (np.concatenate(parts) for parts in zip(*batches, strict=True))
        s, s_next = states(rows), states(rows_next)
        a, r = th.as_tensor(a), th.as_tensor(r)
        bootstrap = th.as_tensor(discount) * (1.0 - th.as_tensor(d))
        with th.no_grad():
            noise = th.clamp(th.randn_like(a) * config.policy_noise, -config.noise_clip, config.noise_clip)
            a_next = th.clamp(actor_mean(actor_target, actor_input(rows_next)) / high + noise, -1.0, 1.0)
            backup = r + bootstrap * th.min(*critic_target(s_next, a_next))
        q1, q2 = critic(s, a)
        critic_loss = th.mean((q1 - backup) ** 2) + th.mean((q2 - backup) ** 2)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        logs = {"critic_loss": critic_loss.item(), "q": q1.mean().item()}
        if updates % config.policy_delay == 0 and not train_actor:
            polyak_update(critic_target, critic, config.polyak)
        elif updates % config.policy_delay == 0:
            mean = actor_mean(policy, actor_input(rows))
            scaled = mean / high
            q_pi = critic.q1(th.cat([s, th.clamp(scaled, -1.0, 1.0)], dim=1)).squeeze(1)
            lam = config.alpha / q_pi.abs().mean().detach() if fixed_lam is None else th.tensor(fixed_lam)
            bc = th.mean(bc_error(mean[:n_prior], th.as_tensor(labels), bc_weights))
            with th.no_grad():
                anchor = actor_mean(actor_target, actor_input(rows))
            trust = th.mean(bc_error(mean, anchor, bc_weights))
            bound = th.mean(th.relu(scaled.abs() - 1.0) ** 2)
            actor_loss = (-lam * q_pi.mean() + weight * bc + trust_weight * trust + config.bound_weight * bound)
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()
            polyak_update(critic_target, critic, config.polyak)
            polyak_update(actor_target, policy, config.polyak)
            logs.update({"actor_loss": actor_loss.item(), "bc_loss": bc.item(), "lambda": lam.item(),
                         "trust": trust.item(), "bound": bound.item()})
        return logs

    def save() -> None:
        output.mkdir(parents=True, exist_ok=True)
        model.save(str(output / POLICY_FILE))
        write_student_spec(output, layout, REWARD_TARGET)
        th.save({"critic": critic.state_dict(), "state_mean": state_mean, "state_std": state_std_t.numpy()},
                output / CRITIC_FILE)

    def validate(step: int) -> None:
        every_step = _lap_summary([drive_lap(racing_line, lap, act, config.validation_steps) for lap in laps],
                                  teacher_laps)
        held = _lap_summary([drive_lap(racing_line, lap, act, config.validation_steps, repeat) for lap in laps],
                            teacher_laps)
        entry = {"step": step, "laps": len(laps), "teacher_completed": sum(t["completed"] for t in teacher_laps),
                 "every_step": every_step, "held": held}
        validations.append(entry)
        LOGGER.info("coach validation at %d: %d/%d laps at 100 Hz, %d at %d Hz (teacher %d); lap time / teacher %s, %s",
                    step, every_step["completed"], len(laps), held["completed"], round(1.0 / (repeat * DT)),
                    entry["teacher_completed"], every_step["lap_time_ratio"], held["lap_time_ratio"])
        save()

    pretrain: dict[str, float] = {}
    fixed_lam: float | None = None
    trust_weight = config.trust_weight if config.init_policy is not None else 0.0
    if config.init_policy is not None:
        for _ in range(config.pretrain_updates):
            for key, val in update(1.0, online=False, train_actor=False).items():
                pretrain[key] = pretrain.get(key, 0.0) + val / config.pretrain_updates
            updates += 1
        if config.fixed_lambda:
            with th.no_grad():
                q_abs = []
                for _ in range(20):
                    (rows, *_), _ = prior.sample(rng, config.batch_size)
                    scaled = th.clamp(actor_mean(policy, actor_input(rows)) / high, -1.0, 1.0)
                    q_abs.append(critic.q1(th.cat([states(rows), scaled], dim=1)).abs().mean().item())
            fixed_lam = config.alpha / float(np.mean(q_abs))
            pretrain["fixed_lambda"] = fixed_lam
        LOGGER.info("coach pretraining (%d updates): %s", config.pretrain_updates,
                    ", ".join(f"{k} {v:.4g}" for k, v in pretrain.items()))
        validate(0)
    obs, guide = reset(0)
    decision, step, logged, validated = 0, 0, 0, 0
    while step < config.env_steps:
        start_obs, guided = obs, decision < guide
        if not guided:
            drift = float(np.clip(decay * drift + sigma * np.sqrt(1.0 - decay ** 2) * rng.normal(),
                                  -2.0 * sigma, 2.0 * sigma))
            mean = act(perceived_friction(obs, drift) if sigma > 0.0 else obs) / ACTION_HIGH
            held_action = np.clip(mean + rng.normal(0.0, config.exploration_noise, 2), -1.0, 1.0)
        reward, action_sum, k, terminated, truncated = 0.0, np.zeros(2), 0, False, False
        while k < repeat and not (terminated or truncated):
            scaled = np.clip(teacher.act(obs), ACTION_LOW, ACTION_HIGH) / ACTION_HIGH if guided else held_action
            obs, r, terminated, truncated, info = env.step((scaled * ACTION_HIGH).astype(np.float32))
            reward += step_gamma ** k * r
            action_sum += scaled
            k += 1
            if not guided:
                stats["coach_steps"] += 1
                stats["coach_vx"] += float(info["state"][3])
                stats["traction_cost"] += float(info["traction_cost"])
        step += k
        stats["steps"] += k
        replay.add(start_obs, (action_sum / k).astype(np.float32), reward, terminated, step_gamma ** k, obs)
        decision += 1
        if terminated or truncated:
            stats["episodes"] += 1
            stats["failures"] += int(terminated)
            obs, guide = reset(step)
            decision = 0
        if step >= config.learning_starts:
            for _ in range(config.updates_per_decision):
                warm = config.init_policy is not None and step < config.actor_warmup_steps
                for key, val in update(bc_weight(step) if config.init_policy is not None else 0.0,
                                       train_actor=not warm).items():
                    sums[key] = sums.get(key, 0.0) + val
                    counts[key] = counts.get(key, 0) + 1
                updates += 1
        if step // config.log_every > logged:
            logged = step // config.log_every
            coach_steps = max(stats["coach_steps"], 1)
            entry = {"step": step, "updates": updates, "episodes": stats["episodes"],
                     "failure_share": stats["failures"] / max(stats["episodes"], 1),
                     "coach_share": stats["coach_steps"] / max(stats["steps"], 1),
                     "coach_vx": stats["coach_vx"] / coach_steps,
                     "traction_cost": stats["traction_cost"] / coach_steps,
                     "guide_horizon": guide_horizon(step, config), "wall_s": time.perf_counter() - wall,
                     "bc_weight": bc_weight(step) if config.init_policy is not None else 0.0,
                     **{k: v / counts[k] for k, v in sums.items()}}
            curve.append(entry)
            LOGGER.info("coach step %d: %s", step, ", ".join(f"{k} {v:.4g}" for k, v in entry.items() if k != "step"))
            sums, counts = {}, {}
            stats = {"steps": 0, "episodes": 0, "failures": 0, "coach_steps": 0, "coach_vx": 0.0, "traction_cost": 0.0}
        if step // config.validate_every > validated or step >= config.env_steps:
            validated = step // config.validate_every
            validate(step)
    env.close()
    vec.close()
    report = {"config": asdict(config), "dataset": str(dataset), "racing_line": str(racing_line), "curve": curve,
              "pretrain": pretrain, "teacher_laps": teacher_laps, "validations": validations}
    (output / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report
