"""Offline reinforcement learning with an asymmetric critic: IQL and TD3+BC on an offline (MDP) dataset.

The critic reads the privileged observation; the actor reads the student observation of a level (docs/REDESIGN.md,
section 7). The actor is the network of behaviour cloning (``learning/policy.py``: SB3 actor MLP 512 x 2 on the
student observation, with its frozen normaliser), so a policy trained here is evaluated, and given a friction
estimate, exactly like a behaviour-cloned student.

    data        transitions (o_t, s_t, a_t, r_t, s_{t+1}, o_{t+1}, d_t) of ``learning/datasets.py``: o is the student
                observation, s the privileged one, a the executed action (``actions_exec``), r the offline reward
                (OFFLINE_REWARD_WEIGHTS), d = 1 only at a failure (a lap that ends by completion or step limit is
                bootstrapped); the teacher label y_t is used by TD3+BC only
    critic      two Q networks Q_i(s, a) (MLP 256 x 2, ReLU) on the normalised privileged observation and the action
                scaled to [-1, 1], with Polyak-averaged targets
    BC loss     the loss of behaviour cloning, sum_j lambda_j (pi_j(o) - target_j)^2, lambda = [20, 1]

With ``failures=False`` (an ablation) the failures carry no signal: a failed lap is bootstrapped like a truncated
one and the crash penalty is removed from the reward of its last transition (``strip_failures``), so the transitions
are those of the dataset but the critic cannot tell a failure from a lap that was cut short.

IQL (Kostrikov et al. 2022; executed actions only, no labels):

    V           expectile regression of min_i Q_target,i(s, a) on V(s), expectile tau
    Q           Q_i(s, a) -> r + gamma (1 - d) V(s')
    actor       advantage-weighted regression on the executed actions:
                weight = min(exp(beta A / sigma_A), max_weight), A = min_i Q_target,i(s, a) - V(s)
                sigma_A is a running standard deviation of A: the rewards are metres per step, the raw advantages
                of order 1e-2, so the temperature beta is set on the normalised advantage

TD3+BC (Fujimoto and Gu 2021), with the BC term on the teacher labels:

    Q           Q_i(s, a) -> r + gamma (1 - d) min_i Q_target,i(s', clip(pi_target(o') + noise))
    actor       every ``policy_delay`` steps: - lambda Q_1(s, pi(o)) + BC(pi(o), y),  lambda = alpha / mean |Q_1|

TD3+BC with the traction cost as a constraint (``cost_limit`` = rho, docs/REDESIGN.md, sections 5 and 19): a second
pair of networks estimates the discounted traction cost of ``learning/datasets.py`` (``traction_costs``), with the
terminals of the reward critic and the pessimistic (largest) target,

    Q_c         Q_c,i(s, a) -> c + gamma (1 - d) max_i Q_c,target,i(s', clip(pi_target(o') + noise))
    constraint  mean Q_c,1(s, pi(o)) <= rho mean Q_c,1(s, y) on each batch: the policy spends at most a fraction rho
                of the cost of the teacher labels on the same states
    actor       - lambda Q_1(s, pi(o)) + nu lambda_c Q_c,1(s, pi(o)) + BC(pi(o), y),  lambda_c = alpha / C,
                C a running mean of mean Q_c,1(s, y), so that nu = 1 weighs a cost like the reward
    multiplier  nu <- clip(nu + cost_lr (mean Q_c,1(s, pi) - rho mean Q_c,1(s, y)) / C, 0, cost_multiplier_max)

A policy directory holds ``policy.zip``, ``vecnormalize.pkl`` and ``student.json`` (as behaviour cloning; target
``exec`` for IQL, ``label`` for TD3+BC), ``critic.pt`` (critic, value and cost networks) and ``offline_rl.json``
(configuration, training curve and the diagnostics of ``critic_diagnostics`` and ``cost_diagnostics``).
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from numpy.typing import NDArray

from contextual_bicycle_pacejka.learning.behaviour_cloning import (
    load_or_fit_normalizer,
    normalise_in_place,
    stratified_split,
)
from contextual_bicycle_pacejka.learning.datasets import next_index, student_dataset, student_dataset_observations
from contextual_bicycle_pacejka.learning.policy import (
    POLICY_FILE,
    actor_mean,
    new_bc_model,
    write_student_spec,
)
from contextual_bicycle_pacejka.learning.statistics import column_moments
from contextual_bicycle_pacejka.model.reward import CRASH_BASE, OFFLINE_REWARD_WEIGHTS
from contextual_bicycle_pacejka.spec.constants import STUDENT_HISTORY_FRAMES, STUDENT_HISTORY_STRIDE
from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, OBS, student_layout

LOGGER = logging.getLogger(__name__)
ALGORITHMS = ("iql", "td3bc")
CRITIC_FILE = "critic.pt"
REPORT_FILE = "offline_rl.json"


@dataclass
class OfflineRLConfig:
    """Hyper-parameters of offline reinforcement learning."""

    algorithm: str = "iql"  # "iql" (executed actions only) or "td3bc" (BC term on the teacher labels)
    seed: int = 0
    level: str = "S2"
    n_frames: int = STUDENT_HISTORY_FRAMES
    stride: int = STUDENT_HISTORY_STRIDE
    oracle: tuple[str, ...] = ("mu",)
    steps: int = 100_000
    batch_size: int = 1024
    gamma: float = 0.995  # horizon of about 2 s at 100 Hz
    critic_hidden: tuple[int, ...] = (256, 256)
    critic_lr: float = 3.0e-4
    actor_lr: float = 3.0e-4
    polyak: float = 0.005
    steering_weight: float = 20.0  # as behaviour cloning
    expectile: float = 0.7  # IQL
    beta: float = 3.0  # IQL, on the normalised advantage
    max_weight: float = 100.0  # IQL
    alpha: float = 2.5  # TD3+BC
    policy_noise: float = 0.2  # TD3+BC, in units of the scaled action
    noise_clip: float = 0.5  # TD3+BC
    policy_delay: int = 2  # TD3+BC
    failures: bool = True  # False: failures bootstrapped as truncations, without crash penalty (ablation)
    cost_limit: float | None = None  # TD3+BC: traction-cost budget rho, a fraction of the labels' cost; None: no cost
    cost_lr: float = 1.0e-3  # step of the multiplier nu on the relative constraint violation
    cost_multiplier_max: float = 10.0
    log_every: int = 5000
    threads: int = 3


def _mlp(n_inputs: int, hidden: tuple[int, ...], n_outputs: int, layer_norm: bool = False) -> th.nn.Sequential:
    layers: list[th.nn.Module] = []
    width = n_inputs
    for size in hidden:
        layers += [th.nn.Linear(width, size), *([th.nn.LayerNorm(size)] if layer_norm else []), th.nn.ReLU()]
        width = size
    layers.append(th.nn.Linear(width, n_outputs))
    return th.nn.Sequential(*layers)


class Critic(th.nn.Module):
    """Two Q networks on (normalised privileged observation, action scaled to [-1, 1]); with ``layer_norm`` every
    hidden layer is normalised (the critics of online RL, ``learning/coach.py`` and ``learning/plan_coach.py``)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], layer_norm: bool = False, action_dim: int = 2) -> None:
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, hidden, 1, layer_norm)
        self.q2 = _mlp(obs_dim + action_dim, hidden, 1, layer_norm)

    def forward(self, s: th.Tensor, a: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        x = th.cat([s, a], dim=1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


def polyak_update(target: th.nn.Module, source: th.nn.Module, rate: float) -> None:
    """Move the parameters of ``target`` towards those of ``source`` by ``rate``."""
    with th.no_grad():
        for t, p in zip(target.parameters(), source.parameters(), strict=True):
            t.mul_(1.0 - rate).add_(rate * p)


def strip_failures(rewards: NDArray, terminals: NDArray, episode_ends: NDArray[np.integer],
                   final_observations: NDArray) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Rewards and terminals of a dataset with the failures turned into truncations.

    The last transition of a failed lap loses its terminal flag and gets back the crash penalty of
    ``OFFLINE_REWARD_WEIGHTS``, CRASH_BASE + w_crash (vx^2 + vy^2), with the speeds of the lap's final observation.
    """
    rewards = np.array(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    for lap, end in enumerate(np.asarray(episode_ends, dtype=np.int64)):
        if terminals[end - 1] > 0.0:
            vx, vy = float(final_observations[lap][OBS.VX]), float(final_observations[lap][OBS.VY])
            rewards[end - 1] += CRASH_BASE + OFFLINE_REWARD_WEIGHTS.w_crash * (vx * vx + vy * vy)
    return rewards, np.zeros_like(terminals)


class _Data:
    """Tensors of the training transitions.

    The privileged rows (and the student rows, when the next student observation is needed) hold the N transitions
    followed by the final observation of each lap; ``next_index`` gives the row of the next state of each transition
    (``learning/datasets.py``), so no second copy of the rows is kept for the next states.
    """

    def __init__(self, dataset: str | Path, layout: Any, with_next_student: bool, failures: bool = True) -> None:
        with np.load(dataset, allow_pickle=False) as data:
            if with_next_student:
                student, _ = student_dataset(data, layout)
            else:
                student = student_dataset_observations(data, layout)
            privileged = np.concatenate([np.asarray(data["observations"], dtype=np.float32),
                                         np.asarray(data["final_observations"], dtype=np.float32)])
            self.ends = np.asarray(data["episode_ends"], dtype=np.int64)
            self.actions = np.asarray(data["actions_exec"], dtype=np.float32)
            self.labels = np.asarray(data["actions"], dtype=np.float32)
            self.rewards = np.asarray(data["rewards"], dtype=np.float32)
            self.costs = np.asarray(data["traction_costs"], dtype=np.float32)
            self.dones = np.asarray(data["terminals"], dtype=np.float32)
            if not failures:
                self.rewards, self.dones = strip_failures(self.rewards, self.dones, self.ends,
                                                          data["final_observations"])
        n = int(self.ends[-1])
        rows = len(privileged) if with_next_student else n
        if not len(self.actions) == n == len(privileged) - len(self.ends) or len(student) != rows:
            raise ValueError("inconsistent offline dataset")
        self.n = n
        self.student = student
        mean, var = column_moments(privileged[:n])
        std = np.maximum(np.sqrt(var), 1.0e-6)
        self.state_mean, self.state_std = mean.astype(np.float32), std.astype(np.float32)
        privileged -= self.state_mean  # normalised in place: the dataset is large
        privileged /= self.state_std
        self.states = th.from_numpy(privileged)
        self.next_index = next_index(self.ends)
        high = ACTION_HIGH.astype(np.float32)
        self.actions_scaled = th.as_tensor(self.actions / high)
        self.labels_scaled = th.as_tensor(self.labels / high)


def bc_error(mean: th.Tensor, target: th.Tensor, weights: th.Tensor) -> th.Tensor:
    """Per-sample loss of behaviour cloning, sum_j lambda_j (mean_j - target_j)^2 / len(lambda)."""
    return th.sum(weights * (mean - target) ** 2, dim=1) / len(weights)


def train_offline_rl(dataset: str | Path, output: str | Path, racing_line: str | Path,
                     config: OfflineRLConfig | None = None) -> dict[str, Any]:
    """Train a student actor with IQL or TD3+BC on ``dataset``; write the policy and its report to ``output``."""
    config = config or OfflineRLConfig()
    if config.algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {config.algorithm!r}; expected one of {ALGORITHMS}")
    th.set_num_threads(int(config.threads))
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output = Path(output)
    layout = student_layout(config.level, config.n_frames, config.stride, config.oracle)
    if layout.privileged:
        raise ValueError("offline RL trains a student actor: choose level S1 or S2")
    td3 = config.algorithm == "td3bc"
    constrained = config.cost_limit is not None
    if constrained and not td3:
        raise ValueError("the traction-cost constraint is implemented for TD3+BC only")
    if constrained and config.cost_limit < 0.0:
        raise ValueError(f"cost_limit={config.cost_limit} must be non-negative")
    data = _Data(dataset, layout, with_next_student=td3, failures=config.failures)

    vec = load_or_fit_normalizer(output, racing_line, data.student[:data.n], layout)
    model = new_bc_model(vec, config.seed)
    policy = model.policy
    obs = normalise_in_place(vec, data.student)  # with td3, also the final observations of the laps
    data.student = None
    train_idx, val_idx = stratified_split(data.ends)
    actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    actor_opt = th.optim.Adam(actor_params, lr=config.actor_lr)
    critic = Critic(data.states.shape[1], config.critic_hidden)
    critic_target = copy.deepcopy(critic)
    critic_opt = th.optim.Adam(critic.parameters(), lr=config.critic_lr)
    value = _mlp(data.states.shape[1], config.critic_hidden, 1)
    value_opt = th.optim.Adam(value.parameters(), lr=config.critic_lr)
    actor_target = copy.deepcopy(policy) if td3 else None
    bc_weights = th.as_tensor([float(config.steering_weight), 1.0])
    high = th.as_tensor(ACTION_HIGH.astype(np.float32))
    adv_var = 1.0
    cost_critic = Critic(data.states.shape[1], config.critic_hidden) if constrained else None
    cost_target = copy.deepcopy(cost_critic) if constrained else None
    cost_opt = th.optim.Adam(cost_critic.parameters(), lr=config.critic_lr) if constrained else None
    multiplier, cost_scale = 0.0, None
    curve: list[dict[str, float]] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}  # actor quantities are logged every policy_delay steps only

    for step in range(1, config.steps + 1):
        index = rng.choice(train_idx, size=config.batch_size)
        index_t = th.as_tensor(index)
        index_next = data.next_index[index]
        s, s_next = data.states[index_t], data.states[th.as_tensor(index_next)]
        a = data.actions_scaled[index_t]
        r = th.as_tensor(data.rewards[index])
        not_done = 1.0 - th.as_tensor(data.dones[index])
        logs: dict[str, float] = {}

        if not td3:
            with th.no_grad():
                q_target = th.min(*critic_target(s, a))
            v = value(s).squeeze(1)
            diff = q_target - v
            weight = th.abs(config.expectile - (diff < 0.0).float())
            value_loss = th.mean(weight * diff ** 2)
            value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_opt.step()
            with th.no_grad():
                backup = r + config.gamma * not_done * value(s_next).squeeze(1)
            q1, q2 = critic(s, a)
            critic_loss = th.mean((q1 - backup) ** 2) + th.mean((q2 - backup) ** 2)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_opt.step()
            with th.no_grad():
                advantage = q_target - value(s).squeeze(1)
                adv_var = 0.99 * adv_var + 0.01 * float(th.mean(advantage ** 2))
                awr = th.clamp(th.exp(config.beta * advantage / math.sqrt(adv_var)), max=config.max_weight)
            mean = actor_mean(policy, obs[index])
            actor_loss = th.mean(awr * bc_error(mean, th.as_tensor(data.actions[index]), bc_weights))
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(actor_params, 1.0)
            actor_opt.step()
            polyak_update(critic_target, critic, config.polyak)
            logs = {"value_loss": value_loss.item(), "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                    "q": q1.mean().item(), "advantage_std": math.sqrt(adv_var), "awr_weight": awr.mean().item()}
        else:
            with th.no_grad():
                noise = th.clamp(th.randn_like(a) * config.policy_noise, -config.noise_clip, config.noise_clip)
                a_next = th.clamp(actor_mean(actor_target, obs[index_next]) / high + noise, -1.0, 1.0)
                backup = r + config.gamma * not_done * th.min(*critic_target(s_next, a_next))
            q1, q2 = critic(s, a)
            critic_loss = th.mean((q1 - backup) ** 2) + th.mean((q2 - backup) ** 2)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_opt.step()
            logs = {"critic_loss": critic_loss.item(), "q": q1.mean().item()}
            if constrained:
                c = th.as_tensor(data.costs[index])
                with th.no_grad():
                    cost_backup = c + config.gamma * not_done * th.max(*cost_target(s_next, a_next))
                qc1, qc2 = cost_critic(s, a)
                cost_loss = th.mean((qc1 - cost_backup) ** 2) + th.mean((qc2 - cost_backup) ** 2)
                cost_opt.zero_grad(set_to_none=True)
                cost_loss.backward()
                cost_opt.step()
                logs.update({"cost_critic_loss": cost_loss.item(), "qc": qc1.mean().item()})
            if step % config.policy_delay == 0:
                mean = actor_mean(policy, obs[index])
                q_pi = critic.q1(th.cat([s, th.clamp(mean / high, -1.0, 1.0)], dim=1)).squeeze(1)
                lam = config.alpha / q_pi.abs().mean().detach()
                bc = th.mean(bc_error(mean, th.as_tensor(data.labels[index]), bc_weights))
                actor_loss = -lam * q_pi.mean() + bc
                if constrained:
                    qc_pi = cost_critic.q1(th.cat([s, th.clamp(mean / high, -1.0, 1.0)], dim=1)).squeeze(1)
                    with th.no_grad():
                        qc_label = cost_critic.q1(th.cat([s, data.labels_scaled[index_t]], dim=1)).squeeze(1).mean()
                    label_cost = max(float(qc_label), 0.0)
                    cost_scale = label_cost if cost_scale is None else 0.99 * cost_scale + 0.01 * label_cost
                    scale = max(cost_scale, 1.0e-6)
                    actor_loss = actor_loss + multiplier * (config.alpha / scale) * qc_pi.mean()
                    violation = (qc_pi.mean().item() - config.cost_limit * float(qc_label)) / scale
                    multiplier = min(max(multiplier + config.cost_lr * violation, 0.0), config.cost_multiplier_max)
                    logs.update({"qc_policy": qc_pi.mean().item(), "qc_label": float(qc_label),
                                 "violation": violation, "multiplier": multiplier})
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(actor_params, 1.0)
                actor_opt.step()
                polyak_update(critic_target, critic, config.polyak)
                polyak_update(actor_target, policy, config.polyak)
                if constrained:
                    polyak_update(cost_target, cost_critic, config.polyak)
                logs.update({"actor_loss": actor_loss.item(), "bc_loss": bc.item(), "lambda": lam.item()})
        for key, val in logs.items():
            sums[key] = sums.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
        if step % config.log_every == 0:
            entry = {"step": step, **{k: v / counts[k] for k, v in sums.items()}}
            entry.update(_actor_errors(policy, obs, data, val_idx, bc_weights))
            curve.append(entry)
            sums, counts = {}, {}
            LOGGER.info("%s step %d: %s", config.algorithm, step,
                        ", ".join(f"{k} {v:.4g}" for k, v in entry.items() if k != "step"))

    output.mkdir(parents=True, exist_ok=True)
    policy.log_std.data.fill_(-4.0)
    model.save(str(output / POLICY_FILE))
    vec.close()
    write_student_spec(output, layout, "label" if td3 else "exec")
    th.save({"critic": critic.state_dict(), "value": value.state_dict(), "state_mean": data.state_mean,
             "state_std": data.state_std, **({"cost_critic": cost_critic.state_dict()} if constrained else {})},
            output / CRITIC_FILE)
    report = {"config": asdict(config), "dataset": str(dataset), "curve": curve,
              "diagnostics": critic_diagnostics(critic, data, val_idx)}
    if constrained:
        report["cost_diagnostics"] = cost_diagnostics(cost_critic, policy, obs, data, val_idx, multiplier)
    (output / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report


def _actor_errors(policy: Any, obs: NDArray[np.float32], data: _Data, index: NDArray[np.int64],
                  bc_weights: th.Tensor) -> dict[str, float]:
    """BC loss of the actor against the executed actions and the labels on the held-out transitions."""
    with th.no_grad():
        mean = actor_mean(policy, obs[index])
        to_exec = th.mean(bc_error(mean, th.as_tensor(data.actions[index]), bc_weights))
        to_label = th.mean(bc_error(mean, th.as_tensor(data.labels[index]), bc_weights))
    return {"val_mse_exec": float(to_exec), "val_mse_label": float(to_label)}


def critic_diagnostics(critic: Critic, data: _Data, index: NDArray[np.int64], top_fraction: float = 0.1,
                       ) -> dict[str, float]:
    """Does the critic prefer the corrective label to the executed action where they differ most?

    On the held-out transitions with the largest label-to-executed distance (``top_fraction``), the fraction where
    min_i Q_i(s, y) > min_i Q_i(s, a_exec) and the mean of the difference. The label is never an action of the data:
    a preference for it comes from the critic's generalisation over actions.
    """
    index_t = th.as_tensor(index)
    with th.no_grad():
        s = data.states[index_t]
        q_exec = th.min(*critic(s, data.actions_scaled[index_t]))
        q_label = th.min(*critic(s, data.labels_scaled[index_t]))
        distance = th.sum((data.labels_scaled[index_t] - data.actions_scaled[index_t]) ** 2, dim=1)
    n = max(1, int(top_fraction * len(index)))
    top = th.argsort(distance, descending=True)[:n]
    diff = (q_label - q_exec)[top]
    return {"n": n, "fraction_label_preferred": float((diff > 0).float().mean()),
            "mean_q_label_minus_exec": float(diff.mean()), "mean_q_exec": float(q_exec.mean()),
            "all_fraction_label_preferred": float((q_label > q_exec).float().mean())}


def cost_diagnostics(cost_critic: Critic, policy: Any, obs: NDArray[np.float32], data: _Data,
                     index: NDArray[np.int64], multiplier: float) -> dict[str, float]:
    """Cost critic on the held-out transitions: mean Q_c,1 of the executed actions, of the labels and of the policy,
    the same means on the transitions with a positive traction cost, and the final multiplier."""
    index_t = th.as_tensor(index)
    high = th.as_tensor(ACTION_HIGH.astype(np.float32))
    with th.no_grad():
        s = data.states[index_t]
        policy_scaled = th.clamp(actor_mean(policy, obs[index]) / high, -1.0, 1.0)
        values = {name: cost_critic.q1(th.cat([s, a], dim=1)).squeeze(1)
                  for name, a in (("exec", data.actions_scaled[index_t]), ("label", data.labels_scaled[index_t]),
                                  ("policy", policy_scaled))}
    costly = th.as_tensor(data.costs[index] > 0.0)
    report = {"n": len(index), "n_costly": int(costly.sum()), "multiplier": float(multiplier)}
    for name, q in values.items():
        report[f"mean_qc_{name}"] = float(q.mean())
        report[f"mean_qc_{name}_costly"] = float(q[costly].mean()) if bool(costly.any()) else None
    return report
