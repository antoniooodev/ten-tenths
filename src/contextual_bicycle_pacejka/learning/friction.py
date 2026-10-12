"""Friction estimator: the adaptation module that gives a student the friction mu from onboard sensors.

A student with the true mu drives like the privileged policy (docs/REDESIGN.md, section 13.20): what it lacks is one
scalar. The estimator recovers it in three stages.

    measurement   a network reads a window of the tyre channels (``FRICTION_CHANNELS``: vx, yaw rate, steering
                  angle, accelerometer; n_frames x stride steps) and returns a Gaussian N(m_t, s_t^2) for mu.
                  The axle forces follow from the accelerometer and the yaw acceleration, the slip-angle difference
                  from delta - L r / vx, and the tyre force is mu Fz f(alpha) with f fixed: in a loaded corner the
                  channels determine mu. Previews and actions are excluded, so the estimate cannot be read from how
                  fast the car is driven.
    memory        a scalar Kalman filter fuses one measurement every ``update_steps`` steps, with its variance
                  inflated by c because consecutive windows overlap:

                      K = P / (P + c s^2),    mu_bar <- mu_bar + K (m - mu_bar),    P <- (1 - K) P

                  On a straight the tyres carry no information, s^2 is large and the estimate is kept: the memory
                  spans the straight, which a window of fixed length does not (section 13.21).
    output        a lower confidence bound, clipped to the randomisation range:

                      mu_hat = clip(mu_bar - k sqrt(P + b^2), MU_LO, MU_HI),    k >= k_min

                  The prior N((MU_LO + MU_HI) / 2, ((MU_HI - MU_LO) / 2)^2) puts mu_hat at MU_LO before the first
                  loaded corner when k >= 1; with k < 1 the estimate starts at about 1 - 0.2 k, above MU_LO.
    rise limit    optional, at inference: mu_hat may rise by at most ``rise_limit`` per second and fall freely,

                      mu_hat_t <- min(mu_hat_t, mu_hat_{t-1} + rise_limit DT)

                  An overestimate feeds back: the policy pushes harder, the states leave the training data and the
                  estimate rises further (section 13.26). Limiting the rise slows that loop; a fall is always safe.

The errors cost asymmetrically (section 13.23): underestimating mu costs lap time, overestimating it by more than
about 6 % costs the car. The constants are calibrated on held-out setups: c minimises the negative log-likelihood of
the true mu under N(mu_bar, P), b is the root mean square of the end-of-lap error of mu_bar (the bias the filter
cannot average away), and k is the smallest margin, at least k_min (``margin_floor``, 1 by default), that keeps
mu_hat <= (1 + overestimate_limit) mu at every held-out step. The limit (3 %) is half the closed-loop requirement,
leaving a margin for the shift from the teacher's laps to the student's own. With k_min = 1 the floor, not the limit,
sets k when the network is accurate. Below the floor the steps that set k are the first second or two of the laps at
MU_LO, where the estimate is still the prior's: k then fixes the overestimate at the start to the limit, while later
in the lap a much smaller margin would do (section 18). With ``bias_side="over"`` b
uses the overestimates only (the positive part of the end-of-lap error): an underestimate is on the safe side and
needs no margin (section 13.32). ``recalibrate`` recomputes the calibration of a trained estimator, with other
calibration settings, without retraining it.

Training keeps, for every lap, only the frames of the tyre channels, not the observations, and normalises the input
windows by chunks (``learning/statistics.py``), so a multi-circuit dataset fits in memory; the numbers are those of
the whole-array computation. The report gives the held-out accuracy overall, per kind of lap and, when the datasets
record it (``circuit_per_lap``), per circuit.

An estimator directory holds ``estimator.json`` (configuration, input normalisation, calibration, report) and
``estimator.npz`` (network weights). Inference is NumPy in float64; the estimate of one episode computed offline
(``estimate_episode``) equals the one computed online step by step (``reset`` and ``update``) to rounding.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from numpy.typing import NDArray

from contextual_bicycle_pacejka.learning.statistics import column_moments, standardised
from contextual_bicycle_pacejka.model.student import channel_frames, previous_actions, stack_window
from contextual_bicycle_pacejka.spec.constants import (
    DT,
    MU_HI,
    MU_LO,
    STUDENT_HISTORY_FRAMES,
    STUDENT_HISTORY_STRIDE,
)
from contextual_bicycle_pacejka.spec.layout import FRICTION_CHANNELS

LOGGER = logging.getLogger(__name__)
ESTIMATOR_FILE = "estimator.json"
WEIGHTS_FILE = "estimator.npz"
PRIOR_MEAN: float = 0.5 * (MU_LO + MU_HI)
PRIOR_STD: float = 0.5 * (MU_HI - MU_LO)
LOG_VAR_RANGE = (-12.0, 4.0)  # bounds of the predicted log-variance, in units of PRIOR_STD^2


@dataclass
class FrictionConfig:
    """Hyper-parameters of the friction estimator."""

    seed: int = 0
    n_frames: int = STUDENT_HISTORY_FRAMES
    stride: int = STUDENT_HISTORY_STRIDE
    hidden: tuple[int, ...] = (128, 128)
    learning_rate: float = 1.0e-3
    batch_size: int = 4096
    epochs: int = 40
    patience: int = 6  # epochs without improvement before stopping (>= epochs: no early stopping)
    update_steps: int = 10  # one filter update every 0.1 s
    inflations: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
    overestimate_limit: float = 0.03
    margin_floor: float = 1.0  # smallest margin k of the lower bound
    validation_setups_per_mu: int = 1
    bias_side: str = "both"  # end-of-lap errors entering b: "both" or "over" (overestimates only)


@dataclass
class Calibration:
    """Constants of the filter and of the lower bound (see the module docstring)."""

    inflation: float
    bias_std: float
    margin: float


# --------------------------------------------------------------------------------------------- network
def _forward(weights: list[tuple[NDArray[np.float64], NDArray[np.float64]]], x: NDArray[np.float64],
             ) -> NDArray[np.float64]:
    """MLP with SiLU hidden layers and a linear head, in float64."""
    h = x
    for i, (w, b) in enumerate(weights):
        h = h @ w + b
        if i < len(weights) - 1:
            h = h / (1.0 + np.exp(-h))
    return h


def _torch_network(n_inputs: int, hidden: tuple[int, ...]) -> th.nn.Sequential:
    layers: list[th.nn.Module] = []
    width = n_inputs
    for size in hidden:
        layers += [th.nn.Linear(width, size), th.nn.SiLU()]
        width = size
    layers.append(th.nn.Linear(width, 2))
    return th.nn.Sequential(*layers)


def _gaussian(raw: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Network output (normalised mean, log-variance) -> (mean, variance) of mu."""
    log_var = np.clip(raw[:, 1], *LOG_VAR_RANGE)
    return PRIOR_MEAN + PRIOR_STD * raw[:, 0], PRIOR_STD ** 2 * np.exp(log_var)


# --------------------------------------------------------------------------------------------- filter
def fuse(measured: NDArray[np.float64], variance: NDArray[np.float64], update_steps: int, n_steps: int,
         inflation: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filtered mean and variance after every step of an episode, from the measurements of its update steps."""
    mean, var = np.empty(n_steps), np.empty(n_steps)
    mu_bar, p = PRIOR_MEAN, PRIOR_STD ** 2
    for t in range(n_steps):
        if t % update_steps == 0:
            j = t // update_steps
            gain = p / (p + inflation * variance[j])
            mu_bar += gain * (measured[j] - mu_bar)
            p *= 1.0 - gain
        mean[t], var[t] = mu_bar, p
    return mean, var


def lower_bound(mean: NDArray[np.float64] | float, var: NDArray[np.float64] | float, calibration: Calibration,
                ) -> NDArray[np.float64]:
    """The estimate passed to the policy: mu_bar - k sqrt(P + b^2), clipped to [MU_LO, MU_HI]."""
    bound = np.asarray(mean) - calibration.margin * np.sqrt(np.asarray(var) + calibration.bias_std ** 2)
    return np.clip(bound, MU_LO, MU_HI)


def limit_rise(estimate: NDArray[np.float64], rise_limit: float | None) -> NDArray[np.float64]:
    """The estimate of every step with its rise limited to ``rise_limit`` per second (unchanged without a limit)."""
    if rise_limit is None:
        return estimate
    limited = np.array(estimate, dtype=np.float64)
    for t in range(1, len(limited)):
        limited[t] = min(limited[t], limited[t - 1] + rise_limit * DT)
    return limited


# --------------------------------------------------------------------------------------------- estimator
class FrictionEstimator:
    """A trained friction estimator; ``estimate_episode`` offline, ``reset`` and ``update`` online.

    Parameters
    ----------
    model_dir : str or Path
        Directory with ``estimator.json`` and ``estimator.npz``.
    rise_limit : float, optional
        Largest rise of the estimate per second (no limit by default).
    """

    def __init__(self, model_dir: str | Path, rise_limit: float | None = None) -> None:
        model_dir = Path(model_dir)
        if rise_limit is not None and rise_limit <= 0.0:
            raise ValueError(f"rise_limit={rise_limit} must be positive")
        self.rise_limit = rise_limit
        spec = json.loads((model_dir / ESTIMATOR_FILE).read_text())
        self.config = FrictionConfig(**{**spec["config"], "hidden": tuple(spec["config"]["hidden"]),
                                        "inflations": tuple(spec["config"]["inflations"])})
        self.calibration = Calibration(**spec["calibration"])
        self.input_mean = np.asarray(spec["input_mean"], dtype=np.float64)
        self.input_std = np.asarray(spec["input_std"], dtype=np.float64)
        with np.load(model_dir / WEIGHTS_FILE) as data:
            n_layers = len(self.config.hidden) + 1
            self.weights = [(np.asarray(data[f"w{i}"], dtype=np.float64), np.asarray(data[f"b{i}"], dtype=np.float64))
                            for i in range(n_layers)]
        self.window = (self.config.n_frames - 1) * self.config.stride + 1
        self.reset()

    # offline
    def windows(self, obs: NDArray[np.floating], imu: NDArray[np.floating], prev_actions: NDArray[np.floating],
                ) -> NDArray[np.float32]:
        """Input windows of every step of one episode, shape (T, n_frames * len(FRICTION_CHANNELS))."""
        frames = channel_frames(obs, imu, prev_actions, FRICTION_CHANNELS)
        return stack_window(frames, self.config.n_frames, self.config.stride)

    def measure(self, windows: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Measurement (mean, variance) of mu for a batch of windows."""
        x = (np.asarray(windows, dtype=np.float64) - self.input_mean) / self.input_std
        return _gaussian(_forward(self.weights, x))

    def estimate_episode(self, obs: NDArray[np.floating], imu: NDArray[np.floating],
                         prev_actions: NDArray[np.floating]) -> dict[str, NDArray[np.float64]]:
        """Measurements, filtered mean and variance, and the lower bound of every step of one episode."""
        windows = self.windows(obs, imu, prev_actions)
        measured, variance = self.measure(windows[::self.config.update_steps])
        mean, var = fuse(measured, variance, self.config.update_steps, len(windows), self.calibration.inflation)
        return {"measured": measured, "variance": variance, "mean": mean, "var": var,
                "estimate": limit_rise(lower_bound(mean, var, self.calibration), self.rise_limit)}

    # online
    def reset(self) -> None:
        """Start a new episode: empty history, prior estimate."""
        self._frames: list[NDArray[np.float32]] = []
        self._step = 0
        self.mean, self.var = PRIOR_MEAN, PRIOR_STD ** 2
        self.estimate: float | None = None

    def update(self, obs: NDArray[np.floating], imu: NDArray[np.floating], previous: NDArray[np.floating]) -> float:
        """Add the channels of one step (observation, IMU, previous action) and return the estimate."""
        frame = channel_frames(obs, imu, previous, FRICTION_CHANNELS)[0]
        self._frames = (self._frames + [frame])[-self.window:]
        if self._step % self.config.update_steps == 0:
            window = stack_window(np.asarray(self._frames), self.config.n_frames, self.config.stride)[-1]
            measured, variance = self.measure(window[None, :])
            gain = self.var / (self.var + self.calibration.inflation * float(variance[0]))
            self.mean += gain * (float(measured[0]) - self.mean)
            self.var *= 1.0 - gain
        self._step += 1
        estimate = float(lower_bound(self.mean, self.var, self.calibration))
        if self.rise_limit is not None and self.estimate is not None:
            estimate = min(estimate, self.estimate + self.rise_limit * DT)
        self.estimate = estimate
        return estimate


# --------------------------------------------------------------------------------------------- data
def _episodes(data: dict[str, NDArray] | np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    """Per-lap frames of the tyre channels of an offline (MDP) dataset (``learning/datasets.py``; not imported,
    since the evaluation runner imports this module and the dataset module imports the runner), with the lap's
    friction, setup, kind and circuit (empty if the dataset does not record it). The observations are not kept."""
    observations, imu, executed = (np.asarray(data[k]) for k in ("observations", "imu", "actions_exec"))
    mus, setups = np.asarray(data["mu_per_lap"]), np.asarray(data["setup_per_lap"])
    kinds = [str(k) for k in data["kind_per_lap"]]
    circuits = [str(c) for c in data["circuit_per_lap"]] if "circuit_per_lap" in data else [""] * len(kinds)
    episodes = []
    ends = np.asarray(data["episode_ends"], dtype=np.int64)
    for lap, (start, end) in enumerate(zip(np.concatenate(([0], ends[:-1])), ends, strict=True)):
        frames = channel_frames(observations[start:end], imu[start:end], previous_actions(executed[start:end]),
                                FRICTION_CHANNELS)
        episodes.append({"frames": frames, "mu": float(mus[lap]), "setup": setups[lap], "kind": kinds[lap],
                         "circuit": circuits[lap]})
    return episodes


def validation_setups(episodes: list[dict[str, Any]], per_mu: int) -> set[bytes]:
    """Held-out setups: the first ``per_mu`` distinct setups of every friction value, in dataset order."""
    chosen: dict[float, list[bytes]] = {}
    for episode in episodes:
        key = np.asarray(episode["setup"], dtype=np.float64).tobytes()
        picked = chosen.setdefault(episode["mu"], [])
        if key not in picked and len(picked) < per_mu:
            picked.append(key)
    return {key for keys in chosen.values() for key in keys}


def _window_rows(episodes: list[dict[str, Any]], config: FrictionConfig) -> tuple[NDArray[np.float32], NDArray]:
    frames = [stack_window(e["frames"], config.n_frames, config.stride) for e in episodes]
    targets = [np.full(len(f), e["mu"], dtype=np.float32) for f, e in zip(frames, episodes, strict=True)]
    return np.concatenate(frames), np.concatenate(targets)


# --------------------------------------------------------------------------------------------- training
CALIBRATION_FIELDS = ("inflations", "overestimate_limit", "margin_floor", "bias_side")


def _datasets(dataset: str | Path | list[str | Path]) -> list[str | Path]:
    return list(dataset) if isinstance(dataset, list) else [dataset]


def _split(dataset: str | Path | list[str | Path], config: FrictionConfig,
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Training and held-out laps of the concatenated datasets (held-out setups chosen on the concatenation)."""
    episodes = []
    for path in _datasets(dataset):
        with np.load(path, allow_pickle=False) as data:
            episodes += _episodes(data)
    held_out = validation_setups(episodes, config.validation_setups_per_mu)
    is_val = [np.asarray(e["setup"], dtype=np.float64).tobytes() in held_out for e in episodes]
    return ([e for e, v in zip(episodes, is_val, strict=True) if not v],
            [e for e, v in zip(episodes, is_val, strict=True) if v])


def _calibrate_and_report(output: Path, spec: dict[str, Any], dataset: str | Path | list[str | Path],
                          train_eps: list[dict[str, Any]], val_eps: list[dict[str, Any]]) -> dict[str, Any]:
    """Calibrate the estimator written in ``output`` on the held-out laps; write and return its specification."""
    estimator = FrictionEstimator(output)
    config = estimator.config
    val_runs = [_measure_episode(estimator, e) for e in val_eps]
    calibration = calibrate(val_runs, config)
    spec["calibration"] = asdict(calibration)
    estimator.calibration = calibration
    spec["report"] = {"datasets": [str(path) for path in _datasets(dataset)],
                      "validation": evaluate_runs(estimator, val_runs, config),
                      "validation_by_kind": validation_by(estimator, val_runs, config, "kind"),
                      "validation_by_circuit": validation_by(estimator, val_runs, config, "circuit"),
                      "n_train_laps": len(train_eps), "n_validation_laps": len(val_eps),
                      "validation_mus": sorted({e["mu"] for e in val_eps})}
    (output / ESTIMATOR_FILE).write_text(json.dumps(spec, indent=2))
    return spec


def train_friction(dataset: str | Path | list[str | Path], output: str | Path, config: FrictionConfig | None = None,
                   ) -> dict[str, Any]:
    """Train, calibrate and evaluate a friction estimator on offline datasets (one or several, concatenated in
    order, e.g. demonstrations and closed-loop laps of ``learning/closed_loop.py``); write it to ``output``.

    The held-out setups are chosen on the concatenation, so a setup is held out in every dataset that has it.
    """
    config = config or FrictionConfig()
    if config.bias_side not in ("both", "over"):
        raise ValueError(f"bias_side={config.bias_side!r} must be 'both' or 'over'")
    output = Path(output)
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train_eps, val_eps = _split(dataset, config)
    x_train, y_train = _window_rows(train_eps, config)
    input_mean, input_var = column_moments(x_train)
    input_std = np.maximum(np.sqrt(input_var), 1.0e-6)

    def tensors(x: NDArray, y: NDArray) -> tuple[th.Tensor, th.Tensor]:
        return (th.as_tensor(standardised(x, input_mean, input_std)),
                th.as_tensor(((y - PRIOR_MEAN) / PRIOR_STD).astype(np.float32)))

    xt, yt = tensors(x_train, y_train)
    del x_train  # the windows are kept once, standardised
    xv, yv = tensors(*_window_rows(val_eps, config))
    network = _torch_network(xt.shape[1], config.hidden)
    optimizer = th.optim.Adam(network.parameters(), lr=config.learning_rate)

    def nll(x: th.Tensor, y: th.Tensor) -> th.Tensor:
        out = network(x)
        log_var = th.clamp(out[:, 1], *LOG_VAR_RANGE)
        return 0.5 * th.mean(log_var + (y - out[:, 0]) ** 2 * th.exp(-log_var))

    best, best_state, stale, epoch = math.inf, None, 0, -1
    for epoch in range(config.epochs):
        order = rng.permutation(len(xt))
        for start in range(0, len(order), config.batch_size):
            index = th.as_tensor(order[start:start + config.batch_size])
            loss = nll(xt[index], yt[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with th.no_grad():
            val = float(nll(xv, yv))
        LOGGER.info("friction epoch %02d: validation NLL %.4f", epoch, val)
        if val < best - 1.0e-4:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("friction estimator training produced no checkpoint")
    network.load_state_dict(best_state)

    output.mkdir(parents=True, exist_ok=True)
    linear = [m for m in network if isinstance(m, th.nn.Linear)]
    arrays = {}
    for i, layer in enumerate(linear):
        arrays[f"w{i}"] = layer.weight.detach().numpy().astype(np.float64).T
        arrays[f"b{i}"] = layer.bias.detach().numpy().astype(np.float64)
    np.savez(output / WEIGHTS_FILE, **arrays)
    spec = {"config": asdict(config), "input_mean": input_mean.tolist(), "input_std": input_std.tolist(),
            "calibration": asdict(Calibration(1.0, 0.0, 1.0)), "training": {"epochs": epoch + 1,
                                                                             "validation_nll": best}}
    (output / ESTIMATOR_FILE).write_text(json.dumps(spec, indent=2))
    return _calibrate_and_report(output, spec, dataset, train_eps, val_eps)


def recalibrate(model_dir: str | Path, dataset: str | Path | list[str | Path], output: str | Path,
                **settings: Any) -> dict[str, Any]:
    """Copy the trained estimator in ``model_dir`` to ``output`` and calibrate it again on the held-out laps of
    ``dataset`` with other calibration settings (``CALIBRATION_FIELDS``, e.g. ``bias_side="over"``).

    The network and the input normalisation are unchanged; ``dataset`` should be the one the estimator was trained
    on, so that the held-out setups are the same.
    """
    unknown = set(settings) - set(CALIBRATION_FIELDS)
    if unknown:
        raise ValueError(f"only the calibration settings {CALIBRATION_FIELDS} can change, not {sorted(unknown)}")
    if settings.get("bias_side", "both") not in ("both", "over"):
        raise ValueError(f"bias_side={settings['bias_side']!r} must be 'both' or 'over'")
    model_dir, output = Path(model_dir), Path(output)
    spec = json.loads((model_dir / ESTIMATOR_FILE).read_text())
    spec["config"] = {**spec["config"], **{k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()}}
    spec["recalibrated_from"] = str(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_dir / WEIGHTS_FILE, output / WEIGHTS_FILE)
    (output / ESTIMATOR_FILE).write_text(json.dumps(spec, indent=2))
    config = FrictionEstimator(output).config
    train_eps, val_eps = _split(dataset, config)
    return _calibrate_and_report(output, spec, dataset, train_eps, val_eps)


def _measure_episode(estimator: FrictionEstimator, episode: dict[str, Any]) -> dict[str, Any]:
    windows = stack_window(episode["frames"], estimator.config.n_frames, estimator.config.stride)
    measured, variance = estimator.measure(windows[::estimator.config.update_steps])
    return {"measured": measured, "variance": variance, "n": len(windows), "mu": episode["mu"],
            "kind": episode["kind"], "circuit": episode["circuit"]}


def validation_by(estimator: FrictionEstimator, runs: list[dict[str, Any]], config: FrictionConfig, key: str,
                  ) -> dict[str, dict[str, Any]]:
    """``evaluate_runs`` for the laps of every value of ``key`` ("kind" or "circuit"; laps without a circuit are
    left out), without the per-lap maxima."""
    groups: dict[str, dict[str, Any]] = {}
    for value in sorted({r[key] for r in runs} - {""}):
        report = evaluate_runs(estimator, [r for r in runs if r[key] == value], config)
        groups[value] = {k: v for k, v in report.items() if k != "per_lap_max_ratio"}
    return groups


def calibrate(runs: list[dict[str, Any]], config: FrictionConfig) -> Calibration:
    """Inflation, bias spread and margin from the measurements of held-out laps (see the module docstring)."""
    def filtered(inflation: float) -> list[tuple[NDArray, NDArray, float]]:
        return [(*fuse(r["measured"], r["variance"], config.update_steps, r["n"], inflation), r["mu"]) for r in runs]

    def nll(inflation: float) -> float:
        terms = [0.5 * (np.log(var) + (mean - mu) ** 2 / var) for mean, var, mu in filtered(inflation)]
        return float(np.mean(np.concatenate(terms)))

    inflation = min(config.inflations, key=nll)
    runs_f = filtered(inflation)
    errors = np.array([mean[-1] - mu for mean, _, mu in runs_f])
    if config.bias_side == "over":
        errors = np.maximum(errors, 0.0)
    bias_std = float(np.sqrt(np.mean(errors ** 2)))
    margin = float(config.margin_floor)
    for mean, var, mu in runs_f:
        needed = (mean - (1.0 + config.overestimate_limit) * mu) / np.sqrt(var + bias_std ** 2)
        margin = max(margin, float(np.max(needed)))
    return Calibration(float(inflation), bias_std, margin)


def evaluate_runs(estimator: FrictionEstimator, runs: list[dict[str, Any]], config: FrictionConfig,
                  ) -> dict[str, Any]:
    """Accuracy of the measurements and of the estimate on laps of known friction."""
    ratios, finals, reach, over, errors = [], [], [], [], []
    for r in runs:
        mean, var = fuse(r["measured"], r["variance"], config.update_steps, r["n"], estimator.calibration.inflation)
        estimate = lower_bound(mean, var, estimator.calibration)
        ratio = estimate / r["mu"]
        ratios.append(ratio)
        finals.append(float(ratio[-1]))
        above = np.flatnonzero(ratio >= 0.95)
        reach.append(float(above[0]) * DT if above.size else None)
        over.append(float(np.max(ratio)))
        errors.append(r["measured"] - r["mu"])
    all_ratios = np.concatenate(ratios)
    reached = [t for t in reach if t is not None]
    return {
        "measurement_rmse": float(np.sqrt(np.mean(np.concatenate(errors) ** 2))),
        "estimate_ratio_mean": float(np.mean(all_ratios)),
        "estimate_ratio_max": float(np.max(all_ratios)),
        "estimate_ratio_final_median": float(np.median(finals)),
        "estimate_ratio_final_min": float(np.min(finals)),
        "laps_reaching_0.95": f"{len(reached)}/{len(reach)}",
        "time_to_0.95_median_s": float(np.median(reached)) if reached else None,
        "fraction_within_5pct": float(np.mean(np.abs(all_ratios - 1.0) <= 0.05)),
        "per_lap_max_ratio": over,
    }
