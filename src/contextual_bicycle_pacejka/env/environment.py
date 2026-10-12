"""Gymnasium environment: one closed track, one vehicle setup per episode, 116-entry observation.

The environment only orchestrates the plant (``model/``): reset samples or receives a setup, places the
vehicle on the reference path, and every step applies the action through ``step_plant``, checks the
termination rules and builds the next observation. A lap does not terminate the episode.

The observation is the privileged one. ``info`` also carries the traction cost of the step and the accelerometer
channels ``imu`` that belong to the new observation, from which the student observation is built
(``model/student.py``, ``env/student.py``).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
from contextual_bicycle_pacejka.model.dynamics import DIAGNOSTIC_KEYS, step_plant
from contextual_bicycle_pacejka.model.observation import build_observation
from contextual_bicycle_pacejka.model.reward import RewardWeights, compute_reward, compute_traction_cost
from contextual_bicycle_pacejka.model.sensors import imu_specific_force
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.termination import compute_termination
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import DT, OBS_DIM, V_MIN
from contextual_bicycle_pacejka.spec.layout import ACTION_HIGH, ACTION_LOW, validate_setup, validate_state

DEFAULT_START_VX: float = V_MIN + 4.0  # [m/s] start speed when the reset options give none


class ContextualBicyclePacejkaEnv(gym.Env):
    """Single-track Pacejka vehicle on a closed racing line.

    Parameters
    ----------
    track : RealTrack
        Track whose reference path defines the Frenet frame.
    reward_weights : RewardWeights, optional
        Weights of the reward (used only by PPO fine-tuning).

    Reset options
    -------------
    ``setup``: setup vector in physical units (otherwise sampled uniformly from the randomisation ranges);
    ``initial_s`` [m] and ``initial_vx`` [m/s]: start position and speed (defaults 0 m and 5 m/s).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, track: RealTrack, reward_weights: RewardWeights | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=ACTION_LOW.astype(np.float32), high=ACTION_HIGH.astype(np.float32),
                                           dtype=np.float32)
        self.track = track
        self.reward_weights = reward_weights or RewardWeights()
        self.state = VehicleState()
        self.memory = VehicleMemory.zeros()
        self.setup = SetupVector()
        self.alpha_sat: tuple[float, float] = (0.0, 0.0)
        self._episode_s0 = 0.0
        self._rng = np.random.default_rng()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        options = options or {}
        if "setup" in options:
            self.setup = SetupVector.from_array(np.asarray(options["setup"], dtype=np.float64))
        else:
            self.setup = sample_setup(self._rng)
        validate_setup(self.setup.to_array())

        s0 = float(options.get("initial_s", 0.0))
        vx0 = float(options.get("initial_vx", DEFAULT_START_VX))
        self._episode_s0 = s0
        self.state = VehicleState(s=s0, vx=vx0)
        validate_state(self.state.to_array(), self.track.kappa(s0))
        self.memory = VehicleMemory.zeros()
        self.alpha_sat = (compute_saturation_threshold(self.setup, "front"),
                          compute_saturation_threshold(self.setup, "rear"))

        diag = dict.fromkeys(DIAGNOSTIC_KEYS, 0.0)
        obs = build_observation(self.state, self.setup, diag, self.track)
        info = {"setup": self.setup.to_array(), "alpha_sat_f": self.alpha_sat[0], "alpha_sat_r": self.alpha_sat[1],
                "imu": imu_specific_force(diag, self.setup)}
        return obs.astype(np.float32), info

    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float64)
        try:
            next_state, next_memory, diag = step_plant(
                self.state, (float(action[0]), float(action[1])), self.setup, self.memory, self.track, DT)
        except AdmissibilityError as exc:
            return self._inadmissible_step(action, str(exc))

        terminated, flags = compute_termination(next_state, diag, self.alpha_sat, self.track)
        length = self.track.track_length
        lap_before = int((self.state.s - self._episode_s0) // length)
        lap_after = int((next_state.s - self._episode_s0) // length)
        traction_cost = compute_traction_cost(diag["alpha_f"], diag["alpha_r"], self.alpha_sat)
        reward = compute_reward(next_state, diag["s_dot"], self.track.ey_max(next_state.s), DT, self.reward_weights,
                                terminated, flags, lap_after > lap_before, traction_cost)
        obs = build_observation(next_state, self.setup, diag, self.track)
        self.state, self.memory = next_state, next_memory
        info = {"event_flags": flags, "diag": diag, "state": next_state.to_array(), "setup": self.setup.to_array(),
                "lap_complete": lap_after > lap_before, "traction_cost": traction_cost,
                "imu": imu_specific_force(diag, self.setup)}
        return obs.astype(np.float32), float(reward), terminated, False, info

    def _inadmissible_step(
        self, action: NDArray[np.float64], message: str,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Terminate cleanly when the plant reaches an inadmissible state; the vehicle state is not advanced."""
        flags = {"track_excursion": False, "slip_saturation": False, "low_speed": False,
                 "admissibility_violation": True}
        diag = dict.fromkeys(DIAGNOSTIC_KEYS, 0.0)
        reward = compute_reward(self.state, 0.0, self.track.ey_max(self.state.s), DT, self.reward_weights,
                                True, flags, False)
        obs = build_observation(self.state, self.setup, diag, self.track)
        info = {"event_flags": flags, "diag": diag, "state": self.state.to_array(), "setup": self.setup.to_array(),
                "lap_complete": False, "traction_cost": 0.0, "imu": imu_specific_force(diag, self.setup),
                "admissibility_error": message, "action_applied": action}
        return obs.astype(np.float32), float(reward), True, False, info
