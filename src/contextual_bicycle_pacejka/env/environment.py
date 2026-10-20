"""Gymnasium racing environment: one circuit, one speed stage, decisions at 20 Hz.

Every decision integrates the rate of the longitudinal command once, then holds the steering reference and the
command for ``DECISION_STEPS`` plant steps; the top speed of the stage fades out the traction before each step. The
integrated command itself is bounded by what the top speed lets through (anti-windup): near the top speed it stays
close to zero instead of winding up to full throttle, so braking acts at once. The
decision stops at the first failure (``model/termination.py``): the episode terminates with the failure penalty, and
the vectorised environment starts the next one at once, at a random point of the centreline (Czechmanowski et al.,
IROS 2025). Truncation is the ``TimeLimit`` of ``env/factory.py``.

``info`` holds the ``failure`` (or None), the ``reward_terms``, the ``progress`` [m] of the decision, the raw
``state`` and ``setup``, and ``laps``: the laps completed since the start of the episode.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.env.setup_sampling import sample_setup
from contextual_bicycle_pacejka.env.stages import SpeedStage, with_top_speed
from contextual_bicycle_pacejka.model.dynamics import step_plant
from contextual_bicycle_pacejka.model.observation import build_observation
from contextual_bicycle_pacejka.model.powertrain import limit_top_speed
from contextual_bicycle_pacejka.model.reward import compute_reward
from contextual_bicycle_pacejka.model.state import AdmissibilityError, SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.termination import check_failure
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold, compute_slip_angles
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    B_NOM,
    DECISION_DT,
    DECISION_STEPS,
    DELTA_MAX,
    DT,
    OBS_DIM,
    START_SPEED_MIN_MPS,
    TAU_RATE_MAX,
    V_MIN,
)
from contextual_bicycle_pacejka.spec.layout import ACTION, ACTION_HIGH, ACTION_LOW, validate_setup, validate_state


class RacingEnv(gym.Env):
    """Single-track Pacejka car on a closed circuit, driven at 20 Hz.

    Parameters
    ----------
    track : RealTrack
        Track whose centreline defines the Frenet frame and the progress.
    stage : SpeedStage
        Speed stage: top speed, preview, observation scales and start speeds.
    b_range : (float, float)
        Range of the balance index b drawn at every reset ((0, 0): the Nominal policy).
    b_exclude : ((float, float), ...)
        Open bands of b never drawn (around the held-out values of b).
    oracle : bool
        Whether the observation ends with b (the Oracle policy).
    edge_margin : float
        Distance [m] inside the track edges at which the car counts as off track; training uses
        ``TRAINING_EDGE_MARGIN_M``, so that the policy keeps a margin, and evaluation the true edges (0).

    Reset options
    -------------
    ``setup``: setup vector in physical units (otherwise the nominal setup with b drawn from ``b_range``);
    ``initial_s`` [m], ``initial_vx`` [m/s], ``initial_ey`` [m], ``initial_epsi`` [rad]: start state (otherwise a
    uniform point of the lap on the centreline, aligned with it, at a speed drawn up to the start-speed bound of the
    stage at that point, ``stage.start_speed_bound``).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        track: RealTrack,
        stage: SpeedStage,
        b_range: tuple[float, float] = (B_NOM, B_NOM),
        oracle: bool = False,
        edge_margin: float = 0.0,
        b_exclude: tuple[tuple[float, float], ...] = (),
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM + int(oracle),),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=ACTION_LOW.astype(np.float32), high=ACTION_HIGH.astype(np.float32),
                                           dtype=np.float32)
        self.track = track
        self.stage = stage
        self.b_range = b_range
        self.b_exclude = b_exclude
        self.oracle = oracle
        self.edge_margin = edge_margin
        self.state = VehicleState()
        self.memory = VehicleMemory.zeros()
        self.setup = SetupVector()
        self._saturation = (1.0, 1.0)
        self.delta_ref = 0.0
        self.tau = 0.0
        self.tau_rate = 0.0
        self._s0 = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        if "setup" in options:
            self.setup = SetupVector.from_array(np.asarray(options["setup"], dtype=np.float64))
        else:
            self.setup = sample_setup(rng, self.b_range, self.b_exclude)
        validate_setup(self.setup.to_array())

        s0 = float(options.get("initial_s", rng.uniform(0.0, self.track.track_length)))
        fastest = self.stage.start_speed_bound(s0, self.track.track_length)
        vx0 = float(options.get("initial_vx", rng.uniform(START_SPEED_MIN_MPS, fastest)))
        self.state = VehicleState(s=s0, ey=float(options.get("initial_ey", 0.0)),
                                  epsi=float(options.get("initial_epsi", 0.0)), vx=vx0)
        validate_state(self.state.to_array(), self.track.kappa(s0))
        self.memory = VehicleMemory.zeros()
        self._saturation = (compute_saturation_threshold(self.setup, "front"),
                            compute_saturation_threshold(self.setup, "rear"))
        self.delta_ref = self.tau = self.tau_rate = 0.0
        self._s0 = s0
        return self._observation(), {"setup": self.setup.to_array(), "state": self.state.to_array()}

    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), ACTION_LOW, ACTION_HIGH)
        delta_ref = float(action[ACTION.DELTA_REF]) * DELTA_MAX
        tau_rate = float(action[ACTION.TAU_RATE]) * TAU_RATE_MAX
        delta_ref_change = delta_ref - self.delta_ref
        self.delta_ref, self.tau_rate = delta_ref, tau_rate
        self.tau = limit_top_speed(min(1.0, max(-1.0, self.tau + tau_rate * DECISION_DT)), self.state.vx,
                                   self.stage.top_speed)  # no windup of the command behind the speed bound

        s_before = self.state.s
        failure = None
        for _ in range(DECISION_STEPS):
            tau_eff = limit_top_speed(self.tau, self.state.vx, self.stage.top_speed)
            try:
                self.state, self.memory, _ = step_plant(self.state, (delta_ref, tau_eff), self.setup, self.memory,
                                                        self.track, DT)
            except AdmissibilityError:  # the state is not advanced
                failure = "inadmissible"
                break
            failure = check_failure(self.state, self.track, self.edge_margin)
            if failure is not None:
                break

        progress = self.state.s - s_before
        terms = compute_reward(progress, self.stage.reward_speed, delta_ref_change, tau_rate, failure is not None)
        info = {"failure": failure, "reward_terms": terms._asdict(), "progress": progress,
                "state": self.state.to_array(), "setup": self.setup.to_array(),
                "laps": int((self.state.s - self._s0) // self.track.track_length),
                "slip_ratio": self._slip_ratio()}
        return self._observation(), terms.total, failure is not None, False, info

    def set_top_speed(self, top_speed: float | None) -> None:
        """Change the top speed of the stage (a curriculum during training); the observation does not change."""
        self.stage = with_top_speed(self.stage, top_speed, self.track)

    def _slip_ratio(self) -> tuple[float, float]:
        """Return |alpha| / alpha_sat of the front and rear axle at the end of the decision (0 below V_MIN)."""
        if self.state.vx < V_MIN:
            return 0.0, 0.0
        alpha_f, alpha_r = compute_slip_angles(self.state)
        return abs(alpha_f) / self._saturation[0], abs(alpha_r) / self._saturation[1]

    def _observation(self) -> NDArray[np.float32]:
        return build_observation(self.state, self.setup, self.delta_ref, self.tau, self.tau_rate, self.track,
                                 self.stage.preview, self.stage.scales, self.oracle).astype(np.float32)
