"""Reward of one decision.

    r_k = ds_k / (v DECISION_DT)
          - ACTION_RATE_WEIGHT [((delta_ref,k - delta_ref,k-1) / (2 DELTA_MAX))^2 + (tau_rate_k / TAU_RATE_MAX)^2]
          - FAILURE_PENALTY 1[failure]

v is the top speed of the speed stage (its sizing speed for the free stage), so the progress term is at most about 1
per decision at every stage and the failure penalty keeps the same weight. The progress is measured along the
centreline, as in Czechmanowski et al. (IROS 2025), equation (3).

The reward returned to the learner is ``REWARD_SCALE`` r_k. With gamma 0.99 the discounted return of r_k reaches
about 100; scaled, about 10, close to the returns of progress in metres in Czechmanowski et al. The scale changes
neither the optimum nor the normalised advantages; it keeps the value loss, and its share of the clipped gradient,
small. The terms are reported unscaled.
"""

from __future__ import annotations

from typing import NamedTuple

from contextual_bicycle_pacejka.spec.constants import (
    ACTION_RATE_WEIGHT,
    DECISION_DT,
    DELTA_MAX,
    FAILURE_PENALTY,
    REWARD_SCALE,
    TAU_RATE_MAX,
)


class RewardTerms(NamedTuple):
    """Terms of the reward of one decision, unscaled; ``total`` is their sum times ``REWARD_SCALE``."""

    progress: float
    action_rate: float
    failure: float

    @property
    def total(self) -> float:
        return REWARD_SCALE * (self.progress + self.action_rate + self.failure)


def compute_reward(
    progress_m: float, speed_scale: float, delta_ref_change: float, tau_rate: float, failed: bool,
) -> RewardTerms:
    """Return the reward terms of a decision that advanced ``progress_m`` [m] along the centreline.

    ``delta_ref_change`` [rad] is the change of the steering reference, ``tau_rate`` [1/s] the commanded rate of the
    longitudinal command, ``speed_scale`` [m/s] the top speed of the stage.
    """
    rate = (delta_ref_change / (2.0 * DELTA_MAX)) ** 2 + (tau_rate / TAU_RATE_MAX) ** 2
    return RewardTerms(
        progress=progress_m / (speed_scale * DECISION_DT),
        action_rate=-ACTION_RATE_WEIGHT * rate,
        failure=-FAILURE_PENALTY if failed else 0.0,
    )
