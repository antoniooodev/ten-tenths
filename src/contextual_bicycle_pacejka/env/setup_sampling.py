"""Sampling of the episode setup: the nominal vehicle with a balance index b drawn per episode.

Mass, centre-of-gravity height, steering lag and friction stay at their nominal values; only b varies. The Nominal
policy trains on b = 0 (the default range), the Randomised and Oracle policies on a range of b.
"""

from __future__ import annotations

import numpy as np

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import B_NOM, HCG_NOM, MASS_NOM, MU_NOM, TAU_D_NOM


def sample_setup(rng: np.random.Generator, b_range: tuple[float, float] = (B_NOM, B_NOM)) -> SetupVector:
    """Return the nominal setup with b drawn uniformly from ``b_range`` = (lo, hi)."""
    return SetupVector(m=MASS_NOM, mu=MU_NOM, hcg=HCG_NOM, tau_d=TAU_D_NOM, b=float(rng.uniform(*b_range)))


def randomized_setup(
    mu: float, rng: np.random.Generator, b_range: tuple[float, float] = (B_NOM, B_NOM),
) -> np.ndarray:
    """Draw a setup with ``sample_setup`` and replace its friction by ``mu``.

    An episode's setup is ``randomized_setup(mu, np.random.default_rng(seed), b_range)``.
    """
    setup = sample_setup(rng, b_range)
    setup.mu = float(mu)
    return setup.to_array()
