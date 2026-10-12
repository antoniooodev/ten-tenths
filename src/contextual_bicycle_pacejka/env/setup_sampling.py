"""Sampling of the episode setup (domain randomisation)."""

from __future__ import annotations

import numpy as np

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import SETUP_DIM, SETUP_RANGES


def sample_setup(rng: np.random.Generator) -> SetupVector:
    """Draw every setup entry independently and uniformly from its range, in setup-vector order."""
    xi = np.empty(SETUP_DIM, dtype=np.float64)
    for i, (lo, hi) in enumerate(SETUP_RANGES):
        xi[i] = rng.uniform(lo, hi)
    return SetupVector.from_array(xi)


def randomized_setup(mu: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a setup with ``sample_setup`` and replace its friction by ``mu``.

    This is the setup of every demonstration lap and every evaluation episode:
    ``randomized_setup(mu, np.random.default_rng(seed))``.
    """
    setup = sample_setup(rng)
    setup.mu = float(mu)
    return setup.to_array()
