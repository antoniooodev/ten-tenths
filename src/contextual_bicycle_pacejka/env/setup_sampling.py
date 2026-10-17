"""Sampling of the episode setup: the nominal vehicle with a balance index b drawn per episode.

Mass, centre-of-gravity height, steering lag and friction stay at their nominal values; only b varies. The Nominal
policy trains on b = 0 (the default range), the Randomised and Oracle policies on a range of b. Held-out values of b
are kept out of training by excluding open bands around them: a draw that falls in a band is drawn again.
"""

from __future__ import annotations

import numpy as np

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import B_NOM, HCG_NOM, HELD_OUT_HALF_WIDTH, MASS_NOM, MU_NOM, TAU_D_NOM

MAX_DRAWS: int = 1000  # draws of b before an excluded range is declared empty


def held_out_bands(
    centres: tuple[float, ...], half_width: float = HELD_OUT_HALF_WIDTH,
) -> tuple[tuple[float, float], ...]:
    """Return the open bands (h - half_width, h + half_width) around the held-out values ``centres``."""
    return tuple((h - half_width, h + half_width) for h in centres)


def sample_setup(
    rng: np.random.Generator,
    b_range: tuple[float, float] = (B_NOM, B_NOM),
    b_exclude: tuple[tuple[float, float], ...] = (),
) -> SetupVector:
    """Return the nominal setup with b drawn uniformly from ``b_range`` = (lo, hi), outside the open bands
    ``b_exclude`` = ((lo, hi), ...)."""
    for _ in range(MAX_DRAWS):
        b = float(rng.uniform(*b_range))
        if not any(lo < b < hi for lo, hi in b_exclude):
            return SetupVector(m=MASS_NOM, mu=MU_NOM, hcg=HCG_NOM, tau_d=TAU_D_NOM, b=b)
    raise ValueError(f"no b of {b_range} outside {b_exclude}")


def randomized_setup(
    mu: float, rng: np.random.Generator, b_range: tuple[float, float] = (B_NOM, B_NOM),
) -> np.ndarray:
    """Draw a setup with ``sample_setup`` and replace its friction by ``mu``.

    An episode's setup is ``randomized_setup(mu, np.random.default_rng(seed), b_range)``.
    """
    setup = sample_setup(rng, b_range)
    setup.mu = float(mu)
    return setup.to_array()
