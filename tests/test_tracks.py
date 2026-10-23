"""Track preparation: the stored centreline reference is reproduced from its TUM centreline CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import REPO

from contextual_bicycle_pacejka.tracks.reference import build_reference

TRACKS = REPO / "data/tracks"


def test_reference_from_tum_csv(tmp_path: Path) -> None:
    out = build_reference(TRACKS / "catalunya/tum_centerline.csv", tmp_path / "reference.npz")
    with np.load(out) as produced, np.load(TRACKS / "catalunya/reference.npz") as stored:
        assert sorted(produced.files) == sorted(stored.files)
        for key in produced.files:
            np.testing.assert_array_equal(produced[key], stored[key], err_msg=key)
