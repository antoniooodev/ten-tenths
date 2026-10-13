"""Shared test helpers: repository paths and access to the golden reference data.

The files in ``tests/golden/`` pin the numerical behaviour: the reference rollouts of the plant
(``plant_catalunya.npz``, see ``test_plant.py``) and the result analysis recorded from the original implementation
(``analysis_*.json``) must be reproduced exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "src"))


def golden_json(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text())


def golden_npz(name: str) -> np.lib.npyio.NpzFile:
    return np.load(GOLDEN / f"{name}.npz", allow_pickle=False)
