"""Shared test helpers: repository paths and access to the golden reference data.

The files in ``tests/golden/`` were recorded from the original implementation of this project (before the
reorganisation) and pin its numerical behaviour: plant, teacher, safety filter, demonstrations, behaviour
cloning, DAgger, evaluation and analysis must reproduce them exactly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "src"))
th.set_num_threads(1)

# File paths recorded inside the golden data (original layout) -> files of this repository.
_LEGACY_PATHS = {entry["legacy"]: entry["new"] for entry in json.loads((GOLDEN / "legacy_paths.json").read_text())}


def repo_path(legacy_relative_path: str) -> Path:
    """Map a file path recorded in the golden data to the corresponding file of this repository."""
    return REPO / _LEGACY_PATHS[legacy_relative_path]


def golden_json(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text())


def golden_npz(name: str) -> np.lib.npyio.NpzFile:
    return np.load(GOLDEN / f"{name}.npz", allow_pickle=False)


LEGACY_MODEL_DIRS = {
    "TreatmentA": "treatment_a",
    "ALL4": "all4",
    "LOCO_spa": "loco_spa",
    "LOCO_catalunya": "loco_catalunya",
    "LOCO_budapest": "loco_budapest",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    datasets_present = (REPO / "datasets" / "all4.npz").exists()
    skip = pytest.mark.skip(reason="training datasets not present in datasets/")
    for item in items:
        if "datasets" in item.keywords and not datasets_present:
            item.add_marker(skip)


def array_digest(array: np.ndarray) -> str:
    """SHA-256 of an array's bytes, dtype and shape (the format of the ``*_hashes.json`` golden files)."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes() + str(array.dtype).encode() + str(array.shape).encode()).hexdigest()
