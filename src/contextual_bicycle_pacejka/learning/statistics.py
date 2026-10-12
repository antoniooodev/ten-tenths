"""Column statistics and standardisation of large row arrays, by chunks of rows.

A multi-circuit dataset holds several million rows: a float64 copy of them, which the whole-array numpy expressions
make, does not fit in memory. These functions work by chunks and give the same numbers bit for bit. The sums run
over the rows in order, as numpy's reduction along the first axis, and the element-wise operations do not depend on
the chunking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

CHUNK_ROWS = 65_536


def column_moments(rows: NDArray[np.floating], chunk: int = CHUNK_ROWS,
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and variance of the columns of ``rows`` in float64: ``np.mean`` and ``np.var`` of the float64 rows
    (axis 0), without a float64 copy of the rows."""
    def column_sum(term: Any) -> NDArray[np.float64]:
        total = None
        for start in range(0, len(rows), chunk):
            part = term(np.asarray(rows[start:start + chunk], dtype=np.float64))
            total = np.add.reduce(part if total is None else np.concatenate([total[None], part]), axis=0)
        return total

    mean = column_sum(lambda part: part) / len(rows)
    return mean, column_sum(lambda part: (part - mean) ** 2) / len(rows)


def standardised(rows: NDArray[np.floating], mean: NDArray[np.float64], std: NDArray[np.float64],
                 chunk: int = CHUNK_ROWS) -> NDArray[np.float32]:
    """``((rows - mean) / std).astype(np.float32)``, without a float64 copy of the rows."""
    out = np.empty(np.shape(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        out[start:start + chunk] = (rows[start:start + chunk] - mean) / std
    return out
