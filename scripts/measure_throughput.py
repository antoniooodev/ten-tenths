"""Measure the decisions per second of the environment with random actions.

    python scripts/measure_throughput.py --output outputs/analysis/throughput.json

For each stage: one environment in this process, then ``--processes`` subprocesses holding ``--envs-per-process``
environments each (``BatchedSubprocVecEnv``). The target is at least 40,000 decisions/s with 10 processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from contextual_bicycle_pacejka.env import make_vec_env

CATALUNYA = "data/tracks/catalunya/reference.npz"


def decisions_per_second(n_envs: int, workers: int, stage: int, decisions: int) -> float:
    """Return the decisions per second of ``n_envs`` environments in ``workers`` subprocesses (0: this process),
    stepped together for ``decisions`` in all."""
    vec_env = make_vec_env(CATALUNYA, n_envs, seed=0, stage=stage, workers=workers)
    rng = np.random.default_rng(0)
    vec_env.reset()
    steps = max(1, decisions // n_envs)
    start = time.perf_counter()
    for _ in range(steps):
        vec_env.step(rng.uniform(-1.0, 1.0, size=(n_envs, 2)).astype(np.float32))
    elapsed = time.perf_counter() - start
    vec_env.close()
    return steps * n_envs / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--envs-per-process", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--decisions", type=int, default=200_000, help="decisions per measurement, in all")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/throughput.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = {}
    for stage in args.stages:
        single = decisions_per_second(1, 0, stage, args.decisions // 10)
        report[f"stage_{stage}"] = {"one_process": single}
        logging.info("stage %d: %.0f decisions/s in one process", stage, single)
        for per_process in args.envs_per_process:
            parallel = decisions_per_second(args.processes * per_process, args.processes, stage, args.decisions)
            report[f"stage_{stage}"][f"{args.processes}_processes_x_{per_process}"] = parallel
            logging.info("stage %d: %.0f decisions/s with %d processes x %d environments", stage, parallel,
                         args.processes, per_process)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
