"""Drive the protocol laps of one policy over the grid of the balance index b, for every seed.

    python scripts/evaluate_balance.py --tag oracle_w1_b2
    python scripts/evaluate_balance.py --tag nominal_b2 --grid 0 0.1 --stochastic-repeats 0

Reads <models>/<tag>/s<seed>/stage<k>.zip and writes <output>/<tag>.json. For every seed and b: the laps with the mean
action (``deterministic``) and ``--stochastic-repeats`` sets of laps with the stochastic policy, the action noise
seeded with 1000 seed + repeat (``stochastic``); every lap carries its corner measurements (evaluation/corners.py).
Also the mean inference time of the policy per decision.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_stage import CATALUNYA, protocol_env
from stable_baselines3 import PPO

from contextual_bicycle_pacejka.evaluation.balance_study import GRID
from contextual_bicycle_pacejka.evaluation.corners import CornerRecorder
from contextual_bicycle_pacejka.evaluation.laps import drive_lap, protocol_starts
from contextual_bicycle_pacejka.model.track import RealTrack


def drive_laps(model: PPO, env: Any, track: RealTrack, deterministic: bool) -> tuple[list[dict], float, int]:
    """Drive every protocol lap; return the lap records with their corners, the inference time [s] and decisions."""
    spent = 0.0

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal spent
        start = time.perf_counter()
        action = model.predict(obs, deterministic=deterministic)[0]
        spent += time.perf_counter() - start
        return action

    laps, decisions = [], 0
    for s in protocol_starts(track):
        recorder = CornerRecorder()
        record = drive_lap(policy, env, s, observer=recorder)
        decisions += record.decisions
        laps.append({**dataclasses.asdict(record), "corners": recorder.corners(track.track_length)})
    return laps, spent, decisions


def evaluate(job: tuple[str, int, int, float, int]) -> dict[str, Any]:
    """Drive the laps of one seed at one b (in a worker process)."""
    model_path, seed, stage, b, repeats = job
    torch.set_num_threads(1)
    track = RealTrack(CATALUNYA)
    model = PPO.load(model_path, device="cpu")
    env = protocol_env(model, track, stage, b)
    laps, spent, decisions = drive_laps(model, env, track, deterministic=True)
    stochastic = []
    for repeat in range(repeats):
        torch.manual_seed(1000 * seed + repeat)  # after loading: PPO.load seeds torch with the training seed
        stochastic.append(drive_laps(model, env, track, deterministic=False)[0])
    return {"seed": seed, "b": b, "deterministic": laps, "stochastic": stochastic, "inference_s": spent,
            "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    parser.add_argument("--stochastic-repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/balance"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    jobs = [(str(args.models / args.tag / f"s{seed}" / f"stage{args.stage}.zip"), seed, args.stage, b,
             args.stochastic_repeats) for b in args.grid for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    by_b = {f"{b:+.2f}": [r for r in results if r["b"] == b] for b in args.grid}
    report = {
        "tag": args.tag, "stage": args.stage, "seeds": args.seeds, "grid": args.grid,
        "deterministic": {key: [r["deterministic"] for r in rows] for key, rows in by_b.items()},
        "stochastic": {key: [r["stochastic"] for r in rows] for key, rows in by_b.items()},
        "inference_ms_per_decision": 1000.0 * sum(r["inference_s"] for r in results)
        / max(1, sum(r["decisions"] for r in results)),
    }
    out = args.output / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report))
    for key, rows in report["deterministic"].items():
        done = sum(lap["completed"] for seed_laps in rows for lap in seed_laps)
        logging.info("%s, b %s: %d/%d laps with the mean action", args.tag, key, done, 10 * len(args.seeds))
    logging.info("saved %s", out)


if __name__ == "__main__":
    main()
