"""Drive the protocol laps of a speed stage with the models of several seeds and apply the stage criterion.

    python scripts/evaluate_stage.py --stage 1 --seeds 0 1 2 3 4

Reads <models>/<tag>/s<seed>/stage<k>.zip and writes <output>/<tag>/stage<k>.json: every lap record, the criterion
(evaluation/stage_gate.py) and the point-mass lap time of the stage as a yardstick.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.evaluation.laps import drive_lap, point_mass_lap_time, protocol_starts
from contextual_bicycle_pacejka.evaluation.stage_gate import stage_gate
from contextual_bicycle_pacejka.model.track import RealTrack

CATALUNYA = "data/tracks/catalunya/reference.npz"


def drive_seed(model_path: str, stage: int) -> list[dict]:
    """Drive every protocol lap with one model (in a worker process)."""
    torch.set_num_threads(1)
    track = RealTrack(CATALUNYA)
    env = RacingEnv(track, make_stage(stage, track))
    model = PPO.load(model_path, device="cpu")

    def policy(obs: np.ndarray) -> np.ndarray:
        return model.predict(obs, deterministic=True)[0]

    return [dataclasses.asdict(drive_lap(policy, env, s)) for s in protocol_starts(track)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tag", default="nominal")
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = [str(args.models / args.tag / f"s{seed}" / f"stage{args.stage}.zip") for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=len(paths)) as pool:
        laps = list(pool.map(drive_seed, paths, [args.stage] * len(paths)))
    completed = np.array([[lap["completed"] for lap in seed_laps] for seed_laps in laps])
    times = np.array([[lap["lap_time"] if lap["completed"] else np.nan for lap in seed_laps] for seed_laps in laps])
    track = RealTrack(CATALUNYA)
    report = {"stage": args.stage, "seeds": args.seeds, "laps": laps, "criterion": stage_gate(completed, times),
              "point_mass_lap_time": point_mass_lap_time(track, make_stage(args.stage, track))}
    out = args.output / args.tag / f"stage{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    completion = report["criterion"]["completion"]
    logging.info("stage %d: %d/%d laps, rate %.3f, interval [%.3f, %.3f]; seeds passing %d, passed %s", args.stage,
                 completed.sum(), completed.size, completion["rate"], *completion["interval"],
                 report["criterion"]["seeds_passing"], report["criterion"]["passed"])
    logging.info("lap time: %s (point mass %.1f s)", report["criterion"]["lap_time"], report["point_mass_lap_time"])


if __name__ == "__main__":
    main()
