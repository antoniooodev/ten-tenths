"""Measure whether agents copy the teacher on one protocol (evaluation/independence.py) and write the measures.

    python scripts/analyze_independence.py --protocol data/protocols/monza_R0_reduced.json \
        --agent copy=outputs/models/multi/S2_td3bc_mu_all4_s0 \
        --agent scaled_copy=outputs/models/multi/S2_td3bc_mu_all4_s0@1.08 \
        --agent coach_s1=outputs/models/plan_coach/P_monza_s1 \
        --output outputs/analysis/independence_monza_R0_reduced.json --workers 8

Every agent and the teacher drive every episode of the protocol (condition U) with their speed traces recorded;
``NAME=DIR@C`` drives the policy in DIR on the teacher's plan scaled by C (``--plan-scale`` of scripts/evaluate.py).
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from contextual_bicycle_pacejka.evaluation.independence import independence, speed_trace
from contextual_bicycle_pacejka.evaluation.protocol import load_protocol


def _trace(job: tuple[str, dict[str, Any], str | None, float]) -> dict[str, Any]:
    racing_line, spec, model_dir, plan_scale = job
    return speed_trace(spec, racing_line, model_dir, plan_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--agent", action="append", required=True, metavar="NAME=DIR[@SCALE]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    protocol = load_protocol(args.protocol)
    racing_line, episodes = protocol["racing_line_path"], protocol["episodes"]
    with np.load(racing_line, allow_pickle=False) as data:
        length = float(data["track_length"])
    agents: dict[str, tuple[str | None, float]] = {"teacher": (None, 1.0)}
    for item in args.agent:
        name, target = item.split("=", 1)
        model_dir, _, scale = target.partition("@")
        agents[name] = (model_dir, float(scale) if scale else 1.0)
    jobs = [(racing_line, spec, model_dir, scale) for model_dir, scale in agents.values() for spec in episodes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        traces = list(pool.map(_trace, jobs))
    by_agent = {name: traces[i * len(episodes):(i + 1) * len(episodes)] for i, name in enumerate(agents)}
    out = {"protocol": str(args.protocol), "agents": {name: {"model_dir": model_dir, "plan_scale": scale}
                                                      for name, (model_dir, scale) in agents.items()},
           "measures": {name: independence(by_agent[name], by_agent["teacher"], length)
                        for name in agents if name != "teacher"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for name, m in out["measures"].items():
        logging.info("%s: robust %d/%d, shape residual %.4f (k %.3f), sector gain %.3f +/- %.3f, slower in %.0f %% of "
                     "sectors, ratio spread over mu %.4f, failures at teacher sites %d/%d, teacher failures survived "
                     "%d/%d", name, m["robust"], m["n"], m["shape_residual"] or 0.0, m["shape_k"] or 0.0,
                     m["sector_gain_mean"] or 0.0, m["sector_gain_std"] or 0.0,
                     100 * (m["sector_slower_share"] or 0.0), m["ratio_spread"] or 0.0,
                     m["failures_at_teacher_sites"], m["failures"], m["teacher_failures_survived"],
                     m["teacher_failures"])


if __name__ == "__main__":
    main()
