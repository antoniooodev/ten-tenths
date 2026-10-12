"""Frozen evaluation protocols: 396 episodes per circuit and racing line, run in parallel and stored as JSONL.

A protocol file (``data/protocols/<circuit>_<line>.json``) lists its episodes in five blocks:

    in_sample_gate   original start, mu in {0.8, 0.9, 1.0, 1.1, 1.2}, setup seeds 42-45            (20)
    seed_only        original start, on-grid mu, held-out setup seeds 100-107                       (40)
    mu_only          original start, off-grid mu in {0.85, 0.95, 1.05, 1.15}, setup seeds 42-45     (16)
    start_only       8 held-out starts, mu in {0.8, 1.2}, setup seeds 42-45                          (64)
    held_out         8 held-out starts x off-grid mu x held-out setup seeds                          (256)

Every episode uses setup = randomized_setup(mu, default_rng(seed)) and reset seed = seed. The result file has one
JSON record per line (see ``episode.run_episode``) and is resumable: episodes already present are skipped.

A reduced protocol (``data/protocols/<circuit>_<line>_reduced.json``, ``reduce_protocol``) is a stratified subset
of 40 episodes, all from held-out starts, meant to be run for several training seeds:

    held_out     for start h_i (i = 1..8) the four off-grid mu, the j-th with setup seed 100 + (i - 1 + j) mod 8  (32)
    start_only   for start h_i, mu 0.8 (i odd) or 1.2 (i even), setup seed 42 + (i - 1) mod 4                    (8)

so every start meets five frictions and every held-out setup seed is used four times. The episodes are copied
from the full protocol, so their identifiers and results are shared with it.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from contextual_bicycle_pacejka.evaluation.episode import run_episode

OFF_GRID_MUS = (0.85, 0.95, 1.05, 1.15)
HELD_OUT_SEEDS = tuple(range(100, 108))
START_ONLY_SEEDS = (42, 43, 44, 45)
HELD_OUT_STARTS = tuple(f"h{i}" for i in range(1, 9))


def reduce_protocol(protocol: dict[str, Any], source_name: str) -> dict[str, Any]:
    """The 40-episode reduced protocol of a full protocol (selection rule in the module docstring)."""
    index = {(e["block"], e["start_label"], round(float(e["mu"]), 3), int(e["seed"])): e for e in protocol["episodes"]}
    episodes = []
    for i, start in enumerate(HELD_OUT_STARTS):
        for j, mu in enumerate(OFF_GRID_MUS):
            episodes.append(index[("held_out", start, mu, HELD_OUT_SEEDS[(i + j) % len(HELD_OUT_SEEDS)])])
    for i, start in enumerate(HELD_OUT_STARTS):
        mu = 0.8 if i % 2 == 0 else 1.2
        episodes.append(index[("start_only", start, mu, START_ONLY_SEEDS[i % len(START_ONLY_SEEDS)])])
    reduced = {key: value for key, value in protocol.items() if key not in ("episodes", "racing_line_path")}
    reduced["blocks"] = {
        "held_out": "start h_i x off-grid mu, the j-th with setup seed 100 + (i - 1 + j) mod 8 (32)",
        "start_only": "start h_i, mu 0.8 (i odd) or 1.2 (i even), setup seed 42 + (i - 1) mod 4 (8)",
    }
    reduced["reduced_from"] = source_name
    reduced["episodes"] = [dict(e) for e in episodes]
    return reduced


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Protocol dictionary with ``racing_line_path`` resolved.

    ``racing_line_file`` is either absolute or relative to the repository holding the protocol
    (``<repository>/data/protocols/<name>.json``).
    """
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    protocol["racing_line_path"] = str(path.parents[2] / protocol["racing_line_file"])
    return protocol


def episode_id(agent: str, protocol: dict[str, Any], spec: dict[str, Any], condition: str) -> str:
    """Unique episode identifier, e.g. ``all4|spa-R1|held_out|U|s3|0.85|101``."""
    line = f"{protocol['circuit']}-{protocol['racing_line']}"
    return f"{agent}|{line}|{spec['block']}|{condition}|{spec['start_label']}|{float(spec['mu']):.2f}|{spec['seed']}"


def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    record = run_episode(job["spec"], job["racing_line"], job["controller"], job["filtered"],
                         model_dir=job["model_dir"], corner_zones=job["corner_zones"],
                         oracle_offset=job["oracle_offset"], mu_estimator_dir=job["mu_estimator"],
                         mu_rise_limit=job["mu_rise_limit"], oracle_clip=job["oracle_clip"],
                         plan_scale=job["plan_scale"])
    return {"id": job["id"], "circuit": job["circuit"], "agent": job["agent"], **record}


def evaluate_protocol(protocol_path: str | Path, agent: str, controller: str, filtered: bool, output: str | Path,
                      model_dir: str | Path | None = None, workers: int = 1, limit: int | None = None,
                      oracle_offset: dict[str, float] | None = None, mu_estimator: str | Path | None = None,
                      mu_rise_limit: float | None = None, oracle_clip: bool = False, plan_scale: float = 1.0,
                      ) -> Path:
    """Run (the first ``limit``) episodes of a protocol for one agent and condition; append records to ``output``.

    ``agent`` labels the records (e.g. ``all4`` or ``teacher``); ``controller`` is ``"policy"`` or ``"teacher"``.
    ``oracle_offset`` adds a constant error to the oracle entries of a diagnostic student policy (clipped to the
    range an estimate can take with ``oracle_clip``), and
    ``mu_estimator`` (a friction-estimator directory) replaces the true mu of its oracle entry by the estimate, whose
    rise ``mu_rise_limit`` limits (per second). ``plan_scale`` multiplies the v_ref preview a student policy reads;
    ``model_dir`` can also hold a plan coach, which chooses that scale itself.
    """
    protocol = load_protocol(protocol_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {json.loads(line)["id"] for line in output.read_text().splitlines() if line.strip()}
    condition = "F" if filtered else "U"
    jobs = []
    for spec in protocol["episodes"][:limit]:
        identifier = episode_id(agent, protocol, spec, condition)
        if identifier not in done:
            jobs.append({"id": identifier, "circuit": protocol["circuit"], "agent": agent, "spec": spec,
                         "racing_line": protocol["racing_line_path"], "controller": controller, "filtered": filtered,
                         "model_dir": None if model_dir is None else str(model_dir),
                         "corner_zones": protocol["corner_zones"], "oracle_offset": dict(oracle_offset or {}),
                         "mu_estimator": None if mu_estimator is None else str(mu_estimator),
                         "mu_rise_limit": mu_rise_limit, "oracle_clip": bool(oracle_clip),
                         "plan_scale": float(plan_scale)})
    with output.open("a") as handle:
        if workers <= 1:
            results = map(_run_one, jobs)
            for record in results:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for record in pool.map(_run_one, jobs):
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
    return output
