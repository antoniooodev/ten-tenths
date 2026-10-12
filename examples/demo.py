"""Demonstration of the trained policies on every circuit, with the safety filter, the teacher and friction changes.

Every episode is a frozen-protocol episode (``data/protocols``) run by the evaluation code itself
(``evaluation.episode.run_episode``), so the numbers are those of the reported evaluation.

    python examples/demo.py                          # all scenarios below (about 3 minutes)
    python examples/demo.py --scenario circuits      # one policy, four circuits
    python examples/demo.py --max-steps 2000         # quick run: every episode stopped after 20 s
    python examples/demo.py --circuit spa --policy loco_spa --condition F --mu 0.8 --seed 43   # one episode
    python examples/demo.py --circuit spa --policy loco_spa --line R0 --start h8 --mu 1.2   # held-out start h8

Scenarios:

    circuits    ALL4 policy (trained on all four circuits) on Monza, Spa, Catalunya, Budapest
    zero-shot   each leave-one-circuit-out policy on the circuit it has never seen
    filter      ALL4 unfiltered (U) vs safety-filtered (F), and the teacher it imitates (F), on Catalunya
    friction    ALL4 on Monza at tyre-road friction 0.8, 1.0 and 1.2 (the policy observes the friction)
    racing-line LOCO Spa on Spa from a held-out start, on the original racing line R0 (curvature kink at La Source)
                and on the repaired line R1

Columns: outcome (robust lap or termination reason), lap time, lap fraction covered, filter projections (F only),
mean |speed error| and lateral RMS error with respect to the teacher's target, and the 99th percentile of the
rear slip ratio |alpha_r| / alpha_sat_r.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch as th  # noqa: E402

from contextual_bicycle_pacejka.control.teacher import MATv0  # noqa: E402
from contextual_bicycle_pacejka.evaluation.episode import run_episode  # noqa: E402
from contextual_bicycle_pacejka.evaluation.protocol import load_protocol  # noqa: E402

CIRCUITS = ("monza", "spa", "catalunya", "budapest")
DEFAULT_LINE = {"monza": "R1", "spa": "R1", "catalunya": "R1", "budapest": "R0"}  # Budapest has no kink
POLICIES = ("all4", "treatment_a", "loco_spa", "loco_catalunya", "loco_budapest", "ppo_pilot_all4_monza")
TRAINED_ON = {"all4": "all four circuits", "treatment_a": "Monza only", "loco_spa": "all except Spa",
              "loco_catalunya": "all except Catalunya", "loco_budapest": "all except Budapest",
              "ppo_pilot_all4_monza": "all four circuits, then PPO on Monza (pilot)"}


@dataclass(frozen=True)
class Episode:
    circuit: str
    controller: str  # a policy name or "teacher"
    condition: str = "U"
    mu: float = 1.0
    seed: int = 42
    line: str = ""
    start: str = "orig"  # protocol start label: "orig" or a held-out start "h1".."h8"

    @property
    def racing_line(self) -> str:
        return self.line or DEFAULT_LINE[self.circuit]


SCENARIOS: dict[str, list[Episode]] = {
    "circuits": [Episode(c, "all4") for c in CIRCUITS],
    "zero-shot": [Episode(c, f"loco_{c}") for c in ("spa", "catalunya", "budapest")],
    "filter": [Episode("catalunya", "all4", "U"), Episode("catalunya", "all4", "F"),
               Episode("catalunya", "teacher", "F")],
    "friction": [Episode("monza", "all4", mu=mu) for mu in (0.8, 1.0, 1.2)],
    "racing-line": [Episode("spa", "loco_spa", mu=1.2, line="R0", start="h8"),
                    Episode("spa", "loco_spa", mu=1.2, line="R1", start="h8")],
}


def protocol_episode(circuit: str, line: str, start: str, mu: float, seed: int) -> tuple[dict, dict]:
    """(protocol, spec) of the episode with start label ``start``, friction ``mu`` and setup seed ``seed``."""
    path = REPO / "data/protocols" / f"{circuit}_{line}.json"
    if not path.exists():
        raise SystemExit(f"no protocol for {circuit} {line}")
    protocol = load_protocol(path)
    for spec in protocol["episodes"]:
        if spec["start_label"] == start and abs(spec["mu"] - mu) < 1e-9 and spec["seed"] == seed:
            return protocol, spec
    raise SystemExit(f"{circuit} {line}: no protocol episode with start={start}, mu={mu}, seed={seed} "
                     "(see the blocks of data/protocols/*.json)")


def run(episode: Episode, max_steps: int | None, print_every: int) -> dict:
    """Run one episode and return its record (see ``evaluation.episode``)."""
    protocol, spec = protocol_episode(episode.circuit, episode.racing_line, episode.start, episode.mu, episode.seed)
    if max_steps:
        spec = dict(spec, max_steps=min(int(spec["max_steps"]), max_steps))
    is_teacher = episode.controller == "teacher"
    length = protocol["track_length_m"]

    def report(step: dict) -> None:
        if print_every and step["step"] % print_every == 0:
            state = step["state"]
            print(f"      t={step['step'] / 100:6.1f} s  progress {step['progress'] / length:6.1%}  "
                  f"vx {state[3]:5.1f} m/s  ey {state[1]:+5.2f} m", flush=True)

    return run_episode(spec, protocol["racing_line_path"], "teacher" if is_teacher else "policy",
                       episode.condition == "F", model_dir=None if is_teacher else REPO / "models" / episode.controller,
                       teacher=MATv0.from_config(REPO / "data/teacher/matv0.json") if is_teacher else None,
                       corner_zones=protocol["corner_zones"], on_step=report)


def row(episode: Episode, record: dict) -> str:
    d = record["diagnostics"]
    outcome = "robust lap" if record["robust"] else record["termination_reason"]
    lap_time = f"{record['lap_time_s']:7.2f}" if record["lap_time_s"] else "      -"
    projections = str(record["projected_steps"]) if episode.condition == "F" else "-"
    return (f"{episode.circuit:<10}{episode.racing_line:<5}{episode.controller:<16}{episode.condition:<3}"
            f"{episode.mu:<5.2f}{outcome:<17}{lap_time}  {record['lap_fraction']:6.1%}  {projections:>5}  "
            f"{d['speed_error_mean_abs_mps']:6.2f}  {d['ey_rms_m']:5.2f}  {d['rear_slip_ratio_p99']:5.2f}")


HEADER = (f"{'circuit':<10}{'line':<5}{'controller':<16}{'c':<3}{'mu':<5}{'outcome':<17}{'time[s]':>7}  "
          f"{'lap':>6}  {'proj':>5}  {'|ev|':>6}  {'ey':>5}  {'ar99':>5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--circuit", choices=CIRCUITS, help="run a single episode on this circuit")
    parser.add_argument("--policy", choices=[*POLICIES, "teacher"], default="all4")
    parser.add_argument("--condition", choices=("U", "F"), default="U")
    parser.add_argument("--line", choices=("R0", "R1"))
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="orig", help="protocol start label: orig (default) or h1..h8")
    parser.add_argument("--max-steps", type=int, help="stop every episode after this many 10 ms steps")
    parser.add_argument("--print-every", type=int, default=2000, help="progress line every N steps (0: none)")
    parser.add_argument("--json", type=Path, help="also write the episode records to this file")
    args = parser.parse_args()
    th.set_num_threads(1)

    if args.circuit:
        plan = {"episode": [Episode(args.circuit, args.policy, args.condition, args.mu, args.seed, args.line or "",
                                    args.start)]}
    else:
        plan = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}

    rows, records = [], []
    for name, episodes in plan.items():
        print(f"\n== {name}", flush=True)
        for episode in episodes:
            trained = "" if episode.controller == "teacher" else f" (trained on {TRAINED_ON[episode.controller]})"
            print(f"   {episode.controller}{trained} on {episode.circuit} {episode.racing_line}, "
                  f"{'filtered' if episode.condition == 'F' else 'unfiltered'}, mu={episode.mu}, seed={episode.seed}, "
                  f"start {episode.start}",
                  flush=True)
            start = time.perf_counter()
            record = run(episode, args.max_steps, args.print_every)
            zone = "" if record["robust"] else f" in zone {record['diagnostics'].get('terminal_zone')}"
            print(f"      -> {record['termination_reason']}{zone} after {record['steps']} steps "
                  f"({time.perf_counter() - start:.0f} s wall)", flush=True)
            rows.append(f"{name:<12}" + row(episode, record))
            records.append({"scenario": name, **episode.__dict__, "record": record})

    print("\n" + f"{'scenario':<12}" + HEADER)
    print("\n".join(rows))
    if args.json:
        args.json.write_text(json.dumps(records, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
