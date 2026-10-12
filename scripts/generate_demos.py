"""Generate the teacher demonstration laps of one circuit and perturbation kind.

    python scripts/generate_demos.py --circuit spa --kind speed
    python scripts/generate_demos.py --circuit monza --kind speed --speed-safety 1.4 --keep-failed
    python scripts/generate_demos.py --circuit monza --kind none --exec-speed-safety 1.4 --keep-failed
    python scripts/generate_demos.py --circuit monza --kind speed_wide --keep-failed --max-attempts 3

Laps are written to outputs/demos/<circuit>/<kind>/lapNN.npz with a generation.json report (kept attempt per lap).
With --keep-failed the attempts that are not kept are saved too (lapNN_attemptKK.npz), for the offline datasets.
A teacher speed factor other than the certified one writes to outputs/demos/<circuit>/<kind>_k<factor>/; it changes
both the labels and the executed actions. --exec-speed-safety changes only the teacher that drives the lap (the
labels stay those of the certified teacher) and writes to outputs/demos/<circuit>/<kind>_x<factor>/.
Demonstrations are always generated on the R0 racing line.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.learning.demonstrations import (
    MAX_ATTEMPTS,
    PERTURBATIONS,
    episode_specs,
    generate_laps,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", required=True, choices=("monza", "spa", "catalunya", "budapest"))
    parser.add_argument("--kind", required=True, choices=sorted(PERTURBATIONS))
    parser.add_argument("--laps", type=int, nargs="+", help="lap indices (default: all 20)")
    parser.add_argument("--speed-safety", type=float, help="teacher speed factor k (default: the certified one)")
    parser.add_argument("--exec-speed-safety", type=float,
                        help="speed factor k of the teacher that drives the lap (default: the labelling one)")
    parser.add_argument("--keep-failed", action="store_true", help="also save the attempts that are not kept")
    parser.add_argument("--max-attempts", type=int, help=f"attempts per lap (default: {MAX_ATTEMPTS})")
    parser.add_argument("--out", type=Path, help="output directory (default: outputs/demos/<circuit>/<kind>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    racing_line = REPO / "data/tracks" / args.circuit / "racing_line_R0.npz"
    teacher = MATv0.from_config(REPO / "data/teacher/matv0.json")
    folder = args.kind
    if args.speed_safety is not None and args.speed_safety != teacher.speed_safety:
        teacher = dataclasses.replace(teacher, speed_safety=args.speed_safety)
        folder = f"{args.kind}_k{args.speed_safety:.2f}"
    exec_teacher = None
    if args.exec_speed_safety is not None and args.exec_speed_safety != teacher.speed_safety:
        exec_teacher = dataclasses.replace(teacher, speed_safety=args.exec_speed_safety)
        folder = f"{folder}_x{args.exec_speed_safety:.2f}"
    out = args.out or REPO / "outputs/demos" / args.circuit / folder
    report = generate_laps(racing_line, args.circuit, args.kind, out, episode_specs(racing_line), args.laps, teacher,
                           keep_failed=args.keep_failed, exec_teacher=exec_teacher, max_attempts=args.max_attempts)
    kept = sum(lap["kept_attempt"] is not None for lap in report["laps"])
    failed = sum(len(lap.get("failed_files", [])) for lap in report["laps"])
    logging.info("%s/%s: %d of %d laps kept, %d failed attempts saved -> %s", args.circuit, folder, kept,
                 len(report["laps"]), failed, out)


if __name__ == "__main__":
    main()
