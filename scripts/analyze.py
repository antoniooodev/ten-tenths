"""Compare a policy with the teacher on one protocol and write the summary and the failure table.

    python scripts/analyze.py --student outputs/evaluation/all4_spa_R1 --teacher outputs/evaluation/teacher_spa_R1 \
        --output outputs/analysis/all4_spa_R1

``--student`` and ``--teacher`` are result-file prefixes; ``_U.jsonl`` and ``_F.jsonl`` are appended (either may be
missing). Writes <output>.json (see evaluation/analysis.py) and <output>_failures.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import compare, failure_rows, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    student = {c: load_records(f"{args.student}_{c}.jsonl") for c in ("U", "F")}
    teacher = {c: load_records(f"{args.teacher}_{c}.jsonl") for c in ("U", "F")}
    summary = compare(student, teacher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    rows = [row for c in ("U", "F") for row in failure_rows(student[c], Path(args.student).name, c,
                                                             student["F" if c == "U" else "U"], teacher[c])]
    if rows:
        with args.output.with_name(args.output.name + "_failures.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for c, s in summary["student"].items():
        logging.info("student %s: robust %d/%d", c, s["robust"]["k"], s["n"])
    for c, s in summary["teacher"].items():
        logging.info("teacher %s: robust %d/%d", c, s["robust"]["k"], s["n"])


if __name__ == "__main__":
    main()
