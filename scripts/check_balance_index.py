"""Measure the balance index b on the plant and apply the criterion of ``evaluation/balance.py``.

    python scripts/check_balance_index.py --output outputs/analysis/balance_index.json

Writes the measurements per b and the criterion (see evaluation/balance.py) and logs them as a table.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.balance import check_balance_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/balance_index.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = check_balance_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    logging.info("    b   K [deg/g]  K lin  ay_max [g]  axle   v_crit [m/s]")
    for row in report["rows"]:
        v_crit = row["critical_speed_mps"]
        logging.info("%+6.2f  %+8.3f  %+6.3f  %8.4f   %-5s  %s", row["b"], row["K_deg_per_g"],
                     row["K_linear_deg_per_g"], row["ay_max_g"], row["saturating_axle"],
                     "-" if v_crit is None else f"{v_crit:.1f}")
    criterion = report["criterion"]
    logging.info("monotonic K: %s; saturating axle: %s", criterion["monotonic_K"]["passed"],
                 criterion["saturating_axle"]["passed"])
    for speed, (lo, hi) in criterion["drivable"].items():
        logging.info("drivable at %s m/s: [%+.2f, %+.2f]", speed, lo, hi)


if __name__ == "__main__":
    main()
