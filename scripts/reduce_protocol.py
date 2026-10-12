"""Write the 40-episode reduced protocol of a frozen protocol (see evaluation/protocol.py).

    python scripts/reduce_protocol.py --protocol data/protocols/monza_R0.json

Writes data/protocols/<name>_reduced.json next to the source protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.protocol import reduce_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="output file (default: <protocol>_reduced.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reduced = reduce_protocol(json.loads(args.protocol.read_text()), args.protocol.name)
    output = args.output or args.protocol.with_name(f"{args.protocol.stem}_reduced.json")
    output.write_text(json.dumps(reduced, indent=1))
    logging.info("%s: %d episodes -> %s", args.protocol.name, len(reduced["episodes"]), output)


if __name__ == "__main__":
    main()
