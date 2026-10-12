"""Run a frozen evaluation protocol (396 episodes) for a policy or the teacher, unfiltered (U) or filtered (F).

    python scripts/evaluate.py --protocol data/protocols/spa_R1.json --model models/all4 --condition U --workers 8
    python scripts/evaluate.py --protocol data/protocols/spa_R1.json --teacher --condition F --workers 8
    python scripts/evaluate.py --protocol data/protocols/monza_R0_reduced.json \
        --model outputs/models/phase2/S2_label_mu_s0 --condition U --oracle-offset mu=-0.05 --workers 8

``--oracle-offset NAME=VALUE`` adds a constant error to an oracle entry of a diagnostic student (sensitivity to an
estimated value); the agent label gets the suffix ``_<NAME><VALUE>``, e.g. ``S2_label_mu_s0_mu-0.05``. With
``--oracle-clip`` the entry is then clipped to the range an estimate can take (mu in [0.8, 1.2]: a friction
estimator never leaves it), and the label gets ``_clip``.
``--mu-estimator DIR`` replaces the true mu of the oracle entry by the estimate of a friction estimator
(scripts/train_friction.py); the agent label gets the suffix ``_muhat``. ``--mu-rise-limit R`` limits the rise of
the estimate to R per second (suffix ``_rise<R>``, e.g. ``_muhat_rise0.05``).
``--plan-scale C`` multiplies the v_ref preview a student reads by C (a constant scale of the teacher's plan; suffix
``_plan<C>``, e.g. ``_plan1.08``). ``--model`` can also be a plan coach (scripts/train_plan_coach.py), which
chooses the scale itself and drives with its executor policy.

Results are appended to outputs/evaluation/<agent>_<protocol>_<condition>.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from contextual_bicycle_pacejka.evaluation.analysis import load_records, summarise
from contextual_bicycle_pacejka.evaluation.protocol import evaluate_protocol

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", type=Path, help="policy directory (policy.zip + vecnormalize.pkl) or plan coach")
    who.add_argument("--teacher", action="store_true", help="evaluate the MATv0 teacher")
    parser.add_argument("--condition", choices=("U", "F"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="run only the first LIMIT episodes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oracle-offset", nargs="*", default=[], metavar="NAME=VALUE",
                        help="constant error added to an oracle entry of a diagnostic student")
    parser.add_argument("--oracle-clip", action="store_true",
                        help="clip the oracle entries to the range an estimate can take (after the offset)")
    parser.add_argument("--mu-estimator", type=Path, help="friction estimator replacing the true mu of the oracle")
    parser.add_argument("--mu-rise-limit", type=float, help="largest rise of the estimate per second")
    parser.add_argument("--plan-scale", type=float, default=1.0, help="scale of the v_ref preview a student reads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    offset = {name: float(value) for name, value in (item.split("=", 1) for item in args.oracle_offset)}
    agent = "teacher" if args.teacher else args.model.name + "".join(f"_{k}{v:+.2f}" for k, v in offset.items())
    if args.oracle_clip and not args.teacher:
        agent += "_clip"
    if args.mu_estimator is not None and not args.teacher:
        agent += "_muhat"
        if args.mu_rise_limit is not None:
            agent += f"_rise{args.mu_rise_limit:.2f}"
    if args.plan_scale != 1.0 and not args.teacher:
        agent += f"_plan{args.plan_scale:.2f}"
    output = args.output or REPO / "outputs/evaluation" / f"{agent}_{args.protocol.stem}_{args.condition}.jsonl"
    evaluate_protocol(args.protocol, agent, "teacher" if args.teacher else "policy", args.condition == "F", output,
                      model_dir=args.model, workers=args.workers, limit=args.limit, oracle_offset=offset,
                      mu_estimator=args.mu_estimator, mu_rise_limit=args.mu_rise_limit, oracle_clip=args.oracle_clip,
                      plan_scale=args.plan_scale)
    summary = summarise(load_records(output), args.condition == "F")
    logging.info("%s: robust %d/%d, completed %d/%d -> %s", agent, summary["robust"]["k"], summary["n"],
                 summary["completed"]["k"], summary["n"], output)


if __name__ == "__main__":
    main()
