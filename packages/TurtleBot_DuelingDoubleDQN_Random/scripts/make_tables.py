#!/usr/bin/env python3
"""Build Dueling-Double-DQN-only random-arm tables (T-R1 … T-R9) from
completed run directories as CSV (data), Markdown (reading) and LaTeX booktabs
(IEEE camera-ready). Any other algorithm identity is rejected.

    python3 scripts/make_tables.py --results <root> --out <dir>

<root> is searched recursively for COMPLETE runs; training runs (phase_type
training) feed the in-run E1 tables, evaluation runs (phase_type evaluation)
feed the held-out E2/E3 tables.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav import analysis  # noqa: E402
from turtlebot3_drl_nav.env_config import load_evaluation_protocol  # noqa: E402

EXPECTED_ALGORITHM = "DuelingDoubleDQN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-incomplete", action="store_true", help="also read runs without a COMPLETE marker (diagnostics only)")
    parser.add_argument("--phase-label", choices=("calibration", "pilot", "controlled"), default="controlled")
    parser.add_argument(
        "--expected-algorithms",
        default=EXPECTED_ALGORITHM,
        help="must be exactly DuelingDoubleDQN; missing/duplicate seeds and tier-2 checkpoints are fatal",
    )
    args = parser.parse_args()
    expected = [name.strip() for name in args.expected_algorithms.split(",") if name.strip()]
    if expected != [EXPECTED_ALGORITHM]:
        parser.error("--expected-algorithms must be exactly DuelingDoubleDQN")
    runs = analysis.discover_runs(args.results, require_complete=not args.include_incomplete)
    runs = [r for r in runs if r.identity["phase_label"] == args.phase_label]
    if not runs:
        raise ValueError(
            f"no {args.phase_label} {EXPECTED_ALGORITHM} runs found under {args.results}"
        )
    protocol = load_evaluation_protocol(os.path.join(ROOT, "config"))
    report = analysis.require_campaign_complete(runs, protocol, args.phase_label, expected)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "campaign_completeness.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    training = [r for r in runs if r.identity["phase_type"] == "training"]
    evaluation = analysis.select_final_evaluation_runs([r for r in runs if r.identity["phase_type"] == "evaluation"])
    print(f"{len(training)} training runs, {len(evaluation)} evaluation runs")
    curves = analysis.learning_curves(training)
    analysis.write_table(analysis.table_main(training, evaluation), args.out, "T-R1_main_results")
    analysis.write_table(analysis.table_efficiency(curves), args.out, "T-R2_learning_efficiency")
    analysis.write_table(analysis.table_generalization(training, evaluation), args.out, "T-R3_generalization")
    analysis.write_table(analysis.table_pairwise(evaluation), args.out, "T-R4_probability_of_improvement")
    analysis.write_table(analysis.table_failure_anatomy(evaluation), args.out, "T-R5_failure_anatomy")
    analysis.write_table(analysis.table_per_seed(training, evaluation), args.out, "T-R6_per_seed")
    analysis.write_table(analysis.table_ledger(runs), args.out, "T-R7_run_ledger")
    analysis.write_table(analysis.table_learner_diagnostics(training), args.out, "T-R8_learner_diagnostics")
    analysis.write_table(analysis.table_initialization(training), args.out, "T-R9_initialization")
    if not curves.empty:
        curves.to_csv(os.path.join(args.out, "learning_curves_E1.csv"), index=False)
    print(f"tables written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
