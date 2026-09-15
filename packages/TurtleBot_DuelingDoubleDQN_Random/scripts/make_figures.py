#!/usr/bin/env python3
"""Draw Dueling-Double-DQN-only random-arm figures (F-R1 … F-R14) from
completed run directories and the frozen world. Any other algorithm identity
is rejected. Output uses IEEE camera-ready form: Times-like 8 pt text, single
(3.5 in) or double (7.16 in) column widths, vector PDF with Type 42 fonts plus
300 dpi PNG.

    python3 scripts/make_figures.py --results <root> --out <dir> [--formats pdf,png] [--style ieee|default]
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav import analysis  # noqa: E402
from turtlebot3_drl_nav.env_config import build_arena, load_common_parameters  # noqa: E402
from turtlebot3_drl_nav.env_config import load_evaluation_protocol  # noqa: E402

EXPECTED_ALGORITHM = "DuelingDoubleDQN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--formats", default="pdf,png", help="comma-separated: pdf, png, svg, eps")
    parser.add_argument("--style", choices=["ieee", "default"], default="ieee")
    parser.add_argument("--phase-label", choices=("calibration", "pilot", "controlled"), default="controlled")
    parser.add_argument(
        "--expected-algorithms",
        default=EXPECTED_ALGORITHM,
        help="must be exactly DuelingDoubleDQN; an incomplete evidence matrix is fatal",
    )
    args = parser.parse_args()
    expected = [name.strip() for name in args.expected_algorithms.split(",") if name.strip()]
    if expected != [EXPECTED_ALGORITHM]:
        parser.error("--expected-algorithms must be exactly DuelingDoubleDQN")
    os.makedirs(args.out, exist_ok=True)
    analysis.FIGURE_FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    if args.style == "ieee":
        analysis.apply_ieee_style()
    else:
        import matplotlib
        matplotlib.use("Agg")
    params = load_common_parameters(os.path.join(ROOT, "config"))
    arena = build_arena(params, os.path.join(ROOT, "worlds", f"{params['world_id']}.world"))
    amplitude = float(params["obstacle_speed"]) * float(params["obstacle_half_period"])
    scenarios = pd.read_csv(os.path.join(ROOT, "config", "evaluation_scenarios_v1.csv"))
    runs = analysis.discover_runs(args.results, require_complete=not args.include_incomplete)
    runs = [r for r in runs if r.identity["phase_label"] == args.phase_label]
    if not runs:
        raise ValueError(
            f"no {args.phase_label} {EXPECTED_ALGORITHM} runs found under {args.results}"
        )
    analysis.require_campaign_complete(
        runs, load_evaluation_protocol(os.path.join(ROOT, "config")), args.phase_label, expected,
    )
    training = [r for r in runs if r.identity["phase_type"] == "training"]
    evaluation_all = [r for r in runs if r.identity["phase_type"] == "evaluation"]
    evaluation = analysis.select_final_evaluation_runs(evaluation_all)
    out = lambda name: os.path.join(args.out, name)  # noqa: E731
    analysis.figure_learning_curves(analysis.learning_curves(training), out("F-R1_learning_curves"))
    analysis.figure_training_window(training, out("F-R2_training_window"))
    analysis.figure_performance_profiles(evaluation, out("F-R3_performance_profiles"))
    analysis.figure_poi_heatmap(evaluation, out("F-R4_probability_of_improvement"))
    analysis.figure_spatial_success(evaluation, arena, amplitude, out("F-R5_spatial_success"))
    analysis.figure_trajectories(evaluation, arena, amplitude, out("F-R6_trajectories"))
    analysis.figure_stratified_success(evaluation, out("F-R7_stratified_success"))
    analysis.figure_reward_composition(training, out("F-R8_reward_composition"))
    analysis.figure_diagnostics(training, out("F-R9_learner_diagnostics"))
    analysis.figure_initialization_coverage(training, arena, amplitude, scenarios, out("F-R10_initialization_coverage"))
    analysis.figure_outcome_composition(training, out("F-R11_outcome_composition"))
    analysis.figure_checkpointwise_heldout(evaluation_all, out("F-R12_checkpointwise_heldout"))
    analysis.figure_efficiency_distributions(evaluation, out("F-R13_efficiency"))
    analysis.figure_timing(training, out("F-R14_timing"))
    print(f"figures written to {args.out}: {sorted(os.listdir(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
