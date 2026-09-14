#!/usr/bin/env python3
"""Regenerate config/evaluation_scenarios_v1.csv from the declared nu_R and the
evaluation seed, or verify (--check) that the frozen file is exactly what the
generator produces. Shared layer."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav.env_config import build_arena, build_law, load_common_parameters, load_evaluation_protocol  # noqa: E402
from turtlebot3_drl_nav.initialization import Sampler, generate_scenarios, read_scenarios, write_scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the frozen file instead of writing it")
    parser.add_argument(
        "--package-root",
        default=ROOT,
        help="package root whose configuration, world and frozen scenario are checked",
    )
    args = parser.parse_args()
    package_root = os.path.realpath(args.package_root)
    if not os.path.isdir(package_root):
        parser.error(f"package root is not a directory: {package_root}")
    config_dir = os.path.join(package_root, "config")
    params = load_common_parameters(config_dir)
    protocol = load_evaluation_protocol(config_dir)
    sampler = Sampler(
        build_law(params),
        build_arena(params, os.path.join(package_root, "worlds", f"{params['world_id']}.world")),
    )
    scenarios = generate_scenarios(sampler, int(protocol["evaluation_seed_default"]), int(protocol["scenario_count"]),
                                   int(protocol["physical_subset_count"]), float(params["obstacle_half_period"]))
    path = os.path.join(config_dir, protocol["scenario_file"])
    if args.check:
        frozen = read_scenarios(path)
        same = [s.to_row() for s in frozen] == [s.to_row() for s in scenarios]
        print("frozen scenario file matches the generator" if same else "MISMATCH: frozen scenario file differs from the generator")
        return 0 if same else 1
    write_scenarios(path, scenarios)
    print(f"wrote {len(scenarios)} scenarios to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
