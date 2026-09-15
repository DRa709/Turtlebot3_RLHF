#!/usr/bin/env python3
"""Read preregistered phase facts from config/common_environment.yaml for the
Slurm scripts (run with the container's Python; the host needs no packages).

    phase_query.py <config.yaml> seed   <phase> <array_index>   -> learning seed
    phase_query.py <config.yaml> budget <phase>                 -> environment budget
    phase_query.py <config.yaml> tier2-steps <phase>            -> allowed checkpoint steps
    phase_query.py <config.yaml> common-seeds                   -> "world init obstacle evaluation"
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav.env_config import load_evaluation_protocol  # noqa: E402


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    protocol = load_evaluation_protocol(os.path.dirname(os.path.abspath(argv[0])))
    command = argv[1]
    if command == "seed":
        phase, index = argv[2], int(argv[3])
        seeds = [int(v) for v in protocol["phases"][phase]["learning_seeds"]]
        if index >= len(seeds):
            print(f"array index {index} exceeds the {len(seeds)} preregistered seeds of phase {phase}", file=sys.stderr)
            return 1
        print(seeds[index])
    elif command == "budget":
        print(int(protocol["phases"][argv[2]]["environment_budget"]))
    elif command == "tier2-steps":
        print(" ".join(str(int(value)) for value in protocol["phases"][argv[2]]["tier2_checkpoint_steps"]))
    elif command == "common-seeds":
        c = protocol["common_seeds"]
        print(c["world_seed"], c["initialization_seed"], c["dynamic_obstacle_seed"], c["evaluation_seed"])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
