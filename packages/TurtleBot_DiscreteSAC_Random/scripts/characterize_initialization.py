#!/usr/bin/env python3
"""Deterministic Monte Carlo characterization of the declared start support.

This is descriptive only: training samples with the exact rejection sampler in
``initialization.py``.  The script makes the area figure in
``RANDOM_INIT_SPEC.md`` reproducible from the authenticated package.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from turtlebot3_drl_nav.env_config import build_arena, build_law, load_common_parameters  # noqa: E402
from turtlebot3_drl_nav.initialization import Sampler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.samples <= 0 or args.seed < 0:
        parser.error("--samples must be positive and --seed non-negative")
    params = load_common_parameters(os.path.join(ROOT, "config"))
    law = build_law(params)
    arena = build_arena(params, os.path.join(ROOT, "worlds", str(params["world_id"]) + ".world"))
    sampler = Sampler(law, arena)
    rng = np.random.default_rng(args.seed)
    accepted = 0
    remaining = args.samples
    # Bound memory while preserving the exact generator sequence.
    while remaining:
        count = min(remaining, 100_000)
        proposals = rng.uniform(
            [law.x_min, law.y_min], [law.x_max, law.y_max], size=(count, 2)
        )
        accepted += sum(sampler.admissible(float(x), float(y)) for x, y in proposals)
        remaining -= count
    box_area = (law.x_max - law.x_min) * (law.y_max - law.y_min)
    fraction = accepted / args.samples
    print(f"seed={args.seed}")
    print(f"samples={args.samples}")
    print(f"accepted={accepted}")
    print(f"acceptance_fraction={fraction:.9f}")
    print(f"estimated_support_area_m2={box_area * fraction:.9f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
