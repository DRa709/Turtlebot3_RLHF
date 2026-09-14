#!/usr/bin/env python3
"""
Offline Discrete SAC Actor Verification Utility (Section 6).

Inspects Discrete SAC checkpoints or validates synthetic observation forward
passes without requiring ROS 2 or Gazebo simulation runtime.
"""

import argparse
import os
import sys
from pathlib import Path

# Add vendor directory to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch
from turtlebot3_drl_nav.discretesac import (
    DiscreteSACPolicy,
    CategoricalActor,
    DiscreteSACConfig,
)
from turtlebot3_drl_nav.state import OBSERVATION_DIM, ACTION_NAMES

ACTION_COUNT = len(ACTION_NAMES)
OBSERVATION_COUNT = OBSERVATION_DIM


def verify_synthetic_forward_pass(checkpoint_path: Path = None):
    """
    Validates synthetic 41-dim observation input and Discrete SAC actor distribution.
    If checkpoint_path is provided and exists, loads the checkpoint.
    Otherwise, tests the CategoricalActor network architecture directly.
    """
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Synthetic input: 36 clear LiDAR beams (1.0 = 3.5m), goal 1 m ahead (0.1 = 1.0m/10),
    # sin(0)=0.0, cos(0)=1.0, prev_v=0.0, prev_w=0.0.
    obs = np.array([1.0] * 36 + [0.1, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert len(obs) == OBSERVATION_COUNT, f"Expected {OBSERVATION_COUNT} features, got {len(obs)}"

    if checkpoint_path and checkpoint_path.is_file():
        print(f"[Offline Check] Loading checkpoint: {checkpoint_path}")
        policy = DiscreteSACPolicy(str(checkpoint_path), device)
        assert policy.observation_dim == OBSERVATION_COUNT
        assert policy.action_dim == ACTION_COUNT
        print(f"  • Parameter Digest: {policy.parameter_digest}")

        with torch.no_grad():
            tensor = torch.from_numpy(obs).unsqueeze(0)
            probabilities, log_probabilities = policy.actor.distribution(tensor)
        
        prob_list = probabilities[0].tolist()
        prob_sum = float(probabilities.sum())
        action_det = policy.act(obs, "deterministic")
        action_stoch = policy.act(obs, "stochastic", sampling_seed=42)

        print(f"  • Probabilities: {[round(p, 4) for p in prob_list]}")
        print(f"  • Probability Sum: {prob_sum:.6f}")
        print(f"  • Deterministic Action: {action_det} ({ACTION_NAMES[action_det]})")
        print(f"  • Stochastic Action (seed 42): {action_stoch} ({ACTION_NAMES[action_stoch]})")
    else:
        print("[Offline Check] No checkpoint file specified; testing raw CategoricalActor architecture...")
        config = DiscreteSACConfig()
        config.validate()
        actor = CategoricalActor(OBSERVATION_COUNT, ACTION_COUNT, config.hidden_size).to(device)
        actor.eval()

        with torch.no_grad():
            tensor = torch.from_numpy(obs).unsqueeze(0)
            probabilities, log_probabilities = actor.distribution(tensor)

        prob_list = probabilities[0].tolist()
        prob_sum = float(probabilities.sum())
        action_det = int(probabilities.argmax(dim=1).item())

        print(f"  • Network hidden size: {config.hidden_size}")
        print(f"  • Probabilities: {[round(p, 4) for p in prob_list]}")
        print(f"  • Probability Sum: {prob_sum:.6f}")
        print(f"  • Deterministic Action: {action_det} ({ACTION_NAMES[action_det]})")

    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities sum {prob_sum} deviates from 1.0"
    assert 0 <= action_det < ACTION_COUNT, f"Action {action_det} out of bounds"
    print("SYNTHETIC_ACTOR_CHECK_PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 Discrete SAC Offline Actor Checker")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic forward pass check")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None:
        default_model = PROJECT_ROOT / "models" / "pretrained.pt"
        if default_model.is_file():
            ckpt_path = default_model

    verify_synthetic_forward_pass(ckpt_path)


if __name__ == "__main__":
    main()
