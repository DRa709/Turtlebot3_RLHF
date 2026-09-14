"""Algorithm-specific factory consumed by the shared test harness and tools.
Every package provides this module with the same three names."""

import torch

from .rainbowdqn import ALGORITHM, RainbowDQNAgent, RainbowDQNConfig, GreedyPolicy, parameter_digest  # noqa: F401

POLICY_MODES = ("greedy",)


def make_learner(observation_dim: int, action_dim: int, seed: int, device: torch.device, **overrides) -> RainbowDQNAgent:
    """A learner with the algorithm's frozen defaults, overridable for tests."""
    config = RainbowDQNConfig(**overrides)
    return RainbowDQNAgent(observation_dim, action_dim, config, device, seed)


def policy_loader(device: torch.device):
    return lambda path: GreedyPolicy(path, device)
