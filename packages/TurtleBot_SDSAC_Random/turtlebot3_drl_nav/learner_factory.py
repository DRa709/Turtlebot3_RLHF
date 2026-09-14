"""Algorithm-specific factory consumed by the shared test harness and tools.
Every package provides this module with the same three names."""

import torch

from .sdsac import (  # noqa: F401
    ALGORITHM,
    SDSACAgent,
    SDSACConfig,
    SDSACPolicy,
    parameter_digest,
)

POLICY_MODES = ("stochastic", "deterministic")


def make_learner(observation_dim: int, action_dim: int, seed: int, device: torch.device, **overrides) -> SDSACAgent:
    """A learner with the algorithm's frozen defaults, overridable for tests."""
    config = SDSACConfig(**overrides)
    return SDSACAgent(observation_dim, action_dim, config, device, seed)


def policy_loader(device: torch.device):
    return lambda path: SDSACPolicy(path, device)
