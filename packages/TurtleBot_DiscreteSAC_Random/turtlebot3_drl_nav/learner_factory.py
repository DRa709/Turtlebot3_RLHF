"""Algorithm-specific factory consumed by the shared test harness and tools.
Every package provides this module with the same three names."""

import torch

from .discretesac import (  # noqa: F401
    ALGORITHM,
    DiscreteSACAgent,
    DiscreteSACConfig,
    DiscreteSACPolicy,
    parameter_digest,
)

POLICY_MODES = ("stochastic", "deterministic")


def make_learner(observation_dim: int, action_dim: int, seed: int, device: torch.device, **overrides) -> DiscreteSACAgent:
    """A learner with the algorithm's frozen defaults, overridable for tests."""
    config = DiscreteSACConfig(**overrides)
    return DiscreteSACAgent(observation_dim, action_dim, config, device, seed)


def policy_loader(device: torch.device):
    return lambda path: DiscreteSACPolicy(path, device)
