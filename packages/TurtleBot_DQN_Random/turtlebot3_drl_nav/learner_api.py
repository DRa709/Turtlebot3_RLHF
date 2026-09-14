"""The interface every algorithm package's learner presents to the shared
orchestrator. Shared layer; no ROS dependency.

    Transition(observation, action, reward, next_observation, terminated)
    ActionChoice(action, epsilon_used, random_action)

Learner (training):
    act(observation, evaluation=False) -> ActionChoice
    observe(Transition) -> None            increments env_steps
    optimize() -> Optional[dict]           one gradient step; keys are updates.csv columns
    save(path, kind, extra) -> None        kind in {"full", "policy_only"}; atomic
    env_steps, gradient_steps, online (nn.Module), replay (len())

Evaluation policy (loaded from a checkpoint file by ``policy_loader(path)``):
    act(observation, policy_mode) -> int
    parameter_digest: str
"""

from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool


class ActionChoice(NamedTuple):
    action: int
    epsilon_used: float
    random_action: bool
