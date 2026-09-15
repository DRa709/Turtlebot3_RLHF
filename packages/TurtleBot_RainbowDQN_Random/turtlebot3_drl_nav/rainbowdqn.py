"""Standalone Phase-1 Rainbow DQN learner (roadmap section 3.4).

The learner combines exactly six mechanisms: Double-Q action selection and
evaluation; a mean-centred dueling categorical network; C51 distributional
learning; proportional prioritized replay with importance sampling;
episode-safe three-step returns; and factorized Gaussian NoisyNet exploration.
"""

import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .learner_api import ActionChoice, Transition

ALGORITHM = "RainbowDQN"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RainbowDQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    target_update_steps: int = 1_000
    hidden_size: int = 256
    gradient_clip_norm: float = 10.0
    atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_decay_steps: int = 500_000
    per_epsilon: float = 1e-6
    noisy_sigma0: float = 0.5
    dueling_aggregation: str = "mean"
    distributional_loss: str = "cross_entropy"
    torch_threads: int = 1

    def validate(self) -> None:
        scalars = (
            self.gamma, self.learning_rate, self.gradient_clip_norm,
            self.v_min, self.v_max, self.per_alpha, self.per_beta_start,
            self.per_beta_end, self.per_epsilon, self.noisy_sigma0,
        )
        if not all(np.isfinite(float(value)) for value in scalars):
            raise ValueError("Rainbow scalar hyperparameters must be finite")
        integers = (
            self.batch_size, self.replay_capacity, self.warmup_steps,
            self.target_update_steps, self.hidden_size, self.atoms, self.n_step,
            self.per_beta_decay_steps, self.torch_threads,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("Rainbow counts and sizes must be integers")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if min(
            self.batch_size, self.replay_capacity, self.target_update_steps,
            self.hidden_size, self.n_step, self.per_beta_decay_steps,
            self.torch_threads,
        ) <= 0:
            raise ValueError("Rainbow sizes, lengths and periods must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.warmup_steps < self.batch_size + self.n_step - 1:
            raise ValueError("warmup_steps must permit a full n-step replay minibatch")
        if self.atoms < 2 or not self.v_max > self.v_min:
            raise ValueError("C51 requires atoms >= 2 and v_max > v_min")
        if not 0.0 <= self.per_alpha <= 1.0:
            raise ValueError("per_alpha must be in [0, 1]")
        if not 0.0 <= self.per_beta_start <= self.per_beta_end <= 1.0:
            raise ValueError("PER beta must satisfy 0 <= start <= end <= 1")
        if self.per_epsilon <= 0.0 or self.noisy_sigma0 <= 0.0:
            raise ValueError("per_epsilon and noisy_sigma0 must be positive")
        if self.dueling_aggregation != "mean":
            raise ValueError("dueling_aggregation must be 'mean'")
        if self.distributional_loss != "cross_entropy":
            raise ValueError("distributional_loss must be 'cross_entropy'")


class RawTransition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    episode_end: bool


class RainbowTransition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    steps: int


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer from Fortunato et al. (2018)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or not math.isfinite(sigma0) or sigma0 <= 0.0:
            raise ValueError("NoisyLinear dimensions and sigma0 must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma0 / math.sqrt(in_features))
        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scaled_noise(self.in_features, self.weight_mu.device)
        epsilon_out = self._scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(value, weight, bias)


class RainbowQNetwork(nn.Module):
    """Noisy, dueling categorical action-value network."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int,
                 atoms: int, v_min: float, v_max: float, sigma0: float) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.atoms = int(atoms)
        self.feature1 = NoisyLinear(observation_dim, hidden_size, sigma0)
        self.feature2 = NoisyLinear(hidden_size, hidden_size, sigma0)
        self.value = NoisyLinear(hidden_size, atoms, sigma0)
        self.advantage = NoisyLinear(hidden_size, action_dim * atoms, sigma0)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    def streams(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.feature1(observation))
        features = torch.relu(self.feature2(features))
        value = self.value(features).view(-1, 1, self.atoms)
        advantage = self.advantage(features).view(-1, self.action_dim, self.atoms)
        return value, advantage

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value, advantage = self.streams(observation)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def probabilities(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(observation), dim=2)

    def q_values(self, observation: torch.Tensor) -> torch.Tensor:
        probabilities = self.probabilities(observation)
        return (probabilities * self.support.view(1, 1, -1)).sum(dim=2)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def sigma_statistics(self) -> Tuple[float, float, float]:
        values = torch.cat([
            parameter.detach().abs().reshape(-1)
            for module in self.modules() if isinstance(module, NoisyLinear)
            for parameter in (module.weight_sigma, module.bias_sigma)
        ])
        return float(values.mean().item()), float(values.min().item()), float(values.max().item())


class PrioritizedReplayBuffer:
    """Proportional PER with logarithmic-time sampling and updates."""

    def __init__(self, capacity: int, alpha: float, rng: random.Random) -> None:
        if capacity <= 0 or not 0.0 <= alpha <= 1.0:
            raise ValueError("PER capacity must be positive and alpha must be in [0,1]")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.rng = rng
        self.buffer: List[Optional[RainbowTransition]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.length = 0
        tree_size = 1
        while tree_size < self.capacity:
            tree_size *= 2
        self.tree_size = tree_size
        self.sum_tree = np.zeros(2 * tree_size, dtype=np.float64)
        self.min_tree = np.full(2 * tree_size, np.inf, dtype=np.float64)
        self.max_tree = np.zeros(2 * tree_size, dtype=np.float64)

    def __len__(self) -> int:
        return self.length

    def _scaled(self, priority: float) -> float:
        return float(priority) ** self.alpha

    def _set_leaf(self, index: int, priority: float) -> None:
        node = self.tree_size + index
        scaled = self._scaled(priority)
        self.sum_tree[node] = scaled
        self.min_tree[node] = scaled
        self.max_tree[node] = priority
        node //= 2
        while node:
            left = 2 * node
            self.sum_tree[node] = self.sum_tree[left] + self.sum_tree[left + 1]
            self.min_tree[node] = min(self.min_tree[left], self.min_tree[left + 1])
            self.max_tree[node] = max(self.max_tree[left], self.max_tree[left + 1])
            node //= 2

    @property
    def max_priority(self) -> float:
        return float(self.max_tree[1]) if self.length else 1.0

    def push(self, transition: RainbowTransition) -> None:
        priority = self.max_priority
        index = self.position
        self.buffer[index] = transition
        self.priorities[index] = priority
        self._set_leaf(index, priority)
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)

    def probabilities(self) -> np.ndarray:
        if not self.length:
            raise ValueError("cannot compute probabilities of an empty replay buffer")
        scaled = np.power(self.priorities[:self.length], self.alpha)
        total = float(scaled.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("PER probability mass is invalid")
        return scaled / total

    def _find_prefix(self, mass: float) -> int:
        total = float(self.sum_tree[1])
        if not 0.0 <= mass < total:
            mass = min(max(mass, 0.0), np.nextafter(total, 0.0))
        node = 1
        while node < self.tree_size:
            left = 2 * node
            if mass < self.sum_tree[left]:
                node = left
            else:
                mass -= self.sum_tree[left]
                node = left + 1
        index = node - self.tree_size
        if index >= self.length or self.buffer[index] is None:
            raise RuntimeError("PER sampled an unoccupied tree leaf")
        return index

    def sample(self, batch_size: int, beta: float) -> Tuple[List[RainbowTransition], List[int], torch.Tensor]:
        if batch_size <= 0 or self.length < batch_size:
            raise ValueError("PER sample requires a positive full minibatch")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("PER beta must be in [0,1]")
        total = float(self.sum_tree[1])
        minimum_scaled = float(self.min_tree[1])
        if not math.isfinite(total) or total <= 0.0 or not math.isfinite(minimum_scaled) or minimum_scaled <= 0.0:
            raise RuntimeError("PER tree mass is invalid")
        indices = [self._find_prefix(self.rng.random() * total) for _ in range(batch_size)]
        probabilities = np.asarray(
            [self.sum_tree[self.tree_size + index] / total for index in indices], dtype=np.float64,
        )
        minimum_probability = minimum_scaled / total
        maximum_weight = (self.length * minimum_probability) ** (-beta)
        weights = np.power(self.length * probabilities, -beta) / maximum_weight
        if not np.isfinite(weights).all() or np.any(weights <= 0.0) or np.any(weights > 1.0 + 1e-12):
            raise RuntimeError("PER importance weights are invalid")
        transitions = [self.buffer[index] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("PER returned an empty transition")
        return list(transitions), indices, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        if len(indices) != len(priorities):
            raise ValueError("PER indices and priorities must have equal length")
        # Sampling is with replacement. If one replay item appears repeatedly
        # in a minibatch, retain the largest new loss-derived priority; this is
        # deterministic and prevents an arbitrary duplicate order from hiding
        # a large distributional error.
        updates: Dict[int, float] = {}
        for index, priority in zip(indices, priorities):
            value = float(priority)
            if not 0 <= int(index) < self.length or not math.isfinite(value) or value <= 0.0:
                raise ValueError("PER indices must be occupied and priorities finite and positive")
            updates[int(index)] = max(value, updates.get(int(index), 0.0))
        for index, value in updates.items():
            self.priorities[index] = value
            self._set_leaf(index, value)

    def state_dict(self) -> Dict:
        count = self.capacity if self.length == self.capacity else self.length
        return {
            "capacity": self.capacity, "alpha": self.alpha,
            "position": self.position, "length": self.length,
            "items": list(self.buffer[:count]),
            "priorities": self.priorities[:count].tolist(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("PER checkpoint state is not a mapping")
        try:
            capacity = int(state["capacity"]); alpha = float(state["alpha"])
            position = int(state["position"]); length = int(state["length"])
            items = list(state["items"])
            priorities = [float(value) for value in state["priorities"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed PER checkpoint state: {error}")
        expected_count = capacity if length == capacity else length
        if capacity != self.capacity or alpha != self.alpha:
            raise ValueError("PER checkpoint configuration is incompatible")
        if not 0 <= length <= capacity or len(items) != expected_count or len(priorities) != expected_count:
            raise ValueError("PER checkpoint length is invalid")
        if not 0 <= position < capacity or (length < capacity and position != length):
            raise ValueError("PER checkpoint position is invalid")
        if any(not math.isfinite(value) or value <= 0.0 for value in priorities):
            raise ValueError("PER checkpoint priorities are invalid")
        self.buffer = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.sum_tree.fill(0.0); self.min_tree.fill(np.inf); self.max_tree.fill(0.0)
        self.position = position; self.length = length
        for index, (item, priority) in enumerate(zip(items, priorities)):
            if not isinstance(item, RainbowTransition):
                raise ValueError("PER checkpoint contains an invalid transition")
            self.buffer[index] = item; self.priorities[index] = priority
            self._set_leaf(index, priority)


def project_c51(target_probabilities: torch.Tensor, rewards: torch.Tensor,
                terminated: torch.Tensor, steps: torch.Tensor, gamma: float,
                support: torch.Tensor, return_stats: bool = False):
    """Project n-step categorical targets onto the fixed C51 support."""
    if target_probabilities.ndim != 2 or support.ndim != 1:
        raise ValueError("C51 probabilities must be [batch, atoms] and support one-dimensional")
    batch, atoms = target_probabilities.shape
    if atoms != support.numel() or atoms < 2:
        raise ValueError("C51 probability and support dimensions disagree")
    if any(tensor.shape != (batch,) for tensor in (rewards, terminated, steps)):
        raise ValueError("C51 reward, mask and step tensors must match the batch")
    delta = (support[-1] - support[0]) / (atoms - 1)
    if not bool(torch.isfinite(delta)) or float(delta.item()) <= 0.0:
        raise ValueError("C51 support must be finite and strictly increasing")
    expected = support[0] + torch.arange(atoms, device=support.device, dtype=support.dtype) * delta
    if not torch.allclose(support, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("C51 support must be evenly spaced")
    discounts = torch.pow(torch.full_like(rewards, float(gamma)), steps) * (1.0 - terminated)
    unbounded = rewards.unsqueeze(1) + discounts.unsqueeze(1) * support.unsqueeze(0)
    low_fraction = float((unbounded < support[0]).float().mean().item())
    high_fraction = float((unbounded > support[-1]).float().mean().item())
    shifted = unbounded.clamp(float(support[0]), float(support[-1]))
    positions = ((shifted - support[0]) / delta).clamp(0.0, float(atoms - 1))
    lower = positions.floor().long().clamp(0, atoms - 1)
    upper = positions.ceil().long().clamp(0, atoms - 1)
    projection = torch.zeros_like(target_probabilities)
    offset = (torch.arange(batch, device=target_probabilities.device) * atoms).unsqueeze(1)
    flat = projection.view(-1)
    flat.index_add_(0, (lower + offset).reshape(-1), (target_probabilities * (upper.float() - positions)).reshape(-1))
    flat.index_add_(0, (upper + offset).reshape(-1), (target_probabilities * (positions - lower.float())).reshape(-1))
    equal = lower == upper
    if bool(equal.any()):
        flat.index_add_(0, (lower + offset)[equal], target_probabilities[equal])
    mass_error = float((projection.sum(dim=1) - 1.0).abs().max().item())
    if not torch.isfinite(projection).all() or bool((projection < -1e-7).any()) or mass_error > 1e-5:
        raise RuntimeError("C51 projection failed finiteness, non-negativity or mass conservation")
    stats = {
        "projection_clip_low_fraction": low_fraction,
        "projection_clip_high_fraction": high_fraction,
        "projection_mass_error_max": mass_error,
    }
    return (projection, stats) if return_stats else projection


def parameter_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: Dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_checkpoint_payload(payload: Dict,
                                allowed_kinds: tuple = ("full", "policy_only")) -> RainbowDQNConfig:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    try:
        if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint format_version mismatch")
    except (TypeError, ValueError):
        raise ValueError("checkpoint format_version mismatch")
    if payload.get("algorithm") != ALGORITHM:
        raise ValueError(f"checkpoint algorithm {payload.get('algorithm')} is not {ALGORITHM}")
    if payload.get("kind") not in allowed_kinds:
        raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not allowed here")
    try:
        observation_dim = int(payload["observation_dim"]); action_dim = int(payload["action_dim"])
        env_steps = int(payload["env_steps"]); gradient_steps = int(payload["gradient_steps"])
        seed = int(payload["seed"]); config = RainbowDQNConfig(**payload["config"])
        online = payload["online_state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint metadata is malformed: {error}")
    config.validate()
    if observation_dim <= 0 or action_dim <= 0 or seed < 0:
        raise ValueError("checkpoint dimensions and seed are invalid")
    if env_steps < 0 or gradient_steps < 0 or gradient_steps > env_steps:
        raise ValueError("checkpoint counters are invalid")
    if not isinstance(online, dict):
        raise ValueError("checkpoint online_state_dict is not a mapping")
    return config


class RainbowDQNAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: RainbowDQNConfig,
                 device: torch.device, seed: int) -> None:
        config.validate()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.observation_dim = int(observation_dim); self.action_dim = int(action_dim)
        self.config = config; self.device = device; self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        torch.set_num_threads(config.torch_threads)
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        args = (self.observation_dim, self.action_dim, config.hidden_size,
                config.atoms, config.v_min, config.v_max, config.noisy_sigma0)
        self.online = RainbowQNetwork(*args).to(device)
        self.target = RainbowQNetwork(*args).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.online.train(); self.target.train()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = PrioritizedReplayBuffer(config.replay_capacity, config.per_alpha, self.rng)
        self.n_step_accumulator: List[RawTransition] = []
        self.env_steps = 0; self.gradient_steps = 0

    def per_beta(self, env_steps: Optional[int] = None) -> float:
        steps = self.env_steps if env_steps is None else int(env_steps)
        progress = min(1.0, steps / self.config.per_beta_decay_steps)
        return self.config.per_beta_start + progress * (self.config.per_beta_end - self.config.per_beta_start)

    def _validate_observation(self, observation: np.ndarray, label: str) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,):
            raise ValueError(f"{label} must have shape {(self.observation_dim,)}")
        if not np.isfinite(value).all() or np.any(value < -1.000001) or np.any(value > 1.000001):
            raise ValueError(f"{label} must be finite and normalized to [-1,1]")
        return value

    def act(self, observation: np.ndarray, evaluation: bool = False) -> ActionChoice:
        value = self._validate_observation(observation, "action observation")
        if not evaluation and self.env_steps < self.config.warmup_steps:
            return ActionChoice(self.rng.randrange(self.action_dim), None, True)
        self.online.train(not evaluation)
        if not evaluation:
            self.online.reset_noise()
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = int(self.online.q_values(tensor).argmax(dim=1).item())
        return ActionChoice(action, None, False)

    def _aggregate_front(self) -> RainbowTransition:
        if not self.n_step_accumulator:
            raise RuntimeError("cannot aggregate an empty n-step accumulator")
        window = self.n_step_accumulator[:self.config.n_step]
        reward = 0.0; last = window[-1]; steps = len(window)
        for index, transition in enumerate(window):
            reward += (self.config.gamma ** index) * transition.reward
            last = transition; steps = index + 1
            if transition.episode_end:
                break
        first = window[0]
        return RainbowTransition(first.observation.copy(), first.action, float(reward),
                                 last.next_observation.copy(), bool(last.terminated), int(steps))

    def observe(self, transition: Transition) -> None:
        observation = self._validate_observation(transition.observation, "transition observation")
        next_observation = self._validate_observation(transition.next_observation, "transition next_observation")
        try:
            action = int(transition.action); action_exact = float(transition.action) == action
        except (TypeError, ValueError, OverflowError):
            action, action_exact = -1, False
        if not action_exact or not 0 <= action < self.action_dim:
            raise ValueError("transition action is outside the discrete action space")
        if not np.isfinite(float(transition.reward)):
            raise ValueError("transition reward must be finite")
        if not isinstance(transition.terminated, (bool, np.bool_)) or not isinstance(transition.episode_end, (bool, np.bool_)):
            raise ValueError("transition termination and episode-end masks must be boolean")
        if bool(transition.terminated) and not bool(transition.episode_end):
            raise ValueError("a terminated transition must end the episode")
        raw = RawTransition(observation.copy(), action, float(transition.reward), next_observation.copy(),
                            bool(transition.terminated), bool(transition.episode_end))
        self.n_step_accumulator.append(raw); self.env_steps += 1
        if raw.episode_end:
            while self.n_step_accumulator:
                self.replay.push(self._aggregate_front()); self.n_step_accumulator.pop(0)
        elif len(self.n_step_accumulator) >= self.config.n_step:
            self.replay.push(self._aggregate_front()); self.n_step_accumulator.pop(0)

    def optimize(self) -> Optional[Dict[str, float]]:
        if self.env_steps < self.config.warmup_steps or len(self.replay) < self.config.batch_size:
            return None
        beta = self.per_beta()
        batch, indices, weights = self.replay.sample(self.config.batch_size, beta)
        weights = weights.to(self.device)
        self.online.train(); self.target.train()
        self.online.reset_noise(); self.target.reset_noise()
        states = torch.as_tensor(np.stack([item.observation for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_observation for item in batch]), dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor([item.terminated for item in batch], dtype=torch.float32, device=self.device)
        steps = torch.as_tensor([item.steps for item in batch], dtype=torch.float32, device=self.device)
        value_logits, advantage_logits = self.online.streams(states)
        centered_advantage = advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
        logits = value_logits + centered_advantage
        chosen_logits = logits[torch.arange(len(batch), device=self.device), actions]
        with torch.no_grad():
            next_actions = self.online.q_values(next_states).argmax(dim=1)
            target_probabilities = self.target.probabilities(next_states)[
                torch.arange(len(batch), device=self.device), next_actions]
            projected, projection_stats = project_c51(
                target_probabilities, rewards, terminated, steps,
                self.config.gamma, self.online.support, return_stats=True)
        per_sample_loss = -(projected * torch.log_softmax(chosen_logits, dim=1)).sum(dim=1)
        loss = (weights * per_sample_loss).mean()
        if not torch.isfinite(per_sample_loss).all() or not torch.isfinite(loss) or not torch.isfinite(weights).all():
            raise RuntimeError("Rainbow update produced a non-finite loss or importance weight")
        self.optimizer.zero_grad(); loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(); raise RuntimeError("Rainbow update produced a non-finite gradient norm")
        self.optimizer.step()
        priorities = (per_sample_loss.detach() + self.config.per_epsilon).cpu().tolist()
        self.replay.update_priorities(indices, priorities)
        self.gradient_steps += 1
        target_synced = self.gradient_steps % self.config.target_update_steps == 0
        if target_synced:
            self.target.load_state_dict(self.online.state_dict())
        chosen_probabilities = torch.softmax(chosen_logits.detach(), dim=1)
        q_taken = (chosen_probabilities * self.online.support).sum(dim=1)
        target_values = (projected * self.online.support).sum(dim=1)
        next_values = (target_probabilities * self.online.support).sum(dim=1)
        value_probabilities = torch.softmax(value_logits.detach().squeeze(1), dim=1)
        value_expectation = (value_probabilities * self.online.support).sum(dim=1)
        entropy = -(chosen_probabilities * chosen_probabilities.clamp_min(1e-12).log()).sum(dim=1)
        sigma_mean, sigma_min, sigma_max = self.online.sigma_statistics()
        return {
            "loss": float(loss.item()), "distributional_loss_mean": float(per_sample_loss.mean().item()),
            "td_error_abs_mean": float((q_taken - target_values).abs().mean().item()),
            "q_taken_mean": float(q_taken.mean().item()), "target_mean": float(target_values.mean().item()),
            "next_max_q_mean": float(next_values.mean().item()), "grad_norm": float(gradient_norm.item()),
            "learning_rate": float(self.config.learning_rate), "target_synced": bool(target_synced),
            "batch_terminal_fraction": float(terminated.mean().item()), "batch_size": int(len(batch)),
            "replay_size": int(len(self.replay)), "beta_is": float(beta),
            "mean_importance_weight": float(weights.mean().item()),
            "max_importance_weight": float(weights.max().item()),
            "mean_priority": float(np.mean(priorities)), "max_priority": float(self.replay.max_priority),
            "noisy_sigma_mean": sigma_mean, "noisy_sigma_min": sigma_min, "noisy_sigma_max": sigma_max,
            "n_step": int(self.config.n_step), "mean_effective_n_step": float(steps.mean().item()),
            "value_stream_expected_mean": float(value_expectation.mean().item()),
            "centered_advantage_abs_mean": float(centered_advantage.detach().abs().mean().item()),
            "chosen_distribution_entropy_mean": float(entropy.mean().item()), **projection_stats,
        }

    def _base_payload(self, kind: str, extra: Optional[Dict]) -> Dict:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION, "algorithm": ALGORITHM, "kind": kind,
            "online_state_dict": self.online.state_dict(), "observation_dim": self.observation_dim,
            "action_dim": self.action_dim, "config": asdict(self.config), "seed": self.seed,
            "env_steps": self.env_steps, "gradient_steps": self.gradient_steps, "extra": extra or {},
        }

    def save(self, path: str, kind: str = "full", extra: Optional[Dict] = None) -> None:
        if kind not in ("full", "policy_only"):
            raise ValueError("kind must be full or policy_only")
        payload = self._base_payload(kind, extra)
        if kind == "full":
            payload.update({
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
                "n_step_accumulator": list(self.n_step_accumulator), "python_rng_state": self.rng.getstate(),
                "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
            })
        atomic_torch_save(payload, path)

    def load(self, path: str, load_optimizer: bool = True) -> Dict:
        payload = _torch_load(path, self.device); saved_config = validate_checkpoint_payload(payload)
        if int(payload["observation_dim"]) != self.observation_dim or int(payload["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint dimensions do not match this agent")
        if saved_config != self.config:
            raise ValueError("checkpoint configuration does not match this agent")
        self.online.load_state_dict(payload["online_state_dict"])
        self.target.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
        if load_optimizer and payload.get("kind") == "full":
            required = ("optimizer_state_dict", "replay", "n_step_accumulator",
                        "python_rng_state", "numpy_rng_state", "torch_rng_state")
            if any(name not in payload for name in required):
                raise ValueError("full checkpoint is missing optimizer, PER, n-step or RNG state")
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.replay.load_state_dict(payload["replay"])
            accumulator = list(payload["n_step_accumulator"])
            if len(accumulator) >= self.config.n_step or any(not isinstance(item, RawTransition) for item in accumulator):
                raise ValueError("checkpoint n-step accumulator is invalid")
            self.n_step_accumulator = accumulator
            self.rng.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"]); self.seed = int(payload["seed"])
        self.env_steps = int(payload["env_steps"]); self.gradient_steps = int(payload["gradient_steps"])
        return payload.get("extra", {})


class GreedyPolicy:
    """Noise-free mean-parameter policy loaded from a Rainbow checkpoint."""

    def __init__(self, path: str, device: torch.device) -> None:
        # Building nn.Modules normally consumes the process-global torch RNG.
        # Tier-1 policies are loaded in the training process, so preserve every
        # global torch stream: evaluation cadence must not change later
        # NoisyNet exploration or updates.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            payload = _torch_load(path, device); config = validate_checkpoint_payload(payload)
            self.observation_dim = int(payload["observation_dim"]); self.action_dim = int(payload["action_dim"])
            self.network = RainbowQNetwork(self.observation_dim, self.action_dim, config.hidden_size,
                                           config.atoms, config.v_min, config.v_max, config.noisy_sigma0).to(device)
            self.network.load_state_dict(payload["online_state_dict"]); self.network.eval()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.device = device; self.parameter_digest = parameter_digest(self.network)

    def act(self, observation: np.ndarray, policy_mode: str = "greedy") -> int:
        if policy_mode != "greedy":
            raise ValueError("Rainbow DQN evaluates greedily with NoisyNet noise disabled")
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.isfinite(value).all():
            raise ValueError("evaluation observation is malformed")
        if np.any(value < -1.000001) or np.any(value > 1.000001):
            raise ValueError("evaluation observation must be normalized to [-1,1]")
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.network.q_values(tensor).argmax(dim=1).item())
