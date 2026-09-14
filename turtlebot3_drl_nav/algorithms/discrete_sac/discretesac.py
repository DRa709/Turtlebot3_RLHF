"""Standalone categorical Discrete Soft Actor-Critic learner.

This is the discrete-action SAC formulation of Christodoulou (2019): a
categorical actor, two action-value critics, clipped double-Q targets and exact
sums over the finite action set. The study fixes alpha instead of optimizing
it. Replay is uniform and targets are one-step. This module has no ROS
dependency and imports no other algorithm implementation.
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

ALGORITHM = "DiscreteSAC"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DiscreteSACConfig:
    gamma: float = 0.99
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    alpha: float = 0.2
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    target_update_steps: int = 1_000
    hidden_size: int = 256
    gradient_clip_norm: float = 10.0
    torch_threads: int = 1

    def validate(self) -> None:
        scalars = (self.gamma, self.actor_learning_rate, self.critic_learning_rate,
                   self.alpha, self.gradient_clip_norm)
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("Discrete SAC scalar hyperparameters must be finite")
        counts = (self.batch_size, self.replay_capacity, self.warmup_steps,
                  self.target_update_steps, self.hidden_size, self.torch_threads)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("Discrete SAC counts and sizes must be integers")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("actor and critic learning rates must be positive")
        if self.alpha <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("alpha and gradient_clip_norm must be positive")
        if min(counts) <= 0:
            raise ValueError("Discrete SAC sizes and periods must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.warmup_steps < self.batch_size:
            raise ValueError("warmup_steps must permit a full replay minibatch")


class ReplayTransition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool


class UniformReplayBuffer:
    """Fixed-capacity uniform replay with deterministic, checkpointed sampling."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity = capacity
        self.rng = rng
        self.buffer: List[Optional[ReplayTransition]] = [None] * capacity
        self.position = 0
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def push(self, transition: ReplayTransition) -> None:
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size: int) -> List[ReplayTransition]:
        if batch_size <= 0 or self.length < batch_size:
            raise ValueError("uniform replay sample requires a positive full minibatch")
        result = [self.buffer[index] for index in self.rng.sample(range(self.length), batch_size)]
        if any(item is None for item in result):
            raise RuntimeError("uniform replay returned an empty slot")
        return list(result)  # type: ignore[arg-type]

    def state_dict(self) -> Dict:
        count = self.capacity if self.length == self.capacity else self.length
        return {"capacity": self.capacity, "position": self.position,
                "length": self.length, "items": list(self.buffer[:count])}

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("replay checkpoint state is not a mapping")
        try:
            capacity = int(state["capacity"]); position = int(state["position"])
            length = int(state["length"]); items = list(state["items"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed replay checkpoint state: {error}")
        count = capacity if length == capacity else length
        if capacity != self.capacity or not 0 <= length <= capacity or len(items) != count:
            raise ValueError("replay checkpoint dimensions are incompatible")
        if not 0 <= position < capacity or (length < capacity and position != length):
            raise ValueError("replay checkpoint position is invalid")
        if any(not isinstance(item, ReplayTransition) for item in items):
            raise ValueError("replay checkpoint contains an invalid transition")
        self.buffer = [None] * self.capacity
        self.buffer[:count] = items
        self.position = position
        self.length = length


class CategoricalActor(nn.Module):
    """MLP categorical policy over all discrete commands."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)

    def distribution(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probabilities = torch.log_softmax(self(observation), dim=-1)
        return log_probabilities.exp(), log_probabilities


class QNetwork(nn.Module):
    """MLP returning one action value for every discrete command."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


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
                                allowed_kinds: tuple = ("full", "policy_only")) -> DiscreteSACConfig:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format_version mismatch")
    if payload.get("algorithm") != ALGORITHM:
        raise ValueError(f"checkpoint algorithm {payload.get('algorithm')} is not {ALGORITHM}")
    if payload.get("kind") not in allowed_kinds:
        raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not allowed here")
    try:
        observation_dim = int(payload["observation_dim"]); action_dim = int(payload["action_dim"])
        env_steps = int(payload["env_steps"]); gradient_steps = int(payload["gradient_steps"])
        seed = int(payload["seed"]); config = DiscreteSACConfig(**payload["config"])
        actor = payload["actor_state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint metadata is malformed: {error}")
    config.validate()
    if observation_dim <= 0 or action_dim <= 1 or seed < 0:
        raise ValueError("checkpoint dimensions or seed are invalid")
    if env_steps < 0 or gradient_steps < 0 or gradient_steps > env_steps:
        raise ValueError("checkpoint counters are invalid")
    if not isinstance(actor, dict):
        raise ValueError("checkpoint actor_state_dict is not a mapping")
    if payload["kind"] == "full":
        required = {"critic1_state_dict", "critic2_state_dict", "target1_state_dict",
                    "target2_state_dict", "actor_optimizer_state_dict",
                    "critic1_optimizer_state_dict", "critic2_optimizer_state_dict",
                    "replay_state", "python_rng_state", "numpy_rng_state", "torch_rng_state"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"full checkpoint is missing {sorted(missing)}")
    return config


def _categorical_sample(probabilities: Sequence[float], rng: random.Random) -> int:
    if not probabilities:
        raise ValueError("categorical probabilities are empty")
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("categorical probabilities must be finite and non-negative")
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0 or abs(total - 1.0) > 1e-5:
        raise ValueError("categorical probabilities must sum to one")
    threshold = rng.random() * total
    cumulative = 0.0
    for index, value in enumerate(values):
        cumulative += value
        if threshold < cumulative:
            return index
    return len(values) - 1


def soft_value(probabilities: torch.Tensor, log_probabilities: torch.Tensor,
               q1: torch.Tensor, q2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact finite-action soft value using clipped double Q."""
    if not (probabilities.shape == log_probabilities.shape == q1.shape == q2.shape):
        raise ValueError("soft-value tensors must have identical shapes")
    return (probabilities * (torch.minimum(q1, q2) - float(alpha) * log_probabilities)).sum(dim=1)


def critic_target(rewards: torch.Tensor, terminated: torch.Tensor,
                  next_soft_values: torch.Tensor, gamma: float) -> torch.Tensor:
    """One-step target; only genuine termination disables bootstrapping."""
    if not (rewards.shape == terminated.shape == next_soft_values.shape):
        raise ValueError("critic-target tensors must have identical shapes")
    return rewards + float(gamma) * (1.0 - terminated) * next_soft_values


def actor_objective(probabilities: torch.Tensor, log_probabilities: torch.Tensor,
                    q1: torch.Tensor, q2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact categorical SAC actor objective averaged over the batch."""
    if not (probabilities.shape == log_probabilities.shape == q1.shape == q2.shape):
        raise ValueError("actor-objective tensors must have identical shapes")
    return (probabilities * (float(alpha) * log_probabilities - torch.minimum(q1, q2))).sum(dim=1).mean()


class DiscreteSACAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: DiscreteSACConfig,
                 device: torch.device, seed: int) -> None:
        config.validate()
        if observation_dim <= 0 or action_dim <= 1:
            raise ValueError("observation_dim must be positive and action_dim must exceed one")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.observation_dim = int(observation_dim); self.action_dim = int(action_dim)
        self.config = config; self.device = device; self.seed = int(seed)
        torch.set_num_threads(config.torch_threads)
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.actor = CategoricalActor(observation_dim, action_dim, config.hidden_size).to(device)
        self.critic1 = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.critic2 = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.target1 = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.target2 = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.target1.load_state_dict(self.critic1.state_dict()); self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.requires_grad_(False); self.target2.requires_grad_(False)
        self.online = self.actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_learning_rate)
        self.replay = UniformReplayBuffer(config.replay_capacity, self.rng)
        self.env_steps = 0; self.gradient_steps = 0

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
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probabilities, _ = self.actor.distribution(tensor)
        action = (int(probabilities.argmax(dim=1).item()) if evaluation else
                  _categorical_sample(probabilities[0].cpu().tolist(), self.rng))
        return ActionChoice(action, None, False)

    def observe(self, transition: Transition) -> None:
        observation = self._validate_observation(transition.observation, "transition observation")
        next_observation = self._validate_observation(transition.next_observation, "transition next_observation")
        try:
            action = int(transition.action); exact = float(transition.action) == action
        except (TypeError, ValueError, OverflowError):
            action, exact = -1, False
        if not exact or not 0 <= action < self.action_dim:
            raise ValueError("transition action is outside the discrete action space")
        if not math.isfinite(float(transition.reward)):
            raise ValueError("transition reward must be finite")
        if not isinstance(transition.terminated, (bool, np.bool_)) or not isinstance(transition.episode_end, (bool, np.bool_)):
            raise ValueError("transition masks must be boolean")
        if bool(transition.terminated) and not bool(transition.episode_end):
            raise ValueError("a terminated transition must end the episode")
        self.replay.push(ReplayTransition(observation.copy(), action, float(transition.reward),
                                          next_observation.copy(), bool(transition.terminated)))
        self.env_steps += 1

    def optimize(self) -> Optional[Dict[str, float]]:
        if self.env_steps < self.config.warmup_steps or len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.observation for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_observation for item in batch]), dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor([item.terminated for item in batch], dtype=torch.float32, device=self.device)
        rows = torch.arange(len(batch), device=self.device)
        with torch.no_grad():
            next_probabilities, next_log_probabilities = self.actor.distribution(next_states)
            next_soft_values = soft_value(
                next_probabilities, next_log_probabilities,
                self.target1(next_states), self.target2(next_states), self.config.alpha,
            )
            targets = critic_target(rewards, terminated, next_soft_values, self.config.gamma)
        q1_taken = self.critic1(states)[rows, actions]
        q2_taken = self.critic2(states)[rows, actions]
        td1 = targets - q1_taken; td2 = targets - q2_taken
        critic1_loss = td1.square().mean(); critic2_loss = td2.square().mean()
        if not torch.isfinite(targets).all() or not torch.isfinite(critic1_loss) or not torch.isfinite(critic2_loss):
            raise RuntimeError("Discrete SAC critic target or loss is non-finite")
        self.critic1_optimizer.zero_grad(set_to_none=True); critic1_loss.backward()
        critic1_grad = nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.gradient_clip_norm)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True); critic2_loss.backward()
        critic2_grad = nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.gradient_clip_norm)
        self.critic2_optimizer.step()
        probabilities, log_probabilities = self.actor.distribution(states)
        with torch.no_grad():
            min_q_policy = torch.minimum(self.critic1(states), self.critic2(states))
        actor_loss = actor_objective(
            probabilities, log_probabilities,
            self.critic1(states).detach(), self.critic2(states).detach(), self.config.alpha,
        )
        if not torch.isfinite(actor_loss):
            raise RuntimeError("Discrete SAC actor loss is non-finite")
        self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()
        self.gradient_steps += 1
        target_synced = self.gradient_steps % self.config.target_update_steps == 0
        if target_synced:
            self.target1.load_state_dict(self.critic1.state_dict()); self.target2.load_state_dict(self.critic2.state_dict())
        entropy = -(probabilities.detach() * log_probabilities.detach()).sum(dim=1)
        return {
            "actor_loss": float(actor_loss.item()), "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "critic_loss_mean": float(((critic1_loss + critic2_loss) * 0.5).item()),
            "td_error_abs_mean": float(torch.maximum(td1.abs(), td2.abs()).mean().item()),
            "q1_taken_mean": float(q1_taken.detach().mean().item()),
            "q2_taken_mean": float(q2_taken.detach().mean().item()),
            "q_gap_abs_mean": float((q1_taken.detach() - q2_taken.detach()).abs().mean().item()),
            "soft_target_mean": float(targets.mean().item()),
            "next_soft_value_mean": float(next_soft_values.mean().item()),
            "min_q_policy_mean": float(min_q_policy.mean().item()),
            "policy_entropy_mean": float(entropy.mean().item()),
            "next_policy_entropy_mean": float((-(next_probabilities * next_log_probabilities).sum(dim=1)).mean().item()),
            "max_action_probability_mean": float(probabilities.detach().max(dim=1).values.mean().item()),
            "min_action_probability_mean": float(probabilities.detach().min(dim=1).values.mean().item()),
            "alpha": float(self.config.alpha), "actor_grad_norm": float(actor_grad),
            "critic1_grad_norm": float(critic1_grad), "critic2_grad_norm": float(critic2_grad),
            "actor_learning_rate": float(self.config.actor_learning_rate),
            "critic_learning_rate": float(self.config.critic_learning_rate),
            "target_synced": target_synced,
            "batch_terminal_fraction": float(terminated.mean().item()),
            # These are integer-valued CSV schema fields. Preserve their type so
            # the exact integer validator never needs a lossy float round trip.
            "batch_size": int(len(batch)), "replay_size": int(len(self.replay)),
        }

    def _checkpoint_payload(self, kind: str, extra: Optional[Dict]) -> Dict:
        if kind not in ("full", "policy_only"):
            raise ValueError("checkpoint kind must be full or policy_only")
        payload: Dict = {"format_version": CHECKPOINT_FORMAT_VERSION, "algorithm": ALGORITHM,
                         "kind": kind, "observation_dim": self.observation_dim,
                         "action_dim": self.action_dim, "config": asdict(self.config),
                         "seed": self.seed, "env_steps": self.env_steps,
                         "gradient_steps": self.gradient_steps,
                         "actor_state_dict": self.actor.state_dict(), "extra": dict(extra or {})}
        if kind == "full":
            payload.update({
                "critic1_state_dict": self.critic1.state_dict(), "critic2_state_dict": self.critic2.state_dict(),
                "target1_state_dict": self.target1.state_dict(), "target2_state_dict": self.target2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "replay_state": self.replay.state_dict(), "python_rng_state": self.rng.getstate(),
                "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
            })
            if torch.cuda.is_available():
                payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def save(self, path: str, kind: str = "full", extra: Optional[Dict] = None) -> None:
        atomic_torch_save(self._checkpoint_payload(kind, extra), path)

    def load(self, path: str, load_optimizer: bool = True) -> Dict:
        payload = _torch_load(path, self.device)
        config = validate_checkpoint_payload(payload, ("full",) if load_optimizer else ("full", "policy_only"))
        if config != self.config or int(payload["observation_dim"]) != self.observation_dim or int(payload["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint configuration or dimensions are incompatible")
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.env_steps = int(payload["env_steps"]); self.gradient_steps = int(payload["gradient_steps"])
        if load_optimizer:
            self.critic1.load_state_dict(payload["critic1_state_dict"]); self.critic2.load_state_dict(payload["critic2_state_dict"])
            self.target1.load_state_dict(payload["target1_state_dict"]); self.target2.load_state_dict(payload["target2_state_dict"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            self.critic1_optimizer.load_state_dict(payload["critic1_optimizer_state_dict"])
            self.critic2_optimizer.load_state_dict(payload["critic2_optimizer_state_dict"])
            self.replay.load_state_dict(payload["replay_state"]); self.rng.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"]); torch.set_rng_state(payload["torch_rng_state"])
            if torch.cuda.is_available() and "torch_cuda_rng_state_all" in payload:
                torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
        return payload


class DiscreteSACPolicy:
    """Frozen actor loaded for stochastic or deterministic evaluation."""

    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            payload = _torch_load(checkpoint_path, device); config = validate_checkpoint_payload(payload)
            self.observation_dim = int(payload["observation_dim"]); self.action_dim = int(payload["action_dim"])
            self.device = device
            self.actor = CategoricalActor(self.observation_dim, self.action_dim, config.hidden_size).to(device)
            self.actor.load_state_dict(payload["actor_state_dict"]); self.actor.eval()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.parameter_digest = parameter_digest(self.actor)

    def act(self, observation: np.ndarray, policy_mode: str = "deterministic",
            sampling_seed: Optional[int] = None) -> int:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.isfinite(value).all():
            raise ValueError("evaluation observation is invalid")
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probabilities, _ = self.actor.distribution(tensor)
        if policy_mode == "deterministic":
            return int(probabilities.argmax(dim=1).item())
        if policy_mode == "stochastic":
            if sampling_seed is None or int(sampling_seed) < 0:
                raise ValueError("stochastic evaluation requires a non-negative sampling_seed")
            return _categorical_sample(probabilities[0].cpu().tolist(), random.Random(int(sampling_seed)))
        raise ValueError(f"unsupported Discrete SAC policy mode {policy_mode!r}")
