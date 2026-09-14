"""Phase-1 Double DQN learner (roadmap §3.2). Algorithm-specific; no ROS dependency.

Update rule, verified bit-exactly against an independent recomputation:
    a*_i = argmax_a' Q_online(o'_i, a')
    y_i = r_i + gamma * (1 - m_i^term) * Q_target(o'_i, a*_i)
    L(theta) = mean_i Huber(y_i - Q_theta(o_i, a_i))          (Huber: quadratic below 1, linear above)
Warm-up of W uniform-random transitions with no gradient step; linear epsilon
schedule counted from transition 0; one Adam step per environment transition
after warm-up; hard target synchronization every C_target gradient steps.
"""

import hashlib
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .learner_api import ActionChoice, Transition

ALGORITHM = "DoubleDQN"
CHECKPOINT_FORMAT_VERSION = 4


@dataclass(frozen=True)
class DoubleDQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    target_update_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    hidden_size: int = 256
    gradient_clip_norm: float = 10.0
    loss: str = "huber"
    torch_threads: int = 1

    def validate(self) -> None:
        scalar_values = (
            self.gamma, self.learning_rate, self.epsilon_start, self.epsilon_end,
            self.gradient_clip_norm,
        )
        if not all(np.isfinite(float(value)) for value in scalar_values):
            raise ValueError("Double DQN scalar hyperparameters must be finite")
        integer_values = (
            self.batch_size, self.replay_capacity, self.warmup_steps,
            self.target_update_steps, self.epsilon_decay_steps,
            self.hidden_size, self.torch_threads,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValueError("Double DQN step counts and sizes must be integers")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if min(self.batch_size, self.replay_capacity, self.target_update_steps, self.hidden_size, self.torch_threads) <= 0:
            raise ValueError("sizes, periods and thread counts must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.warmup_steps < self.batch_size:
            raise ValueError("warmup_steps must be >= batch_size so the first update has a full minibatch")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if self.loss not in {"huber", "mse"}:
            raise ValueError("loss must be 'huber' or 'mse'")


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self._buffer: List[Transition] = []
        self._next_index = 0
        self._rng = rng

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        # Sampling directly from the list is O(batch_size). Converting a
        # 100k-item deque to a list on every gradient step would dominate the
        # 500k-transition ARC run.
        return self._rng.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def state_dict(self) -> Dict:
        return {
            "capacity": self.capacity,
            "next_index": self._next_index,
            "items": list(self._buffer),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("replay checkpoint state is not a mapping")
        try:
            capacity = int(state["capacity"])
            next_index = int(state["next_index"])
            items = list(state["items"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed replay checkpoint state: {error}")
        if capacity != self.capacity or len(items) > self.capacity:
            raise ValueError("replay checkpoint capacity is incompatible")
        if not 0 <= next_index < self.capacity:
            raise ValueError("replay checkpoint write index is invalid")
        if len(items) < self.capacity and next_index != len(items):
            raise ValueError("partially filled replay checkpoint has an invalid write index")
        self._buffer = items
        self._next_index = next_index


def parameter_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: Dict, path: str) -> None:
    """Write to a temporary sibling, fsync, then rename: a crash never leaves a
    truncated file at the canonical name."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch without the keyword
        return torch.load(path, map_location=device)


def validate_checkpoint_payload(payload: Dict, allowed_kinds: tuple = ("full", "policy_only")) -> DoubleDQNConfig:
    """Validate checkpoint metadata before any tensors are installed."""
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
        observation_dim = int(payload["observation_dim"])
        action_dim = int(payload["action_dim"])
        env_steps = int(payload["env_steps"])
        gradient_steps = int(payload["gradient_steps"])
        seed = int(payload["seed"])
        config = DoubleDQNConfig(**payload["config"])
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


class DoubleDQNAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: DoubleDQNConfig, device: torch.device, seed: int) -> None:
        config.validate()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        torch.set_num_threads(config.torch_threads)
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.online = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.target = QNetwork(observation_dim, action_dim, config.hidden_size).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, self.rng)
        self.env_steps = 0
        self.gradient_steps = 0

    # ------------------------------------------------------------ schedule

    def epsilon(self, env_steps: Optional[int] = None) -> float:
        steps = self.env_steps if env_steps is None else env_steps
        progress = min(1.0, steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    # ----------------------------------------------------------------- act

    def act(self, observation: np.ndarray, evaluation: bool = False) -> ActionChoice:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError(f"Expected observation shape {(self.observation_dim,)}, got {observation.shape}")
        if not np.isfinite(observation).all():
            raise ValueError("Action observation must be finite")
        if np.any(observation < -1.000001) or np.any(observation > 1.000001):
            raise ValueError("Action observation must lie in the normalized range [-1, 1]")
        if evaluation:
            return ActionChoice(self._greedy(self.online, observation), 0.0, False)
        epsilon = self.epsilon()
        if self.env_steps < self.config.warmup_steps or self.rng.random() < epsilon:
            return ActionChoice(self.rng.randrange(self.action_dim), epsilon, True)
        return ActionChoice(self._greedy(self.online, observation), epsilon, False)

    def _greedy(self, network: nn.Module, observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(network(tensor).argmax(dim=1).item())

    # ------------------------------------------------------------- observe

    def observe(self, transition: Transition) -> None:
        observation = np.asarray(transition.observation)
        next_observation = np.asarray(transition.next_observation)
        if observation.shape != (self.observation_dim,):
            raise ValueError("Invalid transition observation shape")
        if next_observation.shape != (self.observation_dim,):
            raise ValueError("Invalid transition next_observation shape")
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all():
            raise ValueError("Transition observations must be finite")
        try:
            action = int(transition.action)
            action_exact = float(transition.action) == action
        except (TypeError, ValueError, OverflowError):
            action = -1
            action_exact = False
        if not action_exact or not 0 <= action < self.action_dim:
            raise ValueError("Transition action is outside the discrete action space")
        if not np.isfinite(float(transition.reward)):
            raise ValueError("Transition reward must be finite")
        if not isinstance(transition.terminated, (bool, np.bool_)):
            raise ValueError("Transition terminated mask must be boolean")
        self.replay.push(transition)
        self.env_steps += 1

    # ------------------------------------------------------------ optimize

    def optimize(self) -> Optional[Dict[str, float]]:
        if self.env_steps < self.config.warmup_steps:
            return None
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([x.observation for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([x.next_observation for x in batch]), dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor([x.terminated for x in batch], dtype=torch.float32, device=self.device)

        predictions = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN decouples the maximization: the online network selects
            # the next action and the target network evaluates that action.
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_values
        if self.config.loss == "huber":
            loss = nn.functional.smooth_l1_loss(predictions, targets)
        else:
            loss = nn.functional.mse_loss(predictions, targets)
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all() or not torch.isfinite(loss):
            raise RuntimeError("Double DQN update produced a non-finite prediction, target, or loss")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad()
            raise RuntimeError("Double DQN update produced a non-finite gradient norm")
        self.optimizer.step()
        self.gradient_steps += 1
        synced = self.gradient_steps % self.config.target_update_steps == 0
        if synced:
            self.target.load_state_dict(self.online.state_dict())
        return {
            "loss": float(loss.item()),
            "td_error_abs_mean": float((predictions.detach() - targets).abs().mean().item()),
            "q_taken_mean": float(predictions.detach().mean().item()),
            "target_mean": float(targets.mean().item()),
            # The shared column name is retained across the value-based family;
            # here it is Q_target at the online-network argmax action.
            "next_max_q_mean": float(next_values.mean().item()),
            "grad_norm": float(grad_norm),
            "target_synced": bool(synced),
            "batch_terminal_fraction": float(terminated.mean().item()),
            "batch_size": int(len(batch)),
            "replay_size": int(len(self.replay)),
            "learning_rate": float(self.config.learning_rate),
        }

    # --------------------------------------------------------- checkpoints

    def _base_payload(self, kind: str, extra: Optional[Dict]) -> Dict:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": ALGORITHM,
            "kind": kind,
            "online_state_dict": self.online.state_dict(),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "seed": self.seed,
            "env_steps": self.env_steps,
            "gradient_steps": self.gradient_steps,
            "extra": extra or {},
        }

    def save(self, path: str, kind: str = "full", extra: Optional[Dict] = None) -> None:
        if kind not in ("full", "policy_only"):
            raise ValueError("kind must be full or policy_only")
        payload = self._base_payload(kind, extra)
        if kind == "full":
            payload.update({
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replay": self.replay.state_dict(),
                "python_rng_state": self.rng.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            })
        atomic_torch_save(payload, path)

    def load(self, path: str, load_optimizer: bool = True) -> Dict:
        payload = _torch_load(path, self.device)
        saved_config = validate_checkpoint_payload(payload)
        if int(payload.get("observation_dim", -1)) != self.observation_dim or int(payload.get("action_dim", -1)) != self.action_dim:
            raise ValueError("checkpoint dimensions do not match this agent")
        if saved_config != self.config:
            raise ValueError("checkpoint configuration does not match this agent")
        self.online.load_state_dict(payload["online_state_dict"])
        self.target.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
        if load_optimizer and payload.get("kind") == "full":
            required = ("optimizer_state_dict", "replay", "python_rng_state", "numpy_rng_state", "torch_rng_state")
            if any(name not in payload for name in required):
                raise ValueError("full checkpoint is missing optimizer, replay, or RNG state")
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.replay.load_state_dict(payload["replay"])
            self.rng.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            self.seed = int(payload["seed"])
        self.env_steps = int(payload["env_steps"])
        self.gradient_steps = int(payload["gradient_steps"])
        return payload.get("extra", {})


class GreedyPolicy:
    """A frozen greedy policy loaded from a checkpoint file, used for evaluation
    so the learner's parameters and replay are never touched."""

    def __init__(self, path: str, device: torch.device) -> None:
        payload = _torch_load(path, device)
        config = validate_checkpoint_payload(payload)
        self.observation_dim = int(payload["observation_dim"])
        self.action_dim = int(payload["action_dim"])
        self.env_steps = int(payload["env_steps"])
        self.network = QNetwork(self.observation_dim, self.action_dim, config.hidden_size).to(device)
        self.network.load_state_dict(payload["online_state_dict"])
        self.network.eval()
        self.device = device
        self.parameter_digest = parameter_digest(self.network)

    def act(self, observation: np.ndarray, policy_mode: str = "greedy") -> int:
        if policy_mode != "greedy":
            raise ValueError("Double DQN evaluates greedily only")
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("observation shape mismatch")
        if not np.isfinite(observation).all() or np.any(observation < -1.000001) or np.any(observation > 1.000001):
            raise ValueError("evaluation observation must be finite and normalized")
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.network(tensor).argmax(dim=1).item())
