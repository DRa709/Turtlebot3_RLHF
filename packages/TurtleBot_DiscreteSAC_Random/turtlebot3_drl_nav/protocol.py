"""Wire format shared by the environment node and every agent node.

Shared layer, protocol version 3. Numeric messages travel as
std_msgs/Float32MultiArray; structured control messages travel as JSON text in
std_msgs/String. Any change here is a shared-layer change and moves the
shared-layer digest.
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .state import OBSERVATION_DIM, RewardResult

PROTOCOL_VERSION = 3.0

STATE_HEADER = ["protocol_version", "sequence", "episode_key", "step_index", "sim_time"]
ACTION_FIELDS = ["protocol_version", "sequence", "action_index", "linear", "angular", "final_transition"]
STEP_HEADER = [
    "protocol_version", "sequence", "episode_key", "step_index", "sim_time",
    "hold_sim_s", "hold_odom_s", "scan_age_s", "odom_age_s", "obs1_age_s", "obs2_age_s",
    "decision_gap_sim_s", "decision_latency_wall_s", "obstacle_position_error_max",
    "reward_total", "r_distance", "r_step", "r_collision", "r_goal", "r_angular", "r_near",
    "terminated", "episode_end", "truncated",
    "collision", "static_collision", "dynamic_collision", "safety", "goal",
    "min_lidar", "distance", "heading_error", "x", "y", "yaw",
    "obs1_x", "obs1_y", "obs2_x", "obs2_y",
]
STATE_HEADER_SIZE = len(STATE_HEADER)
STEP_HEADER_SIZE = len(STEP_HEADER)


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is not numeric")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: object, name: str, minimum: int = 0) -> int:
    number = _finite(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _boolean(value: object, name: str) -> bool:
    number = _finite(value, name)
    if number not in (0.0, 1.0):
        raise ValueError(f"{name} must be encoded as 0 or 1")
    return bool(number)


def _json_numbers_are_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    return True


@dataclass(frozen=True)
class StateMessage:
    sequence: int
    episode_key: int
    step_index: int
    sim_time: float
    observation: np.ndarray


@dataclass(frozen=True)
class ActionMessage:
    sequence: int
    action_index: int
    linear: float
    angular: float
    final_transition: bool


@dataclass(frozen=True)
class StepMessage:
    sequence: int
    episode_key: int
    step_index: int
    sim_time: float
    hold_sim_s: float
    hold_odom_s: float
    scan_age_s: float
    odom_age_s: float
    obs1_age_s: float
    obs2_age_s: float
    decision_gap_sim_s: float
    decision_latency_wall_s: float
    obstacle_position_error_max: float
    reward_total: float
    r_distance: float
    r_step: float
    r_collision: float
    r_goal: float
    r_angular: float
    r_near: float
    terminated: bool
    episode_end: bool
    truncated: bool
    collision: bool
    static_collision: bool
    dynamic_collision: bool
    safety: bool
    goal: bool
    min_lidar: float
    distance: float
    heading_error: float
    x: float
    y: float
    yaw: float
    obs1_x: float
    obs1_y: float
    obs2_x: float
    obs2_y: float
    observation: np.ndarray


def encode_state(sequence: int, episode_key: int, step_index: int, sim_time: float, observation: Sequence[float]) -> List[float]:
    if len(observation) != OBSERVATION_DIM:
        raise ValueError("Invalid Phase-1 observation length")
    return [PROTOCOL_VERSION, float(sequence), float(episode_key), float(step_index), float(sim_time)] + [
        float(v) for v in observation
    ]


def decode_state(values: Sequence[float]) -> StateMessage:
    expected = STATE_HEADER_SIZE + OBSERVATION_DIM
    if len(values) != expected:
        raise ValueError(f"Expected {expected} state values, got {len(values)}")
    if _finite(values[0], "protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: {values[0]} != {PROTOCOL_VERSION}")
    sim_time = _finite(values[4], "sim_time")
    observation = np.asarray([_finite(value, "observation") for value in values[STATE_HEADER_SIZE:]], dtype=np.float32)
    return StateMessage(
        sequence=_integer(values[1], "sequence", 1),
        episode_key=_integer(values[2], "episode_key", 1),
        step_index=_integer(values[3], "step_index"),
        sim_time=sim_time,
        observation=observation,
    )


def encode_action(sequence: int, action_index: int, linear: float, angular: float, final_transition: bool) -> List[float]:
    return [PROTOCOL_VERSION, float(sequence), float(action_index), float(linear), float(angular), float(bool(final_transition))]


def decode_action(values: Sequence[float]) -> ActionMessage:
    if len(values) != len(ACTION_FIELDS):
        raise ValueError(f"Expected {len(ACTION_FIELDS)} action values, got {len(values)}")
    if _finite(values[0], "protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Protocol version mismatch in action")
    return ActionMessage(
        sequence=_integer(values[1], "sequence", 1),
        action_index=_integer(values[2], "action_index"),
        linear=_finite(values[3], "linear"),
        angular=_finite(values[4], "angular"),
        final_transition=_boolean(values[5], "final_transition"),
    )


def encode_step(
    sequence: int,
    episode_key: int,
    step_index: int,
    sim_time: float,
    hold_sim_s: float,
    hold_odom_s: float,
    scan_age_s: float,
    odom_age_s: float,
    obs1_age_s: float,
    obs2_age_s: float,
    decision_gap_sim_s: float,
    decision_latency_wall_s: float,
    obstacle_position_error_max: float,
    reward: RewardResult,
    episode_end: bool,
    truncated: bool,
    static_collision: bool,
    dynamic_collision: bool,
    min_lidar: float,
    distance: float,
    heading_error: float,
    pose: Sequence[float],
    obstacle_positions: Sequence[Sequence[float]],
    observation: Sequence[float],
) -> List[float]:
    if len(observation) != OBSERVATION_DIM:
        raise ValueError("Invalid Phase-1 observation length")
    if len(obstacle_positions) != 2:
        raise ValueError("Two dynamic obstacle positions are required")
    header = [
        PROTOCOL_VERSION, float(sequence), float(episode_key), float(step_index), float(sim_time),
        float(hold_sim_s), float(hold_odom_s), float(scan_age_s), float(odom_age_s),
        float(obs1_age_s), float(obs2_age_s), float(decision_gap_sim_s),
        float(decision_latency_wall_s), float(obstacle_position_error_max),
        float(reward.total), float(reward.distance_progress), float(reward.step_penalty),
        float(reward.collision_penalty), float(reward.goal_bonus), float(reward.angular_penalty),
        float(reward.near_penalty),
        float(reward.terminated), float(episode_end), float(truncated),
        float(reward.collision), float(static_collision), float(dynamic_collision),
        float(reward.safety), float(reward.goal),
        float(min_lidar), float(distance), float(heading_error),
        float(pose[0]), float(pose[1]), float(pose[2]),
        float(obstacle_positions[0][0]), float(obstacle_positions[0][1]),
        float(obstacle_positions[1][0]), float(obstacle_positions[1][1]),
    ]
    assert len(header) == STEP_HEADER_SIZE
    return header + [float(v) for v in observation]


def decode_step(values: Sequence[float]) -> StepMessage:
    expected = STEP_HEADER_SIZE + OBSERVATION_DIM
    if len(values) != expected:
        raise ValueError(f"Expected {expected} transition values, got {len(values)}")
    if _finite(values[0], "protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Protocol version mismatch in step")
    v = [_finite(x, STEP_HEADER[index]) for index, x in enumerate(values[:STEP_HEADER_SIZE])]
    observation = np.asarray([_finite(value, "observation") for value in values[STEP_HEADER_SIZE:]], dtype=np.float32)
    return StepMessage(
        sequence=_integer(v[1], "sequence", 1), episode_key=_integer(v[2], "episode_key", 1),
        step_index=_integer(v[3], "step_index", 1), sim_time=v[4],
        hold_sim_s=v[5], hold_odom_s=v[6], scan_age_s=v[7], odom_age_s=v[8],
        obs1_age_s=v[9], obs2_age_s=v[10], decision_gap_sim_s=v[11],
        decision_latency_wall_s=v[12], obstacle_position_error_max=v[13],
        reward_total=v[14], r_distance=v[15], r_step=v[16], r_collision=v[17], r_goal=v[18],
        r_angular=v[19], r_near=v[20],
        terminated=_boolean(v[21], "terminated"), episode_end=_boolean(v[22], "episode_end"),
        truncated=_boolean(v[23], "truncated"), collision=_boolean(v[24], "collision"),
        static_collision=_boolean(v[25], "static_collision"), dynamic_collision=_boolean(v[26], "dynamic_collision"),
        safety=_boolean(v[27], "safety"), goal=_boolean(v[28], "goal"),
        min_lidar=v[29], distance=v[30], heading_error=v[31], x=v[32], y=v[33], yaw=v[34],
        obs1_x=v[35], obs1_y=v[36], obs2_x=v[37], obs2_y=v[38],
        observation=observation,
    )


# ------------------------------------------------------------- JSON control

def encode_json(payload: Dict[str, object]) -> str:
    payload = dict(payload)
    payload["protocol_version"] = PROTOCOL_VERSION
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_json(text: str) -> Dict[str, object]:
    try:
        payload = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {token}")))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON control message: {error}")
    if not isinstance(payload, dict):
        raise ValueError("control message is not a JSON object")
    if not _json_numbers_are_finite(payload):
        raise ValueError("control message contains a non-finite number")
    if _finite(payload.get("protocol_version", -1), "protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Protocol version mismatch in control message")
    return payload


@dataclass(frozen=True)
class EpisodeSpec:
    """What the agent asks the environment to initialize next."""

    phase: str                       # "training" | "evaluation"
    policy_mode: str                 # training: epsilon_greedy/noisy/stochastic; evaluation: greedy/stochastic/deterministic
    condition: Optional[str] = None  # "E1" | "E2" | "E3" for evaluation
    checkpoint_step: Optional[int] = None
    checkpoint_index: Optional[int] = None
    scenario_id: Optional[str] = None
    evaluation_episode: Optional[int] = None

    def validate(self) -> None:
        if self.phase not in ("training", "evaluation"):
            raise ValueError("phase must be training or evaluation")
        if self.policy_mode not in ("epsilon_greedy", "noisy", "greedy", "stochastic", "deterministic"):
            raise ValueError("unknown policy_mode")
        if self.phase == "evaluation":
            if self.policy_mode in ("epsilon_greedy", "noisy"):
                raise ValueError("epsilon_greedy and noisy are training-only policy modes")
            if self.condition not in ("E1", "E2", "E3"):
                raise ValueError("evaluation episodes need condition E1, E2 or E3")
            if self.checkpoint_step is None or self.checkpoint_index is None or self.evaluation_episode is None:
                raise ValueError("evaluation episodes need checkpoint_step, checkpoint_index, evaluation_episode")
            if self.checkpoint_step <= 0 or self.checkpoint_index <= 0 or self.evaluation_episode <= 0:
                raise ValueError("evaluation checkpoint and episode indices must be positive")
            if self.condition == "E2" and not self.scenario_id:
                raise ValueError("E2 episodes need a scenario_id")
            if self.condition != "E2" and self.scenario_id is not None:
                raise ValueError("only E2 episodes carry a scenario_id")
        else:
            if self.policy_mode not in ("epsilon_greedy", "noisy", "stochastic"):
                raise ValueError("training policy_mode must describe its exploratory policy")
            if any(value is not None for value in (
                self.condition, self.checkpoint_step, self.checkpoint_index,
                self.scenario_id, self.evaluation_episode,
            )):
                raise ValueError("training episodes carry no evaluation identifiers")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["cmd"] = "start_episode"
        return encode_json(payload)

    @staticmethod
    def from_payload(payload: Dict[str, object]) -> "EpisodeSpec":
        spec = EpisodeSpec(
            phase=str(payload["phase"]),
            policy_mode=str(payload["policy_mode"]),
            condition=payload.get("condition"),
            checkpoint_step=None if payload.get("checkpoint_step") is None else int(payload["checkpoint_step"]),
            checkpoint_index=None if payload.get("checkpoint_index") is None else int(payload["checkpoint_index"]),
            scenario_id=payload.get("scenario_id"),
            evaluation_episode=None if payload.get("evaluation_episode") is None else int(payload["evaluation_episode"]),
        )
        spec.validate()
        return spec
