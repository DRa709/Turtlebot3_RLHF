"""Phase-1 observation, action, reward, and termination definitions.

This module deliberately has no ROS dependency so its numerical semantics can be
unit tested on a workstation or on ARC without starting Gazebo.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


LIDAR_BINS = 36
OBSERVATION_DIM = LIDAR_BINS + 5
ACTION_NAMES = (
    "forward",
    "forward_left",
    "forward_right",
    "rotate_left",
    "rotate_right",
)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ActionMap:
    forward_linear: float = 0.15
    steering_linear: float = 0.12
    steering_angular: float = 0.6
    rotation_angular: float = 1.0

    def commands(self) -> Tuple[Tuple[float, float], ...]:
        return (
            (self.forward_linear, 0.0),
            (self.steering_linear, self.steering_angular),
            (self.steering_linear, -self.steering_angular),
            (0.0, self.rotation_angular),
            (0.0, -self.rotation_angular),
        )

    @property
    def v_max(self) -> float:
        return max(abs(v) for v, _ in self.commands())

    @property
    def omega_max(self) -> float:
        return max(abs(w) for _, w in self.commands())


@dataclass(frozen=True)
class RewardConfig:
    distance: float = 10.0
    step: float = 0.01
    collision: float = 100.0
    goal: float = 100.0
    angular: float = 0.05
    near: float = 0.25
    safe_distance: float = 0.30
    stop_distance: float = 0.16
    goal_tolerance: float = 0.20

    def validate(self) -> None:
        if not 0.0 < self.stop_distance < self.safe_distance:
            raise ValueError("Require 0 < stop_distance < safe_distance")
        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive")


@dataclass(frozen=True)
class RewardResult:
    total: float
    distance_progress: float
    step_penalty: float
    collision_penalty: float
    goal_bonus: float
    angular_penalty: float
    near_penalty: float
    collision: bool
    safety: bool
    goal: bool

    @property
    def terminated(self) -> bool:
        return self.collision or self.safety or self.goal

    @property
    def event(self) -> str:
        if self.collision:
            return "collision"
        if self.safety:
            return "safety"
        if self.goal:
            return "goal"
        return "running"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def goal_features(pose: Pose2D, goal_x: float, goal_y: float) -> Tuple[float, float]:
    dx = goal_x - pose.x
    dy = goal_y - pose.y
    return math.hypot(dx, dy), normalize_angle(math.atan2(dy, dx) - pose.yaw)


def _clean_range(value: float, range_min: float, range_max: float) -> float:
    # REP 117: +Inf means no return beyond maximum range; -Inf means the
    # obstacle is closer than the minimum measurable range.  NaN/zero are
    # invalid measurements and are treated conservatively as minimum range so
    # a sensor fault cannot masquerade as free space.
    if value == math.inf:
        return range_max
    if not math.isfinite(value) or value <= 0.0:
        return range_min
    return min(max(value, range_min), range_max)


def sample_lidar_nearest(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    bins: int = LIDAR_BINS,
) -> List[float]:
    """Sample a full-revolution scan at canonical uniformly spaced bearings."""
    if not ranges or angle_increment == 0.0 or bins <= 0:
        return [range_max] * bins
    result: List[float] = []
    for index in range(bins):
        target = -math.pi + (2.0 * math.pi * index / bins)
        # The stock Burger LDS reports a full revolution as [0, 2*pi), while
        # our canonical bearings are [-pi, pi). Wrap across the scan seam so
        # negative canonical bearings address the rear/right beams instead of
        # collapsing to ranges[0].
        raw_index = int(round((target - angle_min) / angle_increment)) % len(ranges)
        result.append(_clean_range(ranges[raw_index], range_min, range_max))
    return result


def sample_lidar_sector_min(
    ranges: Sequence[float],
    range_min: float,
    range_max: float,
    bins: int = LIDAR_BINS,
) -> List[float]:
    """Fallback that preserves thin obstacles by taking each sector minimum."""
    if not ranges or bins <= 0:
        return [range_max] * bins
    result: List[float] = []
    for index in range(bins):
        start = int(index * len(ranges) / bins)
        end = max(start + 1, int((index + 1) * len(ranges) / bins))
        values = [_clean_range(x, range_min, range_max) for x in ranges[start:end]]
        result.append(min(values) if values else range_max)
    return result


def min_valid_range(ranges: Iterable[float], fallback: float) -> float:
    finite = [x for x in ranges if math.isfinite(x) and x > 0.0]
    return min(finite) if finite else fallback


def build_observation(
    lidar: Sequence[float],
    distance: float,
    heading_error: float,
    previous_linear: float,
    previous_angular: float,
    lidar_max: float,
    distance_max: float,
    velocity_max: float,
    angular_max: float,
) -> List[float]:
    if len(lidar) != LIDAR_BINS:
        raise ValueError("Phase-1 requires exactly 36 LiDAR samples")
    if min(lidar_max, distance_max, velocity_max, angular_max) <= 0.0:
        raise ValueError("All normalization bounds must be positive")
    normalized_lidar = [min(max(x / lidar_max, 0.0), 1.0) for x in lidar]
    return normalized_lidar + [
        min(max(distance / distance_max, 0.0), 1.0),
        math.sin(heading_error),
        math.cos(heading_error),
        min(max(previous_linear / velocity_max, -1.0), 1.0),
        min(max(previous_angular / angular_max, -1.0), 1.0),
    ]


def compute_reward(
    previous_distance: float,
    current_distance: float,
    min_scan: float,
    executed_angular: float,
    collision_contact: bool,
    config: RewardConfig,
) -> RewardResult:
    """Compute the roadmap reward with collision > safety > goal precedence."""
    config.validate()
    collision = bool(collision_contact)
    safety = (not collision) and min_scan < config.stop_distance
    goal = (
        (not collision)
        and (not safety)
        and current_distance <= config.goal_tolerance
    )
    distance_progress = config.distance * (previous_distance - current_distance)
    step_penalty = -config.step
    collision_penalty = -config.collision if collision else 0.0
    goal_bonus = config.goal if goal else 0.0
    angular_penalty = -config.angular * abs(executed_angular)
    near_penalty = -config.near if min_scan < config.safe_distance else 0.0
    total = sum(
        (
            distance_progress,
            step_penalty,
            collision_penalty,
            goal_bonus,
            angular_penalty,
            near_penalty,
        )
    )
    return RewardResult(
        total=total,
        distance_progress=distance_progress,
        step_penalty=step_penalty,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        angular_penalty=angular_penalty,
        near_penalty=near_penalty,
        collision=collision,
        safety=safety,
        goal=goal,
    )
