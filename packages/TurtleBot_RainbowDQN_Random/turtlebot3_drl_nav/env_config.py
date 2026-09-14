"""Loading of the shared environment configuration into typed objects.

Shared layer. No ROS dependency. ``config/common_environment.yaml`` is the single
authority for every environment value; nodes declare their ROS parameters from
it, tests build engines from it, and the configuration digest covers it.
"""

import math
import os
from typing import Any, Dict, Tuple

import yaml

from .episode_engine import EnvironmentConfig
from .geometry import Arena, parse_world
from .initialization import InitializationLaw, Sampler, law_from_parameters, require_seed
from .state import ActionMap, RewardConfig

COMMON_CONFIG_NAME = "common_environment.yaml"
ENV_NODE_NAME = "drl_environment"
SEED_PARAMETERS = ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed")


def load_common_parameters(config_dir: str) -> Dict[str, Any]:
    path = os.path.join(config_dir, COMMON_CONFIG_NAME)
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        params = document[ENV_NODE_NAME]["ros__parameters"]
    except (KeyError, TypeError):
        raise ValueError(f"{path} lacks {ENV_NODE_NAME}.ros__parameters")
    if not isinstance(params, dict):
        raise ValueError("ros__parameters must be a mapping")
    return dict(params)


def load_evaluation_protocol(config_dir: str) -> Dict[str, Any]:
    path = os.path.join(config_dir, COMMON_CONFIG_NAME)
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        protocol = dict(document["evaluation_protocol"])
    except (KeyError, TypeError):
        raise ValueError(f"{path} lacks evaluation_protocol")
    if protocol.get("tier2_conditions") != ["E2", "E3"]:
        raise ValueError("tier2_conditions must be the held-out E2 list plus the E3 anchor")
    if int(protocol.get("scenario_count", 0)) <= 0 or int(protocol.get("tier2_e3_episodes", 0)) <= 0:
        raise ValueError("evaluation episode counts must be positive")
    common_seeds = protocol.get("common_seeds", {})
    for name in SEED_PARAMETERS:
        require_seed(name, common_seeds.get(name, -1))
    if int(protocol.get("evaluation_seed_default", -1)) != int(common_seeds["evaluation_seed"]):
        raise ValueError("evaluation_seed_default must equal common_seeds.evaluation_seed")
    phases = protocol.get("phases", {})
    if set(phases) != {"calibration", "pilot", "controlled"}:
        raise ValueError("evaluation protocol must define calibration, pilot and controlled phases")
    for label, phase in phases.items():
        budget = int(phase["environment_budget"])
        checkpoint = int(phase["checkpoint_interval_steps"])
        full = int(phase["full_checkpoint_interval_steps"])
        seeds = [int(value) for value in phase["learning_seeds"]]
        tier2 = [int(value) for value in phase["tier2_checkpoint_steps"]]
        if min(budget, checkpoint, full, int(phase["tier1_episodes_per_checkpoint"])) <= 0:
            raise ValueError(f"{label} phase budgets, cadences and evaluation counts must be positive")
        if budget % checkpoint or full % checkpoint:
            raise ValueError(f"{label} checkpoint interval must divide the budget and full interval must be a checkpoint multiple")
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            raise ValueError(f"{label} learning seeds must be unique non-negative integers")
        if tier2 != sorted(set(tier2)) or not tier2 or tier2[-1] != budget or any(step <= 0 or step % checkpoint for step in tier2):
            raise ValueError(f"{label} tier2 checkpoint steps must be unique cadence points ending at the budget")
    return protocol


def action_map_from(params: Dict[str, Any]) -> ActionMap:
    return ActionMap(
        forward_linear=float(params["forward_linear"]),
        steering_linear=float(params["steering_linear"]),
        steering_angular=float(params["steering_angular"]),
        rotation_angular=float(params["rotation_angular"]),
    )


def reward_from(params: Dict[str, Any]) -> RewardConfig:
    reward = RewardConfig(
        distance=float(params["reward_distance"]),
        step=float(params["reward_step"]),
        collision=float(params["reward_collision"]),
        goal=float(params["reward_goal"]),
        angular=float(params["reward_angular"]),
        near=float(params["reward_near"]),
        safe_distance=float(params["safe_distance"]),
        stop_distance=float(params["stop_distance"]),
        goal_tolerance=float(params["goal_tolerance"]),
    )
    reward.validate()
    return reward


def dynamic_axes(params: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    names = list(params["dynamic_obstacle_names"])
    axes = list(params["dynamic_obstacle_axes"])
    if len(names) != 2 or len(axes) != 2:
        raise ValueError("exactly two dynamic obstacles are declared in Phase 1")
    return tuple(zip(names, axes))


def build_arena(params: Dict[str, Any], world_path: str) -> Arena:
    arena = parse_world(world_path, dynamic_axes(params))
    if not arena.has_state_plugin:
        raise ValueError("the world does not load libgazebo_ros_state.so; the transactional reset needs it")
    return arena


def build_law(params: Dict[str, Any]) -> InitializationLaw:
    law = law_from_parameters(params, params["obstacle_speed"], params["obstacle_half_period"])
    law.validate()
    if law.start_clearance_min < float(params["safe_distance"]):
        raise ValueError("init_start_clearance_min must be >= safe_distance so no start begins in the near band")
    return law


def build_environment_config(params: Dict[str, Any], require_seeds: bool = True) -> EnvironmentConfig:
    names = list(params["dynamic_obstacle_names"])
    if len(names) != 2 or len(set(str(name) for name in names)) != 2:
        raise ValueError("exactly two distinct dynamic obstacle names are required")
    if str(params["lidar_sampling"]) not in ("nearest", "sector_min"):
        raise ValueError("lidar_sampling must be nearest or sector_min")
    positive = (
        "lidar_max", "distance_max", "control_period", "episode_max_steps",
        "obstacle_half_period", "init_settle_sim_seconds", "init_position_tolerance",
        "init_yaw_tolerance", "init_odom_tolerance", "init_odom_yaw_tolerance",
        "hold_tolerance_fraction", "sensor_max_age_sim_seconds", "obstacle_position_tolerance",
        "service_timeout_wall_seconds", "obstacle_ack_timeout_wall_seconds",
        "sensor_timeout_wall_seconds", "action_timeout_wall_seconds",
    )
    if any(float(params[name]) <= 0.0 for name in positive):
        raise ValueError("environment dimensions, periods and tolerances must be positive")
    if float(params["decision_gap_tolerance_sim_seconds"]) < 0.0:
        raise ValueError("decision_gap_tolerance_sim_seconds must be non-negative")
    if float(params["sensor_max_age_sim_seconds"]) > float(params["control_period"]):
        raise ValueError("sensor_max_age_sim_seconds must not exceed one control period")
    numeric = [
        float(params[name]) for name in positive + (
            "decision_gap_tolerance_sim_seconds", "goal_x", "goal_y", "obstacle_speed",
            "fixed_start_x", "fixed_start_y", "fixed_start_yaw",
        )
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("environment numerical parameters must be finite")
    if float(params["obstacle_speed"]) <= 0.0:
        raise ValueError("obstacle_speed must be positive")
    seeds = {}
    for name in SEED_PARAMETERS:
        seeds[name] = require_seed(name, params[name]) if require_seeds else int(params[name])
    return EnvironmentConfig(
        goal_x=float(params["goal_x"]),
        goal_y=float(params["goal_y"]),
        lidar_max=float(params["lidar_max"]),
        distance_max=float(params["distance_max"]),
        lidar_sampling=str(params["lidar_sampling"]),
        control_period=float(params["control_period"]),
        episode_max_steps=int(params["episode_max_steps"]),
        action_map=action_map_from(params),
        reward=reward_from(params),
        robot_model_name=str(params["robot_model_name"]),
        obstacle_names=(str(names[0]), str(names[1])),
        obstacle_speed=float(params["obstacle_speed"]),
        obstacle_half_period=float(params["obstacle_half_period"]),
        settle_sim_s=float(params["init_settle_sim_seconds"]),
        position_tolerance=float(params["init_position_tolerance"]),
        yaw_tolerance=float(params["init_yaw_tolerance"]),
        odom_tolerance=float(params["init_odom_tolerance"]),
        odom_yaw_tolerance=float(params["init_odom_yaw_tolerance"]),
        hold_tolerance_fraction=float(params["hold_tolerance_fraction"]),
        decision_gap_tolerance_sim_s=float(params["decision_gap_tolerance_sim_seconds"]),
        sensor_max_age_sim_s=float(params["sensor_max_age_sim_seconds"]),
        obstacle_position_tolerance=float(params["obstacle_position_tolerance"]),
        fixed_start=(float(params["fixed_start_x"]), float(params["fixed_start_y"]), float(params["fixed_start_yaw"])),
        initialization_seed=seeds["initialization_seed"],
        dynamic_obstacle_seed=seeds["dynamic_obstacle_seed"],
        evaluation_seed=seeds["evaluation_seed"],
        service_timeout_wall_s=float(params["service_timeout_wall_seconds"]),
        obstacle_ack_timeout_wall_s=float(params["obstacle_ack_timeout_wall_seconds"]),
        sensor_timeout_wall_s=float(params["sensor_timeout_wall_seconds"]),
        action_timeout_wall_s=float(params["action_timeout_wall_seconds"]),
        fatal_on_reset_contact=bool(params["fatal_on_reset_contact"]),
    )


def build_sampler(params: Dict[str, Any], world_path: str) -> Sampler:
    return Sampler(build_law(params), build_arena(params, world_path))
