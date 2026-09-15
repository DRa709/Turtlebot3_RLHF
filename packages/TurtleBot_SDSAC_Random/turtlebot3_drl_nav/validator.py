"""Run validator (Gate 7). Shared layer; no ROS dependency.

    python3 -m turtlebot3_drl_nav.validator --package-root <pkg> --run-dir <run> [--phase-type training|evaluation]

Reads the five canonical streams, the run identity, and the frozen configuration,
and refuses the run unless every structural identity of DATA_CONTRACT.md holds:
identity uniformity, contiguity, exact budget, mask consistency, precedence,
reward-sum identity, action-map cross-check, hold tolerance, episode/transition
agreement, seeded-initialization reproducibility (every recorded requested start
pose is recomputed from the seeds and must match), update cadence and
applicability, checkpoint schedule and digests, and evaluation completeness.
Writes validation_report.json into the run directory and exits non-zero on any
failure.
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import yaml

from .env_config import build_law, build_arena, load_common_parameters, load_evaluation_protocol, reward_from
from .identity import (
    IDENTITY_FIELDS, config_digest, read_identity, release_digest, sha256_file, shared_layer_digest,
)
from .initialization import (
    Sampler,
    e1_start_seed,
    e1_obstacle_seed,
    e3_obstacle_seed,
    obstacle_phases,
    read_scenarios,
    scenario_by_id,
    training_start_seed,
    training_obstacle_seed,
)
from .obstacle_schedule import obstacle_displacement
from .recorder import (
    CHECKPOINTS_COLUMNS,
    EPISODES_COLUMNS,
    EVALUATION_COLUMNS,
    TRANSITIONS_COLUMNS,
    UPDATE_APPLICABILITY,
    UPDATES_COLUMNS,
    read_stream,
)
from .state import ACTION_NAMES, OBSERVATION_DIM, ActionMap, compute_reward

STREAM_SCHEMAS = {
    "transitions": TRANSITIONS_COLUMNS,
    "episodes": EPISODES_COLUMNS,
    "updates": UPDATES_COLUMNS,
    "evaluation": EVALUATION_COLUMNS,
    "checkpoints": CHECKPOINTS_COLUMNS,
}

REWARD_TOLERANCE = 1e-9
FLOAT_TOLERANCE = 1e-6


class Report:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks: List[str] = []
        self.stats: Dict[str, object] = {}

    def check(self, condition: bool, message: str) -> bool:
        self.checks.append(message)
        if not condition:
            self.failures.append(message)
        return bool(condition)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json(self) -> Dict[str, object]:
        return {"passed": self.passed, "failures": self.failures, "checks_run": len(self.checks), "stats": self.stats}


def _f(value: str) -> Optional[float]:
    return None if value == "" else float(value)


def _i(value: str) -> Optional[int]:
    if value == "":
        return None
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        raise ValueError(f"expected an integer, got {value!r}")
    return int(value, 10)


def _b(value: str) -> Optional[bool]:
    if value == "":
        return None
    if value not in ("0", "1"):
        raise ValueError(f"expected a 0/1 boolean, got {value!r}")
    return value == "1"


def _angle_error(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _matches(recorded: Optional[float], expected: Optional[float], tolerance: float = FLOAT_TOLERANCE) -> bool:
    if expected is None:
        return recorded is None
    return recorded is not None and math.isfinite(recorded) and abs(recorded - expected) <= tolerance


def _summary_matches_transitions(summary, episode_rows, common) -> bool:
    """Reconstruct one episodes/evaluation row from its raw trajectory."""
    try:
        ordered = sorted(episode_rows, key=lambda row: _i(row["step_in_episode"]))
        if not ordered or _i(ordered[0]["step_in_episode"]) != 0:
            return False
        steps = ordered[1:]
        if not steps or not _b(steps[-1]["episode_end"]):
            return False
        final = steps[-1]
        initial = ordered[0]
        if not (
            _matches(_f(summary["init_sim_time"]), _f(initial["sim_time"]))
            and _matches(_f(summary["init_scan_clearance"]), _f(initial["min_lidar"]))
            and _matches(_f(summary["odom_x"]), _f(initial["x"]))
            and _matches(_f(summary["odom_y"]), _f(initial["y"]))
            and _matches(_f(summary["odom_yaw"]), _f(initial["yaw"]))
            and _matches(_f(summary["obs1_reset_x"]), _f(initial["obs1_x"]))
            and _matches(_f(summary["obs1_reset_y"]), _f(initial["obs1_y"]))
            and _matches(_f(summary["obs2_reset_x"]), _f(initial["obs2_x"]))
            and _matches(_f(summary["obs2_reset_y"]), _f(initial["obs2_y"]))
        ):
            return False
        path_length = sum(
            math.hypot(_f(current["x"]) - _f(previous["x"]), _f(current["y"]) - _f(previous["y"]))
            for previous, current in zip(ordered, ordered[1:])
        )
        reward_fields = {
            "sum_r_distance": "r_distance", "sum_r_step": "r_step", "sum_r_collision": "r_collision",
            "sum_r_goal": "r_goal", "sum_r_angular": "r_angular", "sum_r_near": "r_near",
        }
        if _i(summary["length"]) != len(steps) or not _matches(_f(summary["return"]), sum(_f(row["reward_total"]) for row in steps)):
            return False
        if any(not _matches(_f(summary[out]), sum(_f(row[source]) for row in steps)) for out, source in reward_fields.items()):
            return False
        collision, static, dynamic = _b(final["collision"]), _b(final["static_collision"]), _b(final["dynamic_collision"])
        safety, goal, truncated = _b(final["safety"]), _b(final["goal"]), _b(final["truncated"])
        outcome = (
            "collision_both" if collision and static and dynamic else
            "collision_static" if collision and static else
            "collision_dynamic" if collision and dynamic else
            "safety" if safety else "goal" if goal else "timeout"
        )
        if summary["outcome"] != outcome:
            return False
        for field, expected in (
            ("collision", collision), ("static_collision", static), ("dynamic_collision", dynamic),
            ("safety", safety), ("goal", goal), ("truncated", truncated),
        ):
            if _b(summary[field]) != expected:
                return False
        min_clearance = min(_f(row["min_lidar"]) for row in ordered)
        near_steps = sum(_f(row["r_near"]) < 0.0 for row in steps)
        straight = math.hypot(_f(summary["requested_x"]) - _f(summary["goal_x"]), _f(summary["requested_y"]) - _f(summary["goal_y"]))
        efficiency = straight / path_length if goal and path_length > 0.0 else None
        time_to_goal = len(steps) if goal else None
        holds = [_f(row["hold_sim_s"]) for row in steps]
        decisions = [_f(row["decision_latency_wall_s"]) for row in steps]
        gaps = [_f(row["decision_gap_sim_s"]) for row in steps]
        sensor_ages = [_f(row[field]) for row in steps for field in ("scan_age_s", "odom_age_s", "obs1_age_s", "obs2_age_s")]
        obstacle_errors = [_f(row["obstacle_position_error_max"]) for row in steps]
        hold_tolerance = float(common["hold_tolerance_fraction"]) * float(common["control_period"])
        out_of_tolerance = sum(abs(_f(row["hold_odom_s"]) - float(common["control_period"])) > hold_tolerance for row in steps)
        expected_floats = {
            "min_clearance": min_clearance,
            "path_length": path_length,
            "straight_line_distance": straight,
            "path_efficiency": efficiency,
            "time_to_goal_steps": time_to_goal,
            "sim_time_start": _f(ordered[0]["sim_time"]),
            "sim_time_end": _f(final["sim_time"]),
            "mean_hold_sim_s": sum(holds) / len(holds),
            "max_hold_sim_s": max(holds),
            "mean_decision_latency_wall_s": sum(decisions) / len(decisions),
            "max_decision_latency_wall_s": max(decisions),
            "max_decision_gap_sim_s": max(abs(value) for value in gaps),
            "max_sensor_age_s": max(sensor_ages),
            "max_obstacle_position_error": max(obstacle_errors),
        }
        if any(not _matches(_f(summary[field]), expected) for field, expected in expected_floats.items()):
            return False
        if _i(summary["near_penalty_steps"]) != near_steps or _i(summary["hold_out_of_tolerance_steps"]) != out_of_tolerance:
            return False
        wall_time = _f(summary["wall_time_s"])
        duration = _f(summary["sim_time_end"]) - _f(summary["sim_time_start"])
        if wall_time is None or not math.isfinite(wall_time) or wall_time <= 0.0:
            return False
        if not _matches(_f(summary["rtf"]), duration / wall_time):
            return False
        return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _load_algorithm_params(package_root: str, algorithm: str) -> Dict[str, object]:
    path = os.path.join(package_root, "config", f"phase1_{algorithm.lower()}.yaml")
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    node = f"{algorithm.lower()}_agent"
    return dict(document[node]["ros__parameters"])


def _check_header(report: Report, run_dir: str, name: str) -> Optional[List[Dict[str, str]]]:
    path = os.path.join(run_dir, f"{name}.csv")
    if not report.check(os.path.isfile(path), f"{name}.csv present"):
        return None
    with open(path, newline="", encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split(",")
    expected = list(IDENTITY_FIELDS) + list(STREAM_SCHEMAS[name])
    report.check(header == expected, f"{name}.csv header matches the frozen schema")
    return read_stream(path)


def _sidecar_digest(path: str) -> Optional[str]:
    """Read one sha256sum-format line without trusting the recorded path."""
    try:
        with open(path, encoding="utf-8") as stream:
            lines = [line.strip() for line in stream if line.strip()]
        if len(lines) != 1:
            return None
        digest = lines[0].split(None, 1)[0]
        return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else None
    except OSError:
        return None


def _validate_runtime_provenance(report: Report, package_root: str, run_dir: str,
                                 identity: Dict[str, object], budget: int, phase_type: str) -> None:
    """Bind the run metadata and exact simulator inputs to the authenticated package."""
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest: Dict[str, object] = {}
    if report.check(os.path.isfile(manifest_path), "run_manifest.json present"):
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                candidate = json.load(stream)
            if not isinstance(candidate, dict):
                raise ValueError("manifest is not a JSON object")
            manifest = candidate
        except (OSError, ValueError, TypeError) as error:
            report.check(False, f"run_manifest.json is readable JSON: {error}")
    if manifest:
        report.check(
            all(manifest.get(field) == identity[field] for field in IDENTITY_FIELDS),
            "run manifest carries the exact run identity",
        )
        if phase_type == "training":
            report.check(
                type(manifest.get("environment_budget")) is int and manifest.get("environment_budget") == budget,
                "run manifest records the exact training budget",
            )
        slurm_values = [manifest.get(field) for field in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id")]
        report.check(
            all(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) for value in slurm_values),
            "run manifest carries numeric Slurm job, array-job and array-task identities",
        )
        report.check(
            isinstance(manifest.get("hostname"), str) and bool(manifest.get("hostname"))
            and isinstance(manifest.get("started_utc"), str)
            and bool(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[^\r\n]+Z", str(manifest.get("started_utc")))),
            "run manifest carries host and UTC start provenance",
        )

    runtime_path = os.path.join(run_dir, "runtime_manifest.txt")
    runtime_text = ""
    if report.check(os.path.isfile(runtime_path), "runtime_manifest.txt present"):
        try:
            with open(runtime_path, encoding="utf-8") as stream:
                runtime_text = stream.read()
        except OSError:
            runtime_text = ""
    report.check(
        all(token in runtime_text for token in ("Python ", "numpy=", "torch=", "ros_distro=foxy", "Gazebo", "ros_domain_id=", "gazebo_master_uri=")),
        "runtime manifest identifies Python, NumPy, PyTorch, Foxy, Gazebo and leased transports",
    )

    runtime_world = os.path.join(run_dir, "sim", "phase1_mixed.world")
    runtime_robot = os.path.join(run_dir, "sim", "turtlebot3_burger_with_contact.sdf")
    world_sidecar = os.path.join(run_dir, "sim", "runtime_world.sha256")
    robot_sidecar = os.path.join(run_dir, "sim", "runtime_robot_model.sha256")
    files_present = all(os.path.isfile(path) for path in (runtime_world, runtime_robot, world_sidecar, robot_sidecar))
    report.check(files_present, "runtime world, contact-instrumented robot model and digest sidecars are present")
    if files_present:
        package_world = os.path.join(package_root, "worlds", str(identity["world_id"]) + ".world")
        report.check(
            sha256_file(runtime_world) == sha256_file(package_world) == _sidecar_digest(world_sidecar),
            "runtime world is byte-identical to the authenticated package world",
        )
        report.check(
            sha256_file(runtime_robot) == _sidecar_digest(robot_sidecar),
            "runtime contact-instrumented robot model matches its recorded digest",
        )


def validate_run(package_root: str, run_dir: str, phase_type: Optional[str] = None) -> Report:
    report = Report()
    identity_path = os.path.join(run_dir, "run_identity.json")
    if not report.check(os.path.isfile(identity_path), "run_identity.json present"):
        return report
    identity = read_identity(identity_path)
    phase_type = phase_type or str(identity["phase_type"])
    report.check(phase_type == str(identity["phase_type"]), "requested phase_type matches run_identity.json")
    report.check(phase_type in ("training", "evaluation"), "phase_type is training or evaluation")
    algorithm = str(identity["algorithm"])
    marker_path = os.path.join(package_root, "ALGORITHM")
    version_path = os.path.join(package_root, "VERSION")
    with open(marker_path, encoding="utf-8") as stream:
        package_algorithm = stream.read().strip()
    with open(version_path, encoding="utf-8") as stream:
        package_version = stream.read().strip()
    report.check(algorithm == package_algorithm, "algorithm identity matches the package marker")
    report.check(str(identity["package_version"]) == package_version, "package version identity matches VERSION")
    report.check(identity["action_space"] == "discrete" and identity["arm"] == "random", "identity is the discrete random-start arm")
    for marker in ("ENV_FATAL", "AGENT_FATAL"):
        report.check(not os.path.exists(os.path.join(run_dir, marker)), f"no {marker} marker")

    common = load_common_parameters(os.path.join(package_root, "config"))
    try:
        actual_config = config_digest(package_root)
        actual_shared = shared_layer_digest(package_root)
        actual_release = release_digest(package_root)
    except (FileNotFoundError, OSError, ValueError) as error:
        report.check(False, f"package provenance is readable and internally valid: {error}")
        return report
    report.check(actual_config == identity["config_sha256"], "configuration digest matches the package")
    report.check(actual_shared == identity["shared_layer_sha256"], "shared-layer digest matches the package")
    report.check(actual_release == identity["release_sha256"], "release digest matches the exact package inventory")
    report.check(bool(re.fullmatch(r"[0-9a-f]{64}", str(identity["container_sha256"]))), "container digest is a lowercase SHA-256")
    seeds = {k: int(identity[k]) for k in ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed")}
    common.update(seeds)
    world_path = os.path.join(package_root, "worlds", f"{common['world_id']}.world")
    sampler = Sampler(build_law(common), build_arena(common, world_path))
    action_map = ActionMap(
        float(common["forward_linear"]), float(common["steering_linear"]),
        float(common["steering_angular"]), float(common["rotation_angular"]),
    )
    commands = action_map.commands()
    control_period = float(common["control_period"])
    hold_tol = float(common["hold_tolerance_fraction"])
    validation = _load_validation_section(package_root)
    protocol = _load_protocol_section(package_root)
    scenarios = read_scenarios(os.path.join(package_root, "config", protocol["scenario_file"]))
    algo_params = _load_algorithm_params(package_root, algorithm)
    report.check(str(algo_params.get("algorithm")) == algorithm, "algorithm configuration matches the package identity")
    report.check(str(algo_params.get("algorithm_version")) == str(identity["algorithm_version"]), "algorithm version matches its configuration")
    phase_label = str(identity["phase_label"])
    if not report.check(phase_label in protocol["phases"], f"phase_label {phase_label} is preregistered"):
        return report
    phase = dict(protocol["phases"][phase_label])
    budget = int(phase["environment_budget"]) if phase_type == "training" else 0
    report.check(int(identity["learning_seed"]) in [int(v) for v in phase["learning_seeds"]], "learning seed preregistered for the phase")
    report.check(
        all(int(identity[name]) == int(protocol["common_seeds"][name]) for name in ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed")),
        "world, initialization, obstacle and evaluation seeds match their preregistered roles",
    )
    report.check(len(scenarios) == int(protocol["scenario_count"]), "scenario file has the preregistered row count")
    report.check(sum(int(s.physical_subset) for s in scenarios) == int(protocol["physical_subset_count"]), "scenario file has the preregistered physical subset count")
    report.stats["environment_budget"] = budget
    report.stats["phase_label"] = phase_label
    _validate_runtime_provenance(report, package_root, run_dir, identity, budget, phase_type)

    streams: Dict[str, Optional[List[Dict[str, str]]]] = {}
    names = ["transitions", "evaluation"] + (["episodes", "updates", "checkpoints"] if phase_type == "training" else [])
    for name in names:
        streams[name] = _check_header(report, run_dir, name)
    if any(streams[n] is None for n in names):
        return report

    # identity uniformity
    for name in names:
        for row in streams[name]:
            for field in IDENTITY_FIELDS:
                if str(identity[field]) != row[field]:
                    report.failures.append(f"{name}.csv identity field {field} disagrees with run_identity.json")
                    break
            else:
                continue
            break
    report.checks.append("identity block uniform across every row of every stream")

    transitions = streams["transitions"]
    _validate_transitions(report, transitions, commands, common, control_period, hold_tol, validation, budget, phase_type, algorithm)
    if phase_type == "training":
        _validate_episodes(report, streams["episodes"], transitions, sampler, seeds, common, budget)
        _validate_updates(report, streams["updates"], algorithm, algo_params, budget)
        checkpoints = streams["checkpoints"]
        _validate_checkpoints(report, checkpoints, run_dir, budget, phase, algo_params)
        _validate_evaluation_training(report, streams["evaluation"], checkpoints, phase, sampler, seeds, common, transitions)
        _validate_obstacle_trajectories(report, transitions, streams["episodes"], streams["evaluation"], common)
    else:
        _validate_evaluation_posthoc(report, streams["evaluation"], scenarios, protocol, phase, common, transitions, run_dir, algorithm, sampler)
        _validate_obstacle_trajectories(report, transitions, [], streams["evaluation"], common)
    return report


def _load_validation_section(package_root: str) -> Dict[str, object]:
    with open(os.path.join(package_root, "config", "common_environment.yaml"), encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return dict(document.get("validation", {}))


def _load_protocol_section(package_root: str) -> Dict[str, object]:
    return load_evaluation_protocol(os.path.join(package_root, "config"))


def _validate_transitions(report, rows, commands, common, control_period, hold_tol, validation, budget, phase_type, algorithm) -> None:
    training = [r for r in rows if r["phase"] == "training"]
    evaluation = [r for r in rows if r["phase"] == "evaluation"]
    report.check(all(r["phase"] in ("training", "evaluation") for r in rows), "transition phase labels valid")
    steps = [r for r in training if r["step_in_episode"] != "0"]
    if phase_type == "training":
        report.check(len(steps) == budget, f"training transitions == budget ({len(steps)} vs {budget})")
        report.check([_i(r["env_step"]) for r in steps] == list(range(1, len(steps) + 1)), "env_step contiguous 1..B")
    else:
        report.check(not training, "no training rows in an evaluation run")
    report.check(all(r["env_step"] == "" for r in evaluation), "evaluation rows carry no env_step")
    report.check(all(r["checkpoint_step"] != "" and r["condition"] != "" for r in evaluation), "evaluation rows carry checkpoint and condition")
    training_mode = "stochastic"
    context_ok = all(
        r["policy_mode"] == training_mode
        and r["training_episode"] != ""
        and all(r[name] == "" for name in ("checkpoint_step", "condition", "scenario_id", "evaluation_episode"))
        for r in training
    ) and all(
        r["training_episode"] == "" and r["policy_mode"] in ("greedy", "deterministic", "stochastic")
        for r in evaluation
    )
    report.check(context_ok, "training/evaluation context and exploratory policy labels are unambiguous")

    # Per-episode structure and independent recomputation from adjacent states.
    out_of_tol = 0
    by_episode: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_episode.setdefault(r["episode_key"], []).append(r)
    structure_ok = True
    for key, ep in by_episode.items():
        indices = [_i(r["step_in_episode"]) for r in ep]
        if indices != list(range(len(ep))):
            structure_ok = False
        if len(ep) < 2 or ep[-1]["episode_end"] != "1" or any(r["episode_end"] == "1" for r in ep[1:-1]):
            structure_ok = False
    report.check(structure_ok, "every episode has a step-0 row, contiguous steps, and exactly one terminal row at its end")
    masks_ok = precedence_ok = reward_ok = action_ok = obs_ok = collision_ok = True
    timing_ok = freshness_ok = state_link_ok = physical_ok = observation_semantics_ok = True
    reward_cfg = reward_from(common)
    max_gap = float(validation["max_decision_gap_sim_seconds"])
    max_age = float(validation["max_sensor_age_sim_seconds"])
    max_obstacle_error = float(validation["max_obstacle_position_error"])
    initial_blank = True
    for ep in by_episode.values():
        for index, r in enumerate(ep):
            obs = [_f(r[f"obs_{i}"]) for i in range(OBSERVATION_DIM)]
            obs_valid = not any(v is None or not math.isfinite(v) or v < -1.0 - 1e-6 or v > 1.0 + 1e-6 for v in obs)
            if not obs_valid:
                obs_ok = False
            physical = [_f(r[name]) for name in (
                "sim_time", "state_sim_time", "min_lidar", "distance", "heading_error",
                "x", "y", "yaw", "obs1_x", "obs1_y", "obs2_x", "obs2_y",
            )]
            if any(v is None or not math.isfinite(v) for v in physical):
                physical_ok = False
            elif obs_valid:
                _, _, minimum, distance, heading, x, y, yaw, _, _, _, _ = physical
                goal_distance = math.hypot(float(common["goal_x"]) - x, float(common["goal_y"]) - y)
                goal_heading = math.atan2(float(common["goal_y"]) - y, float(common["goal_x"]) - x) - yaw
                goal_heading = math.atan2(math.sin(goal_heading), math.cos(goal_heading))
                linear = 0.0 if index == 0 else _f(r["linear_cmd"])
                angular = 0.0 if index == 0 else _f(r["angular_cmd"])
                if linear is None or angular is None or not math.isfinite(linear) or not math.isfinite(angular):
                    observation_semantics_ok = False
                else:
                    expected_tail = (
                        min(max(distance / float(common["distance_max"]), 0.0), 1.0),
                        math.sin(heading), math.cos(heading),
                        min(max(linear / max(abs(v) for v, _ in commands), -1.0), 1.0),
                        min(max(angular / max(abs(w) for _, w in commands), -1.0), 1.0),
                    )
                    reconstructed_min = min(obs[:36]) * float(common["lidar_max"])
                    if (
                        abs(distance - goal_distance) > FLOAT_TOLERANCE
                        or _angle_error(heading, goal_heading) > FLOAT_TOLERANCE
                        or abs(reconstructed_min - minimum) > FLOAT_TOLERANCE
                        or any(abs(actual - expected) > FLOAT_TOLERANCE for actual, expected in zip(obs[-5:], expected_tail))
                    ):
                        observation_semantics_ok = False
            if index == 0:
                blank_fields = (
                    "hold_sim_s", "hold_odom_s", "scan_age_s", "odom_age_s", "obs1_age_s", "obs2_age_s",
                    "decision_gap_sim_s", "decision_latency_wall_s", "obstacle_position_error_max",
                    "action_index", "action_name", "linear_cmd", "angular_cmd", "reward_total",
                    "r_distance", "r_step", "r_collision", "r_goal", "r_angular", "r_near",
                    "terminated", "episode_end", "truncated", "collision", "static_collision",
                    "dynamic_collision", "safety", "goal",
                )
                if any(r[k] != "" for k in blank_fields) or not _matches(_f(r["state_sim_time"]), _f(r["sim_time"])):
                    initial_blank = False
                continue
            previous = ep[index - 1]
            term, end, trunc = _b(r["terminated"]), _b(r["episode_end"]), _b(r["truncated"])
            col, saf, goal = _b(r["collision"]), _b(r["safety"]), _b(r["goal"])
            if not (end >= term) or trunc != (end and not term):
                masks_ok = False
            if term != (col or saf or goal) or (col and (saf or goal)) or (saf and goal):
                precedence_ok = False
            if col and not (_b(r["static_collision"]) or _b(r["dynamic_collision"])):
                collision_ok = False
            idx = _i(r["action_index"])
            if idx is None or not (0 <= idx < len(commands)) or ACTION_NAMES[idx] != r["action_name"]:
                action_ok = False
                continue
            lin, ang = commands[idx]
            if abs(_f(r["linear_cmd"]) - lin) > 1e-6 or abs(_f(r["angular_cmd"]) - ang) > 1e-6:
                action_ok = False
            expected = compute_reward(
                _f(previous["distance"]), _f(r["distance"]), _f(r["min_lidar"]), ang, bool(col), reward_cfg,
            )
            expected_values = {
                "reward_total": expected.total, "r_distance": expected.distance_progress,
                "r_step": expected.step_penalty, "r_collision": expected.collision_penalty,
                "r_goal": expected.goal_bonus, "r_angular": expected.angular_penalty,
                "r_near": expected.near_penalty,
            }
            if any(not math.isfinite(_f(r[k])) or abs(_f(r[k]) - value) > REWARD_TOLERANCE for k, value in expected_values.items()):
                reward_ok = False
            if (term, col, saf, goal) != (expected.terminated, expected.collision, expected.safety, expected.goal):
                precedence_ok = False
            state_time = _f(r["state_sim_time"])
            sim_time = _f(r["sim_time"])
            previous_time = _f(previous["sim_time"])
            hold_sim = _f(r["hold_sim_s"])
            hold_odom = _f(r["hold_odom_s"])
            gap = _f(r["decision_gap_sim_s"])
            latency = _f(r["decision_latency_wall_s"])
            if any(v is None or not math.isfinite(v) for v in (state_time, sim_time, previous_time, hold_sim, hold_odom, gap, latency)):
                timing_ok = False
            else:
                if abs(state_time - previous_time) > FLOAT_TOLERANCE or abs((sim_time - state_time) - hold_sim) > FLOAT_TOLERANCE:
                    state_link_ok = False
                if abs(gap) > max_gap or latency < 0.0:
                    timing_ok = False
                if abs(hold_sim - control_period) > hold_tol * control_period or abs(hold_odom - control_period) > hold_tol * control_period:
                    out_of_tol += 1
            ages = [_f(r[k]) for k in ("scan_age_s", "odom_age_s", "obs1_age_s", "obs2_age_s")]
            obstacle_error = _f(r["obstacle_position_error_max"])
            if any(v is None or not math.isfinite(v) or v < -1e-6 or v > max_age for v in ages):
                freshness_ok = False
            if obstacle_error is None or not math.isfinite(obstacle_error) or obstacle_error < 0.0 or obstacle_error > max_obstacle_error:
                freshness_ok = False
    report.check(masks_ok, "mask identities: episode_end >= terminated; truncated == episode_end and not terminated")
    report.check(precedence_ok, "termination precedence: terminated == collision|safety|goal, mutually exclusive")
    report.check(collision_ok, "every collision is classified static and/or dynamic")
    report.check(reward_ok, "all six reward components and the total recompute from adjacent states and the executed action")
    report.check(action_ok, "action index, name and velocity pair match the frozen action map")
    report.check(obs_ok, "observations are 41 finite values in [-1, 1]")
    report.check(physical_ok, "recorded clocks, poses, obstacle positions and navigation features are finite")
    report.check(observation_semantics_ok, "distance, heading, LiDAR minimum and five scalar observation features independently recompute")
    report.check(initial_blank, "step-0 rows contain no action, reward, or hold measurements")
    report.check(state_link_ok, "each transition state_sim_time equals its preceding recorded state time")
    report.check(timing_ok, "decision gaps, wall latencies, and simulation times are finite and within contract")
    report.check(freshness_ok, "sensor ages and online obstacle-trajectory errors are finite and bounded")
    total_steps = max(1, sum(1 for r in rows if r["step_in_episode"] != "0"))
    fraction = out_of_tol / total_steps
    max_fraction = float(validation.get("max_hold_out_of_tolerance_fraction", 0.01))
    report.stats["hold_out_of_tolerance_fraction"] = fraction
    report.check(fraction <= max_fraction, f"action holds within tolerance ({fraction:.4f} out-of-tolerance <= {max_fraction})")


def _validate_episodes(report, episodes, transitions, sampler, seeds, common, budget) -> None:
    tr_by_key: Dict[str, List[Dict[str, str]]] = {}
    for r in transitions:
        if r["phase"] == "training":
            tr_by_key.setdefault(r["episode_key"], []).append(r)
    ends = [r for r in transitions if r["phase"] == "training" and r["episode_end"] == "1"]
    report.check(len(episodes) == len(ends), f"episodes.csv rows == training episode ends ({len(episodes)} vs {len(ends)})")
    report.check([_i(e["training_episode"]) for e in episodes] == list(range(1, len(episodes) + 1)), "training_episode contiguous")
    report.check(sum(_i(e["length"]) for e in episodes) == budget, "episode lengths sum to the budget")
    agree = init_ok = repro = summary_ok = True
    if {e["episode_key"] for e in episodes} != set(tr_by_key):
        agree = False
    max_rej = int(common["init_max_rejections"])
    for e in episodes:
        ep = tr_by_key.get(e["episode_key"], [])
        steps = [r for r in ep if r["step_in_episode"] != "0"]
        if len(steps) != _i(e["length"]):
            agree = False
        if abs(sum(_f(r["reward_total"]) for r in steps) - _f(e["return"])) > FLOAT_TOLERANCE:
            agree = False
        if steps and _i(steps[-1]["env_step"]) != _i(e["end_env_step"]):
            agree = False
        if not _summary_matches_transitions(e, ep, common):
            summary_ok = False
        if not (_b(e["init_tolerance_ok"]) and _b(e["init_support_ok"]) and e["reset_status"] == "ok" and not _b(e["contact_during_reset"])
                and _i(e["rejection_count"]) <= max_rej and e["init_kind"] == "train_nu_R"):
            init_ok = False
        if not _initialization_geometry_ok(e, sampler, common, require_support=True):
            init_ok = False
        pos_error = math.hypot(_f(e["realized_x"]) - _f(e["requested_x"]), _f(e["realized_y"]) - _f(e["requested_y"]))
        yaw_error = _angle_error(_f(e["realized_yaw"]), _f(e["requested_yaw"]))
        odom_error = math.hypot(_f(e["odom_x"]) - _f(e["realized_x"]), _f(e["odom_y"]) - _f(e["realized_y"]))
        odom_yaw_error = _angle_error(_f(e["odom_yaw"]), _f(e["realized_yaw"]))
        if (
            pos_error > float(common["init_position_tolerance"])
            or yaw_error > float(common["init_yaw_tolerance"])
            or odom_error > float(common["init_odom_tolerance"])
            or odom_yaw_error > float(common["init_odom_yaw_tolerance"])
            or abs(pos_error - _f(e["init_pos_error"])) > FLOAT_TOLERANCE
            or abs(yaw_error - _f(e["init_yaw_error"])) > FLOAT_TOLERANCE
            or abs(odom_error - _f(e["init_odom_error"])) > FLOAT_TOLERANCE
            or abs(odom_yaw_error - _f(e["init_odom_yaw_error"])) > FLOAT_TOLERANCE
        ):
            init_ok = False
        draw = sampler.draw(training_start_seed(seeds["initialization_seed"], _i(e["training_episode"])))
        if (abs(draw.x - _f(e["requested_x"])) > 1e-9 or abs(draw.y - _f(e["requested_y"])) > 1e-9
                or abs(draw.yaw - _f(e["requested_yaw"])) > 1e-9 or draw.rejection_count != _i(e["rejection_count"])
                or draw.generator_seed != _i(e["init_generator_seed"])):
            repro = False
        if not (
            sampler.admissible(_f(e["requested_x"]), _f(e["requested_y"]))
            and sampler.admissible(_f(e["realized_x"]), _f(e["realized_y"]))
            and sampler.admissible(_f(e["odom_x"]), _f(e["odom_y"]))
            and _f(e["init_scan_clearance"]) >= sampler.law.start_clearance_min
        ):
            repro = False
        phases = obstacle_phases(
            training_obstacle_seed(seeds["dynamic_obstacle_seed"], _i(e["training_episode"])),
            2, float(common["obstacle_half_period"]),
        )
        if (
            phases.generator_seed != _i(e["obstacle_phase_seed"])
            or any(abs(a - b) > 1e-12 for a, b in zip(phases.signs, (_f(e["obs1_sign"]), _f(e["obs2_sign"]))))
            or any(abs(a - b) > 1e-12 for a, b in zip(phases.offsets, (_f(e["obs1_offset"]), _f(e["obs2_offset"]))))
        ):
            repro = False
    report.check(agree, "episode length, return and end_env_step agree with the transition rows")
    report.check(summary_ok, "every training episode outcome and diagnostic summary recomputes from its raw transitions")
    report.check(init_ok, "every training initialization independently satisfies pose, odometry, support, and contact gates")
    report.check(repro, "every training start and obstacle phase is reproduced from its role-specific seed and realized in supp(nu_R)")


def _validate_updates(report, updates, algorithm, params, budget) -> None:
    warmup = int(params["warmup_steps"])
    report.check(len(updates) == budget - warmup + 1, f"one update per transition after warm-up ({len(updates)} vs {budget - warmup + 1})")
    report.check([_i(u["gradient_step"]) for u in updates] == list(range(1, len(updates) + 1)), "gradient_step contiguous")
    report.check(bool(updates) and _i(updates[0]["env_step"]) == warmup, f"first update at env_step == W ({warmup})")
    report.check([_i(u["env_step"]) for u in updates] == list(range(warmup, warmup + len(updates))), "updates cover every env_step from W to B")
    applicable = set(UPDATE_APPLICABILITY[algorithm])
    matrix_ok = True
    for u in updates:
        for column in UPDATES_COLUMNS:
            present = u[column] != ""
            if column in applicable and (not present or not math.isfinite(float(u[column]))):
                matrix_ok = False
            if column not in applicable and present:
                matrix_ok = False
    report.check(matrix_ok, f"applicability matrix for {algorithm}: required fields finite, inapplicable fields empty")
    period = int(params.get("target_update_steps", 0))
    if period:
        synced = [_i(u["gradient_step"]) for u in updates if u["target_synced"] == "1"]
        report.check(synced == [g for g in range(period, len(updates) + 1, period)], "target synchronized exactly every C_target gradient steps")
    if algorithm == "SDSAC":
        batch_size = int(params["batch_size"])
        capacity = int(params["replay_capacity"])
        actor_lr = float(params["actor_learning_rate"])
        critic_lr = float(params["critic_learning_rate"])
        alpha = float(params["alpha"])
        beta = float(params["entropy_penalty_beta"])
        clip_range = float(params["q_clip_range"])
        sac_ok = True
        for row in updates:
            try:
                env_step = _i(row["env_step"])
                replay_size = _i(row["replay_size"])
                invalid = (
                    _i(row["batch_size"]) != batch_size
                    or replay_size != min(env_step, capacity)
                    or abs(_f(row["actor_learning_rate"]) - actor_lr) > 1e-15
                    or abs(_f(row["critic_learning_rate"]) - critic_lr) > 1e-15
                    or abs(_f(row["alpha"]) - alpha) > 1e-12
                    or abs(_f(row["entropy_penalty_beta"]) - beta) > 1e-12
                    or abs(_f(row["q_clip_range"]) - clip_range) > 1e-12
                    or _f(row["entropy_penalty_loss"]) < 0.0
                    or _f(row["entropy_gap_abs_mean"]) < 0.0
                    or abs(_f(row["entropy_penalty_contribution"]) - _f(row["entropy_penalty_loss"])) > 1e-6
                    or abs(_f(row["actor_loss"]) - _f(row["actor_base_loss"]) - _f(row["entropy_penalty_contribution"])) > 1e-5
                    or _f(row["critic1_loss"]) < 0.0
                    or _f(row["critic2_loss"]) < 0.0
                    or _f(row["critic_loss_mean"]) < 0.0
                    or abs(_f(row["critic_loss_mean"]) - 0.5 * (_f(row["critic1_loss"]) + _f(row["critic2_loss"]))) > 1e-5
                    or _f(row["td_error_abs_mean"]) < 0.0
                    or _f(row["q_gap_abs_mean"]) < 0.0
                    or not 0.0 <= _f(row["policy_entropy_mean"]) <= math.log(5.0) + 1e-6
                    or not 0.0 <= _f(row["old_policy_entropy_mean"]) <= math.log(5.0) + 1e-6
                    or not 0.0 <= _f(row["next_policy_entropy_mean"]) <= math.log(5.0) + 1e-6
                    or not 0.2 - 1e-6 <= _f(row["max_action_probability_mean"]) <= 1.0 + 1e-6
                    or not -1e-6 <= _f(row["min_action_probability_mean"]) <= 0.2 + 1e-6
                    or not 0.0 <= _f(row["q1_clip_activation_fraction"]) <= 1.0
                    or not 0.0 <= _f(row["q2_clip_activation_fraction"]) <= 1.0
                    or min(_f(row["actor_grad_norm"]), _f(row["critic1_grad_norm"]), _f(row["critic2_grad_norm"])) < 0.0
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid = True
            if invalid:
                sac_ok = False
                break
        report.check(
            sac_ok,
            "SD-SAC double-average-Q actor, entropy penalty, elementwise Q-clip, fixed alpha, learning rates and uniform replay satisfy the frozen contract",
        )


def _validate_checkpoints(report, checkpoints, run_dir, budget, phase, params) -> None:
    interval = int(phase["checkpoint_interval_steps"])
    full_interval = int(phase["full_checkpoint_interval_steps"])
    expected_policy = list(range(interval, budget + 1, interval))
    expected_full = sorted({s for s in expected_policy if s % full_interval == 0} | {budget})
    policy = sorted(_i(c["env_step"]) for c in checkpoints if c["kind"] == "policy_only")
    full = sorted(_i(c["env_step"]) for c in checkpoints if c["kind"] == "full")
    report.check(all(c["kind"] in ("policy_only", "full") for c in checkpoints), "checkpoint kinds are policy_only or full")
    report.check(policy == expected_policy, f"policy-only checkpoints at every {interval} steps through the budget")
    report.check(full == expected_full, "full checkpoints at every full interval and at the exact budget")
    files_ok = True
    seen_paths = set()
    checkpoint_root = os.path.realpath(os.path.join(run_dir, "checkpoints")) + os.sep
    warmup = int(params["warmup_steps"])
    for c in checkpoints:
        path = c["path"]
        if not os.path.isabs(path):
            path = os.path.join(run_dir, path)
        path = os.path.realpath(path)
        step = _i(c["env_step"])
        expected_gradient_step = max(0, step - warmup + 1)
        if (
            not path.startswith(checkpoint_root)
            or path in seen_paths
            or not os.path.isfile(path)
            or sha256_file(path) != c["sha256"]
            or c["validated"] != "1"
            or os.path.exists(path + ".tmp")
            or _i(c["gradient_step"]) != expected_gradient_step
            or _f(c["wall_time"]) is None
            or not math.isfinite(_f(c["wall_time"]))
        ):
            files_ok = False
        seen_paths.add(path)
    report.check(files_ok, "every checkpoint is unique, confined, cadence-consistent, digest-valid, load-validated, and atomic")


def _initialization_geometry_ok(row, sampler, common, require_support=True) -> bool:
    try:
        values = [_f(row[k]) for k in (
            "requested_x", "requested_y", "requested_yaw", "realized_x", "realized_y", "realized_yaw",
            "odom_x", "odom_y", "odom_yaw", "init_pos_error", "init_yaw_error", "init_odom_error",
            "init_odom_yaw_error", "init_scan_clearance",
        )]
        if any(v is None or not math.isfinite(v) for v in values):
            return False
        requested_x, requested_y, requested_yaw, realized_x, realized_y, realized_yaw, odom_x, odom_y, odom_yaw = values[:9]
        pos_error = math.hypot(realized_x - requested_x, realized_y - requested_y)
        yaw_error = _angle_error(realized_yaw, requested_yaw)
        odom_error = math.hypot(odom_x - realized_x, odom_y - realized_y)
        odom_yaw_error = _angle_error(odom_yaw, realized_yaw)
        recorded = values[9:13]
        errors = (pos_error, yaw_error, odom_error, odom_yaw_error)
        tolerances = (
            float(common["init_position_tolerance"]), float(common["init_yaw_tolerance"]),
            float(common["init_odom_tolerance"]), float(common["init_odom_yaw_tolerance"]),
        )
        if any(error > tolerance or abs(error - saved) > FLOAT_TOLERANCE for error, tolerance, saved in zip(errors, tolerances, recorded)):
            return False
        if not (_b(row["init_tolerance_ok"]) and row["reset_status"] == "ok" and not _b(row["contact_during_reset"])):
            return False
        if (
            not _matches(_f(row["goal_x"]), float(common["goal_x"]))
            or not _matches(_f(row["goal_y"]), float(common["goal_y"]))
            or not _matches(_f(row["settle_sim_s"]), float(common["init_settle_sim_seconds"]))
            or _f(row["init_sim_time"]) is None
            or not math.isfinite(_f(row["init_sim_time"]))
        ):
            return False
        for index, dynamic in enumerate(sampler.arena.dynamic_obstacles):
            reset_x, reset_y = sampler.arena.dynamic_reset_pose(dynamic.name)
            if (
                abs(_f(row[f"obs{index + 1}_reset_x"]) - reset_x) > float(common["init_position_tolerance"])
                or abs(_f(row[f"obs{index + 1}_reset_y"]) - reset_y) > float(common["init_position_tolerance"])
            ):
                return False
        if require_support and not (
            _b(row["init_support_ok"])
            and sampler.admissible(realized_x, realized_y)
            and sampler.admissible(odom_x, odom_y)
            and values[13] >= sampler.law.start_clearance_min
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _phase_row_matches(row, phases) -> bool:
    try:
        return (
            _i(row["obstacle_phase_seed"]) == phases.generator_seed
            and abs(_f(row["obs1_sign"]) - phases.signs[0]) <= 1e-12
            and abs(_f(row["obs2_sign"]) - phases.signs[1]) <= 1e-12
            and abs(_f(row["obs1_offset"]) - phases.offsets[0]) <= 1e-12
            and abs(_f(row["obs2_offset"]) - phases.offsets[1]) <= 1e-12
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_obstacle_trajectories(report, transitions, episodes, evaluation, common) -> None:
    init_by_key = {str(row["episode_key"]): row for row in list(episodes) + list(evaluation)}
    axes = list(common["dynamic_obstacle_axes"])
    speed = float(common["obstacle_speed"])
    half_period = float(common["obstacle_half_period"])
    tolerance = float(common["obstacle_position_tolerance"])
    ok = True
    for row in transitions:
        init = init_by_key.get(str(row["episode_key"]))
        if init is None:
            ok = False
            continue
        try:
            elapsed = max(0.0, _f(row["sim_time"]) - _f(init["init_sim_time"]))
            errors = []
            for index in range(2):
                dx, dy = obstacle_displacement(
                    axes[index], _f(init[f"obs{index + 1}_sign"]), _f(init[f"obs{index + 1}_offset"]),
                    speed, half_period, elapsed,
                )
                expected_x = _f(init[f"obs{index + 1}_reset_x"]) + dx
                expected_y = _f(init[f"obs{index + 1}_reset_y"]) + dy
                errors.append(math.hypot(_f(row[f"obs{index + 1}_x"]) - expected_x, _f(row[f"obs{index + 1}_y"]) - expected_y))
            maximum = max(errors)
            if maximum > tolerance:
                ok = False
            if row["step_in_episode"] != "0" and abs(maximum - _f(row["obstacle_position_error_max"])) > FLOAT_TOLERANCE:
                ok = False
        except (KeyError, TypeError, ValueError):
            ok = False
    report.check(ok, "recorded obstacle poses independently match the seeded simulation-time trajectories")


def _validate_evaluation_training(report, evaluation, checkpoints, phase, sampler, seeds, common, transitions) -> None:
    n = int(phase["tier1_episodes_per_checkpoint"])
    expected_modes = ("stochastic", "deterministic")
    policy_cps = {_i(c["env_step"]): c["sha256"] for c in checkpoints if c["kind"] == "policy_only"}
    per_cp: Dict[int, List[Dict[str, str]]] = {}
    for e in evaluation:
        per_cp.setdefault(_i(e["checkpoint_step"]), []).append(e)
    report.check(set(per_cp) == set(policy_cps), "every policy checkpoint has an evaluation block and no block lacks a checkpoint")
    complete = clean = repro = summaries = True
    transition_by_key: Dict[str, List[Dict[str, str]]] = {}
    for row in transitions:
        if row["phase"] == "evaluation":
            transition_by_key.setdefault(row["episode_key"], []).append(row)
    for step, rows in per_cp.items():
        for mode in expected_modes:
            numbered = sorted(_i(r["evaluation_episode"]) for r in rows if r["policy_mode"] == mode)
            if numbered != list(range(1, n + 1)):
                complete = False
        for r in rows:
            if r["condition"] != "E1" or r["policy_mode"] not in expected_modes or r["learner_state_unchanged"] != "1" or r["checkpoint_sha256"] != policy_cps.get(step):
                clean = False
            if not _initialization_geometry_ok(r, sampler, common, require_support=True):
                clean = False
            if not _summary_matches_transitions(r, transition_by_key.get(r["episode_key"], []), common):
                summaries = False
            draw = sampler.draw(e1_start_seed(seeds["evaluation_seed"], _i(r["checkpoint_index"]), _i(r["evaluation_episode"])))
            expected_index = step // int(phase["checkpoint_interval_steps"])
            if (
                abs(draw.x - _f(r["requested_x"])) > 1e-9
                or abs(draw.y - _f(r["requested_y"])) > 1e-9
                or abs(draw.yaw - _f(r["requested_yaw"])) > 1e-9
                or draw.rejection_count != _i(r["rejection_count"])
                or draw.generator_seed != _i(r["init_generator_seed"])
                or _i(r["checkpoint_index"]) != expected_index
                or r["init_kind"] != "E1_nu_R"
            ):
                repro = False
            phases = obstacle_phases(
                e1_obstacle_seed(seeds["evaluation_seed"], _i(r["checkpoint_index"]), _i(r["evaluation_episode"])),
                2, float(common["obstacle_half_period"]),
            )
            if not _phase_row_matches(r, phases):
                repro = False
    report.check(complete, f"exactly {n} E1 episodes per checkpoint and policy mode, numbered 1..{n}")
    report.check(clean, "evaluation rows are E1 stochastic/deterministic, learner untouched, checkpoint digest matches, initialization verified")
    report.check(repro, "every E1 start pose is reproduced from (evaluation_seed, checkpoint_index, evaluation_episode)")
    report.check(summaries, "every E1 outcome and diagnostic summary recomputes from its raw transitions")
    eval_keys = {r["episode_key"] for r in transitions if r["phase"] == "evaluation"}
    report.check(eval_keys == {r["episode_key"] for r in evaluation}, "evaluation episodes have trajectory rows and vice versa")


def _validate_evaluation_posthoc(report, evaluation, scenarios, protocol, phase, common, transitions, run_dir, algorithm, sampler) -> None:
    modes = sorted({r["policy_mode"] for r in evaluation})
    expected_modes = ["deterministic", "stochastic"]
    report.check(modes == expected_modes, f"post-hoc policy modes are exactly {expected_modes}")
    checkpoint_steps = {_i(row["checkpoint_step"]) for row in evaluation}
    allowed_steps = {int(value) for value in phase["tier2_checkpoint_steps"]}
    report.check(len(checkpoint_steps) == 1 and checkpoint_steps <= allowed_steps, "post-hoc checkpoint step is preregistered for the phase")
    expected_index = next(iter(checkpoint_steps), 0) // int(phase["checkpoint_interval_steps"])
    report.check(all(_i(row["checkpoint_index"]) == expected_index for row in evaluation), "post-hoc checkpoint index matches its checkpoint step")
    conditions = list(protocol["tier2_conditions"])
    report.check(all(r["condition"] in conditions for r in evaluation), "post-hoc evaluation contains no undeclared condition")
    manifest_ok = False
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        steps = {_i(r["checkpoint_step"]) for r in evaluation}
        digests = {r["checkpoint_sha256"] for r in evaluation}
        manifest_ok = (
            steps == {int(manifest.get("checkpoint_step", -1))}
            and digests == {str(manifest.get("checkpoint_sha256", ""))}
            and len(next(iter(digests), "")) == 64
        )
    report.check(manifest_ok, "evaluation rows match the exact checkpoint step and digest in run_manifest.json")
    transition_by_key: Dict[str, List[Dict[str, str]]] = {}
    for row in transitions:
        if row["phase"] == "evaluation":
            transition_by_key.setdefault(row["episode_key"], []).append(row)
    report.check(
        all(_summary_matches_transitions(row, transition_by_key.get(row["episode_key"], []), common) for row in evaluation),
        "every post-hoc outcome and diagnostic summary recomputes from its raw transitions",
    )
    for mode in modes:
        rows = [r for r in evaluation if r["policy_mode"] == mode]
        if "E2" in conditions:
            ids = sorted(r["scenario_id"] for r in rows if r["condition"] == "E2")
            report.check(ids == sorted(s.scenario_id for s in scenarios), f"E2 covers every scenario exactly once for mode {mode}")
            fixed_ok = True
            scenario_episode = {scenario.scenario_id: index + 1 for index, scenario in enumerate(scenarios)}
            for r in rows:
                if r["condition"] != "E2":
                    continue
                s = scenario_by_id(scenarios, r["scenario_id"])
                if abs(_f(r["requested_x"]) - s.x) > 1e-9 or abs(_f(r["requested_y"]) - s.y) > 1e-9 or abs(_f(r["requested_yaw"]) - s.yaw) > 1e-9:
                    fixed_ok = False
                if (
                    r["difficulty"] != s.difficulty
                    or _i(r["distance_bin"]) != s.distance_bin
                    or _i(r["clearance_bin"]) != s.clearance_bin
                    or _i(r["heading_bin"]) != s.heading_bin
                    or _i(r["evaluation_episode"]) != scenario_episode[s.scenario_id]
                    or _i(r["init_generator_seed"]) != s.start_generator_seed
                    or _i(r["rejection_count"]) != 0
                    or r["init_kind"] != "E2_scenario"
                ):
                    fixed_ok = False
                phases = obstacle_phases(s.obstacle_phase_seed, 2, float(common["obstacle_half_period"]))
                if not _phase_row_matches(r, phases):
                    fixed_ok = False
                if not _initialization_geometry_ok(r, sampler, common, True):
                    fixed_ok = False
            report.check(fixed_ok, f"E2 rows use the scenario poses and labels for mode {mode}")
        if "E3" in conditions:
            e3 = sorted(_i(r["evaluation_episode"]) for r in rows if r["condition"] == "E3")
            report.check(e3 == list(range(1, int(protocol["tier2_e3_episodes"]) + 1)), f"E3 has the scheduled episode count for mode {mode}")
            fixed = (float(common["fixed_start_x"]), float(common["fixed_start_y"]), float(common["fixed_start_yaw"]))
            fixed_start_ok = all(
                abs(_f(r["requested_x"]) - fixed[0]) < 1e-9
                and abs(_f(r["requested_y"]) - fixed[1]) < 1e-9
                and _angle_error(_f(r["requested_yaw"]), fixed[2]) < 1e-9
                for r in rows if r["condition"] == "E3"
            )
            report.check(fixed_start_ok, f"E3 uses the complete fixed start pose for mode {mode}")
            report.check(all(_initialization_geometry_ok(r, sampler, common, False) for r in rows if r["condition"] == "E3"), f"E3 initialization geometry/contact verified for mode {mode}")
            e3_phase_ok = True
            for r in rows:
                if r["condition"] != "E3":
                    continue
                phases = obstacle_phases(
                    e3_obstacle_seed(int(common["evaluation_seed"]), _i(r["checkpoint_index"]), _i(r["evaluation_episode"])),
                    2, float(common["obstacle_half_period"]),
                )
                if (
                    not _phase_row_matches(r, phases)
                    or _i(r["init_generator_seed"]) != phases.generator_seed
                    or _i(r["rejection_count"]) != 0
                    or r["init_kind"] != "E3_fixed"
                ):
                    e3_phase_ok = False
            report.check(e3_phase_ok, f"E3 obstacle phases reproduce from the evaluation seed for mode {mode}")
    eval_keys = {r["episode_key"] for r in transitions if r["phase"] == "evaluation"}
    report.check(eval_keys == {r["episode_key"] for r in evaluation}, "evaluation episodes have trajectory rows and vice versa")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase-type", choices=["training", "evaluation"], default=None)
    args = parser.parse_args(argv)
    try:
        report = validate_run(args.package_root, args.run_dir, args.phase_type)
    except Exception as error:  # the run directory is untrusted input; fail closed with a durable report
        report = Report()
        report.check(False, f"validator could not parse the run safely: {type(error).__name__}: {error}")
    out = os.path.join(args.run_dir, "validation_report.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(report.to_json(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, out)
    for failure in report.failures:
        print(f"FAIL: {failure}")
    print(f"validation {'PASSED' if report.passed else 'FAILED'}: {len(report.checks)} checks, {len(report.failures)} failures")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
