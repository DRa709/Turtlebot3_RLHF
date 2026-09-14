"""Canonical recording: the frozen schemas of the run streams and a create-only,
fsync'ed, strictly typed CSV writer that prefixes every row with the identity
block.

Shared layer. No ROS dependency. Streams (DATA_CONTRACT.md):

    transitions.csv   one row per environment step (training and evaluation)
    episodes.csv      one row per training episode, with the initialization block
    updates.csv       one row per gradient step
    evaluation.csv    one row per evaluation episode, with the initialization block
    checkpoints.csv   one row per checkpoint file
"""

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence

from .identity import IDENTITY_FIELDS
from .state import OBSERVATION_DIM

INIT_BLOCK = [
    "init_kind", "init_generator_seed", "requested_x", "requested_y", "requested_yaw",
    "realized_x", "realized_y", "realized_yaw", "odom_x", "odom_y", "odom_yaw",
    "init_pos_error", "init_yaw_error", "init_odom_error", "init_odom_yaw_error",
    "init_scan_clearance", "init_support_ok", "init_tolerance_ok",
    "goal_x", "goal_y", "obstacle_phase_seed", "obs1_sign", "obs1_offset", "obs2_sign", "obs2_offset",
    "obs1_reset_x", "obs1_reset_y", "obs2_reset_x", "obs2_reset_y",
    "rejection_count", "reset_status", "settle_sim_s", "contact_during_reset", "init_sim_time",
]

TRANSITIONS_COLUMNS = [
    "phase", "policy_mode", "episode_key", "training_episode", "step_in_episode", "env_step", "sim_time",
    "state_sim_time", "hold_sim_s", "hold_odom_s", "scan_age_s", "odom_age_s", "obs1_age_s", "obs2_age_s",
    "decision_gap_sim_s", "decision_latency_wall_s", "obstacle_position_error_max",
    "action_index", "action_name", "linear_cmd", "angular_cmd",
    "reward_total", "r_distance", "r_step", "r_collision", "r_goal", "r_angular", "r_near",
    "terminated", "episode_end", "truncated", "collision", "static_collision", "dynamic_collision",
    "safety", "goal", "min_lidar", "distance", "heading_error", "x", "y", "yaw",
    "obs1_x", "obs1_y", "obs2_x", "obs2_y",
    "checkpoint_step", "condition", "scenario_id", "evaluation_episode",
] + [f"obs_{i}" for i in range(OBSERVATION_DIM)]

EPISODE_OUTCOME = [
    "length", "return", "sum_r_distance", "sum_r_step", "sum_r_collision", "sum_r_goal", "sum_r_angular",
    "sum_r_near", "outcome", "collision", "static_collision", "dynamic_collision", "safety", "goal",
    "truncated", "near_penalty_steps", "min_clearance", "path_length", "straight_line_distance",
    "path_efficiency", "time_to_goal_steps", "sim_time_start", "sim_time_end", "wall_time_s", "rtf",
    "mean_hold_sim_s", "max_hold_sim_s", "hold_out_of_tolerance_steps",
    "mean_decision_latency_wall_s", "max_decision_latency_wall_s", "max_decision_gap_sim_s",
    "max_sensor_age_s", "max_obstacle_position_error",
]

EPISODES_COLUMNS = ["episode_key", "training_episode", "start_env_step", "end_env_step"] + EPISODE_OUTCOME + INIT_BLOCK

UPDATES_COLUMNS = [
    "gradient_step", "env_step", "target_synced",
    "batch_terminal_fraction", "batch_size", "replay_size",
    # SD-SAC: double-average Q, behavior-entropy penalty and Q-clip diagnostics.
    "actor_loss", "actor_base_loss", "entropy_penalty_loss", "entropy_penalty_contribution",
    "critic1_loss", "critic2_loss", "critic_loss_mean",
    "td_error_abs_mean", "q1_taken_mean", "q2_taken_mean", "q_gap_abs_mean",
    "soft_target_mean", "next_soft_value_mean", "average_q_policy_mean",
    "policy_entropy_mean", "old_policy_entropy_mean", "entropy_gap_abs_mean", "next_policy_entropy_mean",
    "max_action_probability_mean", "min_action_probability_mean", "alpha",
    "entropy_penalty_beta", "q_clip_range", "q1_clipped_mean", "q2_clipped_mean",
    "q1_clip_activation_fraction", "q2_clip_activation_fraction",
    "actor_grad_norm", "critic1_grad_norm", "critic2_grad_norm",
    "actor_learning_rate", "critic_learning_rate",
]

EVALUATION_COLUMNS = [
    "checkpoint_step", "checkpoint_index", "checkpoint_sha256", "condition", "scenario_id",
    "evaluation_episode", "policy_mode", "episode_key", "learner_state_unchanged",
] + EPISODE_OUTCOME + INIT_BLOCK + ["distance_bin", "clearance_bin", "heading_bin", "difficulty"]

CHECKPOINTS_COLUMNS = [
    "env_step", "gradient_step", "kind", "path", "sha256", "sim_time", "wall_time", "validated",
]

STREAMS = {
    "transitions": TRANSITIONS_COLUMNS,
    "episodes": EPISODES_COLUMNS,
    "updates": UPDATES_COLUMNS,
    "evaluation": EVALUATION_COLUMNS,
    "checkpoints": CHECKPOINTS_COLUMNS,
}

# Algorithm applicability of the update-row optimizer fields. A field listed for
# an algorithm must be present and finite on every row; a field not listed must
# be empty on every row (validated in both directions).
UPDATE_APPLICABILITY: Dict[str, List[str]] = {
    "SDSAC": list(UPDATES_COLUMNS),
}


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


class CsvStream:
    """Create-only CSV writer: refuses an existing file, writes the header first,
    checks every row against the schema, flushes and fsyncs periodically."""

    def __init__(self, path: str, columns: Sequence[str], identity: Dict[str, object], flush_every: int = 200) -> None:
        if os.path.exists(path):
            raise FileExistsError(f"refusing to append to existing stream {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self.columns = list(columns)
        self.identity = [format_value(identity[k]) for k in IDENTITY_FIELDS]
        self.flush_every = max(1, int(flush_every))
        self.rows_written = 0
        self._stream = open(path, "x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._stream, lineterminator="\n")
        self._writer.writerow(list(IDENTITY_FIELDS) + self.columns)
        self._sync()

    def write(self, row: Dict[str, object]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"unknown columns for {self.path}: {sorted(unknown)}")
        values = [format_value(row.get(column)) for column in self.columns]
        self._writer.writerow(self.identity + values)
        self.rows_written += 1
        if self.rows_written % self.flush_every == 0:
            self._sync()

    def _sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._sync()
            self._stream.close()


def open_stream(run_dir: str, name: str, identity: Dict[str, object], flush_every: int = 200) -> CsvStream:
    if name not in STREAMS:
        raise KeyError(name)
    return CsvStream(os.path.join(run_dir, f"{name}.csv"), STREAMS[name], identity, flush_every)


def read_stream(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
