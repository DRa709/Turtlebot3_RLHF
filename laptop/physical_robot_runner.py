"""Physical robot runner for TurtleBot3 Discrete SAC.

Executes policy neural network inference on the laptop workstation,
constructs the 41-dimensional observation vector, calculates the six
roadmap reward components, requests velocity commands via the local
supervisor, and logs episode and step telemetry to CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add vendor path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_VENDOR_DIR = _PROJECT_ROOT / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import numpy as np
import torch

from turtlebot3_drl_nav.discretesac import CategoricalActor, DiscreteSACPolicy
from turtlebot3_drl_nav.state import (
    ACTION_NAMES,
    LIDAR_BINS,
    OBSERVATION_DIM,
    ActionMap,
    Pose2D,
    RewardConfig,
    RewardResult,
    build_observation,
    compute_reward,
    goal_features,
    min_valid_range,
    quaternion_to_yaw,
    sample_lidar_nearest,
)


@dataclass
class EpisodeTelemetry:
    run_id: str
    episode_id: int
    checkpoint_id: str
    eval_mode: str
    policy_mode: str
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    outcome: str
    total_return: float
    episode_length: int
    wall_seconds: float
    final_distance: float
    path_length: float
    interventions: int


@dataclass
class StepTelemetry:
    episode_id: int
    step_id: int
    timestamp: float
    raw_min_scan: float
    action_idx: int
    action_name: str
    req_linear: float
    req_angular: float
    exec_linear: float
    exec_angular: float
    dist_progress: float
    step_penalty: float
    collision_penalty: float
    goal_bonus: float
    angular_penalty: float
    near_penalty: float
    total_step_reward: float
    pose_x: float
    pose_y: float
    pose_yaw: float
    goal_dist: float
    goal_bearing: float
    terminated: bool
    event: str


class PhysicalRobotRunnerEngine:
    """Pure-Python engine decoupled from ROS for deterministic unit testing.
    
    Handles observation preparation, policy evaluation, step/episode transitions,
    and CSV logging.
    """

    EPISODE_HEADER = [
        "run_id",
        "episode_id",
        "checkpoint_id",
        "eval_mode",
        "policy_mode",
        "start_x",
        "start_y",
        "start_yaw",
        "goal_x",
        "goal_y",
        "outcome",
        "total_return",
        "episode_length",
        "wall_seconds",
        "final_distance",
        "path_length",
        "interventions",
    ]

    STEP_HEADER = [
        "episode_id",
        "step_id",
        "timestamp",
        "raw_min_scan",
        "action_idx",
        "action_name",
        "req_linear",
        "req_angular",
        "exec_linear",
        "exec_angular",
        "dist_progress",
        "step_penalty",
        "collision_penalty",
        "goal_bonus",
        "angular_penalty",
        "near_penalty",
        "total_step_reward",
        "pose_x",
        "pose_y",
        "pose_yaw",
        "goal_dist",
        "goal_bearing",
        "terminated",
        "event",
    ]

    def __init__(
        self,
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        goal_tolerance: float = 0.20,
        timeout_sec: float = 60.0,
        action_duration_sec: float = 0.10,
        policy_mode: str = "deterministic",
        checkpoint_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        run_id: str = "physical_pilot",
        reward_config: Optional[RewardConfig] = None,
        action_map: Optional[ActionMap] = None,
    ):
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_tolerance = float(goal_tolerance)
        self.timeout_sec = float(timeout_sec)
        self.action_duration_sec = float(action_duration_sec)
        self.policy_mode = policy_mode
        self.checkpoint_path = checkpoint_path
        self.checkpoint_id = Path(checkpoint_path).stem if checkpoint_path else "synthetic_actor"
        self.run_id = run_id

        self.reward_config = reward_config or RewardConfig(
            distance=10.0,
            step=0.01,
            collision=100.0,
            goal=100.0,
            angular=0.05,
            near=0.25,
            safe_distance=0.30,
            stop_distance=0.16,
            goal_tolerance=goal_tolerance,
        )
        self.reward_config.validate()
        self.action_map = action_map or ActionMap()

        self.out_dir = Path(out_dir) if out_dir else _PROJECT_ROOT / "runs"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_csv_path = self.out_dir / "episodes.csv"
        self.steps_csv_path = self.out_dir / "steps.csv"

        self._init_csv_headers()

        # Policy model
        device = torch.device("cpu")
        if checkpoint_path and Path(checkpoint_path).is_file():
            self.policy: Optional[DiscreteSACPolicy] = DiscreteSACPolicy(
                checkpoint_path, device=device
            )
            self.actor: Optional[CategoricalActor] = None
        else:
            self.policy = None
            self.actor = CategoricalActor(
                observation_dim=OBSERVATION_DIM,
                action_dim=len(ACTION_NAMES),
                hidden_size=256,
            ).to(device)

        # State tracking
        self.episode_id = 0
        self.step_id = 0
        self.start_pose: Optional[Pose2D] = None
        self.current_pose: Optional[Pose2D] = None
        self.previous_pose: Optional[Pose2D] = None
        self.previous_distance: Optional[float] = None
        self.current_distance: float = 0.0
        self.current_bearing: float = 0.0
        self.prev_linear: float = 0.0
        self.prev_angular: float = 0.0
        self.total_return: float = 0.0
        self.path_length: float = 0.0
        self.start_time: float = 0.0
        self.interventions: int = 0
        self.is_active: bool = False

    def _init_csv_headers(self) -> None:
        if not self.episodes_csv_path.exists() or self.episodes_csv_path.stat().st_size == 0:
            with open(self.episodes_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.EPISODE_HEADER)
        if not self.steps_csv_path.exists() or self.steps_csv_path.stat().st_size == 0:
            with open(self.steps_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.STEP_HEADER)

    def start_episode(self, start_x: float, start_y: float, start_yaw: float, timestamp: float) -> None:
        self.episode_id += 1
        self.step_id = 0
        self.start_pose = Pose2D(x=start_x, y=start_y, yaw=start_yaw)
        self.current_pose = self.start_pose
        self.previous_pose = self.start_pose
        dist, bearing = goal_features(self.start_pose, self.goal_x, self.goal_y)
        self.previous_distance = dist
        self.current_distance = dist
        self.current_bearing = bearing
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.total_return = 0.0
        self.path_length = 0.0
        self.start_time = timestamp
        self.interventions = 0
        self.is_active = True

    def process_lidar(
        self,
        raw_ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float = 0.12,
        range_max: float = 3.5,
    ) -> Tuple[List[float], float]:
        """Sample 36 canonical beams and calculate minimum valid range."""
        lidar_36 = sample_lidar_nearest(
            ranges=raw_ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=range_min,
            range_max=range_max,
            bins=LIDAR_BINS,
        )
        min_scan = min_valid_range(raw_ranges, fallback=range_max)
        return lidar_36, min_scan

    def construct_observation(
        self,
        lidar_36: Sequence[float],
        distance: float,
        bearing: float,
        prev_linear: float,
        prev_angular: float,
    ) -> List[float]:
        """Build 41-dim observation vector with exact normalizations from state.py."""
        return build_observation(
            lidar=lidar_36,
            distance=distance,
            heading_error=bearing,
            previous_linear=prev_linear,
            previous_angular=prev_angular,
            lidar_max=3.5,
            distance_max=10.0,
            velocity_max=self.action_map.v_max,
            angular_max=self.action_map.omega_max,
        )

    def select_action(self, obs: Sequence[float]) -> Tuple[int, str, float, float]:
        """Infers discrete action index and velocity commands."""
        obs_array = np.asarray(obs, dtype=np.float32)
        if self.policy is not None:
            action_idx = self.policy.act(
                obs_array,
                policy_mode=self.policy_mode,
                sampling_seed=42 if self.policy_mode == "stochastic" else None,
            )
        elif self.actor is not None:
            with torch.no_grad():
                tensor_obs = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                probabilities, _ = self.actor.distribution(tensor_obs)
                if self.policy_mode == "deterministic":
                    action_idx = int(probabilities.argmax(dim=1).item())
                else:
                    dist = torch.distributions.Categorical(probs=probabilities)
                    action_idx = int(dist.sample().item())
        else:
            action_idx = 0

        action_name = ACTION_NAMES[action_idx]
        cmd_v, cmd_w = self.action_map.commands()[action_idx]
        return action_idx, action_name, cmd_v, cmd_w

    def step(
        self,
        raw_ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        timestamp: float,
        supervisor_event: Optional[str] = None,
        contact_collision: bool = False,
    ) -> Tuple[RewardResult, bool, str, float, float]:
        """Advances one decision step, computes rewards, checks termination, and logs telemetry.
        
        Returns:
            (reward_result, is_terminated, outcome_str, req_linear, req_angular)
        """
        if not self.is_active:
            raise RuntimeError("Episode is not active. Call start_episode() first.")

        self.step_id += 1
        self.previous_pose = self.current_pose
        self.current_pose = Pose2D(x=pose_x, y=pose_y, yaw=pose_yaw)

        # Path length tracking
        if self.previous_pose is not None:
            step_disp = math.hypot(
                self.current_pose.x - self.previous_pose.x,
                self.current_pose.y - self.previous_pose.y,
            )
            self.path_length += step_disp

        # Goal geometry
        dist, bearing = goal_features(self.current_pose, self.goal_x, self.goal_y)
        self.current_distance = dist
        self.current_bearing = bearing

        # LiDAR processing
        lidar_36, min_scan = self.process_lidar(
            raw_ranges=raw_ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
        )

        # Build 41-dim observation vector
        obs = self.construct_observation(
            lidar_36=lidar_36,
            distance=self.current_distance,
            bearing=self.current_bearing,
            prev_linear=self.prev_linear,
            prev_angular=self.prev_angular,
        )

        # Action selection
        action_idx, action_name, req_v, req_w = self.select_action(obs)

        # Interventions / supervisor event check
        if supervisor_event in ("protective_stop", "collision", "timeout", "emergency_stop"):
            self.interventions += 1
            exec_v, exec_w = 0.0, 0.0
        else:
            exec_v, exec_w = req_v, req_w

        # Compute reward
        prev_d = self.previous_distance if self.previous_distance is not None else self.current_distance
        rew = compute_reward(
            previous_distance=prev_d,
            current_distance=self.current_distance,
            min_scan=min_scan,
            executed_angular=exec_w,
            collision_contact=contact_collision,
            config=self.reward_config,
        )
        self.total_return += rew.total
        self.previous_distance = self.current_distance

        # Termination checks
        wall_elapsed = timestamp - self.start_time
        terminated = False
        outcome = "running"

        if contact_collision or rew.collision:
            terminated = True
            outcome = "collision"
        elif supervisor_event == "protective_stop" or rew.safety:
            terminated = True
            outcome = "protective_stop"
        elif rew.goal or self.current_distance <= self.goal_tolerance:
            terminated = True
            outcome = "goal"
        elif wall_elapsed >= self.timeout_sec:
            terminated = True
            outcome = "timeout"

        # Update previous commands for next step
        self.prev_linear = exec_v
        self.prev_angular = exec_w

        # Record step telemetry
        step_entry = StepTelemetry(
            episode_id=self.episode_id,
            step_id=self.step_id,
            timestamp=timestamp,
            raw_min_scan=min_scan,
            action_idx=action_idx,
            action_name=action_name,
            req_linear=req_v,
            req_angular=req_w,
            exec_linear=exec_v,
            exec_angular=exec_w,
            dist_progress=rew.distance_progress,
            step_penalty=rew.step_penalty,
            collision_penalty=rew.collision_penalty,
            goal_bonus=rew.goal_bonus,
            angular_penalty=rew.angular_penalty,
            near_penalty=rew.near_penalty,
            total_step_reward=rew.total,
            pose_x=self.current_pose.x,
            pose_y=self.current_pose.y,
            pose_yaw=self.current_pose.yaw,
            goal_dist=self.current_distance,
            goal_bearing=self.current_bearing,
            terminated=terminated,
            event=outcome,
        )
        self._write_step_csv(step_entry)

        if terminated:
            self.is_active = False
            start_x = self.start_pose.x if self.start_pose else 0.0
            start_y = self.start_pose.y if self.start_pose else 0.0
            start_yaw = self.start_pose.yaw if self.start_pose else 0.0
            ep_entry = EpisodeTelemetry(
                run_id=self.run_id,
                episode_id=self.episode_id,
                checkpoint_id=self.checkpoint_id,
                eval_mode="physical_eval",
                policy_mode=self.policy_mode,
                start_x=start_x,
                start_y=start_y,
                start_yaw=start_yaw,
                goal_x=self.goal_x,
                goal_y=self.goal_y,
                outcome=outcome,
                total_return=self.total_return,
                episode_length=self.step_id,
                wall_seconds=wall_elapsed,
                final_distance=self.current_distance,
                path_length=self.path_length,
                interventions=self.interventions,
            )
            self._write_episode_csv(ep_entry)

        return rew, terminated, outcome, req_v, req_w

    def _write_step_csv(self, s: StepTelemetry) -> None:
        with open(self.steps_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                s.episode_id,
                s.step_id,
                f"{s.timestamp:.4f}",
                f"{s.raw_min_scan:.4f}",
                s.action_idx,
                s.action_name,
                f"{s.req_linear:.3f}",
                f"{s.req_angular:.3f}",
                f"{s.exec_linear:.3f}",
                f"{s.exec_angular:.3f}",
                f"{s.dist_progress:.5f}",
                f"{s.step_penalty:.5f}",
                f"{s.collision_penalty:.1f}",
                f"{s.goal_bonus:.1f}",
                f"{s.angular_penalty:.5f}",
                f"{s.near_penalty:.5f}",
                f"{s.total_step_reward:.5f}",
                f"{s.pose_x:.4f}",
                f"{s.pose_y:.4f}",
                f"{s.pose_yaw:.4f}",
                f"{s.goal_dist:.4f}",
                f"{s.goal_bearing:.4f}",
                s.terminated,
                s.event,
            ])

    def _write_episode_csv(self, e: EpisodeTelemetry) -> None:
        with open(self.episodes_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                e.run_id,
                e.episode_id,
                e.checkpoint_id,
                e.eval_mode,
                e.policy_mode,
                f"{e.start_x:.4f}",
                f"{e.start_y:.4f}",
                f"{e.start_yaw:.4f}",
                f"{e.goal_x:.4f}",
                f"{e.goal_y:.4f}",
                e.outcome,
                f"{e.total_return:.4f}",
                e.episode_length,
                f"{e.wall_seconds:.2f}",
                f"{e.final_distance:.4f}",
                f"{e.path_length:.4f}",
                e.interventions,
            ])


# ROS 2 Node Wrapper (only instantiated if rclpy is available)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


if _ROS2_AVAILABLE:
    class PhysicalRobotRunnerNode(Node):
        """ROS 2 node that wraps the engine and connects to robot topics."""

        def __init__(self, engine: PhysicalRobotRunnerEngine):
            super().__init__("tb3_discrete_sac_runner")
            self.engine = engine

            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self.scan_sub = self.create_subscription(
                LaserScan, "/scan", self._scan_callback, sensor_qos
            )
            self.odom_sub = self.create_subscription(
                Odometry, "/odom", self._odom_callback, 10
            )

            self.req_pub = self.create_publisher(Twist, "/tb3_sac/cmd_vel_request", 10)
            self.timer = self.create_timer(self.engine.action_duration_sec, self._control_loop)

            self.latest_scan: Optional[LaserScan] = None
            self.latest_odom: Optional[Odometry] = None
            self.has_started = False
            self.get_logger().info(
                f"PhysicalRobotRunnerNode initialized. Goal: ({self.engine.goal_x}, {self.engine.goal_y})"
            )

        def _scan_callback(self, msg: LaserScan) -> None:
            self.latest_scan = msg

        def _odom_callback(self, msg: Odometry) -> None:
            self.latest_odom = msg

        def _control_loop(self) -> None:
            if self.latest_scan is None or self.latest_odom is None:
                self.get_logger().info("Waiting for /scan and /odom...", throttle_duration_sec=2.0)
                return

            now = time.monotonic()
            pos = self.latest_odom.pose.pose.position
            ori = self.latest_odom.pose.pose.orientation
            yaw = quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)

            if not self.has_started:
                self.engine.start_episode(pos.x, pos.y, yaw, now)
                self.has_started = True
                self.get_logger().info(f"Episode {self.engine.episode_id} started at ({pos.x:.2f}, {pos.y:.2f})")

            rew, terminated, outcome, req_v, req_w = self.engine.step(
                raw_ranges=self.latest_scan.ranges,
                angle_min=self.latest_scan.angle_min,
                angle_increment=self.latest_scan.angle_increment,
                pose_x=pos.x,
                pose_y=pos.y,
                pose_yaw=yaw,
                timestamp=now,
            )

            twist = Twist()
            if not terminated:
                twist.linear.x = req_v
                twist.angular.z = req_w
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            self.req_pub.publish(twist)

            if terminated:
                self.get_logger().info(
                    f"Episode {self.engine.episode_id} TERMINATED: {outcome}. Total return: {self.engine.total_return:.2f}"
                )
                self.timer.cancel()
                self.has_started = False


def main() -> None:
    parser = argparse.ArgumentParser(description="TurtleBot3 Discrete SAC Physical Runner")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch checkpoint")
    parser.add_argument("--goal-x", type=float, default=1.0, help="Goal X coordinate (meters)")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal Y coordinate (meters)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Episode timeout (seconds)")
    parser.add_argument("--policy-mode", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--out-dir", type=str, default="runs", help="Output telemetry directory")
    parser.add_argument("--synthetic", action="store_true", help="Run 1 synthetic offline step for testing")
    args = parser.parse_args()

    engine = PhysicalRobotRunnerEngine(
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        timeout_sec=args.timeout,
        policy_mode=args.policy_mode,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
    )

    if args.synthetic:
        print("[Runner] Running synthetic step test...")
        engine.start_episode(0.0, 0.0, 0.0, time.monotonic())
        fake_ranges = [1.5] * 360
        rew, term, outcome, v, w = engine.step(
            raw_ranges=fake_ranges,
            angle_min=0.0,
            angle_increment=2.0 * math.pi / 360.0,
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            timestamp=time.monotonic() + 0.1,
        )
        print(f"  Step outcome: {outcome}, reward: {rew.total:.4f}, v: {v:.2f}, w: {w:.2f}")
        print("RUNNER_SYNTHETIC_TEST_PASSED")
        return

    if not _ROS2_AVAILABLE:
        print("ERROR: ROS 2 (rclpy) is not installed in this environment. Cannot run live ROS node.")
        sys.exit(1)

    rclpy.init()
    node = PhysicalRobotRunnerNode(engine)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
