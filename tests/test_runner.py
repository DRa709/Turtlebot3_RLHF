"""Unit tests for Laptop Physical Robot Runner (Gate 3)."""

import csv
import math
import sys
import tempfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
LAPTOP_DIR = PROJECT_ROOT / "laptop"
for d in (str(PROJECT_ROOT), str(VENDOR_DIR), str(LAPTOP_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from turtlebot3_drl_nav.state import (
    LIDAR_BINS,
    OBSERVATION_DIM,
    ActionMap,
    RewardConfig,
    sample_lidar_nearest,
    build_observation,
    compute_reward,
)
from physical_robot_runner import PhysicalRobotRunnerEngine


def test_lidar_36_resampling():
    """G3.1: 36-beam LiDAR resampling aligns exactly with trained policy bearings.
    
    Bearings are -180 to +170 deg in 10-deg increments.
    Index 18 points forward (0 deg), 27 points left (+90 deg), 9 points right (-90 deg), 0 points back (-180 deg).
    """
    num_samples = 360
    ranges = [1.0 + (i / 1000.0) for i in range(num_samples)]
    angle_min = 0.0
    angle_inc = 2.0 * math.pi / num_samples

    sampled = sample_lidar_nearest(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_inc,
        range_min=0.12,
        range_max=3.5,
        bins=36,
    )

    assert len(sampled) == 36

    # Forward beam is index 18 (target angle 0.0 rad, or 0 deg in [0, 2pi))
    assert math.isclose(sampled[18], 1.0 + (0 / 1000.0), abs_tol=1e-3)

    # Left beam is index 27 (target angle +pi/2 rad, or 90 deg)
    assert math.isclose(sampled[27], 1.0 + (90 / 1000.0), abs_tol=1e-3)

    # Right beam is index 9 (target angle -pi/2 rad, wrapped to 270 deg)
    assert math.isclose(sampled[9], 1.0 + (270 / 1000.0), abs_tol=1e-3)

    # Backward beam is index 0 (target angle -pi rad, wrapped to 180 deg)
    assert math.isclose(sampled[0], 1.0 + (180 / 1000.0), abs_tol=1e-3)


def test_observation_normalization():
    """G3.2: Observation vector (41 dimensions) normalization and clipping matches state.py."""
    engine = PhysicalRobotRunnerEngine(goal_x=5.0, goal_y=0.0)

    lidar_36 = [4.0 if i % 2 == 0 else 1.75 for i in range(36)]
    dist = 15.0  # exceeds 10.0m max
    bearing = math.pi / 2.0  # sin=1.0, cos=0.0
    prev_v = 0.30  # exceeds 0.15m/s max
    prev_w = -2.0  # exceeds -1.0 rad/s min

    obs = engine.construct_observation(
        lidar_36=lidar_36,
        distance=dist,
        bearing=bearing,
        prev_linear=prev_v,
        prev_angular=prev_w,
    )

    assert len(obs) == 41

    # First 36 are LiDAR ranges normalized by 3.5 and clipped to [0, 1]
    for i in range(36):
        if i % 2 == 0:
            assert math.isclose(obs[i], 1.0)  # 4.0 clipped to 1.0
        else:
            assert math.isclose(obs[i], 0.5)  # 1.75 / 3.5 = 0.5

    # 36: distance / 10 clipped to [0, 1]
    assert math.isclose(obs[36], 1.0)  # 15.0 clipped to 1.0

    # 37: sin(pi/2) = 1.0
    assert math.isclose(obs[37], 1.0, abs_tol=1e-5)

    # 38: cos(pi/2) = 0.0
    assert math.isclose(obs[38], 0.0, abs_tol=1e-5)

    # 39: prev_v / 0.15 clipped to 1.0
    assert math.isclose(obs[39], 1.0)

    # 40: prev_w / 1.0 clipped to -1.0
    assert math.isclose(obs[40], -1.0)


def test_reward_computation():
    """G3.3: Six reward components calculation fidelity and protective stop non-penalty handling."""
    config = RewardConfig(
        distance=10.0,
        step=0.01,
        collision=100.0,
        goal=100.0,
        angular=0.05,
        near=0.25,
        safe_distance=0.30,
        stop_distance=0.16,
        goal_tolerance=0.20,
    )

    # Scenario 1: Normal progress step
    res1 = compute_reward(
        previous_distance=1.0,
        current_distance=0.9,
        min_scan=1.5,
        executed_angular=0.6,
        collision_contact=False,
        config=config,
    )
    assert math.isclose(res1.distance_progress, 1.0)  # 10.0 * (1.0 - 0.9) = 1.0
    assert math.isclose(res1.step_penalty, -0.01)
    assert math.isclose(res1.collision_penalty, 0.0)
    assert math.isclose(res1.goal_bonus, 0.0)
    assert math.isclose(res1.angular_penalty, -0.03)  # -0.05 * 0.6 = -0.03
    assert math.isclose(res1.near_penalty, 0.0)
    assert math.isclose(res1.total, 1.0 - 0.01 - 0.03)
    assert not res1.terminated

    # Scenario 2: Near obstacle penalty
    res2 = compute_reward(
        previous_distance=1.0,
        current_distance=1.0,
        min_scan=0.25,
        executed_angular=0.0,
        collision_contact=False,
        config=config,
    )
    assert math.isclose(res2.near_penalty, -0.25)
    assert math.isclose(res2.total, -0.01 - 0.25)
    assert not res2.terminated

    # Scenario 3: Goal arrival
    res3 = compute_reward(
        previous_distance=0.3,
        current_distance=0.15,
        min_scan=1.5,
        executed_angular=0.0,
        collision_contact=False,
        config=config,
    )
    assert res3.goal
    assert math.isclose(res3.goal_bonus, 100.0)
    assert res3.terminated

    # Scenario 4: Contact collision
    res4 = compute_reward(
        previous_distance=0.5,
        current_distance=0.5,
        min_scan=0.10,
        executed_angular=0.0,
        collision_contact=True,
        config=config,
    )
    assert res4.collision
    assert math.isclose(res4.collision_penalty, -100.0)
    assert res4.terminated

    # Scenario 5: Protective Stop (Crucial Specification Check)
    # Proximity breach min_scan < stop_distance (0.15 < 0.16), BUT no contact collision
    # MUST receive NO -100 collision penalty!
    res5 = compute_reward(
        previous_distance=0.5,
        current_distance=0.5,
        min_scan=0.15,
        executed_angular=0.0,
        collision_contact=False,
        config=config,
    )
    assert res5.safety
    assert not res5.collision
    assert math.isclose(res5.collision_penalty, 0.0), "Protective stop must NOT apply -100 penalty"
    assert math.isclose(res5.near_penalty, -0.25)  # 0.15 < 0.30
    assert res5.terminated


def test_csv_logging_schema():
    """G3.4: CSV output logging schema matches Section 11 specifications (episodes.csv, steps.csv)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PhysicalRobotRunnerEngine(
            goal_x=1.0,
            goal_y=0.0,
            out_dir=tmpdir,
            run_id="test_run_01",
        )

        ep_csv = Path(tmpdir) / "episodes.csv"
        step_csv = Path(tmpdir) / "steps.csv"

        assert ep_csv.exists()
        assert step_csv.exists()

        with open(ep_csv, "r", encoding="utf-8") as f:
            ep_reader = csv.reader(f)
            ep_header = next(ep_reader)
            assert ep_header == [
                "run_id", "episode_id", "checkpoint_id", "eval_mode", "policy_mode",
                "start_x", "start_y", "start_yaw", "goal_x", "goal_y", "outcome",
                "total_return", "episode_length", "wall_seconds", "final_distance",
                "path_length", "interventions"
            ]

        with open(step_csv, "r", encoding="utf-8") as f:
            step_reader = csv.reader(f)
            step_header = next(step_reader)
            assert step_header == [
                "episode_id", "step_id", "timestamp", "raw_min_scan", "action_idx",
                "action_name", "req_linear", "req_angular", "exec_linear", "exec_angular",
                "dist_progress", "step_penalty", "collision_penalty", "goal_bonus",
                "angular_penalty", "near_penalty", "total_step_reward", "pose_x",
                "pose_y", "pose_yaw", "goal_dist", "goal_bearing", "terminated", "event"
            ]

        engine.start_episode(start_x=0.0, start_y=0.0, start_yaw=0.0, timestamp=100.0)
        fake_ranges = [1.5] * 360

        rew1, term1, out1, _, _ = engine.step(
            raw_ranges=fake_ranges,
            angle_min=0.0,
            angle_increment=2 * math.pi / 360,
            pose_x=0.5,
            pose_y=0.0,
            pose_yaw=0.0,
            timestamp=100.1,
        )
        assert not term1

        rew2, term2, out2, _, _ = engine.step(
            raw_ranges=fake_ranges,
            angle_min=0.0,
            angle_increment=2 * math.pi / 360,
            pose_x=0.95,
            pose_y=0.0,
            pose_yaw=0.0,
            timestamp=100.2,
        )
        assert term2
        assert out2 == "goal"

        with open(ep_csv, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert len(rows) == 2
            ep_row = rows[1]
            assert ep_row[0] == "test_run_01"
            assert ep_row[1] == "1"
            assert ep_row[10] == "goal"

        with open(step_csv, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert len(rows) == 3
            assert rows[1][0] == "1" and rows[1][1] == "1"
            assert rows[2][0] == "1" and rows[2][1] == "2"
            assert rows[2][22] == "True"
            assert rows[2][23] == "goal"
