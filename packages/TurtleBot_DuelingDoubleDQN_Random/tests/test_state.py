import math
import unittest

from turtlebot3_drl_nav.state import (
    ACTION_NAMES,
    OBSERVATION_DIM,
    ActionMap,
    RewardConfig,
    build_observation,
    compute_reward,
    sample_lidar_nearest,
)


class StateTests(unittest.TestCase):
    def test_observation_is_41_dimensional_and_statically_normalized(self):
        observation = build_observation([1.75] * 36, 5.0, math.pi / 2.0, 0.075, -0.5, 3.5, 10.0, 0.15, 1.0)
        self.assertEqual(len(observation), OBSERVATION_DIM)
        self.assertAlmostEqual(observation[0], 0.5)
        for actual, wanted in zip(observation[-5:], [0.5, 1.0, 0.0, 0.5, -0.5]):
            self.assertAlmostEqual(actual, wanted)

    def test_nearest_beam_uses_angle_metadata(self):
        ranges = [float(index) for index in range(360)]
        sampled = sample_lidar_nearest(ranges, -math.pi, 2.0 * math.pi / 360.0, 0.0, 400.0)
        self.assertEqual(sampled[0], 0.0)  # an invalid zero is conservative, never free space
        self.assertEqual(sampled[1], 10.0)
        self.assertEqual(sampled[18], 180.0)

    def test_nearest_beam_wraps_stock_burger_zero_based_scan(self):
        ranges = [float(index) for index in range(360)]
        sampled = sample_lidar_nearest(ranges, 0.0, 2.0 * math.pi / 360.0, 0.0, 400.0)
        self.assertEqual(sampled[0], 180.0)
        self.assertEqual(sampled[1], 190.0)
        self.assertEqual(sampled[19], 10.0)
        self.assertEqual(len(set(sampled[:18])), 18)

    def test_rep117_and_invalid_ranges_fail_safe(self):
        sampled = sample_lidar_nearest(
            [float("-inf"), float("inf"), float("nan"), 0.0],
            -math.pi, math.pi / 2.0, 0.12, 3.5, bins=4,
        )
        self.assertEqual(sampled, [0.12, 3.5, 0.12, 0.12])

    def test_nearest_beam_tolerates_the_gazebo_increment(self):
        # Gazebo's burger LDS reports 360 samples over [0, 6.28] -> increment 6.28/359
        ranges = [float(index) for index in range(360)]
        sampled = sample_lidar_nearest(ranges, 0.0, 6.28 / 359.0, 0.0, 400.0)
        self.assertEqual(sampled[19], 10.0)
        self.assertEqual(sampled[17], 350.0)

    def test_event_precedence_and_reward_components(self):
        config = RewardConfig()
        result = compute_reward(1.0, 0.1, 0.1, 0.0, True, config)
        self.assertTrue(result.collision and not result.safety and not result.goal)
        self.assertEqual(result.event, "collision")
        safety = compute_reward(1.0, 0.1, 0.1, 0.0, False, config)
        self.assertTrue(safety.safety and not safety.goal)
        goal = compute_reward(1.0, 0.1, 1.0, 0.5, False, config)
        self.assertTrue(goal.goal and goal.terminated)
        self.assertAlmostEqual(goal.total, 10.0 * 0.9 - 0.01 + 100.0 - 0.05 * 0.5)
        self.assertAlmostEqual(result.total, sum((result.distance_progress, result.step_penalty, result.collision_penalty,
                                                  result.goal_bonus, result.angular_penalty, result.near_penalty)))

    def test_near_penalty_band_and_frozen_action_map(self):
        config = RewardConfig()
        near = compute_reward(1.0, 0.95, 0.25, 0.0, False, config)
        self.assertLess(near.near_penalty, 0.0)
        self.assertFalse(near.terminated)
        commands = ActionMap().commands()
        self.assertEqual(commands, ((0.15, 0.0), (0.12, 0.6), (0.12, -0.6), (0.0, 1.0), (0.0, -1.0)))
        self.assertEqual(len(ACTION_NAMES), 5)


if __name__ == "__main__":
    unittest.main()
