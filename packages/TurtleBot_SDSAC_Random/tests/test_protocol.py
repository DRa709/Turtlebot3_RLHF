import unittest

import numpy as np

from turtlebot3_drl_nav.protocol import (
    EpisodeSpec,
    decode_action,
    decode_json,
    decode_state,
    decode_step,
    encode_action,
    encode_json,
    encode_state,
    encode_step,
)
from turtlebot3_drl_nav.state import RewardConfig, compute_reward


class ProtocolTests(unittest.TestCase):
    def test_state_and_action_round_trip(self):
        obs = list(np.linspace(-1, 1, 41))
        state = decode_state(encode_state(7, 3, 0, 12.5, obs))
        self.assertEqual((state.sequence, state.episode_key, state.step_index, state.sim_time), (7, 3, 0, 12.5))
        self.assertEqual(len(state.observation), 41)
        action = decode_action(encode_action(7, 2, 0.12, -0.6, True))
        self.assertEqual((action.sequence, action.action_index, action.final_transition), (7, 2, True))
        self.assertAlmostEqual(action.angular, -0.6, places=6)

    def test_step_round_trip_preserves_both_masks_and_pose(self):
        result = compute_reward(2.0, 1.9, 1.0, 0.0, False, RewardConfig())
        values = encode_step(9, 3, 4, 13.0, 0.1, 0.1, 0.02, 0.01, 0.02, 0.02, 0.0, 0.4, 0.01,
                             result, True, True, False, False, 1.0, 1.9, 0.3,
                             (0.5, -0.2, 1.1), ((0.0, -1.0), (-1.0, 0.4)), [0.0] * 41)
        step = decode_step(values)
        self.assertFalse(step.terminated)
        self.assertTrue(step.episode_end and step.truncated)
        self.assertEqual((step.x, step.y), (0.5, -0.2))
        self.assertAlmostEqual(step.yaw, 1.1, places=6)
        self.assertEqual((step.obs2_x, step.obs2_y), (-1.0, 0.4))
        self.assertAlmostEqual(step.reward_total, result.total, places=5)

    def test_wrong_length_or_version_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_state([2.0, 1, 1, 0, 0] + [0.0] * 40)
        values = encode_state(1, 1, 0, 0.0, [0.0] * 41)
        values[0] = 1.0
        with self.assertRaises(ValueError):
            decode_state(values)
        for corrupt in (float("nan"), float("inf")):
            values = encode_state(1, 1, 0, 0.0, [0.0] * 41)
            values[-1] = corrupt
            with self.assertRaises(ValueError):
                decode_state(values)
        action = encode_action(1, 0, 0.15, 0.0, False)
        action[-1] = 2.0
        with self.assertRaises(ValueError):
            decode_action(action)
        action = encode_action(1, 0, 0.15, 0.0, False)
        action[1] = 1.5
        with self.assertRaises(ValueError):
            decode_action(action)

    def test_episode_spec_validation_and_json(self):
        spec = EpisodeSpec(phase="evaluation", policy_mode="greedy", condition="E2", checkpoint_step=25000, checkpoint_index=1, scenario_id="S001", evaluation_episode=1)
        payload = decode_json(spec.to_json())
        self.assertEqual(payload["cmd"], "start_episode")
        self.assertEqual(EpisodeSpec.from_payload(payload), spec)
        with self.assertRaises(ValueError):
            EpisodeSpec(phase="evaluation", policy_mode="greedy", condition="E2", checkpoint_step=1, checkpoint_index=1, evaluation_episode=1).validate()
        with self.assertRaises(ValueError):
            EpisodeSpec(phase="training", policy_mode="epsilon_greedy", condition="E1").validate()
        with self.assertRaises(ValueError):
            decode_json(encode_json({"a": 1}).replace("3.0", "1.0"))
        with self.assertRaises(ValueError):
            decode_json('{"protocol_version":3.0,"x":NaN}')
        with self.assertRaises(ValueError):
            decode_json('{"protocol_version":3.0,"x":1e999}')
        with self.assertRaises(ValueError):
            EpisodeSpec(phase="training", policy_mode="epsilon_greedy", checkpoint_step=1).validate()


if __name__ == "__main__":
    unittest.main()
