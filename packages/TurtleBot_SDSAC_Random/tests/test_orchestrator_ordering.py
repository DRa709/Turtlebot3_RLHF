import unittest

import numpy as np

from turtlebot3_drl_nav.learner_api import ActionChoice
from turtlebot3_drl_nav.orchestrator import Orchestrator, OrchestratorConfig, SendEpisodeControl
from turtlebot3_drl_nav.protocol import StateMessage, StepMessage


class _CountingLearner:
    def __init__(self):
        self.env_steps = 0
        self.gradient_steps = 0
        self.replay = []
        self.online = object()

    def act(self, observation, evaluation=False):
        return ActionChoice(0, None, True)

    def observe(self, transition):
        self.env_steps += 1

    def optimize(self):
        return None


class EventOrderingTests(unittest.TestCase):
    def test_terminal_summary_arriving_before_step_is_buffered(self):
        learner = _CountingLearner()
        cfg = OrchestratorConfig(
            algorithm="SDSAC", mode="train", environment_budget=2, checkpoint_interval=2,
            full_checkpoint_interval=2, tier1_episodes=1, tier2_e3_episodes=1,
            tier2_conditions=("E2", "E3"), checkpoint_dir="/unused", learning_seed=101,
            action_commands=((0.15, 0.0), (0.12, 0.6), (0.12, -0.6), (0.0, 1.2), (0.0, -1.2)),
            config_sha256="c" * 64,
        )
        orchestrator = Orchestrator(cfg, learner, lambda path: None, [], lambda path: "0" * 64,
                                    lambda model: "p", wall_clock=lambda: 0.0)
        orchestrator.on_env_ready({"config_sha256": "c" * 64})
        observation = np.zeros(41, dtype=np.float32)
        orchestrator.on_state(StateMessage(7, 1, 0, 10.0, observation))
        self.assertEqual(orchestrator.on_episode_summary({"episode_key": 1}), [])
        self.assertEqual(learner.env_steps, 0)
        step = StepMessage(
            sequence=7, episode_key=1, step_index=1, sim_time=10.1,
            hold_sim_s=0.1, hold_odom_s=0.1, scan_age_s=0.0, odom_age_s=0.0,
            obs1_age_s=0.0, obs2_age_s=0.0, decision_gap_sim_s=0.0,
            decision_latency_wall_s=0.01, obstacle_position_error_max=0.0,
            reward_total=-0.01, r_distance=0.0, r_step=-0.01, r_collision=0.0,
            r_goal=0.0, r_angular=0.0, r_near=0.0, terminated=False,
            episode_end=True, truncated=True, collision=False, static_collision=False,
            dynamic_collision=False, safety=False, goal=False, min_lidar=1.0,
            distance=2.0, heading_error=0.0, x=0.0, y=0.0, yaw=0.0,
            obs1_x=0.0, obs1_y=-1.45, obs2_x=-1.45, obs2_y=0.45,
            observation=observation,
        )
        effects = orchestrator.on_step(step)
        self.assertEqual(learner.env_steps, 1)
        self.assertTrue(any(isinstance(effect, SendEpisodeControl) for effect in effects))
        self.assertIsNone(orchestrator.pending_summary)


if __name__ == "__main__":
    unittest.main()
