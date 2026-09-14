"""Executable mathematical, interaction and checkpoint tests for Discrete SAC."""

import copy
import os
import random
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is required for the Discrete SAC learner tests") from exc

from turtlebot3_drl_nav.discretesac import (
    ALGORITHM,
    CategoricalActor,
    DiscreteSACAgent,
    DiscreteSACConfig,
    DiscreteSACPolicy,
    QNetwork,
    ReplayTransition,
    UniformReplayBuffer,
    _categorical_sample,
    _torch_load,
    actor_objective,
    critic_target,
    parameter_digest,
    soft_value,
    validate_checkpoint_payload,
)
from turtlebot3_drl_nav.learner_api import Transition


def obs(size, value=0.0):
    return np.full(size, value, dtype=np.float32)


def transition(size=4, action=0, reward=0.2, value=0.0,
               terminated=False, episode_end=False):
    return Transition(obs(size, value), action, reward, obs(size, value + 0.01),
                      terminated, episode_end)


def small_config(**overrides):
    values = dict(
        gamma=0.99, actor_learning_rate=1e-4, critic_learning_rate=1e-4,
        alpha=0.2, batch_size=2, replay_capacity=32, warmup_steps=2,
        target_update_steps=3, hidden_size=8, gradient_clip_norm=10.0,
        torch_threads=1,
    )
    values.update(overrides)
    return DiscreteSACConfig(**values)


class DiscreteSACMathTests(unittest.TestCase):
    def test_actor_and_each_critic_have_frozen_architecture(self):
        actor = CategoricalActor(41, 5, 256)
        critic = QNetwork(41, 5, 256)
        x = torch.zeros(7, 41)
        probabilities, logs = actor.distribution(x)
        self.assertEqual(tuple(probabilities.shape), (7, 5))
        self.assertEqual(tuple(critic(x).shape), (7, 5))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(7)))
        self.assertTrue(torch.allclose(probabilities.log(), logs, atol=1e-6))
        self.assertEqual(sum(p.numel() for p in actor.parameters()), 77829)
        self.assertEqual(sum(p.numel() for p in critic.parameters()), 77829)

    def test_soft_value_uses_exact_sum_entropy_and_minimum_critic(self):
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        logs = probabilities.log()
        q1 = torch.tensor([[4.0, 2.0], [-2.0, 7.0]])
        q2 = torch.tensor([[1.0, 3.0], [5.0, 6.0]])
        actual = soft_value(probabilities, logs, q1, q2, 0.2)
        expected = torch.stack([
            0.25 * (1.0 - 0.2 * logs[0, 0]) + 0.75 * (2.0 - 0.2 * logs[0, 1]),
            0.60 * (-2.0 - 0.2 * logs[1, 0]) + 0.40 * (6.0 - 0.2 * logs[1, 1]),
        ])
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(float(actual[0]), float((probabilities[0] * q1[0]).sum()))

    def test_target_bootstraps_truncation_but_not_termination(self):
        rewards = torch.tensor([3.0, 3.0])
        terminated = torch.tensor([1.0, 0.0])
        next_values = torch.tensor([8.0, 8.0])
        actual = critic_target(rewards, terminated, next_values, 0.99)
        self.assertTrue(torch.equal(actual, torch.tensor([3.0, 10.92])))

    def test_actor_objective_is_exact_sum_with_clipped_double_q(self):
        p = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
        logp = p.log()
        q1 = torch.tensor([[1.0, 4.0, -1.0]])
        q2 = torch.tensor([[2.0, 3.0, 6.0]])
        actual = actor_objective(p, logp, q1, q2, 0.2)
        expected = (p * (0.2 * logp - torch.tensor([[1.0, 3.0, -1.0]]))).sum()
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(all(value.grad is None for value in (q1, q2)))

    def test_seeded_categorical_sampler_is_reproducible(self):
        probabilities = [0.1, 0.2, 0.3, 0.4]
        first = [_categorical_sample(probabilities, random.Random(seed)) for seed in range(20)]
        second = [_categorical_sample(probabilities, random.Random(seed)) for seed in range(20)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)


class ReplayTests(unittest.TestCase):
    def test_uniform_sampling_ring_and_checkpoint_round_trip(self):
        replay = UniformReplayBuffer(3, random.Random(7))
        for index in range(4):
            replay.push(ReplayTransition(obs(2, index / 10), index % 2, float(index),
                                         obs(2, index / 10 + 0.01), False))
        self.assertEqual((len(replay), replay.position), (3, 1))
        self.assertEqual(float(replay.buffer[0].reward), 3.0)
        state = replay.state_dict()
        restored = UniformReplayBuffer(3, random.Random(7))
        restored.load_state_dict(state)
        self.assertEqual((len(restored), restored.position), (3, 1))
        self.assertEqual([item.reward for item in restored.buffer],
                         [item.reward for item in replay.buffer])
        self.assertEqual([item.reward for item in restored.sample(3)],
                         [item.reward for item in replay.sample(3)])


class DiscreteSACAgentTests(unittest.TestCase):
    def test_warmup_is_uniform_random_and_has_no_gradient_step(self):
        agent = DiscreteSACAgent(4, 3, small_config(warmup_steps=4, batch_size=2),
                                 torch.device("cpu"), 11)
        actions = []
        for step in range(4):
            choice = agent.act(obs(4))
            actions.append(choice.action)
            self.assertTrue(choice.random_action)
            self.assertIsNone(choice.epsilon_used)
            agent.observe(transition(action=choice.action, value=step / 100))
            diagnostics = agent.optimize()
            if step < 3:
                self.assertIsNone(diagnostics)
        self.assertIsNotNone(diagnostics)
        self.assertEqual(agent.gradient_steps, 1)
        choice = agent.act(obs(4, 0.2))
        self.assertFalse(choice.random_action)
        self.assertIn(choice.action, range(3))

    def test_one_update_per_transition_and_hard_target_sync(self):
        agent = DiscreteSACAgent(4, 3, small_config(), torch.device("cpu"), 12)
        syncs = []
        for step in range(8):
            agent.observe(transition(action=step % 3, value=step / 100))
            diagnostics = agent.optimize()
            if diagnostics is not None and diagnostics["target_synced"]:
                syncs.append(agent.gradient_steps)
        self.assertEqual(agent.gradient_steps, 7)
        self.assertEqual(syncs, [3, 6])
        self.assertTrue(all(parameter.grad is None for parameter in agent.target1.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in agent.target2.parameters()))

    def test_optimize_reports_complete_finite_sac_diagnostics(self):
        agent = DiscreteSACAgent(4, 3, small_config(), torch.device("cpu"), 13)
        agent.observe(transition(action=0, reward=1.0, terminated=True, episode_end=True))
        agent.observe(transition(action=1, reward=-0.5, episode_end=True))
        diagnostics = agent.optimize()
        required = {
            "actor_loss", "critic1_loss", "critic2_loss", "critic_loss_mean",
            "td_error_abs_mean", "q1_taken_mean", "q2_taken_mean", "q_gap_abs_mean",
            "soft_target_mean", "next_soft_value_mean", "min_q_policy_mean",
            "policy_entropy_mean", "next_policy_entropy_mean",
            "max_action_probability_mean", "min_action_probability_mean", "alpha",
            "actor_grad_norm", "critic1_grad_norm", "critic2_grad_norm",
            "actor_learning_rate", "critic_learning_rate", "target_synced",
            "batch_terminal_fraction", "batch_size", "replay_size",
        }
        self.assertEqual(set(diagnostics), required)
        self.assertTrue(all(np.isfinite(float(value)) for value in diagnostics.values()))
        self.assertAlmostEqual(diagnostics["alpha"], 0.2)
        self.assertAlmostEqual(diagnostics["batch_terminal_fraction"], 0.5)
        self.assertIs(type(diagnostics["batch_size"]), int)
        self.assertIs(type(diagnostics["replay_size"]), int)
        self.assertAlmostEqual(diagnostics["critic_loss_mean"],
                               0.5 * (diagnostics["critic1_loss"] + diagnostics["critic2_loss"]), places=6)

    def test_full_checkpoint_restores_all_networks_optimizers_replay_and_rng(self):
        config = small_config()
        original = DiscreteSACAgent(4, 3, config, torch.device("cpu"), 14)
        for step in range(7):
            action = original.act(obs(4, step / 100)).action
            original.observe(transition(action=action, reward=step / 10, value=step / 100,
                                          episode_end=step == 4))
            original.optimize()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "full.pt")
            original.save(path, kind="full", extra={"action_map": [[0.0, 0.0]]})
            restored = DiscreteSACAgent(4, 3, config, torch.device("cpu"), 99)
            payload = restored.load(path)
            self.assertEqual(payload["extra"], {"action_map": [[0.0, 0.0]]})
            for left, right in (
                (original.actor, restored.actor), (original.critic1, restored.critic1),
                (original.critic2, restored.critic2), (original.target1, restored.target1),
                (original.target2, restored.target2),
            ):
                self.assertEqual(parameter_digest(left), parameter_digest(right))
            self.assertEqual(original.replay.state_dict()["position"], restored.replay.state_dict()["position"])
            observation = obs(4, 0.2)
            left_action = original.act(observation).action
            right_action = restored.act(observation).action
            self.assertEqual(left_action, right_action)
            sample = transition(action=left_action, reward=0.7, value=0.2)
            original.observe(sample); restored.observe(sample)
            left_diag = original.optimize(); right_diag = restored.optimize()
            self.assertEqual(left_diag, right_diag)
            self.assertEqual(parameter_digest(original.actor), parameter_digest(restored.actor))
            self.assertEqual(parameter_digest(original.critic1), parameter_digest(restored.critic1))

    def test_policy_checkpoint_supports_two_separate_reproducible_modes(self):
        agent = DiscreteSACAgent(4, 3, small_config(), torch.device("cpu"), 15)
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.zero_()
            agent.actor.network[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.pt")
            agent.save(path, kind="policy_only")
            state_before = torch.get_rng_state().clone()
            policy = DiscreteSACPolicy(path, torch.device("cpu"))
            self.assertTrue(torch.equal(torch.get_rng_state(), state_before))
            deterministic = [policy.act(obs(4), "deterministic", seed) for seed in range(20)]
            stochastic = [policy.act(obs(4), "stochastic", seed) for seed in range(20)]
            self.assertEqual(set(deterministic), {1})
            self.assertEqual(stochastic, [policy.act(obs(4), "stochastic", seed) for seed in range(20)])
            self.assertGreater(len(set(stochastic)), 1)
            self.assertEqual(policy.parameter_digest, parameter_digest(agent.actor))

    def test_checkpoint_rejects_foreign_algorithm_and_incomplete_full_state(self):
        agent = DiscreteSACAgent(4, 3, small_config(), torch.device("cpu"), 16)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.pt")
            agent.save(path, kind="policy_only")
            payload = _torch_load(path, torch.device("cpu"))
            foreign = copy.deepcopy(payload); foreign["algorithm"] = "RainbowDQN"
            with self.assertRaisesRegex(ValueError, "not DiscreteSAC"):
                validate_checkpoint_payload(foreign)
            fake_full = copy.deepcopy(payload); fake_full["kind"] = "full"
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_checkpoint_payload(fake_full)

    def test_configuration_and_observation_fail_closed(self):
        invalid = (
            dict(alpha=0.0), dict(actor_learning_rate=0.0), dict(critic_learning_rate=0.0),
            dict(gamma=1.0), dict(batch_size=0), dict(replay_capacity=1, batch_size=2),
            dict(warmup_steps=1, batch_size=2), dict(target_update_steps=0),
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                small_config(**override).validate()
        agent = DiscreteSACAgent(4, 3, small_config(), torch.device("cpu"), 17)
        with self.assertRaises(ValueError):
            agent.act(np.zeros(3, dtype=np.float32))
        bad = obs(4); bad[0] = np.nan
        with self.assertRaises(ValueError):
            agent.act(bad)


if __name__ == "__main__":
    unittest.main()
