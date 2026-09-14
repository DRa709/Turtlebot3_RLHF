"""Executable Rainbow component, interaction and checkpoint tests."""

import copy
import glob
import math
import os
import random
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is required for the RainbowDQN learner tests") from exc

from turtlebot3_drl_nav.learner_api import Transition
from turtlebot3_drl_nav.rainbowdqn import (
    ALGORITHM,
    GreedyPolicy,
    NoisyLinear,
    PrioritizedReplayBuffer,
    RainbowDQNAgent,
    RainbowDQNConfig,
    RainbowQNetwork,
    RainbowTransition,
    parameter_digest,
    project_c51,
)

torch.use_deterministic_algorithms(True)


def obs(size, value=0.0):
    return np.full(size, value, dtype=np.float32)


def tr(size, action=0, reward=0.0, value=0.0, terminated=False, episode_end=False):
    return Transition(obs(size, value), action, reward, obs(size, value + 0.01), terminated, episode_end)


def small_config(**updates):
    values = dict(
        gamma=0.99, learning_rate=1e-4, batch_size=2, replay_capacity=32,
        warmup_steps=4, target_update_steps=3, hidden_size=8,
        gradient_clip_norm=10.0, atoms=5, v_min=-2.0, v_max=2.0,
        n_step=3, per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0,
        per_beta_decay_steps=40, per_epsilon=1e-6, noisy_sigma0=0.5,
        torch_threads=1,
    )
    values.update(updates)
    return RainbowDQNConfig(**values)


def slow_projection(probabilities, rewards, terminated, steps, gamma, support):
    out = torch.zeros_like(probabilities)
    delta = float((support[-1] - support[0]) / (support.numel() - 1))
    for row in range(probabilities.shape[0]):
        discount = (gamma ** float(steps[row])) * (1.0 - float(terminated[row]))
        for atom, probability in enumerate(probabilities[row]):
            shifted = float(rewards[row]) + discount * float(support[atom])
            shifted = min(float(support[-1]), max(float(support[0]), shifted))
            position = min(support.numel() - 1, max(0.0, (shifted - float(support[0])) / delta))
            lower, upper = math.floor(position), math.ceil(position)
            if lower == upper:
                out[row, lower] += probability
            else:
                out[row, lower] += probability * (upper - position)
                out[row, upper] += probability * (position - lower)
    return out


class RainbowComponentTests(unittest.TestCase):
    def test_network_is_dueling_categorical_and_mean_centered(self):
        network = RainbowQNetwork(3, 4, 7, 5, -2.0, 2.0, 0.5)
        network.eval()
        observation = torch.tensor([[0.2, -0.1, 0.7], [0.0, 0.5, -0.4]])
        with torch.no_grad():
            value, advantage = network.streams(observation)
            logits = network(observation)
            probabilities = network.probabilities(observation)
            q_values = network.q_values(observation)
        self.assertEqual(tuple(value.shape), (2, 1, 5))
        self.assertEqual(tuple(advantage.shape), (2, 4, 5))
        self.assertEqual(tuple(logits.shape), (2, 4, 5))
        self.assertTrue(torch.allclose(logits.mean(dim=1, keepdim=True), value, atol=1e-6))
        self.assertTrue(torch.allclose(probabilities.sum(dim=2), torch.ones(2, 4), atol=1e-7))
        self.assertEqual(tuple(q_values.shape), (2, 4))
        self.assertTrue(torch.equal(network.support, torch.linspace(-2.0, 2.0, 5)))

    def test_factorized_noisy_layer_training_and_evaluation(self):
        torch.manual_seed(3)
        layer = NoisyLinear(4, 3, 0.5)
        expected_sigma = 0.5 / math.sqrt(4)
        self.assertTrue(torch.allclose(layer.weight_sigma, torch.full_like(layer.weight_sigma, expected_sigma)))
        self.assertTrue(torch.allclose(layer.bias_sigma, torch.full_like(layer.bias_sigma, expected_sigma)))
        value = torch.ones(2, 4)
        layer.train(); first = layer(value); layer.reset_noise(); second = layer(value)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(torch.linalg.matrix_rank(layer.weight_epsilon).item(), 1)
        layer.eval(); mean_first = layer(value); layer.reset_noise(); mean_second = layer(value)
        self.assertTrue(torch.equal(mean_first, mean_second))

    def test_n_step_stops_and_flushes_at_termination(self):
        agent = RainbowDQNAgent(4, 2, small_config(gamma=0.5), torch.device("cpu"), seed=1)
        agent.observe(tr(4, reward=1.0))
        agent.observe(tr(4, reward=2.0, value=0.1, terminated=True, episode_end=True))
        self.assertEqual(len(agent.replay), 2)
        first, second = agent.replay.buffer[:2]
        self.assertEqual((first.reward, first.steps, first.terminated), (2.0, 2, True))
        self.assertEqual((second.reward, second.steps, second.terminated), (2.0, 1, True))
        self.assertTrue(np.array_equal(first.next_observation, obs(4, 0.11)))
        self.assertEqual(agent.n_step_accumulator, [])

    def test_truncation_flushes_but_bootstraps_from_pre_reset_observation(self):
        agent = RainbowDQNAgent(4, 2, small_config(gamma=0.5), torch.device("cpu"), seed=2)
        agent.observe(tr(4, reward=1.0))
        final = tr(4, reward=2.0, value=0.2, terminated=False, episode_end=True)
        agent.observe(final)
        first, second = agent.replay.buffer[:2]
        self.assertEqual((first.reward, first.steps, first.terminated), (2.0, 2, False))
        self.assertEqual((second.reward, second.steps, second.terminated), (2.0, 1, False))
        self.assertTrue(np.array_equal(first.next_observation, final.next_observation))
        agent.observe(tr(4, reward=9.0))
        self.assertEqual(len(agent.replay), 2)
        self.assertEqual(len(agent.n_step_accumulator), 1)

    def test_c51_projection_matches_slow_reference_and_conserves_mass(self):
        torch.manual_seed(5)
        support = torch.linspace(-2.0, 2.0, 5)
        probabilities = torch.softmax(torch.randn(7, 5), dim=1)
        rewards = torch.tensor([-3.0, -1.1, 0.0, 0.7, 2.8, 0.2, -0.3])
        terminated = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        steps = torch.tensor([3.0, 2.0, 1.0, 3.0, 1.0, 2.0, 3.0])
        actual, stats = project_c51(probabilities, rewards, terminated, steps, 0.99, support, True)
        expected = slow_projection(probabilities, rewards, terminated, steps, 0.99, support)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual.sum(dim=1), torch.ones(7), atol=1e-6))
        self.assertTrue(torch.all(actual >= 0.0))
        self.assertGreater(stats["projection_clip_low_fraction"], 0.0)
        self.assertGreater(stats["projection_clip_high_fraction"], 0.0)
        self.assertLessEqual(stats["projection_mass_error_max"], 1e-5)

    def test_projection_exact_atom_and_terminal_edges(self):
        support = torch.linspace(-2.0, 2.0, 5)
        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2], [0.0, 0.0, 1.0, 0.0, 0.0]])
        projected = project_c51(probabilities, torch.tensor([1.0, 0.0]),
                                torch.tensor([1.0, 0.0]), torch.tensor([3.0, 1.0]), 1.0, support)
        self.assertTrue(torch.allclose(projected[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(projected[1], probabilities[1]))

    def test_proportional_replay_probabilities_weights_ring_and_round_trip(self):
        replay = PrioritizedReplayBuffer(3, 1.0, random.Random(9))
        item = RainbowTransition(obs(2), 0, 0.0, obs(2), False, 1)
        for _ in range(3):
            replay.push(item)
        replay.update_priorities([0, 1, 2], [1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(replay.probabilities(), [1 / 6, 2 / 6, 3 / 6]))
        _, indices, weights = replay.sample(3, 0.4)
        probabilities = replay.probabilities()
        minimum = probabilities.min()
        expected = [(3 * probabilities[i]) ** -0.4 / ((3 * minimum) ** -0.4) for i in indices]
        self.assertTrue(np.allclose(weights.numpy(), expected))
        self.assertTrue(torch.all((weights > 0.0) & (weights <= 1.0)))
        replay.update_priorities([1, 1], [0.5, 4.0])
        self.assertEqual(replay.priorities[1], 4.0)
        replay.push(RainbowTransition(obs(2), 1, 4.0, obs(2), True, 1))
        self.assertEqual(replay.priorities[0], 4.0)
        state = replay.state_dict()
        restored = PrioritizedReplayBuffer(3, 1.0, random.Random(9))
        restored.load_state_dict(state)
        self.assertEqual((restored.position, len(restored)), (replay.position, len(replay)))
        self.assertTrue(np.array_equal(restored.priorities, replay.priorities))
        self.assertEqual(restored.max_priority, replay.max_priority)


class RainbowIntegrationTests(unittest.TestCase):
    def test_double_selection_uses_online_action_and_target_distribution(self):
        cfg = small_config(gamma=0.5, atoms=3, v_min=-1.0, v_max=1.0,
                           n_step=1, warmup_steps=2, target_update_steps=100)
        agent = RainbowDQNAgent(2, 2, cfg, torch.device("cpu"), seed=11)
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
            # Online selects action 1 (+1 distribution); target would select
            # action 0, but Double-Q must evaluate target action 1 (-1).
            # NoisyLinear exposes bias_mu/bias_sigma, not nn.Linear.bias.
            # Sigma and sampled noise are already zeroed above, so these means
            # isolate the intended online-selector/target-evaluator split.
            agent.online.advantage.bias_mu.copy_(torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0, 10.0]))
            agent.target.advantage.bias_mu.copy_(torch.tensor([-10.0, -10.0, 10.0, 10.0, -10.0, -10.0]))
        item = RainbowTransition(obs(2), 0, 0.0, obs(2), False, 1)
        agent.replay.push(item); agent.replay.push(item); agent.env_steps = 2
        diagnostics = agent.optimize()
        self.assertLess(diagnostics["next_max_q_mean"], -0.99)
        self.assertLess(diagnostics["target_mean"], -0.49)
        self.assertNotAlmostEqual(diagnostics["target_mean"], 0.5, places=2)
        self.assertTrue(all(parameter.grad is None for parameter in agent.target.parameters()))

    def test_warmup_noisy_actions_beta_schedule_and_target_sync(self):
        cfg = small_config()
        agent = RainbowDQNAgent(4, 3, cfg, torch.device("cpu"), seed=12)
        synced = []
        for step in range(1, 11):
            before_noise = agent.online.feature1.weight_epsilon.clone()
            choice = agent.act(obs(4))
            self.assertIsNone(choice.epsilon_used)
            if step <= cfg.warmup_steps:
                self.assertTrue(choice.random_action)
            else:
                self.assertFalse(choice.random_action)
                self.assertFalse(torch.equal(before_noise, agent.online.feature1.weight_epsilon))
            agent.observe(tr(4, action=choice.action))
            diagnostics = agent.optimize()
            if step < cfg.warmup_steps:
                self.assertIsNone(diagnostics)
            else:
                self.assertIsNotNone(diagnostics)
                if diagnostics["target_synced"]:
                    synced.append(agent.gradient_steps)
        self.assertEqual(synced, [3, 6])
        self.assertAlmostEqual(agent.per_beta(0), 0.4)
        self.assertAlmostEqual(agent.per_beta(20), 0.7)
        self.assertAlmostEqual(agent.per_beta(100), 1.0)

    def test_update_diagnostics_are_complete_and_finite(self):
        agent = RainbowDQNAgent(4, 3, small_config(), torch.device("cpu"), seed=13)
        for step in range(4):
            agent.observe(tr(4, action=step % 3, reward=0.1 * step, episode_end=step == 3))
        diagnostics = agent.optimize()
        required = {
            "loss", "distributional_loss_mean", "td_error_abs_mean", "q_taken_mean", "target_mean",
            "next_max_q_mean", "grad_norm", "learning_rate", "target_synced",
            "batch_terminal_fraction", "batch_size", "replay_size", "beta_is",
            "mean_importance_weight", "max_importance_weight", "mean_priority", "max_priority",
            "noisy_sigma_mean", "noisy_sigma_min", "noisy_sigma_max", "n_step",
            "mean_effective_n_step", "value_stream_expected_mean", "centered_advantage_abs_mean",
            "chosen_distribution_entropy_mean", "projection_clip_low_fraction",
            "projection_clip_high_fraction", "projection_mass_error_max",
        }
        self.assertEqual(set(diagnostics), required)
        self.assertTrue(all(np.isfinite(float(value)) for value in diagnostics.values()))
        self.assertEqual(diagnostics["n_step"], 3)
        self.assertGreater(diagnostics["mean_priority"], 0.0)

    def test_evaluation_disables_noise_and_is_deterministic(self):
        agent = RainbowDQNAgent(4, 3, small_config(), torch.device("cpu"), seed=14)
        choices = [agent.act(obs(4, 0.2), evaluation=True) for _ in range(20)]
        self.assertEqual(len({choice.action for choice in choices}), 1)
        self.assertTrue(all(choice.epsilon_used is None and not choice.random_action for choice in choices))
        self.assertFalse(agent.online.training)

    def test_full_checkpoint_restores_per_n_step_rng_and_exact_continuation(self):
        cfg = small_config()
        original = RainbowDQNAgent(4, 3, cfg, torch.device("cpu"), seed=15)
        for step in range(7):
            original.observe(tr(4, action=step % 3, reward=0.2, episode_end=step == 4))
            original.optimize()
        self.assertEqual(len(original.n_step_accumulator), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "full.pt")
            original.save(path, kind="full", extra={"components": "rainbow"})
            restored = RainbowDQNAgent(4, 3, cfg, torch.device("cpu"), seed=99)
            self.assertEqual(restored.load(path), {"components": "rainbow"})
            self.assertEqual(parameter_digest(original.online), parameter_digest(restored.online))
            self.assertEqual(parameter_digest(original.target), parameter_digest(restored.target))
            self.assertTrue(np.array_equal(original.replay.priorities, restored.replay.priorities))
            self.assertEqual(original.replay.position, restored.replay.position)
            self.assertEqual(len(original.n_step_accumulator), len(restored.n_step_accumulator))
            for left, right in zip(original.n_step_accumulator, restored.n_step_accumulator):
                self.assertTrue(np.array_equal(left.observation, right.observation))
                self.assertTrue(np.array_equal(left.next_observation, right.next_observation))
                self.assertEqual(left.action, right.action)
                self.assertEqual(left.reward, right.reward)
                self.assertEqual(left.terminated, right.terminated)
                self.assertEqual(left.episode_end, right.episode_end)
            torch_state = torch.get_rng_state().clone()
            choice_original = original.act(obs(4, 0.3)); original.observe(tr(4, choice_original.action, 0.4, 0.3))
            result_original = original.optimize(); original_digest = parameter_digest(original.online)
            torch.set_rng_state(torch_state)
            choice_restored = restored.act(obs(4, 0.3)); restored.observe(tr(4, choice_restored.action, 0.4, 0.3))
            result_restored = restored.optimize(); restored_digest = parameter_digest(restored.online)
            self.assertEqual(choice_original, choice_restored)
            self.assertEqual(result_original, result_restored)
            self.assertEqual(original_digest, restored_digest)
            self.assertTrue(np.array_equal(original.replay.priorities, restored.replay.priorities))

    def test_atomic_policy_checkpoint_and_algorithm_isolation(self):
        agent = RainbowDQNAgent(4, 3, small_config(), torch.device("cpu"), seed=16)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.pt")
            agent.save(path, kind="policy_only", extra={"action_map": [[0.1, 0.0]]})
            self.assertEqual(glob.glob(os.path.join(directory, "*.tmp")), [])
            rng_before_load = torch.get_rng_state().clone()
            policy = GreedyPolicy(path, torch.device("cpu"))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before_load))
            self.assertEqual(policy.parameter_digest, parameter_digest(agent.online))
            self.assertEqual(policy.act(obs(4), "greedy"), agent.act(obs(4), evaluation=True).action)
            with self.assertRaises(ValueError):
                policy.act(obs(4), "stochastic")
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["algorithm"], ALGORITHM)
            self.assertNotIn("replay", payload)
            payload["algorithm"] = "DuelingDoubleDQN"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                GreedyPolicy(path, torch.device("cpu"))

    def test_configuration_and_transition_validation_fail_closed(self):
        invalid = [
            dict(atoms=1), dict(v_min=2.0, v_max=1.0), dict(n_step=0),
            dict(per_alpha=1.1), dict(per_beta_start=0.8, per_beta_end=0.4),
            dict(per_epsilon=0.0), dict(noisy_sigma0=0.0),
            dict(dueling_aggregation="max"), dict(distributional_loss="mse"),
            dict(batch_size=8, warmup_steps=5, n_step=3),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                small_config(**values).validate()
        agent = RainbowDQNAgent(4, 3, small_config(), torch.device("cpu"), seed=17)
        bad = obs(4); bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            agent.observe(Transition(bad, 0, 0.0, obs(4), False, False))
        with self.assertRaisesRegex(ValueError, "action"):
            agent.observe(Transition(obs(4), 3, 0.0, obs(4), False, False))
        with self.assertRaisesRegex(ValueError, "reward"):
            agent.observe(Transition(obs(4), 0, float("inf"), obs(4), False, False))
        with self.assertRaisesRegex(ValueError, "must end"):
            agent.observe(Transition(obs(4), 0, 0.0, obs(4), True, False))


if __name__ == "__main__":
    unittest.main()
