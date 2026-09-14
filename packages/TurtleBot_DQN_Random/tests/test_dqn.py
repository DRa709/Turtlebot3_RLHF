"""DQN learner tests (roadmap §3.1): independent recomputation of the target,
loss and optimizer step on a fixed sampled batch; mask semantics; warm-up;
epsilon schedule; target synchronization; atomic checkpoints; greedy policy."""

import copy
import glob
import os
import random
import tempfile
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is required for the DQN learner tests") from exc
from torch import nn

from turtlebot3_drl_nav.dqn import ALGORITHM, DQNAgent, DQNConfig, GreedyPolicy, parameter_digest
from turtlebot3_drl_nav.learner_api import Transition

torch.use_deterministic_algorithms(True)
OBS, ACT = 41, 5


def make_batch(rng, n):
    out = []
    for _ in range(n):
        kind = rng.integers(0, 3)  # 0 running, 1 terminated, 2 truncated (m_term = 0)
        out.append(Transition(rng.random(OBS).astype(np.float32), int(rng.integers(0, ACT)), float(rng.normal()),
                              rng.random(OBS).astype(np.float32), kind == 1))
    return out


def reference_loss(online, target, batch, gamma, loss_name):
    s = torch.tensor(np.stack([b.observation for b in batch]))
    a = torch.tensor([b.action for b in batch])
    r = torch.tensor([b.reward for b in batch], dtype=torch.float32)
    s2 = torch.tensor(np.stack([b.next_observation for b in batch]))
    m_term = torch.tensor([float(b.terminated) for b in batch])
    with torch.no_grad():
        y = r + gamma * (1.0 - m_term) * target(s2).max(dim=1).values
    q = online(s)[torch.arange(len(batch)), a]
    d = q - y
    per = torch.where(d.abs() < 1.0, 0.5 * d * d, d.abs() - 0.5) if loss_name == "huber" else d * d
    return per.mean(), y


class DQNUpdateTests(unittest.TestCase):
    def test_update_matches_independent_recomputation_bit_exactly(self):
        cfg = DQNConfig(gamma=0.99, batch_size=8, warmup_steps=8, replay_capacity=64, target_update_steps=3, hidden_size=16, loss="huber")
        agent = DQNAgent(OBS, ACT, cfg, torch.device("cpu"), seed=7)
        rng = np.random.default_rng(0)
        for t in make_batch(rng, 32):
            agent.replay.push(t)
        agent.env_steps = 32
        drawn = random.Random()
        drawn.setstate(agent.rng.getstate())
        sampled = drawn.sample(agent.replay._buffer, cfg.batch_size)
        online_ref, target_ref = copy.deepcopy(agent.online), copy.deepcopy(agent.target)
        ref_loss, y_ref = reference_loss(online_ref, target_ref, sampled, cfg.gamma, cfg.loss)
        opt = torch.optim.Adam(online_ref.parameters(), lr=cfg.learning_rate)
        opt.load_state_dict(agent.optimizer.state_dict())
        opt.zero_grad()
        ref_loss.backward()
        nn.utils.clip_grad_norm_(online_ref.parameters(), cfg.gradient_clip_norm)
        opt.step()
        diagnostics = agent.optimize()
        self.assertEqual(diagnostics["loss"], ref_loss.item())
        for (n1, p1), (_, p2) in zip(agent.online.named_parameters(), online_ref.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), n1)
        for b, y in zip(sampled, y_ref):
            if b.terminated:
                self.assertEqual(y.item(), float(np.float32(b.reward)))  # genuine termination: no bootstrap
        self.assertAlmostEqual(diagnostics["target_mean"], y_ref.mean().item(), places=6)
        self.assertGreaterEqual(diagnostics["grad_norm"], 0.0)

    def test_truncation_bootstraps_and_termination_does_not(self):
        cfg = DQNConfig(gamma=0.5, batch_size=2, warmup_steps=2, hidden_size=4, replay_capacity=2, target_update_steps=100, loss="mse")
        agent = DQNAgent(2, 2, cfg, torch.device("cpu"), seed=1)
        with torch.no_grad():
            for p in agent.online.parameters():
                p.zero_()
            for p in agent.target.parameters():
                p.zero_()
            agent.target.net[-1].bias.fill_(2.0)
        o = np.zeros(2, dtype=np.float32)
        agent.observe(Transition(o, 0, 1.0, o, True))    # target 1.0
        agent.observe(Transition(o, 0, 1.0, o, False))   # target 1.0 + 0.5 * 2.0 = 2.0
        self.assertAlmostEqual(agent.optimize()["loss"], 2.5, places=5)

    def test_warmup_target_sync_and_epsilon_schedule(self):
        cfg = DQNConfig(batch_size=2, warmup_steps=5, hidden_size=4, replay_capacity=16, target_update_steps=3, epsilon_decay_steps=20)
        agent = DQNAgent(4, 3, cfg, torch.device("cpu"), seed=2)
        o = np.zeros(4, dtype=np.float32)
        synced = []
        for i in range(1, 12):
            choice = agent.act(o)
            if i <= cfg.warmup_steps:
                self.assertTrue(choice.random_action)
            agent.observe(Transition(o, choice.action, 0.0, o, False))
            result = agent.optimize()
            if i < cfg.warmup_steps:
                self.assertIsNone(result)
            else:
                self.assertIsNotNone(result)
                if result["target_synced"]:
                    synced.append(agent.gradient_steps)
        self.assertEqual(synced, [3, 6])
        self.assertAlmostEqual(agent.epsilon(0), 1.0)
        self.assertAlmostEqual(agent.epsilon(10), 1.0 + 0.5 * (0.05 - 1.0))
        self.assertAlmostEqual(agent.epsilon(40), 0.05)
        # epsilon reported by act() is the value used for that action (schedule at the current env_steps)
        self.assertAlmostEqual(agent.act(o).epsilon_used, agent.epsilon(agent.env_steps))
        counts = np.bincount([agent.act(o).action for _ in range(3000)], minlength=3)
        self.assertTrue(all(c > 0 for c in counts))

    def test_evaluation_is_greedy_and_deterministic(self):
        cfg = DQNConfig(batch_size=2, warmup_steps=2, hidden_size=4, replay_capacity=8)
        agent = DQNAgent(4, 3, cfg, torch.device("cpu"), seed=3)
        o = np.linspace(0, 1, 4).astype(np.float32)
        choices = {agent.act(o, evaluation=True).action for _ in range(20)}
        self.assertEqual(len(choices), 1)
        self.assertEqual(agent.act(o, evaluation=True).epsilon_used, 0.0)

    def test_atomic_checkpoint_and_greedy_policy_round_trip(self):
        cfg = DQNConfig(batch_size=2, warmup_steps=2, hidden_size=4, replay_capacity=8)
        agent = DQNAgent(4, 3, cfg, torch.device("cpu"), seed=4)
        o = np.zeros(4, dtype=np.float32)
        for _ in range(3):
            agent.observe(Transition(o, 0, 1.0, o, False))
            agent.optimize()
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = os.path.join(tmp, "policy.pt")
            full_path = os.path.join(tmp, "full.pt")
            agent.save(policy_path, kind="policy_only", extra={"note": 1})
            agent.save(full_path, kind="full")
            self.assertEqual(glob.glob(os.path.join(tmp, "*.tmp")), [])
            policy = GreedyPolicy(policy_path, torch.device("cpu"))
            self.assertEqual(policy.parameter_digest, parameter_digest(agent.online))
            self.assertEqual(policy.act(o, "greedy"), agent.act(o, evaluation=True).action)
            with self.assertRaises(ValueError):
                policy.act(o, "stochastic")
            restored = DQNAgent(4, 3, cfg, torch.device("cpu"), seed=99)
            extra = restored.load(full_path)
            self.assertEqual((restored.env_steps, restored.gradient_steps), (agent.env_steps, agent.gradient_steps))
            self.assertEqual(len(restored.replay), len(agent.replay))
            self.assertEqual(parameter_digest(restored.online), parameter_digest(agent.online))
            self.assertEqual(extra, {})
            self.assertEqual(restored.seed, agent.seed)
            self.assertEqual(restored.replay._next_index, agent.replay._next_index)
            payload = torch.load(policy_path, weights_only=False)
            self.assertEqual(payload["algorithm"], ALGORITHM)
            self.assertNotIn("replay", payload)
            payload["format_version"] = -1
            torch.save(payload, policy_path)
            with self.assertRaisesRegex(ValueError, "format_version"):
                GreedyPolicy(policy_path, torch.device("cpu"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            DQNConfig(warmup_steps=10, batch_size=64).validate()
        with self.assertRaises(ValueError):
            DQNConfig(epsilon_start=0.1, epsilon_end=0.5).validate()
        with self.assertRaises(ValueError):
            DQNConfig(loss="l1").validate()
        with self.assertRaises(ValueError):
            DQNConfig(learning_rate=0.0).validate()
        with self.assertRaises(ValueError):
            DQNConfig(batch_size=64.0).validate()

    def test_nonfinite_or_out_of_range_transition_is_refused(self):
        agent = DQNAgent(4, 3, DQNConfig(batch_size=2, warmup_steps=2, hidden_size=4, replay_capacity=8), torch.device("cpu"), seed=5)
        observation = np.zeros(4, dtype=np.float32)
        bad = observation.copy()
        bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            agent.observe(Transition(bad, 0, 0.0, observation, False))
        with self.assertRaisesRegex(ValueError, "action"):
            agent.observe(Transition(observation, 3, 0.0, observation, False))
        with self.assertRaisesRegex(ValueError, "reward"):
            agent.observe(Transition(observation, 0, float("inf"), observation, False))
        with self.assertRaisesRegex(ValueError, "action"):
            agent.observe(Transition(observation, 1.5, 0.0, observation, False))
        with self.assertRaisesRegex(ValueError, "mask"):
            agent.observe(Transition(observation, 0, 0.0, observation, 1))

    def test_replay_ring_overwrites_without_copying_the_whole_buffer(self):
        rng = random.Random(1)
        from turtlebot3_drl_nav.dqn import ReplayBuffer
        replay = ReplayBuffer(3, rng)
        observation = np.zeros(4, dtype=np.float32)
        for reward in range(5):
            replay.push(Transition(observation, 0, float(reward), observation, False))
        self.assertEqual(len(replay), 3)
        self.assertEqual(sorted(item.reward for item in replay._buffer), [2.0, 3.0, 4.0])
        state = replay.state_dict()
        restored = ReplayBuffer(3, random.Random(1))
        restored.load_state_dict(state)
        self.assertEqual(restored._next_index, replay._next_index)
        self.assertEqual([item.reward for item in restored._buffer], [item.reward for item in replay._buffer])


if __name__ == "__main__":
    unittest.main()
