"""PyTorch-independent structural guards for the Discrete SAC implementation."""

import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "turtlebot3_drl_nav", "discretesac.py")


def class_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def function_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def method_node(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def segment(source, node):
    return ast.get_source_segment(source, node) or ""


class DiscreteSACSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PATH, encoding="utf-8") as stream:
            cls.source = stream.read()
        cls.tree = ast.parse(cls.source)
        cls.agent = class_node(cls.tree, "DiscreteSACAgent")

    def test_is_standalone_and_has_one_named_algorithm(self):
        imports = [node for node in ast.walk(self.tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        text = "\n".join(segment(self.source, node) for node in imports)
        for forbidden in (".dqn", ".doubledqn", ".duelingdoubledqn", ".rainbowdqn"):
            self.assertNotIn(forbidden, text)
        assignment = next(node for node in self.tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "ALGORITHM" for target in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value), "DiscreteSAC")

    def test_networks_are_categorical_actor_and_two_plain_q_mlps(self):
        actor = class_node(self.tree, "CategoricalActor")
        critic = class_node(self.tree, "QNetwork")
        self.assertIn("log_softmax", segment(self.source, method_node(actor, "distribution")))
        agent_init = segment(self.source, method_node(self.agent, "__init__"))
        for name in ("self.actor", "self.critic1", "self.critic2", "self.target1", "self.target2"):
            self.assertIn(name, agent_init)
        for forbidden in ("NoisyLinear", "C51", "Prioritized", "dueling"):
            self.assertNotIn(forbidden, segment(self.source, actor) + segment(self.source, critic))

    def test_exact_soft_value_clips_double_q_and_includes_entropy(self):
        source = segment(self.source, function_node(self.tree, "soft_value"))
        self.assertIn("torch.minimum(q1, q2)", source)
        self.assertIn("alpha", source)
        self.assertIn("log_probabilities", source)
        self.assertIn(".sum(dim=1)", source)

    def test_only_termination_mask_disables_bootstrap(self):
        source = segment(self.source, function_node(self.tree, "critic_target"))
        self.assertIn("1.0 - terminated", source)
        self.assertNotIn("episode_end", source)

    def test_actor_objective_sums_all_actions_without_sampled_log_prob(self):
        source = segment(self.source, function_node(self.tree, "actor_objective"))
        self.assertIn("torch.minimum(q1, q2)", source)
        self.assertIn(".sum(dim=1).mean()", source)
        self.assertNotIn("gather", source)

    def test_uniform_replay_and_random_warmup_are_explicit(self):
        replay = class_node(self.tree, "UniformReplayBuffer")
        self.assertIn("self.rng.sample", segment(self.source, method_node(replay, "sample")))
        action = segment(self.source, method_node(self.agent, "act"))
        self.assertIn("self.env_steps < self.config.warmup_steps", action)
        self.assertIn("self.rng.randrange", action)
        self.assertIn("_categorical_sample", action)

    def test_temperature_is_fixed_and_not_optimized(self):
        init = segment(self.source, method_node(self.agent, "__init__"))
        self.assertNotIn("alpha_optimizer", self.source)
        self.assertNotIn("target_entropy", self.source)
        self.assertNotIn("log_alpha", self.source)
        self.assertIn("self.config", init)
        optimize = segment(self.source, method_node(self.agent, "optimize"))
        self.assertIn("self.config.alpha", optimize)

    def test_targets_are_hard_synchronized_by_gradient_step(self):
        source = segment(self.source, method_node(self.agent, "optimize"))
        self.assertIn("self.gradient_steps % self.config.target_update_steps == 0", source)
        self.assertIn("self.target1.load_state_dict(self.critic1.state_dict())", source)
        self.assertIn("self.target2.load_state_dict(self.critic2.state_dict())", source)

    def test_full_checkpoint_contains_all_training_state_and_rng(self):
        source = segment(self.source, method_node(self.agent, "_checkpoint_payload"))
        required = (
            "actor_state_dict", "critic1_state_dict", "critic2_state_dict",
            "target1_state_dict", "target2_state_dict", "actor_optimizer_state_dict",
            "critic1_optimizer_state_dict", "critic2_optimizer_state_dict",
            "replay_state", "python_rng_state", "numpy_rng_state", "torch_rng_state",
        )
        for key in required:
            self.assertIn(key, source)

    def test_integer_csv_diagnostics_are_not_serialized_as_floats(self):
        source = segment(self.source, method_node(self.agent, "optimize"))
        self.assertIn('"batch_size": int(len(batch))', source)
        self.assertIn('"replay_size": int(len(self.replay))', source)
        self.assertNotIn('"batch_size": float(', source)
        self.assertNotIn('"replay_size": float(', source)

    def test_evaluation_requires_explicit_mode_and_seeded_stochastic_sampling(self):
        policy = class_node(self.tree, "DiscreteSACPolicy")
        source = segment(self.source, method_node(policy, "act"))
        self.assertIn('policy_mode == "deterministic"', source)
        self.assertIn('policy_mode == "stochastic"', source)
        self.assertIn("sampling_seed", source)
        self.assertIn("random.Random(int(sampling_seed))", source)


if __name__ == "__main__":
    unittest.main()
