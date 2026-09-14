"""PyTorch-independent structural guards for the complete Rainbow bundle.

The numerical tests in ``test_rainbowdqn.py`` must run inside the sealed ARC
image. These AST guards remain executable on a host without PyTorch and catch
accidental removal of any Rainbow mechanism or dependency on another learner.
"""

import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "turtlebot3_drl_nav", "rainbowdqn.py")


def class_node(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def method_node(node, name):
    return next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name)


class RainbowDQNSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SOURCE, encoding="utf-8") as stream:
            cls.source = stream.read()
        cls.tree = ast.parse(cls.source, SOURCE)
        cls.agent = class_node(cls.tree, "RainbowDQNAgent")
        cls.network = class_node(cls.tree, "RainbowQNetwork")
        cls.noisy = class_node(cls.tree, "NoisyLinear")
        cls.replay = class_node(cls.tree, "PrioritizedReplayBuffer")
        cls.optimize = method_node(cls.agent, "optimize")

    def test_module_is_standalone_and_does_not_import_other_learners(self):
        imported = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for forbidden in (".dqn", ".doubledqn", ".duelingdoubledqn", "turtlebot_dqn"):
            self.assertFalse(any(name.endswith(forbidden) for name in imported), imported)

    def test_factorized_noisy_linear_has_mu_sigma_and_rank_one_noise(self):
        attributes = {
            target.attr
            for node in ast.walk(method_node(self.noisy, "__init__"))
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        self.assertTrue({"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"} <= attributes)
        reset = method_node(self.noisy, "reset_noise")
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "outer" for node in ast.walk(reset)))
        forward = method_node(self.noisy, "forward")
        self.assertTrue(any(isinstance(node, ast.If) for node in ast.walk(forward)))

    def test_network_is_noisy_dueling_and_categorical(self):
        init = method_node(self.network, "__init__")
        assigned = {
            target.attr
            for node in ast.walk(init) if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name) and target.value.id == "self"
        }
        self.assertTrue({"feature1", "feature2", "value", "advantage"} <= assigned)
        self.assertGreaterEqual(sum(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "NoisyLinear"
            for node in ast.walk(init)
        ), 4)
        forward = method_node(self.network, "forward")
        means = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "mean"]
        self.assertTrue(any(
            getattr(next((kw.value for kw in call.keywords if kw.arg == "dim"), None), "value", None) == 1
            and getattr(next((kw.value for kw in call.keywords if kw.arg == "keepdim"), None), "value", None) is True
            for call in means
        ))
        probabilities = method_node(self.network, "probabilities")
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "softmax" for node in ast.walk(probabilities)))

    def test_c51_projection_clamps_indices_and_conserves_mass(self):
        projection = next(node for node in ast.walk(self.tree)
                          if isinstance(node, ast.FunctionDef) and node.name == "project_c51")
        calls = [node.func.attr for node in ast.walk(projection) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertIn("clamp", calls)
        self.assertGreaterEqual(calls.count("index_add_"), 2)
        self.assertIn("sum", calls)
        names = {node.id for node in ast.walk(projection) if isinstance(node, ast.Name)}
        self.assertTrue({"terminated", "steps", "gamma"} <= names)

    def test_double_selection_and_target_distribution_are_in_no_grad(self):
        blocks = [node for node in ast.walk(self.optimize) if isinstance(node, ast.With)
                  and any(isinstance(item.context_expr, ast.Call)
                          and isinstance(item.context_expr.func, ast.Attribute)
                          and item.context_expr.func.attr == "no_grad" for item in node.items)]
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        assignments = {node.targets[0].id: node.value for node in ast.walk(block)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)}
        self.assertIn("next_actions", assignments)
        self.assertIn("target_probabilities", assignments)
        selected_names = {node.id for node in ast.walk(assignments["next_actions"]) if isinstance(node, ast.Name)}
        evaluated_names = {node.id for node in ast.walk(assignments["target_probabilities"]) if isinstance(node, ast.Name)}
        self.assertIn("self", selected_names)
        self.assertIn("self", evaluated_names)
        selected_attrs = {node.attr for node in ast.walk(assignments["next_actions"]) if isinstance(node, ast.Attribute)}
        evaluated_attrs = {node.attr for node in ast.walk(assignments["target_probabilities"]) if isinstance(node, ast.Attribute)}
        self.assertTrue({"online", "q_values", "argmax"} <= selected_attrs)
        self.assertTrue({"target", "probabilities"} <= evaluated_attrs)
        self.assertIn("next_actions", evaluated_names)

    def test_per_uses_alpha_beta_and_logarithmic_trees(self):
        attributes = {node.attr for node in ast.walk(self.replay) if isinstance(node, ast.Attribute)}
        names = {node.id for node in ast.walk(self.replay) if isinstance(node, ast.Name)}
        powers = [node for node in ast.walk(self.replay) if isinstance(node, ast.Pow)]
        self.assertGreaterEqual(len(powers), 2)
        self.assertIn("alpha", attributes)
        self.assertIn("beta", names)
        self.assertTrue({"sum_tree", "min_tree", "max_tree"} <= attributes)
        self.assertIn("_find_prefix", attributes)

    def test_n_step_stops_on_episode_end_but_bootstraps_truncation(self):
        observe = method_node(self.agent, "observe")
        aggregate = method_node(self.agent, "_aggregate_front")
        observe_names = {node.attr for node in ast.walk(observe) if isinstance(node, ast.Attribute)}
        aggregate_names = {node.attr for node in ast.walk(aggregate) if isinstance(node, ast.Attribute)}
        self.assertIn("episode_end", observe_names)
        self.assertIn("episode_end", aggregate_names)
        self.assertIn("terminated", aggregate_names)
        self.assertIn("next_observation", aggregate_names)

    def test_checkpoint_contains_per_n_step_and_all_rng_states(self):
        save = method_node(self.agent, "save")
        constants = {node.value for node in ast.walk(save) if isinstance(node, ast.Constant)
                     and isinstance(node.value, str)}
        required = {"target_state_dict", "optimizer_state_dict", "replay", "n_step_accumulator",
                    "python_rng_state", "numpy_rng_state", "torch_rng_state"}
        self.assertTrue(required <= constants)


if __name__ == "__main__":
    unittest.main()
