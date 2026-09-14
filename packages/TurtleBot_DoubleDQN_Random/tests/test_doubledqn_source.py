"""PyTorch-independent structural guard for the Double DQN target.

The numerical discriminator lives in test_doubledqn.py and must run in the ARC
image. This AST test still fails locally if the implementation is accidentally
changed back to a target-network maximum or if selection/evaluation leave the
no-gradient target block.
"""

import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "turtlebot3_drl_nav", "doubledqn.py")


def is_network_call(node, attribute):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == attribute
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "next_states"
    )


class DoubleDQNSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SOURCE, encoding="utf-8") as stream:
            cls.tree = ast.parse(stream.read(), SOURCE)
        cls.optimize = next(node for node in ast.walk(cls.tree) if isinstance(node, ast.FunctionDef) and node.name == "optimize")

    def test_online_argmax_and_target_gather_are_inside_no_grad(self):
        target_block = None
        for node in ast.walk(self.optimize):
            if not isinstance(node, ast.With):
                continue
            if any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "torch"
                and item.context_expr.func.attr == "no_grad"
                for item in node.items
            ):
                target_block = node
                break
        self.assertIsNotNone(target_block)
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(target_block)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        selected = assignments["next_actions"]
        self.assertIsInstance(selected, ast.Call)
        self.assertIsInstance(selected.func, ast.Attribute)
        self.assertEqual(selected.func.attr, "argmax")
        self.assertTrue(is_network_call(selected.func.value, "online"))

        evaluated = assignments["next_values"]
        self.assertIsInstance(evaluated, ast.Call)
        self.assertEqual(evaluated.func.attr, "squeeze")
        gather = evaluated.func.value
        self.assertIsInstance(gather, ast.Call)
        self.assertEqual(gather.func.attr, "gather")
        self.assertTrue(is_network_call(gather.func.value, "target"))
        self.assertTrue(any(isinstance(arg, ast.Name) and arg.id == "next_actions" for arg in gather.args))

    def test_target_network_maximum_is_not_used(self):
        forbidden = []
        for node in ast.walk(self.optimize):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "max"
                and is_network_call(node.func.value, "target")
            ):
                forbidden.append(node.lineno)
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
