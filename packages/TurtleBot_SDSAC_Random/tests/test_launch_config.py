import os
import re
import unittest

from turtlebot3_drl_nav.launch_config import LAUNCH_ARGUMENTS, build_node_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"package_root": "/pkg", "run_dir": "/run", "mode": "train", "learning_seed": "101", "world_seed": "7000",
        "initialization_seed": "7001", "dynamic_obstacle_seed": "7002", "evaluation_seed": "9001", "phase_label": "pilot", "checkpoint_path": ""}


class LaunchConfigTests(unittest.TestCase):
    def test_every_launch_argument_reaches_a_node(self):
        params = build_node_parameters(dict(BASE))
        env, agent = params["environment"], params["agent"]
        self.assertEqual(env["initialization_seed"], 7001)
        self.assertEqual(env["world_seed"], 7000)
        self.assertEqual(agent["learning_seed"], 101)
        self.assertEqual(agent["phase_label"], "pilot")
        self.assertEqual(agent["mode"], "train")
        self.assertTrue(env["use_sim_time"] and agent["use_sim_time"])
        for name in LAUNCH_ARGUMENTS:
            self.assertTrue(name in env or name in agent or name in ("mode", "checkpoint_path"), name)

    def test_unknown_missing_and_invalid_arguments_fail(self):
        with self.assertRaises(ValueError):
            build_node_parameters({**BASE, "environment_budget": "10"})
        with self.assertRaises(ValueError):
            build_node_parameters({k: v for k, v in BASE.items() if k != "phase_label"})
        with self.assertRaises(ValueError):
            build_node_parameters({**BASE, "phase_label": "full"})
        with self.assertRaises(ValueError):
            build_node_parameters({**BASE, "learning_seed": "abc"})
        with self.assertRaises(ValueError):
            build_node_parameters({**BASE, "checkpoint_path": "/x.pt"})  # train mode never resumes
        with self.assertRaises(ValueError):
            build_node_parameters({**BASE, "mode": "eval"})  # eval needs a checkpoint
        self.assertEqual(build_node_parameters({**BASE, "mode": "eval", "checkpoint_path": "/x.pt"})["agent"]["checkpoint_path"], "/x.pt")

    def test_launch_file_declares_exactly_the_documented_arguments(self):
        with open(os.path.join(ROOT, "launch", "drl_training.launch.py"), encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("LAUNCH_ARGUMENTS", source)
        self.assertIn("build_node_parameters", source)
        self.assertNotIn("environment_budget", source)


if __name__ == "__main__":
    unittest.main()
