import json
import os
import tempfile
import unittest

from turtlebot3_drl_nav.analysis import Run, iqm, select_final_evaluation_runs, standalone_completeness


def evaluation_run(root, name, step, seed=101):
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "evaluation.csv"), "w", encoding="utf-8") as stream:
        stream.write("checkpoint_step\n")
        stream.write(f"{step}\n")
    return Run(run_dir, {
        "algorithm": "RainbowDQN", "learning_seed": seed, "phase_label": "controlled", "phase_type": "evaluation",
    })


def standalone_run(root, name, phase_type, step=None, seed=101):
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir)
    identity = {
        "algorithm": "RainbowDQN", "learning_seed": seed, "phase_label": "controlled",
        "phase_type": phase_type, "run_id": f"RainbowDQN-{phase_type}-{seed}",
        "world_seed": 7000, "initialization_seed": 7001,
        "dynamic_obstacle_seed": 7002, "evaluation_seed": 9001,
        "config_sha256": "c" * 64, "container_sha256": "d" * 64,
        "shared_layer_sha256": "a" * 64, "release_sha256": "b" * 64,
    }
    if step is not None:
        with open(os.path.join(run_dir, "evaluation.csv"), "w", encoding="utf-8") as stream:
            stream.write("checkpoint_step\n")
            stream.write(f"{step}\n")
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump({"training_run_id": "RainbowDQN-training-101"}, stream)
    return Run(run_dir, identity)


class AnalysisCoreTests(unittest.TestCase):
    def test_iqm_uses_fractional_quartile_weights(self):
        self.assertAlmostEqual(iqm([0, 0, 0, 1, 1]), 0.3, places=12)
        self.assertAlmostEqual(iqm([1, 2, 3, 4]), 2.5, places=12)

    def test_final_checkpoint_selection_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            early = evaluation_run(tmp, "early", 25000)
            final = evaluation_run(tmp, "final", 500000)
            self.assertEqual(select_final_evaluation_runs([early, final]), [final])
            duplicate = evaluation_run(tmp, "duplicate", 500000)
            with self.assertRaisesRegex(ValueError, "duplicate final"):
                select_final_evaluation_runs([early, final, duplicate])

    def test_standalone_completeness_rejects_missing_and_extra_evidence(self):
        protocol = {
            "common_seeds": {"world_seed": 7000, "initialization_seed": 7001, "dynamic_obstacle_seed": 7002, "evaluation_seed": 9001},
            "phases": {"controlled": {"learning_seeds": [101], "tier2_checkpoint_steps": [100, 200]}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            runs = [
                standalone_run(tmp, "train", "training"),
                standalone_run(tmp, "eval100", "evaluation", 100),
                standalone_run(tmp, "eval200", "evaluation", 200),
            ]
            self.assertTrue(standalone_completeness(runs, protocol, "controlled")["passed"])
            report = standalone_completeness(runs[:-1], protocol, "controlled")
            self.assertFalse(report["passed"])
            self.assertTrue(any("step 200" in error for error in report["errors"]))

            unexpected = standalone_run(tmp, "eval300", "evaluation", 300)
            report = standalone_completeness(runs + [unexpected], protocol, "controlled")
            self.assertFalse(report["passed"])
            self.assertTrue(any("extra evaluation identities" in error and "300" in error for error in report["errors"]))

            two_seed_protocol = {
                "common_seeds": dict(protocol["common_seeds"]),
                "phases": {
                    "controlled": {
                        "learning_seeds": [101, 202],
                        "tier2_checkpoint_steps": [100, 200],
                    }
                },
            }
            report = standalone_completeness(runs, two_seed_protocol, "controlled")
            self.assertFalse(report["passed"])
            self.assertTrue(any("training run for RainbowDQN seed 202" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
