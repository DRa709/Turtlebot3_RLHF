import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from loop_harness import LoopHarness  # noqa: E402

from turtlebot3_drl_nav.recorder import read_stream  # noqa: E402


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = os.path.join(cls.tmp.name, "run")
        cls.h = LoopHarness(cls.run_dir, budget=60, checkpoint_interval=20, full_interval=40, tier1_episodes=2, episode_max_steps=7)
        cls.h.run()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_finishes_at_the_exact_budget(self):
        self.assertIsNone(self.h.fatal, self.h.fatal)
        self.assertEqual(self.h.finish, "budget_reached")
        self.assertEqual(self.h.learner.env_steps, 60)
        rows = read_stream(os.path.join(self.run_dir, "transitions.csv"))
        training = [r for r in rows if r["phase"] == "training" and r["step_in_episode"] != "0"]
        self.assertEqual(len(training), 60)
        self.assertEqual([int(r["env_step"]) for r in training], list(range(1, 61)))
        # the budget transition is a truncation that ends its episode
        self.assertEqual(training[-1]["truncated"], "1")
        self.assertEqual(training[-1]["episode_end"], "1")
        self.assertEqual(training[-1]["terminated"], "0")

    def test_episode_rows_match_transition_structure(self):
        episodes = read_stream(os.path.join(self.run_dir, "episodes.csv"))
        rows = read_stream(os.path.join(self.run_dir, "transitions.csv"))
        training = [r for r in rows if r["phase"] == "training" and r["step_in_episode"] != "0"]
        ends = [r for r in training if r["episode_end"] == "1"]
        self.assertEqual(len(episodes), len(ends))
        self.assertEqual(sum(int(e["length"]) for e in episodes), 60)
        self.assertEqual([int(e["training_episode"]) for e in episodes], list(range(1, len(episodes) + 1)))
        for e in episodes:
            self.assertEqual(e["init_kind"], "train_nu_R")
            self.assertEqual(e["reset_status"], "ok")
            self.assertEqual(e["init_tolerance_ok"], "1")
            self.assertNotEqual(e["requested_x"], "")

    def test_updates_start_after_warmup_and_log_the_epsilon_used(self):
        updates = read_stream(os.path.join(self.run_dir, "updates.csv"))
        self.assertEqual(int(updates[0]["env_step"]), 8)  # warm-up of 8 transitions
        self.assertEqual(len(updates), 60 - 8 + 1)
        self.assertEqual([int(u["gradient_step"]) for u in updates], list(range(1, len(updates) + 1)))
        synced = [int(u["gradient_step"]) for u in updates if u["target_synced"] == "1"]
        self.assertEqual(synced, [g for g in range(5, len(updates) + 1, 5)])
        for u in updates:
            self.assertNotEqual(u["epsilon"], "")
            self.assertNotEqual(u["state_value_mean"], "")
            self.assertNotEqual(u["centered_advantage_abs_mean"], "")
        # epsilon logged is the value used for the action taken at that env_step (schedule from step 0)
        first = updates[0]
        expected = 1.0 + min(1.0, (int(first["env_step"]) - 1) / 40) * (0.05 - 1.0)
        self.assertAlmostEqual(float(first["epsilon"]), expected, places=9)

    def test_checkpoints_every_interval_with_full_at_multiples_and_final(self):
        checkpoints = read_stream(os.path.join(self.run_dir, "checkpoints.csv"))
        kinds = [(int(c["env_step"]), c["kind"]) for c in checkpoints]
        self.assertEqual(kinds, [(20, "policy_only"), (40, "policy_only"), (40, "full"), (60, "policy_only"), (60, "full")])
        for c in checkpoints:
            self.assertTrue(os.path.isfile(c["path"]))
            self.assertEqual(c["validated"], "1")
            self.assertEqual(len(c["sha256"]), 64)
            self.assertFalse(os.path.exists(c["path"] + ".tmp"))

    def test_tier1_evaluation_of_every_checkpoint_without_learner_contamination(self):
        evaluation = read_stream(os.path.join(self.run_dir, "evaluation.csv"))
        self.assertEqual(len(evaluation), 3 * 2)
        self.assertEqual(sorted({int(e["checkpoint_step"]) for e in evaluation}), [20, 40, 60])
        for e in evaluation:
            self.assertEqual(e["condition"], "E1")
            self.assertEqual(e["policy_mode"], "greedy")
            self.assertEqual(e["learner_state_unchanged"], "1")
            self.assertEqual(e["init_kind"], "E1_nu_R")
            self.assertIn(e["outcome"], {"goal", "timeout", "safety", "collision_static", "collision_dynamic", "collision_both"})
        # evaluation transitions carry no env_step and are excluded from the budget
        rows = read_stream(os.path.join(self.run_dir, "transitions.csv"))
        eval_rows = [r for r in rows if r["phase"] == "evaluation"]
        self.assertTrue(eval_rows)
        self.assertTrue(all(r["env_step"] == "" for r in eval_rows))
        self.assertTrue(all(r["checkpoint_step"] != "" for r in eval_rows))

    def test_identity_block_on_every_row_of_every_stream(self):
        for name in ("transitions", "episodes", "updates", "evaluation", "checkpoints"):
            with open(os.path.join(self.run_dir, f"{name}.csv"), newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                self.assertEqual(header[:2], ["experiment", "run_id"])
                for row in reader:
                    self.assertEqual(row[1], self.h.identity["run_id"])
                    self.assertEqual(len(row), len(header))


class Tier2Tests(unittest.TestCase):
    def test_post_hoc_evaluation_covers_all_scenarios_and_e3(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = os.path.join(tmp, "train")
            h = LoopHarness(train_dir, budget=20, checkpoint_interval=20, full_interval=20, tier1_episodes=1, episode_max_steps=5)
            h.run()
            self.assertIsNone(h.fatal, h.fatal)
            final = [c for c in h.checkpoint_rows if c["kind"] == "policy_only"][-1]["path"]
            eval_dir = os.path.join(tmp, "eval")
            e = LoopHarness(eval_dir, budget=20, checkpoint_interval=20, full_interval=20, tier1_episodes=1,
                            episode_max_steps=5, mode="eval", eval_checkpoint=final, scenario_count=12, tier2_e3=2)
            e.run()
            self.assertIsNone(e.fatal, e.fatal)
            self.assertEqual(e.finish, "evaluation_complete")
            rows = read_stream(os.path.join(eval_dir, "evaluation.csv"))
            self.assertEqual(sum(r["condition"] == "E2" for r in rows), 12)
            self.assertEqual(sum(r["condition"] == "E3" for r in rows), 2)
            self.assertEqual(sorted(r["scenario_id"] for r in rows if r["condition"] == "E2"), sorted(s.scenario_id for s in e.scenarios))
            e2 = [r for r in rows if r["condition"] == "E2"][0]
            self.assertIn(e2["difficulty"], {"open", "obstructed"})
            self.assertEqual(e2["init_kind"], "E2_scenario")
            e3 = [r for r in rows if r["condition"] == "E3"][0]
            self.assertEqual(float(e3["requested_x"]), 0.0)
            self.assertEqual(e.learner.env_steps, 0)
            self.assertFalse(os.path.exists(os.path.join(eval_dir, "updates.csv")) and read_stream(os.path.join(eval_dir, "updates.csv")))


if __name__ == "__main__":
    unittest.main()
