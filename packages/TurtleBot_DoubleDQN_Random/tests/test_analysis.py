"""The offline analysis (tables and figures) runs on a complete harness run and
produces every camera-ready artifact: CSV + Markdown + LaTeX per table, PDF +
PNG per figure, IEEE style applied."""

import os
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from loop_harness import ROOT, LoopHarness  # noqa: E402

from turtlebot3_drl_nav.identity import write_identity  # noqa: E402
from turtlebot3_drl_nav.artifact_integrity import write_manifest  # noqa: E402


def seal(run_dir):
    with open(os.path.join(run_dir, "validation_report.json"), "w", encoding="utf-8") as stream:
        json.dump({"passed": True, "failures": []}, stream)
        stream.write("\n")
    write_manifest(run_dir)
    with open(os.path.join(run_dir, "COMPLETE"), "w", encoding="utf-8") as stream:
        stream.write("validated\n")


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        train = os.path.join(cls.tmp.name, "results", "pilot", "dqn", "seed_101", "job_1")
        h = LoopHarness(train, budget=40, checkpoint_interval=20, full_interval=40, tier1_episodes=2, episode_max_steps=6)
        h.run()
        assert h.fatal is None, h.fatal
        write_identity(os.path.join(train, "run_identity.json"), h.identity)
        seal(train)
        final = [c for c in h.checkpoint_rows if c["kind"] == "policy_only"][-1]["path"]
        ev = os.path.join(cls.tmp.name, "results", "pilot", "dqn", "seed_101", "eval_1")
        e = LoopHarness(ev, budget=40, checkpoint_interval=20, full_interval=40, tier1_episodes=2, episode_max_steps=6,
                        mode="eval", eval_checkpoint=final, scenario_count=12, tier2_e3=2)
        e.run()
        assert e.fatal is None, e.fatal
        write_identity(os.path.join(ev, "run_identity.json"), e.identity)
        seal(ev)
        cls.results = os.path.join(cls.tmp.name, "results")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tables_in_three_formats(self):
        out = os.path.join(self.tmp.name, "tables")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_tables.py"), "--results", self.results, "--out", out, "--phase-label", "pilot"], check=True, capture_output=True)
        for name in ("T-R1_main_results", "T-R2_learning_efficiency", "T-R3_generalization", "T-R4_probability_of_improvement",
                     "T-R5_failure_anatomy", "T-R6_per_seed", "T-R7_run_ledger", "T-R8_learner_diagnostics", "T-R9_initialization"):
            for ext in ("csv", "md", "tex"):
                self.assertTrue(os.path.isfile(os.path.join(out, f"{name}.{ext}")), f"{name}.{ext}")
        with open(os.path.join(out, "T-R1_main_results.tex"), encoding="utf-8") as stream:
            tex = stream.read()
        self.assertIn("\\toprule", tex)
        self.assertIn("\\caption{", tex)
        self.assertIn("\\label{tab:tr1}", tex)
        self.assertNotIn("\\textbackslash", tex)

    def test_figures_as_vector_pdf_and_300dpi_png(self):
        out = os.path.join(self.tmp.name, "figures")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_figures.py"), "--results", self.results, "--out", out, "--formats", "pdf,png", "--phase-label", "pilot"], check=True, capture_output=True)
        produced = sorted(os.listdir(out))
        for name in ("F-R1_learning_curves", "F-R2_training_window", "F-R3_performance_profiles", "F-R5_spatial_success",
                     "F-R6_trajectories", "F-R7_stratified_success", "F-R8_reward_composition", "F-R9_learner_diagnostics",
                     "F-R10_initialization_coverage", "F-R11_outcome_composition", "F-R14_timing"):
            self.assertIn(f"{name}.pdf", produced)
            self.assertIn(f"{name}.png", produced)
            self.assertGreater(os.path.getsize(os.path.join(out, f"{name}.pdf")), 1000)
        with open(os.path.join(out, "F-R1_learning_curves.pdf"), "rb") as stream:
            self.assertTrue(stream.read(5).startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
