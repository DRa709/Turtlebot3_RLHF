import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(relative):
    with open(os.path.join(ROOT, relative), encoding="utf-8") as stream:
        return stream.read()


class ArcScriptContractTests(unittest.TestCase):
    def test_double_dqn_identity_and_entry_points_are_not_dqn_aliases(self):
        train_array = source("arc/doubledqn_array.sbatch")
        eval_array = source("arc/doubledqn_eval_array.sbatch")
        train_wrapper = source("arc/run_doubledqn_seed.sh")
        preflight = source("scripts/preflight.sh")
        container = source("apptainer/tb3_phase1_foxy.def")
        runbook = source("ARC_RUNBOOK.md")
        self.assertIn('algorithm="DoubleDQN"', train_wrapper)
        self.assertNotIn('algorithm="DQN"', train_wrapper)
        self.assertIn("/doubledqn/seed_", train_array)
        self.assertIn("/doubledqn/seed_", eval_array)
        self.assertIn("run_doubledqn_seed.sh", train_array)
        self.assertIn("run_doubledqn_eval.sh", eval_array)
        self.assertIn("TurtleBot_DoubleDQN_Random", container)
        self.assertNotIn("TurtleBot_DQN_Random", container)
        self.assertNotIn("phase1_dqn.yaml", preflight)
        self.assertIn("--expected-algorithms DoubleDQN", runbook)
        self.assertNotIn("--expected-algorithms DQN,", runbook)
        self.assertIn("must not contain whitespace under ROS 2 Foxy", source("arc/run_doubledqn_eval.sh"))

    def test_arrays_are_exclusive_cleanenv_and_forward_slurm_identity(self):
        for relative in ("arc/doubledqn_array.sbatch", "arc/doubledqn_eval_array.sbatch"):
            text = source(relative)
            self.assertIn("#SBATCH --exclusive", text)
            self.assertIn("exec --cleanenv --containall", text)
            self.assertIn('CONTAINER_SHA256="$(sha256sum "$IMAGE"', text)
            self.assertIn("APPTAINERENV_SLURM_JOB_ID", text)
            self.assertNotIn("% 100", text)

    def test_wrappers_seal_outputs_before_complete(self):
        for relative in ("arc/run_doubledqn_seed.sh", "arc/run_doubledqn_eval.sh"):
            text = source(relative)
            self.assertIn("turtlebot3_drl_nav.artifact_integrity write", text)
            self.assertIn("turtlebot3_drl_nav.artifact_integrity verify", text)
            self.assertLess(text.index("artifact_integrity verify"), text.index('STATUS="COMPLETE"'))
            self.assertIn("validation failed", text)
        self.assertIn("scripts/select_checkpoint.py", source("arc/run_doubledqn_eval.sh"))

    def test_container_test_gate_cannot_hide_skips_in_dev_null(self):
        text = source("scripts/run_tests.sh")
        self.assertIn("mktemp", text)
        self.assertIn("export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", text)
        self.assertLess(text.index("export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"), text.index("python3 -m pytest"))
        self.assertIn("skipped tests are not accepted", text)
        self.assertNotIn('tee "${TEST_LOG:-/dev/null}"', text)

    def test_foxy_readiness_uses_a_bounded_subscriber_and_advancing_clock(self):
        shell = source("scripts/wait_for_sim.sh")
        helper = source("scripts/wait_for_topics.py")
        self.assertNotIn("ros2 topic echo", shell)
        self.assertNotIn("--once", shell)
        self.assertIn('python3 "$SCRIPT_DIR/wait_for_topics.py"', shell)
        for topic in ("/clock", "/scan", "/odom", "/bumper_states", "/drl/obstacle_status"):
            self.assertIn(f'"{topic}"', helper)
        self.assertIn("last_clock_ns > first_clock_ns", helper)
        self.assertIn("time.monotonic()", helper)


if __name__ == "__main__":
    unittest.main()
