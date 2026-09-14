import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from loop_harness import ROOT, LoopHarness  # noqa: E402

from turtlebot3_drl_nav.identity import (  # noqa: E402
    IDENTITY_FIELDS, config_digest, release_digest, sha256_file, shared_layer_digest, write_identity,
)
from turtlebot3_drl_nav.validator import validate_run  # noqa: E402

PROTOCOL_OVERRIDE = {"tier2_e3_episodes": 2, "scenario_count": 12, "physical_subset_count": 3}
PHASE_OVERRIDE = {
    "environment_budget": 60, "checkpoint_interval_steps": 20, "full_checkpoint_interval_steps": 40,
    "tier2_checkpoint_steps": [60], "tier1_episodes_per_checkpoint": 2, "learning_seeds": [101],
}


def package_with_protocol(tmp: str, harness: LoopHarness) -> str:
    """A copy of the package whose evaluation protocol matches the tiny harness run."""
    root = os.path.join(tmp, "pkg")
    # SHARED_LAYER_FILES.txt includes shared tests. Preserve that authenticated
    # inventory so manifest regeneration tests the validator rather than
    # failing because the fixture removed declared files.
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache"))
    import yaml
    common_path = os.path.join(root, "config", "common_environment.yaml")
    with open(common_path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    document["evaluation_protocol"].update(PROTOCOL_OVERRIDE)
    document["evaluation_protocol"]["phases"]["pilot"].update(PHASE_OVERRIDE)
    with open(common_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    from turtlebot3_drl_nav.initialization import write_scenarios
    write_scenarios(os.path.join(root, "config", "evaluation_scenarios_v1.csv"), harness.scenarios)
    from turtlebot3_drl_nav.launch_config import package_algorithm
    algorithm, config_name, executable = package_algorithm(root)
    algo_path = os.path.join(root, "config", config_name)
    with open(algo_path, encoding="utf-8") as stream:
        algo = yaml.safe_load(stream)
    algo[executable]["ros__parameters"].update({
        "warmup_steps": 8, "target_update_steps": 5, "batch_size": 4,
        "hidden_size": 8, "atoms": 5, "v_min": -2.0, "v_max": 2.0,
    })
    with open(algo_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(algo, stream, sort_keys=False)
    import subprocess
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make_manifests.py")], check=True, capture_output=True)
    return root


def rewrite_csv(path: str, mutate) -> None:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header, body = rows[0], rows[1:]
    body = mutate(header, body)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(body)


def write_runtime_provenance(package_root: str, run_dir: str, identity, budget: int) -> None:
    manifest = dict(identity)
    manifest.update({
        "environment_budget": budget, "slurm_job_id": "1", "slurm_array_job_id": "1",
        "slurm_array_task_id": "0", "hostname": "test-node", "started_utc": "2026-09-03T00:00:00Z",
    })
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream)
    sim_dir = os.path.join(run_dir, "sim")
    os.makedirs(sim_dir, exist_ok=True)
    world = os.path.join(sim_dir, "phase1_mixed.world")
    robot = os.path.join(sim_dir, "turtlebot3_burger_with_contact.sdf")
    shutil.copyfile(os.path.join(package_root, "worlds", "phase1_mixed.world"), world)
    with open(robot, "w", encoding="utf-8") as stream:
        stream.write("<sdf version='1.6'><model name='burger'/></sdf>\n")
    for payload, sidecar in ((world, "runtime_world.sha256"), (robot, "runtime_robot_model.sha256")):
        with open(os.path.join(sim_dir, sidecar), "w", encoding="utf-8") as stream:
            stream.write(f"{sha256_file(payload)}  {payload}\n")
    with open(os.path.join(run_dir, "runtime_manifest.txt"), "w", encoding="utf-8") as stream:
        stream.write("Python 3.8.10\nnumpy=1.24.4\ntorch=2.4.1+cpu\nros_distro=foxy\nGazebo multi-robot simulator, version 11\nros_domain_id=1 gazebo_master_uri=http://127.0.0.1:11345\n")


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = os.path.join(cls.tmp.name, "run")
        cls.h = LoopHarness(cls.run_dir, budget=60, checkpoint_interval=20, full_interval=40, tier1_episodes=2, episode_max_steps=7)
        cls.h.run()
        assert cls.h.fatal is None, cls.h.fatal
        cls.pkg = package_with_protocol(cls.tmp.name, cls.h)
        cls.h.identity.update({
            "config_sha256": config_digest(cls.pkg),
            "shared_layer_sha256": shared_layer_digest(cls.pkg),
            "release_sha256": release_digest(cls.pkg),
        })
        for name in ("transitions", "episodes", "updates", "evaluation", "checkpoints"):
            path = os.path.join(cls.run_dir, f"{name}.csv")
            def set_identity(header, body):
                for row in body:
                    for field in IDENTITY_FIELDS:
                        row[header.index(field)] = str(cls.h.identity[field])
                return body
            rewrite_csv(path, set_identity)
        write_identity(os.path.join(cls.run_dir, "run_identity.json"), cls.h.identity)
        write_runtime_provenance(cls.pkg, cls.run_dir, cls.h.identity, 60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh_copy(self) -> str:
        target = tempfile.mkdtemp(dir=self.tmp.name)
        shutil.rmtree(target)
        shutil.copytree(self.run_dir, target)
        rewrite_csv(
            os.path.join(target, "checkpoints.csv"),
            lambda header, body: [
                [target + row[header.index("path")][len(self.run_dir):] if i == header.index("path") else value for i, value in enumerate(row)]
                for row in body
            ],
        )
        return target

    def test_complete_run_passes(self):
        with open(os.path.join(self.run_dir, "transitions.csv"), newline="", encoding="utf-8") as stream:
            transitions = list(csv.DictReader(stream))
        episode_keys = {(row["phase"], row["episode_key"]) for row in transitions}
        step_zero_keys = [
            (row["phase"], row["episode_key"])
            for row in transitions
            if row["step_in_episode"] == "0"
        ]
        self.assertEqual(set(step_zero_keys), episode_keys)
        self.assertEqual(len(step_zero_keys), len(episode_keys))
        report = validate_run(self.pkg, self.run_dir)
        self.assertEqual(report.failures, [])
        self.assertGreater(len(report.checks), 30)

    def test_missing_transition_row_fails_budget_and_structure(self):
        run = self.fresh_copy()
        rewrite_csv(os.path.join(run, "transitions.csv"), lambda h, b: [r for r in b if not (r[h.index("phase")] == "training" and r[h.index("env_step")] == "30")])
        report = validate_run(self.pkg, run)
        self.assertTrue(any("budget" in f for f in report.failures))

    def test_tampered_reward_component_fails_identity(self):
        run = self.fresh_copy()

        def mutate(h, b):
            i = h.index("r_step")
            for r in b:
                if r[h.index("step_in_episode")] == "3":
                    r[i] = "-0.5"
            return b
        rewrite_csv(os.path.join(run, "transitions.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("six reward components" in f for f in report.failures))

    def test_impossible_mask_fails(self):
        run = self.fresh_copy()

        def mutate(h, b):
            for r in b:
                if r[h.index("step_in_episode")] == "2":
                    r[h.index("truncated")] = "1"
                    break
            return b
        rewrite_csv(os.path.join(run, "transitions.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("mask identities" in f for f in report.failures))

    def test_forged_start_pose_fails_reproducibility(self):
        run = self.fresh_copy()

        def mutate(h, b):
            b[0][h.index("requested_x")] = str(float(b[0][h.index("requested_x")]) + 0.01)
            return b
        rewrite_csv(os.path.join(run, "episodes.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("every training start" in f for f in report.failures))

    def test_forged_realized_pose_outside_support_fails(self):
        run = self.fresh_copy()

        def mutate(header, body):
            row = body[0]
            row[header.index("realized_x")] = "0.0"
            row[header.index("realized_y")] = "0.0"
            row[header.index("odom_x")] = "0.0"
            row[header.index("odom_y")] = "0.0"
            row[header.index("init_pos_error")] = str(math.hypot(
                float(row[header.index("requested_x")]), float(row[header.index("requested_y")])
            ))
            row[header.index("init_odom_error")] = "0.0"
            return body

        import math
        rewrite_csv(os.path.join(run, "episodes.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("supp(nu_R)" in f or "initialization" in f for f in report.failures))

    def test_nan_observation_fails_finite_gate(self):
        run = self.fresh_copy()

        def mutate(header, body):
            body[1][header.index("obs_0")] = "nan"
            return body

        rewrite_csv(os.path.join(run, "transitions.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("41 finite" in f for f in report.failures))

    def test_state_time_linkage_and_obstacle_trajectory_tampering_fail(self):
        run = self.fresh_copy()

        def mutate(header, body):
            for row in body:
                if row[header.index("step_in_episode")] == "1":
                    row[header.index("state_sim_time")] = str(float(row[header.index("state_sim_time")]) + 0.02)
                    row[header.index("obs1_x")] = str(float(row[header.index("obs1_x")]) + 0.20)
                    break
            return body

        rewrite_csv(os.path.join(run, "transitions.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("state_sim_time" in f for f in report.failures))
        self.assertTrue(any("obstacle poses" in f for f in report.failures))

    def test_missing_checkpoint_file_fails(self):
        run = self.fresh_copy()
        def mutate(header, body):
            for row in body:
                if row[header.index("kind")] == "full":
                    row[header.index("path")] += ".missing"
                    break
            return body
        rewrite_csv(os.path.join(run, "checkpoints.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("every checkpoint" in f for f in report.failures))

    def test_wrong_identity_on_a_row_fails(self):
        run = self.fresh_copy()

        def mutate(h, b):
            b[5][h.index("learning_seed")] = "999"
            return b
        rewrite_csv(os.path.join(run, "updates.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("identity field" in f for f in report.failures))

    def test_evaluation_contamination_flag_fails(self):
        run = self.fresh_copy()

        def mutate(h, b):
            b[0][h.index("learner_state_unchanged")] = "0"
            return b
        rewrite_csv(os.path.join(run, "evaluation.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("learner untouched" in f for f in report.failures))

    def test_forged_episode_and_evaluation_outcomes_fail_raw_recomputation(self):
        run = self.fresh_copy()

        def mutate_outcome(header, body):
            body[0][header.index("outcome")] = "goal" if body[0][header.index("outcome")] != "goal" else "timeout"
            return body

        rewrite_csv(os.path.join(run, "episodes.csv"), mutate_outcome)
        rewrite_csv(os.path.join(run, "evaluation.csv"), mutate_outcome)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("training episode outcome" in f for f in report.failures))
        self.assertTrue(any("E1 outcome" in f for f in report.failures))

    def test_fatal_marker_fails(self):
        run = self.fresh_copy()
        with open(os.path.join(run, "ENV_FATAL"), "w") as stream:
            stream.write("x")
        report = validate_run(self.pkg, run)
        self.assertTrue(any("ENV_FATAL" in f for f in report.failures))

    def test_runtime_world_tampering_fails_provenance(self):
        run = self.fresh_copy()
        with open(os.path.join(run, "sim", "phase1_mixed.world"), "a", encoding="utf-8") as stream:
            stream.write("<!-- tampered -->\n")
        report = validate_run(self.pkg, run)
        self.assertTrue(any("runtime world" in f for f in report.failures))

    def test_missing_slurm_identity_fails_provenance(self):
        run = self.fresh_copy()
        path = os.path.join(run, "run_manifest.json")
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["slurm_job_id"] = ""
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("numeric Slurm" in f for f in report.failures))

    def test_applicability_violation_fails(self):
        run = self.fresh_copy()

        def mutate(h, b):
            b[2][h.index("distributional_loss_mean")] = ""
            return b
        rewrite_csv(os.path.join(run, "updates.csv"), mutate)
        report = validate_run(self.pkg, run)
        self.assertTrue(any("applicability" in f for f in report.failures))


if __name__ == "__main__":
    unittest.main()
