"""Package-level gates that must hold for every archive: no notebooks, LF line
endings, every shell script parses, every documented file exists, the shared
layer list covers the common modules, no fixed-arm dependencies."""

import os
import re
import shlex
import shutil
import subprocess
import glob
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import yaml

from turtlebot3_drl_nav.launch_config import package_algorithm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_SUFFIXES = (".py", ".sh", ".sbatch", ".yaml", ".md", ".world", ".def", ".xml", ".cfg", ".txt", ".csv")


def walk(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".pytest_cache"}]
        for name in files:
            yield os.path.join(base, name)


class PackageHygieneTests(unittest.TestCase):
    def test_no_symbolic_links_in_release(self):
        self.assertEqual([path for path in walk(ROOT) if os.path.islink(path)], [])

    def test_archive_contains_one_algorithm_implementation_only(self):
        algorithm, config_name, executable = package_algorithm(ROOT)
        with open(os.path.join(ROOT, "ALGORITHM"), encoding="utf-8") as stream:
            self.assertEqual(stream.read().strip(), algorithm)
        configs = [os.path.basename(path) for path in glob.glob(os.path.join(ROOT, "config", "phase1_*.yaml"))]
        agents = [os.path.basename(path) for path in glob.glob(os.path.join(ROOT, "turtlebot3_drl_nav", "*_agent_node.py"))]
        self.assertEqual(configs, [config_name])
        self.assertEqual(agents, [f"{executable}_node.py"])
        learner_module = executable[:-len("_agent")]
        implementations = [
            name for name in ("dqn.py", "doubledqn.py", "duelingdoubledqn.py", "rainbowdqn.py", "discretesac.py", "sdsac.py")
            if os.path.isfile(os.path.join(ROOT, "turtlebot3_drl_nav", name))
        ]
        self.assertEqual(implementations, [f"{learner_module}.py"])

    def test_package_and_algorithm_versions_are_consistent(self):
        algorithm, config_name, executable = package_algorithm(ROOT)
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as stream:
            version = stream.read().strip()
        self.assertRegex(version, r"^[0-9]+\.[0-9]+\.[0-9]+$")
        package_version = ET.parse(os.path.join(ROOT, "package.xml")).getroot().findtext("version")
        self.assertEqual(package_version, version)
        with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as stream:
            setup_source = stream.read()
        self.assertIn(f'version="{version}"', setup_source)
        self.assertIn(f'"{executable} = turtlebot3_drl_nav.{executable}_node:main"', setup_source)
        with open(os.path.join(ROOT, "config", config_name), encoding="utf-8") as stream:
            config = yaml.safe_load(stream)[executable]["ros__parameters"]
        self.assertEqual(config["algorithm"], algorithm)
        self.assertEqual(config["algorithm_version"], version)
        self.assertEqual(config["action_space"], "discrete")

    def test_no_notebooks_anywhere(self):
        self.assertEqual([p for p in walk(ROOT) if p.endswith(".ipynb")], [])

    def test_lf_line_endings_everywhere(self):
        crlf = []
        for path in walk(ROOT):
            if path.endswith(TEXT_SUFFIXES):
                with open(path, "rb") as stream:
                    if b"\r" in stream.read():
                        crlf.append(os.path.relpath(path, ROOT))
        self.assertEqual(crlf, [])

    def test_every_shell_and_slurm_script_parses(self):
        for path in walk(ROOT):
            if path.endswith((".sh", ".sbatch")):
                result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_documented_files_exist(self):
        missing = []
        for doc in ("README.md", "ARC_RUNBOOK.md", "DATA_CONTRACT.md", "RANDOM_INIT_SPEC.md", "VERIFICATION_SCOPE.md", "ALGORITHM.md", "AUDIT_CORRECTIONS.md", "PROVENANCE.md"):
            with open(os.path.join(ROOT, doc), encoding="utf-8") as stream:
                text = stream.read()
            for rel in set(re.findall(r"`((?:arc|scripts|config|worlds|launch|apptainer|turtlebot3_drl_nav|tests)/[A-Za-z0-9_./-]+)`", text)):
                if not os.path.exists(os.path.join(ROOT, rel)):
                    missing.append(f"{doc}: {rel}")
        self.assertEqual(missing, [])

    def test_shared_layer_list_covers_the_common_modules(self):
        with open(os.path.join(ROOT, "SHARED_LAYER_FILES.txt"), encoding="utf-8") as stream:
            listed = {line.strip() for line in stream if line.strip() and not line.startswith("#")}
        for module in ("state", "protocol", "collision", "geometry", "initialization", "episode_engine", "recorder", "validator",
                       "identity", "artifact_integrity", "orchestrator", "env_config", "obstacle_schedule", "launch_config", "analysis", "learner_api", "ros_qos",
                       "drl_environment_node", "dynamic_obstacle_node"):
            self.assertIn(f"turtlebot3_drl_nav/{module}.py", listed)
        for shared in ("config/common_environment.yaml", "config/evaluation_scenarios_v1.csv", "worlds/phase1_mixed.world",
                       "scripts/run_sim_headless.sh", "scripts/wait_for_sim.sh", "scripts/wait_for_topics.py", "scripts/select_checkpoint.py",
                       "arc/task_isolation.py", "arc/phase_query.py"):
            self.assertIn(shared, listed)
        _, config_name, executable = package_algorithm(ROOT)
        learner_module = executable[:-len("_agent")] if executable.endswith("_agent") else executable
        for algorithm_specific in (f"turtlebot3_drl_nav/{learner_module}.py", f"turtlebot3_drl_nav/{executable}_node.py", f"config/{config_name}"):
            self.assertNotIn(algorithm_specific, listed)

    def test_analysis_entry_points_are_sdsac_only(self):
        for relative in ("scripts/make_tables.py", "scripts/make_figures.py"):
            with open(os.path.join(ROOT, relative), encoding="utf-8") as stream:
                source = stream.read()
            self.assertIn('ALGORITHM = "SDSAC"', source, relative)
            self.assertIn("standalone analysis refuses non-", source, relative)
            self.assertNotIn("--expected-algorithms", source, relative)
        with open(os.path.join(ROOT, "tests", "test_analysis.py"), encoding="utf-8") as stream:
            fixture = stream.read()
        self.assertIn("shutil.copytree(", fixture)
        self.assertIn('"tier2_checkpoint_steps": [40]', fixture)
        self.assertIn('"learning_seeds": [101]', fixture)
        self.assertIn('"training_run_id": h.identity["run_id"]', fixture)
        self.assertIn('os.path.join(self.package_root, "scripts", "make_tables.py")', fixture)
        self.assertIn('os.path.join(self.package_root, "scripts", "make_figures.py")', fixture)

    def test_ros_distribution_is_consistent(self):
        texts = {}
        relatives = ["apptainer/tb3_phase1_foxy.def", "package.xml", "scripts/preflight.sh"]
        relatives.extend(os.path.relpath(path, ROOT) for path in glob.glob(os.path.join(ROOT, "arc", "run_*.sh")))
        for rel in relatives:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as stream:
                texts[rel] = stream.read()
        for rel, text in texts.items():
            self.assertIn("foxy", text.lower(), rel)
            self.assertNotIn("humble", text.lower(), rel)

    def test_container_recipe_has_no_unauthenticated_or_fail_open_apt(self):
        with open(os.path.join(ROOT, "apptainer", "tb3_phase1_foxy.def"), encoding="utf-8") as stream:
            text = stream.read()
        self.assertNotIn("AllowInsecureRepositories", text)
        self.assertNotRegex(text, r"apt-get[^\n]*\|\|\s*true")
        self.assertIn("signed-by=/usr/share/keyrings/ros-archive-keyring.gpg", text)

    def test_container_isolates_pinned_scientific_stack(self):
        with open(os.path.join(ROOT, "apptainer", "tb3_phase1_foxy.def"), encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("python3 -m venv /opt/tb3-python", text)
        self.assertIn("export PYTHONNOUSERSITE=1", text)
        self.assertIn("/opt/tb3-python/lib/python3.8/site-packages", text)
        self.assertIn("import matplotlib.pyplot", text)
        self.assertIn("import mpl_toolkits.mplot3d", text)
        self.assertIn("import pytest", text)
        self.assertIn("for module in (torch, numpy, yaml, pandas, matplotlib, matplotlib.pyplot, mpl_toolkits.mplot3d, pytest)", text)
        self.assertIn("mixed scientific Python installation", text)
        self.assertNotIn("python3 -m pip install", text)

    def test_foxy_launch_paths_fail_closed_on_whitespace(self):
        with open(os.path.join(ROOT, "scripts", "run_sim_headless.sh"), encoding="utf-8") as stream:
            simulator = stream.read()
        with open(os.path.join(ROOT, "scripts", "preflight.sh"), encoding="utf-8") as stream:
            preflight = stream.read()
        self.assertIn('[[ "$SIM_RUNTIME_DIR" =~ [[:space:]] ]]', simulator)
        self.assertIn("must not contain whitespace", simulator)
        self.assertIn('[[ ! "$ROOT" =~ [[:space:]] && ! "$RUN_DIR" =~ [[:space:]] ]]', preflight)

    def test_valid_preflight_preserves_cache_free_release_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            package_root = os.path.join(temporary, "TurtleBot_SDSAC_Random")
            shutil.copytree(
                ROOT,
                package_root,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".git"),
            )
            fake_bin = os.path.join(temporary, "bin")
            run_dir = os.path.join(temporary, "run")
            os.makedirs(fake_bin)
            os.makedirs(run_dir)

            def executable(name, body):
                path = os.path.join(fake_bin, name)
                with open(path, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(body)
                os.chmod(path, 0o755)

            real_python = shlex.quote(sys.executable)
            executable(
                "python3",
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'if [[ "$*" == *"import torch, numpy, yaml, rclpy, gazebo_msgs.srv"* ]]; then exit 0; fi\n'
                f'exec {real_python} "$@"\n',
            )
            executable("ros2", "#!/usr/bin/env bash\nexit 0\n")
            executable("gzserver", "#!/usr/bin/env bash\nexit 0\n")

            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.update(
                {
                    "PATH": fake_bin + os.pathsep + environment.get("PATH", ""),
                    "PYTHONPATH": package_root + os.pathsep + environment.get("PYTHONPATH", ""),
                    "ROS_DISTRO": "foxy",
                    "WORLD_SEED": "7000",
                    "INITIALIZATION_SEED": "7001",
                    "DYNAMIC_OBSTACLE_SEED": "7002",
                    "EVALUATION_SEED": "9001",
                    "LEARNING_SEED": "101",
                    "PHASE_LABEL": "calibration",
                    "CONTAINER_SHA256": "c" * 64,
                }
            )
            preflight = subprocess.run(
                ["bash", os.path.join(package_root, "scripts", "preflight.sh"), package_root, run_dir],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)

            cache_artifacts = []
            for base, directories, files in os.walk(package_root):
                cache_artifacts.extend(os.path.join(base, name) for name in directories if name == "__pycache__")
                cache_artifacts.extend(
                    os.path.join(base, name) for name in files if name.endswith((".pyc", ".pyo"))
                )
            self.assertEqual(cache_artifacts, [])

            verification = subprocess.run(
                ["bash", os.path.join(package_root, "scripts", "verify_package.sh")],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verification.returncode, 0, verification.stdout + verification.stderr)

    def test_python_files_compile_and_are_py38_compatible(self):
        forbidden = re.compile("|".join([r"\bmatch\s+\w+\s*:", "remove" + "prefix", "remove" + "suffix", r"\blist\[", r"\bdict\[", r"\btuple\[", r"\bstr \| None"]))
        for path in walk(ROOT):
            if path.endswith(".py") and not path.endswith("test_package_hygiene.py"):
                with open(path, encoding="utf-8") as stream:
                    source = stream.read()
                compile(source, path, "exec")
                self.assertIsNone(forbidden.search(source), f"{path} uses syntax newer than Python 3.8")


if __name__ == "__main__":
    unittest.main()
