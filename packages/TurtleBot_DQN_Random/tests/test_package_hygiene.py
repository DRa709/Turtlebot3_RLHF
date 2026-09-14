"""Package-level gates that must hold for every archive: no notebooks, LF line
endings, every shell script parses, every documented file exists, the shared
layer list covers the common modules, no fixed-arm dependencies."""

import os
import re
import subprocess
import glob
import unittest

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
        for doc in ("README.md", "ARC_RUNBOOK.md", "DATA_CONTRACT.md", "RANDOM_INIT_SPEC.md", "VERIFICATION_SCOPE.md", "ALGORITHM.md", "AUDIT_CORRECTIONS.md"):
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
                       "scripts/run_sim_headless.sh", "scripts/wait_for_sim.sh", "scripts/select_checkpoint.py",
                       "arc/task_isolation.py", "arc/phase_query.py"):
            self.assertIn(shared, listed)
        _, config_name, executable = package_algorithm(ROOT)
        learner_module = executable[:-len("_agent")] if executable.endswith("_agent") else executable
        for algorithm_specific in (f"turtlebot3_drl_nav/{learner_module}.py", f"turtlebot3_drl_nav/{executable}_node.py", f"config/{config_name}"):
            self.assertNotIn(algorithm_specific, listed)

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

    def test_container_isolates_pinned_matplotlib_from_ubuntu_toolkits(self):
        with open(os.path.join(ROOT, "apptainer", "tb3_phase1_foxy.def"), encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("python3 -m venv /opt/tb3-python", text)
        self.assertIn("export PYTHONNOUSERSITE=1", text)
        self.assertIn("/opt/tb3-python/lib/python3.8/site-packages", text)
        self.assertIn("import mpl_toolkits.mplot3d", text)
        self.assertIn("mixed Matplotlib installation", text)
        self.assertNotIn("python3 -m pip install", text)

    def test_foxy_launch_paths_fail_closed_on_whitespace(self):
        with open(os.path.join(ROOT, "scripts", "run_sim_headless.sh"), encoding="utf-8") as stream:
            simulator = stream.read()
        with open(os.path.join(ROOT, "scripts", "preflight.sh"), encoding="utf-8") as stream:
            preflight = stream.read()
        self.assertIn('[[ "$SIM_RUNTIME_DIR" =~ [[:space:]] ]]', simulator)
        self.assertIn("must not contain whitespace", simulator)
        self.assertIn('[[ ! "$ROOT" =~ [[:space:]] && ! "$RUN_DIR" =~ [[:space:]] ]]', preflight)

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
