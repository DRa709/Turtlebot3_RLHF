import hashlib
import importlib.util
import marshal
import os
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(relative):
    with open(os.path.join(ROOT, relative), encoding="utf-8") as stream:
        return stream.read()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ArcScriptContractTests(unittest.TestCase):
    def test_v117_identity_is_consistent_across_algorithm_package_files(self):
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as stream:
            self.assertEqual(stream.read().strip(), "1.1.7")
        for relative in ("README.md", "ALGORITHM.md", "ARC_RUNBOOK.md", "AUDIT_CORRECTIONS.md", "CHANGELOG.md"):
            self.assertIn("1.1.7", source(relative), relative)
        self.assertIn('version="1.1.7"', source("setup.py"))
        self.assertIn("<version>1.1.7</version>", source("package.xml"))
        self.assertIn('algorithm_version: "1.1.7"', source("config/phase1_dqn.yaml"))
        with open(os.path.join(ROOT, "config", "phase1_dqn.yaml"), encoding="utf-8") as stream:
            dqn_config = yaml.safe_load(stream)
        self.assertEqual(dqn_config["dqn_agent"]["ros__parameters"]["algorithm_version"], "1.1.7")
        self.assertIn("dqn-v1.1.7", source("arc/submit_dqn.sh"))
        definition = source("apptainer/tb3_phase1_foxy.def")
        self.assertIn("TurtleBot_DQN_Random_Runtime_Foxy.sif", definition)
        self.assertIn("%files", definition)
        self.assertIn(". /opt/turtlebot_dqn_random", definition)
        self.assertIn("colcon build --merge-install", definition)
        self.assertIn("bash /opt/turtlebot_dqn_random/scripts/run_tests.sh", definition)
        self.assertNotIn("TurtleBot_DQN_Random_v1.1.2.sif", definition)
        self.assertNotIn("TurtleBot_DQN_Random_v1.1.3.sif", definition)

    def test_arrays_are_exclusive_cleanenv_and_forward_slurm_identity(self):
        for relative in ("arc/dqn_array.sbatch", "arc/dqn_eval_array.sbatch"):
            text = source(relative)
            self.assertIn("#SBATCH --exclusive", text)
            self.assertIn("exec --cleanenv --containall", text)
            self.assertIn('CONTAINER_SHA256="$(sha256sum "$IMAGE"', text)
            self.assertIn("APPTAINERENV_SLURM_JOB_ID", text)
            self.assertIn("clear_container_runtime_environment", text)
            self.assertIn("assert_no_container_runtime_environment", text)
            self.assertIn("assert_only_owned_container_environment", text)
            self.assertIn("verify_image_identity", text)
            self.assertIn("verify_embedded_release", text)
            self.assertIn("TB3_EXPECTED_IMAGE_SHA256", text)
            self.assertIn("TB3_EXPECTED_RELEASE_SHA256", text)
            self.assertIn("TB3_EXPECTED_BATCH_SCRIPT_SHA256", text)
            self.assertIn("APPTAINERENV_PYTHONDONTWRITEBYTECODE", text)
            self.assertIn('readonly EMBEDDED_REPO=/opt/turtlebot_dqn_random', text)
            self.assertIn('--bind "$OUTPUT_ROOT:$OUTPUT_ROOT"', text)
            self.assertNotIn('--bind "$REPO:$REPO"', text)
            self.assertNotIn("TB3_REPO", text)
            self.assertNotIn("% 100", text)

    def test_array_submission_is_guarded_and_has_no_relative_slurm_logs(self):
        for relative in ("arc/dqn_array.sbatch", "arc/dqn_eval_array.sbatch"):
            text = source(relative)
            self.assertIn("TB3_SUBMISSION_GUARD", text)
            self.assertIn("Use arc/submit_dqn.sh", text)
            self.assertNotIn("#SBATCH --output=", text)
            self.assertNotIn("#SBATCH --error=", text)
            self.assertIn('bash "$EMBEDDED_REPO/scripts/verify_package.sh"', text)
            self.assertIn('sha256sum "${BASH_SOURCE[0]}"', text)

    def test_batch_snapshot_digest_comes_from_the_authenticated_image(self):
        text = source("arc/submit_dqn.sh")
        self.assertIn(
            'awk -v path="$BATCH_RELATIVE" \'$2 == path { print $1 }\' "$EMBEDDED_REPO/RELEASE_MANIFEST.sha256"',
            text,
        )
        self.assertNotIn(
            'EXPECTED_BATCH_SHA256="$(awk -v path="$BATCH_RELATIVE"',
            text,
        )
        self.assertIn(
            '[[ "$FINAL_EXTERNAL_RELEASE_SHA256" == "$EMBEDDED_RELEASE_SHA256" ]]',
            text,
        )

    def test_runtime_wrappers_use_only_the_installed_in_image_release(self):
        for relative in ("arc/run_dqn_seed.sh", "arc/run_dqn_eval.sh"):
            text = source(relative)
            self.assertIn("readonly REPO=/opt/turtlebot_dqn_random", text)
            self.assertIn("readonly INSTALL_PREFIX=/opt/turtlebot_dqn_random_ws/install", text)
            self.assertNotIn("colcon build", text)
            self.assertNotIn("TB3_WS", text)
            self.assertIn('identity verify-release "$REPO"', text)

    def test_submission_wrapper_places_logs_outside_release_and_owns_protocol_options(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            output = os.path.join(tmp, "results")
            log_root = os.path.join(tmp, "slurm-logs")
            image = os.path.join(tmp, "runtime.sif")
            image_link = os.path.join(tmp, "runtime-link.sif")
            alternate_image = os.path.join(tmp, "alternate.sif")
            repo_link = os.path.join(tmp, "repo-link")
            alternate_repo = os.path.join(tmp, "alternate-repo")
            capture = os.path.join(tmp, "sbatch-arguments.txt")
            capture_env = os.path.join(tmp, "sbatch-environment.txt")
            os.makedirs(bin_dir)
            os.makedirs(alternate_repo)
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            with open(alternate_image, "wb") as stream:
                stream.write(b"different image bytes\n")
            os.chmod(image, 0o444)
            os.chmod(alternate_image, 0o444)
            os.symlink(image, image_link)
            os.symlink(ROOT, repo_link)

            apptainer = os.path.join(bin_dir, "apptainer")
            with open(apptainer, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("""#!/usr/bin/env bash
set -euo pipefail
while IFS= read -r name; do
  case "$name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
      echo "ambient container control reached wrapper authentication: $name" >&2
      exit 92
      ;;
  esac
done < <(compgen -e)
[[ "$1" == "exec" ]]
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanenv|--containall) shift ;;
    --bind) shift 2 ;;
    *) shift; break ;;
  esac
done
# Model the immutable SIF and the fixed read-only inspection mount.
mapped=()
for arg in "$@"; do
  arg="${arg//\\/opt\\/turtlebot_dqn_random/$FAKE_EMBEDDED_ROOT}"
  arg="${arg//\\/mnt\\/tb3-submission-source/$FAKE_EXTERNAL_ROOT}"
  mapped+=("$arg")
done
exec "${mapped[@]}"
""")
            sbatch = os.path.join(bin_dir, "sbatch")
            with open(sbatch, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(f"""#!/usr/bin/env bash
set -euo pipefail
while IFS='=' read -r name _; do
  case "$name" in
    SBATCH_*|SLURM_*|APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*|TB3_*|BASH_ENV|ENV|LD_PRELOAD)
      echo "unreviewed caller environment reached sbatch: $name" >&2
      exit 91
      ;;
  esac
done < <(env)
printf '%s\\n' "$@" > {shlex.quote(capture)}
env | LC_ALL=C sort > {shlex.quote(capture_env)}
# Model the queued interval: retarget the caller's original symlinks after the
# wrapper has authenticated them but before the allocation starts.
ln -sfn {shlex.quote(alternate_repo)} {shlex.quote(repo_link)}
ln -sfn {shlex.quote(alternate_image)} {shlex.quote(image_link)}
printf 'Submitted batch job 999\\n'
""")
            for path in (apptainer, sbatch):
                os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env.update({
                "PATH": bin_dir + os.pathsep + env["PATH"],
                "TB3_REPO": repo_link,
                "TB3_IMAGE": image_link,
                "TB3_OUTPUT": output,
                "TB3_SLURM_LOGS": log_root,
                "TB3_SUBMISSION_GUARD": "ambient-value",
                "APPTAINERENV_PYTHONPATH": "/tmp/untrusted-pythonpath",
                "APPTAINER_BIND": "/tmp:/unexpected-apptainer-bind",
                "SINGULARITYENV_LD_LIBRARY_PATH": "/tmp/untrusted-library-path",
                "SINGULARITY_BINDPATH": "/var/tmp:/unexpected-singularity-bind",
                "BASH_ENV": "",
                "LD_PRELOAD": "",
                "SBATCH_OUTPUT": os.path.join(ROOT, "environment-bypass.out"),
                "SBATCH_ARRAY_INX": "9",
                "SBATCH_WRAP": "alternate-command",
                "SLURM_CLUSTERS": "other-cluster",
                "SLURM_HINT": "multithread",
                "FAKE_EMBEDDED_ROOT": ROOT,
                "FAKE_EXTERNAL_ROOT": ROOT,
            })
            result = subprocess.run(
                ["bash", wrapper, "train", "calibration", "--account=test-account", "--partition=test-partition"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(capture, encoding="utf-8") as stream:
                arguments = [line.rstrip("\n") for line in stream]
            expected_log_dir = os.path.join(log_root, "train", "calibration")
            self.assertIn(f"--chdir={expected_log_dir}", arguments)
            self.assertIn(f"--output={expected_log_dir}/%x-%A_%a.out", arguments)
            self.assertIn(f"--error={expected_log_dir}/%x-%A_%a.err", arguments)
            export_argument = next(value for value in arguments if value.startswith("--export="))
            exported_items = export_argument[len("--export="):].split(",")
            self.assertNotIn("ALL", exported_items)
            exported = dict(item.split("=", 1) for item in exported_items)
            self.assertEqual(exported["PHASE_LABEL"], "calibration")
            self.assertEqual(exported["TB3_SUBMISSION_GUARD"], "dqn-v1.1.7")
            self.assertNotIn("TB3_REPO", exported)
            self.assertEqual(exported["TB3_IMAGE"], image)
            self.assertEqual(exported["TB3_OUTPUT"], output)
            self.assertEqual(exported["TB3_LEASES"], os.path.join(output, ".leases"))
            self.assertEqual(exported["TB3_EXPECTED_IMAGE_SHA256"], sha256_file(image))
            self.assertEqual(exported["TB3_EXPECTED_RELEASE_SHA256"], sha256_file(os.path.join(ROOT, "RELEASE_MANIFEST.sha256")))
            self.assertEqual(
                exported["TB3_EXPECTED_BATCH_SCRIPT_SHA256"],
                sha256_file(os.path.join(ROOT, "arc", "dqn_array.sbatch")),
            )
            self.assertIn("--array=0", arguments)
            self.assertLess(arguments.index("--account=test-account"), arguments.index(f"--chdir={expected_log_dir}"))
            self.assertLess(arguments.index("--partition=test-partition"), arguments.index(f"--chdir={expected_log_dir}"))
            self.assertEqual(arguments[-2], "--")
            submitted_script = arguments[-1]
            self.assertEqual(os.path.dirname(submitted_script), expected_log_dir)
            self.assertIn(".dqn-train-calibration.", os.path.basename(submitted_script))
            self.assertEqual(sha256_file(submitted_script), sha256_file(os.path.join(ROOT, "arc", "dqn_array.sbatch")))
            self.assertEqual(os.stat(submitted_script).st_mode & 0o222, 0)
            with open(capture_env, encoding="utf-8") as stream:
                submitted_environment = stream.read()
            for forbidden in ("TB3_", "SBATCH_", "SLURM_", "APPTAINER", "SINGULARITY", "BASH_ENV=", "LD_PRELOAD="):
                self.assertNotIn(forbidden, submitted_environment)

            # Retargeting the original symlinks cannot change the canonical
            # repo/image paths or the digests already placed in --export.
            self.assertEqual(os.path.realpath(repo_link), alternate_repo)
            self.assertEqual(os.path.realpath(image_link), alternate_image)
            os.unlink(repo_link)
            os.unlink(image_link)
            os.symlink(ROOT, repo_link)
            os.symlink(image, image_link)

            result = subprocess.run(
                ["bash", wrapper, "eval", "controlled", "12345", "200000", "--account=test-account"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(capture, encoding="utf-8") as stream:
                arguments = [line.rstrip("\n") for line in stream]
            expected_log_dir = os.path.join(log_root, "eval", "controlled")
            self.assertIn(f"--output={expected_log_dir}/%x-%A_%a.out", arguments)
            self.assertIn(f"--error={expected_log_dir}/%x-%A_%a.err", arguments)
            export_argument = next(value for value in arguments if value.startswith("--export="))
            exported_items = export_argument[len("--export="):].split(",")
            self.assertNotIn("ALL", exported_items)
            exported = dict(item.split("=", 1) for item in exported_items)
            self.assertEqual(exported["PHASE_LABEL"], "controlled")
            self.assertEqual(exported["TB3_SUBMISSION_GUARD"], "dqn-v1.1.7")
            self.assertEqual(exported["TRAIN_JOB"], "12345")
            self.assertEqual(exported["CHECKPOINT_STEP"], "200000")
            self.assertIn("--array=0-4", arguments)
            self.assertIn("--dependency=afterok:12345", arguments)
            self.assertIn("--kill-on-invalid-dep=yes", arguments)
            self.assertLess(arguments.index("--account=test-account"), arguments.index(f"--chdir={expected_log_dir}"))
            self.assertEqual(arguments[-2], "--")
            submitted_script = arguments[-1]
            self.assertEqual(os.path.dirname(submitted_script), expected_log_dir)
            self.assertIn(".dqn-eval-controlled.", os.path.basename(submitted_script))
            self.assertEqual(sha256_file(submitted_script), sha256_file(os.path.join(ROOT, "arc", "dqn_eval_array.sbatch")))
            self.assertEqual(os.stat(submitted_script).st_mode & 0o222, 0)

    def test_submission_wrapper_rejects_output_inside_release(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "runtime.sif")
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            env = os.environ.copy()
            env.update({"TB3_REPO": ROOT, "TB3_IMAGE": image, "TB3_OUTPUT": os.path.join(ROOT, "forbidden-results")})
            result = subprocess.run(["bash", wrapper, "train", "calibration"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TB3_OUTPUT must be outside TB3_REPO", result.stderr)
            self.assertFalse(os.path.exists(os.path.join(ROOT, "forbidden-results")))

    def test_submission_wrapper_rejects_every_owned_long_option_prefix_and_operand_bypass(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        attacks = []
        for option in ("--array", "--export", "--export-file", "--chdir", "--output", "--error", "--wrap", "--dependency", "--kill-on-invalid-dep"):
            attacks.extend(option[:length] + "=attacker" for length in range(3, len(option) + 1))
        attacks.extend(("--", ":", "alternate.sbatch", "-oattack", "-eattack", "-D/tmp", "-a9", "-Jother"))
        for attack in sorted(set(attacks)):
            with self.subTest(attack=attack):
                result = subprocess.run(
                    ["bash", wrapper, "train", "calibration", attack],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DQN SUBMISSION FAIL", result.stderr)

    def test_submission_wrapper_rejects_duplicate_or_detached_site_options(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        cases = (
            ("--account=test", "--account=other"),
            ("--account", "test"),
            ("--partition=test partition",),
            ("--account=",),
            ("--acc=test",),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", wrapper, "train", "calibration", *arguments],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DQN SUBMISSION FAIL", result.stderr)

    def test_submission_wrapper_checks_final_log_directory_and_percent_tokens(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "runtime.sif")
            output = os.path.join(tmp, "results")
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            os.chmod(image, 0o444)
            env = os.environ.copy()
            env.update({"TB3_REPO": ROOT, "TB3_IMAGE": image, "TB3_OUTPUT": output})

            env["TB3_SLURM_LOGS"] = os.path.join(tmp, "logs-%j")
            result = subprocess.run(["bash", wrapper, "train", "calibration"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not contain Slurm percent substitutions", result.stderr)

            percent_target = os.path.join(tmp, "resolved-logs-%j")
            percent_link = os.path.join(tmp, "apparently-safe-logs")
            os.makedirs(percent_target)
            os.symlink(percent_target, percent_link)
            env["TB3_SLURM_LOGS"] = percent_link
            result = subprocess.run(["bash", wrapper, "train", "calibration"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("resolved TB3_SLURM_LOGS must not contain Slurm percent substitutions", result.stderr)

            log_root = os.path.join(tmp, "logs")
            os.makedirs(os.path.join(log_root, "train"))
            os.symlink(ROOT, os.path.join(log_root, "train", "calibration"))
            env["TB3_SLURM_LOGS"] = log_root
            result = subprocess.run(["bash", wrapper, "train", "calibration"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("derived Slurm log directory must be outside TB3_REPO", result.stderr)

    def test_submission_wrapper_rejects_shell_metacharacters_after_canonicalization(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "runtime.sif")
            unsafe_target = os.path.join(tmp, "results;unreviewed")
            safe_link = os.path.join(tmp, "results-link")
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            os.makedirs(unsafe_target)
            os.symlink(unsafe_target, safe_link)
            env = os.environ.copy()
            env.update({"TB3_REPO": ROOT, "TB3_IMAGE": image, "TB3_OUTPUT": safe_link})
            result = subprocess.run(["bash", wrapper, "train", "calibration"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("absolute literal-safe ARC path", result.stderr)

    def test_batch_scripts_remove_ambient_container_controls_at_every_apptainer_call(self):
        allowed_final = {
            "APPTAINERENV_PHASE_LABEL", "APPTAINERENV_LEARNING_SEED", "APPTAINERENV_WORLD_SEED",
            "APPTAINERENV_INITIALIZATION_SEED", "APPTAINERENV_DYNAMIC_OBSTACLE_SEED",
            "APPTAINERENV_EVALUATION_SEED", "APPTAINERENV_TRAIN_RUN_DIR",
            "APPTAINERENV_CHECKPOINT_STEP", "APPTAINERENV_RUN_DIR",
            "APPTAINERENV_LEASE_DIR", "APPTAINERENV_CONTAINER_SHA256", "APPTAINERENV_JOB_TOKEN",
            "APPTAINERENV_SLURM_JOB_ID", "APPTAINERENV_SLURM_ARRAY_JOB_ID",
            "APPTAINERENV_SLURM_ARRAY_TASK_ID", "APPTAINERENV_ROS_LOCALHOST_ONLY",
            "APPTAINERENV_TURTLEBOT3_MODEL", "APPTAINERENV_PYTHONUNBUFFERED",
            "APPTAINERENV_PYTHONDONTWRITEBYTECODE", "APPTAINERENV_OMP_NUM_THREADS",
        }
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            output = os.path.join(tmp, "results")
            image = os.path.join(tmp, "runtime.sif")
            log = os.path.join(tmp, "apptainer-calls.txt")
            os.makedirs(bin_dir)
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            os.chmod(image, 0o444)
            allowed_case = "|".join(sorted(allowed_final))
            apptainer = os.path.join(bin_dir, "apptainer")
            with open(apptainer, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(f"""#!/usr/bin/env bash
set -euo pipefail
final=0
case " $* " in
  *run_dqn_seed.sh*|*run_dqn_eval.sh*) final=1 ;;
esac
while IFS= read -r name; do
  case "$name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
      if [[ "$final" == 0 ]]; then
        echo "container control reached non-final call: $name" >&2
        exit 93
      fi
      case "$name" in
        {allowed_case}) ;;
        *) echo "unapproved container control reached final call: $name" >&2; exit 94 ;;
      esac
      ;;
  esac
done < <(compgen -e)
printf '%s\\t%s\\n' "$final" "$*" >> {shlex.quote(log)}
[[ "$final" == 0 ]] || exit 0
[[ "$1" == exec ]]
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanenv|--containall) shift ;;
    --bind) shift 2 ;;
    *) shift; break ;;
  esac
done
mapped=()
for arg in "$@"; do
  arg="${{arg//\\/opt\\/turtlebot_dqn_random/$FAKE_EMBEDDED_ROOT}}"
  mapped+=("$arg")
done
exec "${{mapped[@]}}"
""")
            os.chmod(apptainer, os.stat(apptainer).st_mode | stat.S_IXUSR)
            base_env = os.environ.copy()
            base_env.update({
                # The fake container executes on the host; include the current
                # test interpreter to model the SIF's pinned Python/YAML stack.
                "PATH": bin_dir + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + "/usr/local/bin:/usr/bin:/bin",
                "TB3_SUBMISSION_GUARD": "dqn-v1.1.7",
                "PHASE_LABEL": "calibration",
                "TB3_IMAGE": image,
                "TB3_OUTPUT": output,
                "TB3_LEASES": os.path.join(output, ".leases"),
                "TB3_EXPECTED_IMAGE_SHA256": sha256_file(image),
                "TB3_EXPECTED_RELEASE_SHA256": sha256_file(os.path.join(ROOT, "RELEASE_MANIFEST.sha256")),
                "FAKE_EMBEDDED_ROOT": ROOT,
                "SLURM_ARRAY_JOB_ID": "9001",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_JOB_ID": "9002",
                "APPTAINERENV_PYTHONPATH": "/tmp/untrusted-pythonpath",
                "APPTAINER_BIND": "/tmp:/unexpected-apptainer-bind",
                "SINGULARITYENV_LD_LIBRARY_PATH": "/tmp/untrusted-library-path",
                "SINGULARITY_BINDPATH": "/var/tmp:/unexpected-singularity-bind",
            })
            for relative in ("arc/dqn_array.sbatch", "arc/dqn_eval_array.sbatch"):
                with self.subTest(relative=relative):
                    env = base_env.copy()
                    env["TB3_EXPECTED_BATCH_SCRIPT_SHA256"] = sha256_file(os.path.join(ROOT, relative))
                    if "eval" in relative:
                        env.update({"TRAIN_JOB": "8001", "CHECKPOINT_STEP": "10000"})
                    result = subprocess.run(
                        ["bash", os.path.join(ROOT, relative)], cwd=tmp, env=env,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(log, encoding="utf-8") as stream:
                calls = [line.rstrip("\n") for line in stream]
            self.assertEqual(sum(line.startswith("1\t") for line in calls), 2)
            self.assertGreaterEqual(sum(line.startswith("0\t") for line in calls), 6)

    def test_batch_scripts_refuse_image_or_spooled_script_mismatch_before_apptainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            image = os.path.join(tmp, "runtime.sif")
            marker = os.path.join(tmp, "apptainer-was-called")
            os.makedirs(bin_dir)
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            os.chmod(image, 0o444)
            apptainer = os.path.join(bin_dir, "apptainer")
            with open(apptainer, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(f"#!/usr/bin/env bash\ntouch {shlex.quote(marker)}\nexit 95\n")
            os.chmod(apptainer, os.stat(apptainer).st_mode | stat.S_IXUSR)
            base_env = os.environ.copy()
            base_env.update({
                "PATH": bin_dir + os.pathsep + "/usr/local/bin:/usr/bin:/bin",
                "TB3_SUBMISSION_GUARD": "dqn-v1.1.7", "PHASE_LABEL": "calibration",
                "TB3_IMAGE": image, "TB3_OUTPUT": os.path.join(tmp, "results"),
                "TB3_EXPECTED_IMAGE_SHA256": sha256_file(image),
                "TB3_EXPECTED_RELEASE_SHA256": sha256_file(os.path.join(ROOT, "RELEASE_MANIFEST.sha256")),
                "TB3_EXPECTED_BATCH_SCRIPT_SHA256": sha256_file(os.path.join(ROOT, "arc", "dqn_array.sbatch")),
                "SLURM_ARRAY_JOB_ID": "9001",
                "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "9002",
            })
            cases = (
                ("TB3_EXPECTED_IMAGE_SHA256", "wrapper-authenticated SHA-256"),
                ("TB3_EXPECTED_BATCH_SCRIPT_SHA256", "Slurm-spooled batch script"),
            )
            for variable, message in cases:
                with self.subTest(variable=variable):
                    env = base_env.copy()
                    env[variable] = "0" * 64
                    result = subprocess.run(
                        ["bash", os.path.join(ROOT, "arc/dqn_array.sbatch")], cwd=tmp, env=env,
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(os.path.exists(marker))

    def test_post_authentication_external_mutation_is_rejected_before_sbatch(self):
        cases = (
            ("arc/run_dqn_seed.sh", "RELEASE_MANIFEST.sha256 does not match the tree"),
            ("arc/dqn_array.sbatch", "batch script changed after authentication"),
        )
        for mutated_relative, expected_message in cases:
            with self.subTest(mutated_relative=mutated_relative), tempfile.TemporaryDirectory() as tmp:
                package_copy = os.path.join(tmp, "TurtleBot_DQN_Random")
                shutil.copytree(
                    ROOT,
                    package_copy,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "build", "install", "log", "*.pyc", "*.egg-info"),
                )
                bin_dir = os.path.join(tmp, "bin")
                os.makedirs(bin_dir)
                image = os.path.join(tmp, "runtime.sif")
                with open(image, "wb") as stream:
                    stream.write(b"test image placeholder\n")
                os.chmod(image, 0o444)
                marker = os.path.join(tmp, "changed-entrypoint-executed")
                sbatch_called = os.path.join(tmp, "sbatch-called")
                mutate_target = os.path.join(package_copy, mutated_relative)

                apptainer = os.path.join(bin_dir, "apptainer")
                with open(apptainer, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(f"""#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == exec ]]
shift
external=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanenv|--containall) shift ;;
    --bind)
      [[ "$2" == *:/mnt/tb3-submission-source:ro ]] && external=1
      shift 2
      ;;
    *) shift; break ;;
  esac
done
mapped=()
for arg in "$@"; do
  arg="${{arg//\\/opt\\/turtlebot_dqn_random/{ROOT}}}"
  arg="${{arg//\\/mnt\\/tb3-submission-source/{package_copy}}}"
  mapped+=("$arg")
done
set +e
"${{mapped[@]}}"
rc=$?
set -e
if [[ "$external" == 1 && ! -e {shlex.quote(os.path.join(tmp, "mutation-done"))} ]]; then
  printf '#!/usr/bin/env bash\ntouch %s\nexit 0\n' {shlex.quote(marker)} > {shlex.quote(mutate_target)}
  chmod u+x {shlex.quote(mutate_target)}
  touch {shlex.quote(os.path.join(tmp, "mutation-done"))}
fi
exit "$rc"
""")
                sbatch = os.path.join(bin_dir, "sbatch")
                with open(sbatch, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(f"#!/usr/bin/env bash\ntouch {shlex.quote(sbatch_called)}\nexit 0\n")
                for path in (apptainer, sbatch):
                    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

                env = os.environ.copy()
                env.update({
                    "PATH": bin_dir + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + "/usr/local/bin:/usr/bin:/bin",
                    "TB3_REPO": package_copy,
                    "TB3_IMAGE": image,
                    "TB3_OUTPUT": os.path.join(tmp, "results"),
                })
                result = subprocess.run(
                    ["bash", os.path.join(package_copy, "arc", "submit_dqn.sh"), "train", "calibration"],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(expected_message, result.stdout + result.stderr)
                self.assertFalse(os.path.exists(sbatch_called))
                self.assertFalse(os.path.exists(marker))

    def test_writable_sif_is_refused(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "runtime.sif")
            with open(image, "wb") as stream:
                stream.write(b"writable image\n")
            env = os.environ.copy()
            env.update({
                "TB3_REPO": ROOT,
                "TB3_IMAGE": image,
                "TB3_OUTPUT": os.path.join(tmp, "results"),
            })
            result = subprocess.run(
                ["bash", wrapper, "train", "calibration"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("all write bits removed", result.stderr)

    def test_trusted_verifier_does_not_import_external_identity_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_copy = os.path.join(tmp, "external")
            shutil.copytree(
                ROOT,
                package_copy,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "build", "install", "log", "*.pyc", "*.egg-info"),
            )
            with open(
                os.path.join(package_copy, "turtlebot3_drl_nav", "identity.py"),
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                stream.write("raise SystemExit(0)\n")
            with open(os.path.join(package_copy, "arc", "run_dqn_seed.sh"), "a", encoding="utf-8") as stream:
                stream.write("# unmanifested mutation\n")
            result = subprocess.run(
                ["bash", os.path.join(ROOT, "scripts", "verify_package.sh"), package_copy],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RELEASE_MANIFEST.sha256 does not match the tree", result.stdout + result.stderr)

    def test_checked_hash_bytecode_is_rejected_before_any_payload_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_copy = os.path.join(tmp, "TurtleBot_DQN_Random")
            shutil.copytree(
                ROOT,
                package_copy,
                ignore=shutil.ignore_patterns(
                    "__pycache__", ".pytest_cache", ".git", "build", "install",
                    "log", "*.pyc", "*.pyo", "*.egg-info",
                ),
            )
            identity_path = os.path.join(package_copy, "turtlebot3_drl_nav", "identity.py")
            cache_dir = os.path.join(package_copy, "turtlebot3_drl_nav", "__pycache__")
            pyc_path = os.path.join(cache_dir, "identity.cpython-38.pyc")
            marker = os.path.join(tmp, "bytecode-payload-executed")
            os.makedirs(cache_dir)
            with open(identity_path, "rb") as stream:
                legitimate_source = stream.read()
            payload_source = legitimate_source + (
                "\nwith open(%r, 'w', encoding='utf-8') as _stream:\n"
                "    _stream.write('executed\\n')\n" % marker
            ).encode("utf-8")
            code = compile(payload_source, identity_path, "exec")
            checked_hash_header = (
                importlib.util.MAGIC_NUMBER
                + struct.pack("<I", 3)
                + importlib.util.source_hash(legitimate_source)
            )
            with open(pyc_path, "wb") as stream:
                stream.write(checked_hash_header)
                marshal.dump(code, stream)

            commands = (
                ["bash", os.path.join(ROOT, "scripts", "verify_package.sh"), package_copy],
                ["bash", os.path.join(package_copy, "scripts", "verify_package.sh"), package_copy],
                [sys.executable, os.path.join(package_copy, "scripts", "make_manifests.py")],
            )
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        command,
                        cwd=package_copy,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("forbidden", (result.stdout + result.stderr).lower())
                    self.assertFalse(os.path.exists(marker))

    def test_manifest_materializer_copies_exact_release_and_refuses_dirty_source(self):
        materializer = os.path.join(ROOT, "scripts", "materialize_release.py")
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "clean-release")
            result = subprocess.run(
                [sys.executable, materializer, "--source", ROOT, "--destination", destination],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(os.path.join(ROOT, "RELEASE_MANIFEST.sha256"), encoding="utf-8") as stream:
                expected = {line.rstrip("\n").partition("  ")[2] for line in stream if line.strip()}
            expected.add("RELEASE_MANIFEST.sha256")
            actual = {
                os.path.relpath(os.path.join(base, name), destination)
                for base, _, names in os.walk(destination)
                for name in names
            }
            self.assertEqual(actual, expected)
            verify = subprocess.run(
                ["bash", os.path.join(destination, "scripts", "verify_package.sh"), destination],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)

            dirty_source = os.path.join(tmp, "dirty-source")
            shutil.copytree(destination, dirty_source)
            os.makedirs(os.path.join(dirty_source, ".pytest_cache"))
            dirty_destination = os.path.join(tmp, "must-not-exist")
            dirty = subprocess.run(
                [sys.executable, materializer, "--source", dirty_source, "--destination", dirty_destination],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn("forbidden release paths", dirty.stderr)
            self.assertFalse(os.path.exists(dirty_destination))

    def test_sif_builder_builds_only_from_new_manifest_materialization(self):
        builder = os.path.join(ROOT, "scripts", "build_sif.sh")
        text = source("scripts/build_sif.sh")
        self.assertLess(text.index("verify_package.sh"), text.index("materialize_release.py"))
        self.assertLess(text.index("materialize_release.py"), text.index('"$APPTAINER_CMD" build'))
        self.assertIn('cd "$STAGED_ROOT"', text)
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            staging_base = os.path.join(tmp, "staging")
            output_parent = os.path.join(tmp, "images")
            os.makedirs(bin_dir)
            os.makedirs(staging_base)
            os.makedirs(output_parent)
            build_cwd = os.path.join(tmp, "build-cwd.txt")
            build_files = os.path.join(tmp, "build-files.txt")
            fake = os.path.join(bin_dir, "apptainer")
            with open(fake, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(f"""#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == build && "$3" == apptainer/tb3_phase1_foxy.def ]]
pwd > {shlex.quote(build_cwd)}
find . -type f -printf '%P\\n' | LC_ALL=C sort > {shlex.quote(build_files)}
printf 'fake immutable image\\n' > "$2"
""")
            os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
            output = os.path.join(output_parent, "runtime.sif")
            result = subprocess.run(
                ["bash", builder, output],
                env=dict(
                    os.environ,
                    PATH=bin_dir + os.pathsep + os.environ["PATH"],
                    TMPDIR=staging_base,
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(build_cwd, encoding="utf-8") as stream:
                used_root = stream.read().strip()
            self.assertNotEqual(os.path.realpath(used_root), os.path.realpath(ROOT))
            self.assertTrue(used_root.startswith(os.path.realpath(staging_base) + os.sep))
            with open(build_files, encoding="utf-8") as stream:
                actual = {line.rstrip("\n") for line in stream if line.strip()}
            with open(os.path.join(ROOT, "RELEASE_MANIFEST.sha256"), encoding="utf-8") as stream:
                expected = {line.rstrip("\n").partition("  ")[2] for line in stream if line.strip()}
            expected.add("RELEASE_MANIFEST.sha256")
            self.assertEqual(actual, expected)

    def test_external_lease_override_is_rejected_and_batches_bind_only_results(self):
        wrapper = os.path.join(ROOT, "arc", "submit_dqn.sh")
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "runtime.sif")
            with open(image, "wb") as stream:
                stream.write(b"test image placeholder\n")
            os.chmod(image, 0o444)
            output = os.path.join(tmp, "results")
            outside = os.path.join(tmp, "external-leases")
            result = subprocess.run(
                ["bash", wrapper, "train", "calibration"],
                env=dict(
                    os.environ,
                    TB3_REPO=ROOT,
                    TB3_IMAGE=image,
                    TB3_OUTPUT=output,
                    TB3_LEASES=outside,
                ),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must resolve exactly to TB3_OUTPUT/.leases", result.stderr)
            self.assertFalse(os.path.exists(outside))
        for relative in ("arc/dqn_array.sbatch", "arc/dqn_eval_array.sbatch"):
            text = source(relative)
            self.assertIn('[[ "$LEASE_DIR" == "$OUTPUT_ROOT/.leases" ]]', text)
            self.assertIn('--bind "$OUTPUT_ROOT:$OUTPUT_ROOT"', text)
            self.assertNotIn('--bind "$LEASE_DIR:$LEASE_DIR"', text)

    def test_wrappers_seal_outputs_before_complete(self):
        for relative in ("arc/run_dqn_seed.sh", "arc/run_dqn_eval.sh"):
            text = source(relative)
            self.assertIn("turtlebot3_drl_nav.artifact_integrity write", text)
            self.assertIn("turtlebot3_drl_nav.artifact_integrity verify", text)
            self.assertLess(text.index("artifact_integrity verify"), text.index('STATUS="COMPLETE"'))
            self.assertIn("validation failed", text)
        self.assertIn("scripts/select_checkpoint.py", source("arc/run_dqn_eval.sh"))

    def test_container_test_gate_cannot_hide_skips_in_dev_null(self):
        text = source("scripts/run_tests.sh")
        self.assertIn("mktemp", text)
        self.assertIn("export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", text)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", text)
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
