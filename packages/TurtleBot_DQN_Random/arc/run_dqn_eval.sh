#!/usr/bin/env bash
# ARC post-hoc (tier-2) evaluation wrapper: evaluates one retained checkpoint of
# a completed training run on the held-out scenario list (E2) and the anchor
# scenario (E3). Runs inside the Apptainer container; fail-closed like training.
#
# Required environment (set by arc/dqn_eval_array.sbatch):
#   TRAIN_RUN_DIR CHECKPOINT_STEP RUN_DIR CONTAINER_SHA256 LEASE_DIR
# Optional: JOB_TOKEN
set -eo pipefail
readonly REPO=/opt/turtlebot_dqn_random
readonly INSTALL_PREFIX=/opt/turtlebot_dqn_random_ws/install
[[ -f "$REPO/VERSION" && -f "$INSTALL_PREFIX/setup.bash" ]] || {
  echo "immutable DQN package/install missing from the authenticated SIF" >&2
  exit 2
}
for v in TRAIN_RUN_DIR CHECKPOINT_STEP RUN_DIR CONTAINER_SHA256 LEASE_DIR; do
  if [[ -z "${!v:-}" ]]; then echo "missing environment variable $v" >&2; exit 2; fi
done
JOB_TOKEN="${JOB_TOKEN:-${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_DIR"
SIM_PID=""; LAUNCH_PID=""; LEASE_PATH=""; STATUS="FAILED"; REASON="not started"

write_marker() {
  printf '%s\n' "$2" > "$RUN_DIR/$1.tmp"
  sync "$RUN_DIR/$1.tmp" 2>/dev/null || true
  mv -f "$RUN_DIR/$1.tmp" "$RUN_DIR/$1"
}
finish() {
  local code=$?
  trap - EXIT
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then kill -INT "$LAUNCH_PID" 2>/dev/null || true; sleep 5; fi
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then kill -TERM "$SIM_PID" 2>/dev/null || true; sleep 3; kill -KILL "$SIM_PID" 2>/dev/null || true; fi
  pkill -KILL -f "$RUN_DIR/sim" 2>/dev/null || true
  if [[ -n "$LEASE_PATH" ]]; then python3 "$REPO/arc/task_isolation.py" release "$LEASE_PATH" || true; fi
  ros2 daemon stop >/dev/null 2>&1 || true
  if [[ "$STATUS" == "COMPLETE" ]]; then write_marker COMPLETE "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REASON"; exit 0; fi
  write_marker "$STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REASON (exit $code)"; echo "run $STATUS: $REASON" >&2; exit 1
}
trap finish EXIT

on_walltime_warning() {
  STATUS="INTERRUPTED"; REASON="walltime signal"
  if [[ -n "$LAUNCH_PID" ]]; then kill -INT "$LAUNCH_PID" 2>/dev/null || true; fi
}
trap on_walltime_warning USR1

source /opt/ros/foxy/setup.bash
# shellcheck disable=SC1090
source "$INSTALL_PREFIX/setup.bash"
export TURTLEBOT3_MODEL=burger ROS_LOCALHOST_ONLY=1
export TURTLEBOT3_WS=/opt/turtlebot_dqn_random_ws

REASON="evaluation preflight failed"
python3 -c "import torch, numpy, yaml, rclpy, gazebo_msgs.srv"
command -v ros2 >/dev/null
command -v gzserver >/dev/null
for pkg in gazebo_ros turtlebot3_gazebo turtlebot3_description; do
  ros2 pkg prefix "$pkg" >/dev/null 2>&1
done
python3 -m turtlebot3_drl_nav.identity verify-shared-layer "$REPO"
python3 -m turtlebot3_drl_nav.identity verify-release "$REPO"
[[ "$CONTAINER_SHA256" =~ ^[0-9a-f]{64}$ ]]

REASON="training run not complete"
[[ -f "$TRAIN_RUN_DIR/COMPLETE" && -f "$TRAIN_RUN_DIR/run_identity.json" ]]
python3 -m turtlebot3_drl_nav.artifact_integrity verify "$TRAIN_RUN_DIR"
python3 - "$TRAIN_RUN_DIR/run_identity.json" "$RUN_DIR" "$REPO" "$CHECKPOINT_STEP" "$CONTAINER_SHA256" "$JOB_TOKEN" <<'PY'
import json, os, sys
identity_path, run_dir, repo, step, container, token = sys.argv[1:7]
sys.path.insert(0, repo)
from turtlebot3_drl_nav.identity import build_identity, write_identity, config_digest, release_digest, shared_layer_digest, read_identity
train = read_identity(identity_path)
if os.path.exists(run_dir) and os.listdir(run_dir):
    sys.exit("evaluation run directory is not empty")
os.makedirs(run_dir, exist_ok=True)
config_sha = config_digest(repo); shared_sha = shared_layer_digest(repo); release_sha = release_digest(repo)
if config_sha != train["config_sha256"] or shared_sha != train["shared_layer_sha256"] or release_sha != train["release_sha256"]:
    sys.exit("package digests differ from the training run's identity; evaluate with the same package")
if container != train["container_sha256"]:
    sys.exit("evaluation container differs from the training container")
identity = build_identity(
    experiment=train["experiment"], algorithm=train["algorithm"], algorithm_version=train["algorithm_version"], arm=train["arm"],
    learning_seed=int(train["learning_seed"]), world_id=train["world_id"], world_seed=int(train["world_seed"]),
    initialization_seed=int(train["initialization_seed"]), dynamic_obstacle_seed=int(train["dynamic_obstacle_seed"]),
    evaluation_seed=int(train["evaluation_seed"]), config_sha256=config_sha, container_sha256=container,
    shared_layer_sha256=shared_sha, release_sha256=release_sha, package_version=train["package_version"], phase_type="evaluation",
    phase_label=train["phase_label"], job_token=token)
write_identity(os.path.join(run_dir, "run_identity.json"), identity)
manifest = dict(identity); manifest.update({"training_run_id": train["run_id"], "checkpoint_step": int(step),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""), "hostname": os.uname().nodename,
    "started_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z"})
with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True); f.write("\n")
PY
mkdir -p "$RUN_DIR/sim" "$RUN_DIR/logs"
{
  python3 --version; python3 -c 'import numpy, torch; print("numpy=" + numpy.__version__); print("torch=" + torch.__version__)'
  echo "ros_distro=${ROS_DISTRO}"; python3 -c "import subprocess; print(subprocess.getoutput('gzserver --version').splitlines()[0])" 2>/dev/null || true
} > "$RUN_DIR/runtime_manifest.txt"
mapfile -t CHECKPOINT_INFO < <(python3 "$REPO/scripts/select_checkpoint.py" "$TRAIN_RUN_DIR" "$CHECKPOINT_STEP")
[[ "${#CHECKPOINT_INFO[@]}" -eq 2 ]] || { REASON="checkpoint selection did not return one path and digest"; exit 1; }
CHECKPOINT="${CHECKPOINT_INFO[0]}"
CHECKPOINT_SHA256="${CHECKPOINT_INFO[1]}"
python3 - "$RUN_DIR/run_manifest.json" "$CHECKPOINT" "$CHECKPOINT_SHA256" <<'PY'
import json, os, sys
path, checkpoint, digest = sys.argv[1:4]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)
data.update({"checkpoint_path": os.path.realpath(checkpoint), "checkpoint_sha256": digest})
tmp = path + ".tmp"
with open(tmp, "x", encoding="utf-8") as stream:
    json.dump(data, stream, indent=2, sort_keys=True); stream.write("\n")
os.replace(tmp, path)
PY
IDENT() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$RUN_DIR/run_identity.json" "$1"; }

REASON="transport lease failed"
eval "$(python3 "$REPO/arc/task_isolation.py" acquire "$LEASE_DIR")"; export ROS_DOMAIN_ID GAZEBO_MASTER_URI; LEASE_PATH="$TRANSPORT_LEASE"
echo "ros_domain_id=$ROS_DOMAIN_ID gazebo_master_uri=$GAZEBO_MASTER_URI" >> "$RUN_DIR/runtime_manifest.txt"
REASON="simulator failed to start"
SIM_RUNTIME_DIR="$RUN_DIR/sim" WORLD_SEED="$(IDENT world_seed)" bash "$REPO/scripts/run_sim_headless.sh" > "$RUN_DIR/logs/simulator.log" 2>&1 &
SIM_PID=$!
bash "$REPO/scripts/wait_for_sim.sh" 300 > "$RUN_DIR/logs/readiness.log" 2>&1
REASON="evaluation launch failed"
ros2 launch turtlebot3_drl_nav drl_training.launch.py \
  package_root:="$REPO" run_dir:="$RUN_DIR" mode:=eval learning_seed:="$(IDENT learning_seed)" \
  world_seed:="$(IDENT world_seed)" initialization_seed:="$(IDENT initialization_seed)" \
  dynamic_obstacle_seed:="$(IDENT dynamic_obstacle_seed)" evaluation_seed:="$(IDENT evaluation_seed)" \
  phase_label:="$(IDENT phase_label)" checkpoint_path:="$CHECKPOINT" > "$RUN_DIR/logs/evaluation_console.log" 2>&1 &
LAUNCH_PID=$!
launch_rc=0
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  if [[ -f "$RUN_DIR/ENV_FATAL" || -f "$RUN_DIR/AGENT_FATAL" ]]; then
    REASON="fatal marker present"
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    REASON="simulator or dynamic-obstacle controller exited during evaluation"
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
wait "$LAUNCH_PID" || launch_rc=$?
LAUNCH_PID=""
if [[ "$STATUS" == "INTERRUPTED" ]]; then exit 1; fi
if [[ -f "$RUN_DIR/ENV_FATAL" || -f "$RUN_DIR/AGENT_FATAL" ]]; then REASON="fatal marker present"; exit 1; fi
[[ -f "$RUN_DIR/AGENT_DONE" ]] || { REASON="agent did not report completion"; exit 1; }
(( launch_rc == 0 )) || { REASON="evaluation launch exited non-zero ($launch_rc)"; exit 1; }
if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
  kill -TERM "$SIM_PID" 2>/dev/null || true
  wait "$SIM_PID" 2>/dev/null || true
fi
SIM_PID=""
REASON="validation failed"
python3 "$REPO/scripts/validate_run.py" --package-root "$REPO" --run-dir "$RUN_DIR" --phase-type evaluation > "$RUN_DIR/logs/validation.log" 2>&1
python3 -m turtlebot3_drl_nav.artifact_integrity write "$RUN_DIR"
python3 -m turtlebot3_drl_nav.artifact_integrity verify "$RUN_DIR"
# Re-authenticate the immutable in-image package at the completion boundary.
python3 -m turtlebot3_drl_nav.identity verify-release "$REPO"
STATUS="COMPLETE"; REASON="validated"
