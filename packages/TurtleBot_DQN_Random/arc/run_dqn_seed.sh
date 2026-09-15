#!/usr/bin/env bash
# ARC training wrapper for one (algorithm=DQN, learning seed) run. Runs inside
# the Apptainer container. Fail-closed: the run directory receives COMPLETE only
# after the validator passes; every other exit leaves FAILED (or INTERRUPTED) and
# a non-zero status.
#
# Required environment (set by arc/dqn_array.sbatch):
#   PHASE_LABEL (calibration|pilot|controlled) LEARNING_SEED WORLD_SEED INITIALIZATION_SEED
#   DYNAMIC_OBSTACLE_SEED EVALUATION_SEED RUN_DIR CONTAINER_SHA256 LEASE_DIR
# Optional: JOB_TOKEN
# The budget, checkpoint cadence and admissible learning seeds come from the
# phase block of config/common_environment.yaml, never from the command line.
set -eo pipefail
readonly REPO=/opt/turtlebot_dqn_random
readonly INSTALL_PREFIX=/opt/turtlebot_dqn_random_ws/install
[[ -f "$REPO/VERSION" && -f "$INSTALL_PREFIX/setup.bash" ]] || {
  echo "immutable DQN package/install missing from the authenticated SIF" >&2
  exit 2
}
for v in PHASE_LABEL LEARNING_SEED WORLD_SEED INITIALIZATION_SEED DYNAMIC_OBSTACLE_SEED EVALUATION_SEED RUN_DIR CONTAINER_SHA256 LEASE_DIR; do
  if [[ -z "${!v:-}" ]]; then echo "missing environment variable $v" >&2; exit 2; fi
done
JOB_TOKEN="${JOB_TOKEN:-${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_DIR"

SIM_PID=""
LAUNCH_PID=""
LEASE_PATH=""
STATUS="FAILED"
REASON="not started"

write_marker() {  # name, text — atomic, written last
  local name="$1" text="$2"
  printf '%s\n' "$text" > "$RUN_DIR/$name.tmp"
  sync "$RUN_DIR/$name.tmp" 2>/dev/null || true
  mv -f "$RUN_DIR/$name.tmp" "$RUN_DIR/$name"
}

finish() {
  local code=$?
  trap - EXIT
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then kill -INT "$LAUNCH_PID" 2>/dev/null || true; sleep 5; fi
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then kill -TERM "$SIM_PID" 2>/dev/null || true; sleep 3; kill -KILL "$SIM_PID" 2>/dev/null || true; fi
  pkill -KILL -f "$RUN_DIR/sim" 2>/dev/null || true
  if [[ -n "$LEASE_PATH" ]]; then python3 "$REPO/arc/task_isolation.py" release "$LEASE_PATH" || true; fi
  ros2 daemon stop >/dev/null 2>&1 || true
  if [[ "$STATUS" == "COMPLETE" ]]; then
    write_marker COMPLETE "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REASON"
    echo "run COMPLETE: $RUN_DIR"
    exit 0
  fi
  write_marker "$STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ) $REASON (exit $code)"
  echo "run $STATUS: $REASON" >&2
  exit 1
}
trap finish EXIT

on_walltime_warning() {
  echo "walltime warning received; forwarding SIGUSR1 to the agent" >&2
  pkill -USR1 -f "lib/turtlebot3_drl_nav/dqn_agent" 2>/dev/null || true
  STATUS="INTERRUPTED"; REASON="walltime signal"
}
trap on_walltime_warning USR1

# ---------------------------------------------------------------- setup
source /opt/ros/foxy/setup.bash
# shellcheck disable=SC1090
source "$INSTALL_PREFIX/setup.bash"
export TURTLEBOT3_MODEL=burger
export ROS_LOCALHOST_ONLY=1
export TURTLEBOT3_WS=/opt/turtlebot_dqn_random_ws

REASON="preflight failed"
# Emit exact paths/digests before the concise common preflight wrapper so ARC
# evidence identifies the cause of any future provenance failure.
python3 -m turtlebot3_drl_nav.identity verify-shared-layer "$REPO"
python3 -m turtlebot3_drl_nav.identity verify-release "$REPO"
bash "$REPO/scripts/preflight.sh" "$REPO" "$RUN_DIR"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/sim" "$RUN_DIR/logs"

REASON="identity failed"
CONFIG_SHA="$(python3 -m turtlebot3_drl_nav.identity config-digest "$REPO")"
SHARED_SHA="$(python3 -m turtlebot3_drl_nav.identity shared-layer-digest "$REPO")"
RELEASE_SHA="$(python3 -c 'import sys; from turtlebot3_drl_nav.identity import release_digest; print(release_digest(sys.argv[1]))' "$REPO")"
PACKAGE_VERSION="$(tr -d '[:space:]' < "$REPO/VERSION")"
python3 - "$RUN_DIR/run_identity.json" <<PY
import json, os, sys
sys.path.insert(0, "$REPO")
from turtlebot3_drl_nav.identity import build_identity, write_identity
identity = build_identity(
    experiment="phase1-random-arm", algorithm="DQN", algorithm_version="$PACKAGE_VERSION", arm="random",
    learning_seed=$LEARNING_SEED, world_id="phase1_mixed", world_seed=$WORLD_SEED,
    initialization_seed=$INITIALIZATION_SEED, dynamic_obstacle_seed=$DYNAMIC_OBSTACLE_SEED,
    evaluation_seed=$EVALUATION_SEED, config_sha256="$CONFIG_SHA", container_sha256="$CONTAINER_SHA256",
    shared_layer_sha256="$SHARED_SHA", release_sha256="$RELEASE_SHA", package_version="$PACKAGE_VERSION", phase_type="training",
    phase_label="$PHASE_LABEL", job_token="$JOB_TOKEN")
write_identity(sys.argv[1], identity)
import yaml
protocol = yaml.safe_load(open("$REPO/config/common_environment.yaml"))["evaluation_protocol"]
phase = protocol["phases"]["$PHASE_LABEL"]
if $LEARNING_SEED not in [int(v) for v in phase["learning_seeds"]]:
    sys.exit("learning seed $LEARNING_SEED is not preregistered for phase $PHASE_LABEL")
for name in ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed"):
    if int(identity[name]) != int(protocol["common_seeds"][name]):
        sys.exit(f"{name}={identity[name]} differs from the preregistered common seed {protocol['common_seeds'][name]}")
manifest = dict(identity)
manifest.update({"environment_budget": int(phase["environment_budget"]),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
                 "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""), "hostname": os.uname().nodename,
                 "started_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z"})
with open(os.path.join(os.path.dirname(sys.argv[1]), "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True); f.write("\n")
PY
{
  python3 --version; python3 -c 'import numpy, torch; print("numpy=" + numpy.__version__); print("torch=" + torch.__version__)'
  echo "ros_distro=${ROS_DISTRO}"; python3 -c "import subprocess; print(subprocess.getoutput('gzserver --version').splitlines()[0])" 2>/dev/null || true
} > "$RUN_DIR/runtime_manifest.txt"

REASON="transport lease failed"
eval "$(python3 "$REPO/arc/task_isolation.py" acquire "$LEASE_DIR")"
export ROS_DOMAIN_ID GAZEBO_MASTER_URI
LEASE_PATH="$TRANSPORT_LEASE"
echo "ros_domain_id=$ROS_DOMAIN_ID gazebo_master_uri=$GAZEBO_MASTER_URI" >> "$RUN_DIR/runtime_manifest.txt"

# ------------------------------------------------------------ simulator
REASON="simulator failed to start"
SIM_RUNTIME_DIR="$RUN_DIR/sim" WORLD_SEED="$WORLD_SEED" bash "$REPO/scripts/run_sim_headless.sh" > "$RUN_DIR/logs/simulator.log" 2>&1 &
SIM_PID=$!
bash "$REPO/scripts/wait_for_sim.sh" 300 > "$RUN_DIR/logs/readiness.log" 2>&1

# ------------------------------------------------------------- training
REASON="training launch failed"
ros2 launch turtlebot3_drl_nav drl_training.launch.py \
  package_root:="$REPO" run_dir:="$RUN_DIR" mode:=train learning_seed:="$LEARNING_SEED" \
  world_seed:="$WORLD_SEED" initialization_seed:="$INITIALIZATION_SEED" dynamic_obstacle_seed:="$DYNAMIC_OBSTACLE_SEED" \
  evaluation_seed:="$EVALUATION_SEED" phase_label:="$PHASE_LABEL" \
  > "$RUN_DIR/logs/training_console.log" 2>&1 &
LAUNCH_PID=$!
launch_rc=0
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  if [[ -f "$RUN_DIR/ENV_FATAL" || -f "$RUN_DIR/AGENT_FATAL" ]]; then
    REASON="fatal marker present"
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    REASON="simulator or dynamic-obstacle controller exited during training"
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
wait "$LAUNCH_PID" || launch_rc=$?
if [[ "$STATUS" == "INTERRUPTED" ]]; then
  # the agent writes its emergency checkpoint and exits on its own; give it time
  for _ in $(seq 1 300); do kill -0 "$LAUNCH_PID" 2>/dev/null || break; sleep 1; done
fi
LAUNCH_PID=""

if [[ "$STATUS" == "INTERRUPTED" ]]; then exit 1; fi
if [[ -f "$RUN_DIR/ENV_FATAL" || -f "$RUN_DIR/AGENT_FATAL" ]]; then REASON="fatal marker present"; exit 1; fi
if [[ ! -f "$RUN_DIR/AGENT_DONE" ]]; then REASON="agent did not report completion"; exit 1; fi
if (( launch_rc != 0 )); then REASON="training launch exited non-zero ($launch_rc)"; exit 1; fi

# Stop every writer before validation and immutable-output hashing.
if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
  kill -TERM "$SIM_PID" 2>/dev/null || true
  wait "$SIM_PID" 2>/dev/null || true
fi
SIM_PID=""

# ------------------------------------------------------------ validation
REASON="validation failed"
python3 "$REPO/scripts/validate_run.py" --package-root "$REPO" --run-dir "$RUN_DIR" --phase-type training > "$RUN_DIR/logs/validation.log" 2>&1
sha256sum "$RUN_DIR"/checkpoints/*.pt > "$RUN_DIR/checkpoints.sha256"
python3 -m turtlebot3_drl_nav.artifact_integrity write "$RUN_DIR"
python3 -m turtlebot3_drl_nav.artifact_integrity verify "$RUN_DIR"
# Re-authenticate the immutable in-image package at the completion boundary.
python3 -m turtlebot3_drl_nav.identity verify-release "$REPO"
STATUS="COMPLETE"; REASON="validated"
