#!/bin/bash
#SBATCH --job-name=tb3-rainbowdqn-random
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Direct-ARC training array for TurtleBot_RainbowDQN_Random. The only entry point is
#     sbatch [--array=...] [--account=... --partition=...] arc/rainbowdqn_array.sbatch
# Phase selection is explicit:
#     sbatch --export=ALL,PHASE_LABEL=calibration --array=0   arc/rainbowdqn_array.sbatch
#     sbatch --export=ALL,PHASE_LABEL=pilot       --array=0-1 arc/rainbowdqn_array.sbatch
#     sbatch --export=ALL,PHASE_LABEL=controlled  --array=0-4 arc/rainbowdqn_array.sbatch
# The budget, cadence and seed list of each phase are frozen in
# config/common_environment.yaml; an array index outside the phase's seed list
# is refused here before anything starts. Nothing in this file derives an
# identity by modulo: transport isolation is leased inside the container.
set -euo pipefail

PHASE_LABEL="${PHASE_LABEL:-}"
[[ "$PHASE_LABEL" =~ ^(calibration|pilot|controlled)$ ]] || { echo "PHASE_LABEL must be calibration, pilot or controlled" >&2; exit 2; }
REPO="${TB3_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${TB3_IMAGE:?TB3_IMAGE must point at the Apptainer image (.sif) of this package}"
OUTPUT_ROOT="${TB3_OUTPUT:?TB3_OUTPUT must point at the results root}"
LEASE_DIR="${TB3_LEASES:-$OUTPUT_ROOT/.leases}"

if ! command -v apptainer >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load apptainer 2>/dev/null || module load containers/apptainer 2>/dev/null || true
fi
APPTAINER_CMD="$(command -v apptainer || command -v singularity)"
[[ -n "$APPTAINER_CMD" ]] || { echo "Apptainer/Singularity is unavailable" >&2; exit 2; }

# Learning seed and common seeds from the frozen phase block (no seed literal
# lives in this file). The container's Python parses the YAML.
CQ=("$APPTAINER_CMD" exec --cleanenv --bind "$REPO:$REPO" "$IMAGE" python3 "$REPO/arc/phase_query.py" "$REPO/config/common_environment.yaml")
LEARNING_SEED="$("${CQ[@]}" seed "$PHASE_LABEL" "${SLURM_ARRAY_TASK_ID:-0}")"
read -r WORLD_SEED INITIALIZATION_SEED DYNAMIC_OBSTACLE_SEED EVALUATION_SEED <<< "$("${CQ[@]}" common-seeds)"

# Always hash the exact bytes submitted. A sidecar is never an authority.
CONTAINER_SHA256="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
RELEASE_TOKEN="$(sha256sum "$REPO/RELEASE_MANIFEST.sha256" | cut -c1-12)"

RUN_DIR="$OUTPUT_ROOT/$PHASE_LABEL/rainbowdqn/seed_${LEARNING_SEED}/job_${SLURM_ARRAY_JOB_ID:-local}"
WS="$OUTPUT_ROOT/ws/rainbowdqn_${RELEASE_TOKEN}_${PHASE_LABEL}_seed_${LEARNING_SEED}_job_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$OUTPUT_ROOT/$PHASE_LABEL/rainbowdqn/seed_${LEARNING_SEED}" "$LEASE_DIR" "$WS"

# Everything the wrapper needs crosses --cleanenv explicitly, Slurm identity included.
export APPTAINERENV_PHASE_LABEL="$PHASE_LABEL"
export APPTAINERENV_LEARNING_SEED="$LEARNING_SEED"
export APPTAINERENV_WORLD_SEED="$WORLD_SEED"
export APPTAINERENV_INITIALIZATION_SEED="$INITIALIZATION_SEED"
export APPTAINERENV_DYNAMIC_OBSTACLE_SEED="$DYNAMIC_OBSTACLE_SEED"
export APPTAINERENV_EVALUATION_SEED="$EVALUATION_SEED"
export APPTAINERENV_RUN_DIR="$RUN_DIR"
export APPTAINERENV_TB3_WS="$WS"
export APPTAINERENV_LEASE_DIR="$LEASE_DIR"
export APPTAINERENV_CONTAINER_SHA256="$CONTAINER_SHA256"
export APPTAINERENV_JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export APPTAINERENV_SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
export APPTAINERENV_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
export APPTAINERENV_ROS_LOCALHOST_ONLY=1
export APPTAINERENV_TURTLEBOT3_MODEL=burger
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS=1

# Forward the walltime warning into the container as SIGUSR1 to the wrapper.
CONTAINER_PID=""
forward_usr1() { if [[ -n "$CONTAINER_PID" ]]; then kill -USR1 "$CONTAINER_PID" 2>/dev/null || true; fi; }
trap forward_usr1 USR1

set +e
"$APPTAINER_CMD" exec --cleanenv --containall \
  --bind "$REPO:$REPO" --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" \
  "$IMAGE" bash "$REPO/arc/run_rainbowdqn_seed.sh" &
CONTAINER_PID=$!
rc=1
while true; do
  wait "$CONTAINER_PID"; rc=$?
  kill -0 "$CONTAINER_PID" 2>/dev/null || break   # a trapped signal interrupted wait; keep waiting
done
echo "container exited with status $rc"
exit "$rc"
