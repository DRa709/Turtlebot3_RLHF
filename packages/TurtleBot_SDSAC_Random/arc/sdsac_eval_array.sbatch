#!/bin/bash
#SBATCH --job-name=tb3-sdsac-random-eval
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Tier-2 post-hoc evaluation array: one task per learning seed of the phase,
# evaluating one retained checkpoint (CHECKPOINT_STEP, default = the phase budget)
# of the COMPLETE training run on the held-out list (E2) and the anchor (E3).
#     sbatch --export=ALL,PHASE_LABEL=controlled,TRAIN_JOB=<array job id> --array=0-4 arc/sdsac_eval_array.sbatch
set -euo pipefail
PHASE_LABEL="${PHASE_LABEL:-}"
[[ "$PHASE_LABEL" =~ ^(calibration|pilot|controlled)$ ]] || { echo "PHASE_LABEL must be calibration, pilot or controlled" >&2; exit 2; }
TRAIN_JOB="${TRAIN_JOB:?TRAIN_JOB (the training array job id) is required}"
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
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
CQ=("$APPTAINER_CMD" exec --cleanenv --bind "$REPO:$REPO" "$IMAGE" python3 "$REPO/arc/phase_query.py" "$REPO/config/common_environment.yaml")
LEARNING_SEED="$("${CQ[@]}" seed "$PHASE_LABEL" "${SLURM_ARRAY_TASK_ID:-0}")"
BUDGET="$("${CQ[@]}" budget "$PHASE_LABEL")"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-$BUDGET}"
TIER2_STEPS="$("${CQ[@]}" tier2-steps "$PHASE_LABEL")"
[[ " $TIER2_STEPS " == *" $CHECKPOINT_STEP "* ]] || {
  echo "CHECKPOINT_STEP=$CHECKPOINT_STEP is not preregistered for $PHASE_LABEL (allowed: $TIER2_STEPS)" >&2
  exit 2
}
CONTAINER_SHA256="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
RELEASE_TOKEN="$(sha256sum "$REPO/RELEASE_MANIFEST.sha256" | cut -c1-12)"
TRAIN_RUN_DIR="$OUTPUT_ROOT/$PHASE_LABEL/sdsac/seed_${LEARNING_SEED}/job_${TRAIN_JOB}"
RUN_DIR="$OUTPUT_ROOT/$PHASE_LABEL/sdsac/seed_${LEARNING_SEED}/eval_step${CHECKPOINT_STEP}_job_${SLURM_ARRAY_JOB_ID:-local}"
WS="$OUTPUT_ROOT/ws/sdsac_${RELEASE_TOKEN}_${PHASE_LABEL}_seed_${LEARNING_SEED}_eval_${CHECKPOINT_STEP}_job_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$LEASE_DIR" "$WS"
export APPTAINERENV_TRAIN_RUN_DIR="$TRAIN_RUN_DIR" APPTAINERENV_CHECKPOINT_STEP="$CHECKPOINT_STEP" APPTAINERENV_RUN_DIR="$RUN_DIR"
export APPTAINERENV_TB3_WS="$WS" APPTAINERENV_LEASE_DIR="$LEASE_DIR" APPTAINERENV_CONTAINER_SHA256="$CONTAINER_SHA256"
export APPTAINERENV_JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}" APPTAINERENV_SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}" APPTAINERENV_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
export APPTAINERENV_ROS_LOCALHOST_ONLY=1 APPTAINERENV_TURTLEBOT3_MODEL=burger APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_OMP_NUM_THREADS=1
CONTAINER_PID=""
forward_usr1() { if [[ -n "$CONTAINER_PID" ]]; then kill -USR1 "$CONTAINER_PID" 2>/dev/null || true; fi; }
trap forward_usr1 USR1
set +e
"$APPTAINER_CMD" exec --cleanenv --containall --bind "$REPO:$REPO" --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" "$IMAGE" bash "$REPO/arc/run_sdsac_eval.sh" &
CONTAINER_PID=$!
rc=1
while true; do
  wait "$CONTAINER_PID"; rc=$?
  kill -0 "$CONTAINER_PID" 2>/dev/null || break
done
exit "$rc"
