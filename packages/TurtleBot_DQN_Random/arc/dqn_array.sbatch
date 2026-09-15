#!/bin/bash
#SBATCH --job-name=tb3-dqn-random
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#
# Direct-ARC training array for TurtleBot_DQN_Random. Submit only through
# arc/submit_dqn.sh. The authenticated SIF contains the complete release at
# /opt/turtlebot_dqn_random; no external package tree is mounted or executed.
set -euo pipefail

readonly EMBEDDED_REPO=/opt/turtlebot_dqn_random

is_container_runtime_name() {
  case "$1" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*) return 0 ;;
    *) return 1 ;;
  esac
}
clear_container_runtime_environment() {
  local name
  while IFS= read -r name; do
    is_container_runtime_name "$name" && unset "$name"
  done < <(compgen -e)
  return 0
}
assert_no_container_runtime_environment() {
  local name
  while IFS= read -r name; do
    if is_container_runtime_name "$name"; then
      echo "unapproved container-runtime variable reached an authentication call: $name" >&2
      return 1
    fi
  done < <(compgen -e)
  return 0
}
assert_only_owned_container_environment() {
  local name
  while IFS= read -r name; do
    case "$name" in
      APPTAINERENV_PHASE_LABEL|APPTAINERENV_LEARNING_SEED|APPTAINERENV_WORLD_SEED|APPTAINERENV_INITIALIZATION_SEED|APPTAINERENV_DYNAMIC_OBSTACLE_SEED|APPTAINERENV_EVALUATION_SEED|APPTAINERENV_RUN_DIR|APPTAINERENV_LEASE_DIR|APPTAINERENV_CONTAINER_SHA256|APPTAINERENV_JOB_TOKEN|APPTAINERENV_SLURM_JOB_ID|APPTAINERENV_SLURM_ARRAY_JOB_ID|APPTAINERENV_SLURM_ARRAY_TASK_ID|APPTAINERENV_ROS_LOCALHOST_ONLY|APPTAINERENV_TURTLEBOT3_MODEL|APPTAINERENV_PYTHONUNBUFFERED|APPTAINERENV_PYTHONDONTWRITEBYTECODE|APPTAINERENV_OMP_NUM_THREADS) ;;
      APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
        echo "unapproved container-runtime variable reached the scientific call: $name" >&2
        return 1
        ;;
    esac
  done < <(compgen -e)
  return 0
}
require_safe_arc_path() {
  [[ "$2" =~ ^/[A-Za-z0-9._/+@-]+$ ]] || {
    echo "$1 is not an absolute literal-safe ARC path: $2" >&2
    exit 2
  }
}
require_read_only_image() {
  local mode
  mode="$(stat -c '%A' "$IMAGE")"
  [[ "$mode" != *w* ]] || {
    echo "TB3_IMAGE must have all write bits removed (run chmod a-w): $IMAGE" >&2
    exit 2
  }
}
verify_image_identity() {
  local image_sha
  image_sha="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
  [[ "$image_sha" == "$EXPECTED_IMAGE_SHA256" ]] || {
    echo "TB3_IMAGE no longer matches the wrapper-authenticated SHA-256" >&2
    return 1
  }
}
verify_embedded_release() {
  local release_sha
  assert_no_container_runtime_environment
  release_sha="$(
    "$APPTAINER_CMD" exec --cleanenv "$IMAGE" \
      sha256sum "$EMBEDDED_REPO/RELEASE_MANIFEST.sha256" | cut -d' ' -f1
  )"
  [[ "$release_sha" == "$EXPECTED_RELEASE_SHA256" ]] || {
    echo "SIF-embedded release differs from the wrapper-authenticated release" >&2
    return 1
  }
  "$APPTAINER_CMD" exec --cleanenv "$IMAGE" \
    bash "$EMBEDDED_REPO/scripts/verify_package.sh" "$EMBEDDED_REPO"
}

clear_container_runtime_environment

[[ "${TB3_SUBMISSION_GUARD:-}" == "dqn-v1.1.7" ]] || {
  echo "Use arc/submit_dqn.sh; direct sbatch submission is refused" >&2
  exit 2
}
PHASE_LABEL="${PHASE_LABEL:-}"
[[ "$PHASE_LABEL" =~ ^(calibration|pilot|controlled)$ ]] || {
  echo "PHASE_LABEL must be calibration, pilot or controlled" >&2
  exit 2
}
IMAGE="${TB3_IMAGE:?TB3_IMAGE must point at the immutable v1.1.7 Apptainer SIF}"
OUTPUT_ROOT="${TB3_OUTPUT:?TB3_OUTPUT must point at the results root}"
LEASE_DIR="${TB3_LEASES:-$OUTPUT_ROOT/.leases}"
EXPECTED_IMAGE_SHA256="${TB3_EXPECTED_IMAGE_SHA256:?wrapper-authenticated image SHA-256 is required}"
EXPECTED_RELEASE_SHA256="${TB3_EXPECTED_RELEASE_SHA256:?wrapper-authenticated embedded-release SHA-256 is required}"
EXPECTED_BATCH_SCRIPT_SHA256="${TB3_EXPECTED_BATCH_SCRIPT_SHA256:?wrapper-authenticated batch-script SHA-256 is required}"
[[ "$EXPECTED_IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid wrapper-authenticated image SHA-256" >&2
  exit 2
}
[[ "$EXPECTED_RELEASE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid wrapper-authenticated release SHA-256" >&2
  exit 2
}
[[ "$EXPECTED_BATCH_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid wrapper-authenticated batch-script SHA-256" >&2
  exit 2
}
[[ "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)" == "$EXPECTED_BATCH_SCRIPT_SHA256" ]] || {
  echo "Slurm-spooled batch script differs from the authenticated snapshot" >&2
  exit 2
}
for slurm_id in "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}"; do
  [[ -z "$slurm_id" || "$slurm_id" =~ ^[0-9]+$ ]] || {
    echo "Slurm identity fields must be numeric" >&2
    exit 2
  }
done

[[ -f "$IMAGE" ]] || { echo "TB3_IMAGE is not a file: $IMAGE" >&2; exit 2; }
IMAGE="$(realpath -e "$IMAGE")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
LEASE_DIR="$(realpath -m "$LEASE_DIR")"
require_safe_arc_path "TB3_IMAGE" "$IMAGE"
require_safe_arc_path "TB3_OUTPUT" "$OUTPUT_ROOT"
require_safe_arc_path "TB3_LEASES" "$LEASE_DIR"
[[ "$LEASE_DIR" == "$OUTPUT_ROOT/.leases" ]] || {
  echo "TB3_LEASES must resolve exactly to TB3_OUTPUT/.leases" >&2
  exit 2
}
require_read_only_image

if ! command -v apptainer >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load apptainer 2>/dev/null || module load containers/apptainer 2>/dev/null || true
fi
APPTAINER_CMD="$(command -v apptainer || command -v singularity || true)"
[[ -n "$APPTAINER_CMD" ]] || { echo "Apptainer/Singularity is unavailable" >&2; exit 2; }

clear_container_runtime_environment
assert_no_container_runtime_environment
verify_image_identity
verify_embedded_release
verify_image_identity

# Read every phase value from the package embedded in the authenticated image.
CQ=("$APPTAINER_CMD" exec --cleanenv "$IMAGE" python3 "$EMBEDDED_REPO/arc/phase_query.py" "$EMBEDDED_REPO/config/common_environment.yaml")
LEARNING_SEED="$("${CQ[@]}" seed "$PHASE_LABEL" "${SLURM_ARRAY_TASK_ID:-0}")"
read -r WORLD_SEED INITIALIZATION_SEED DYNAMIC_OBSTACLE_SEED EVALUATION_SEED <<< "$("${CQ[@]}" common-seeds)"
for seed_value in "$LEARNING_SEED" "$WORLD_SEED" "$INITIALIZATION_SEED" "$DYNAMIC_OBSTACLE_SEED" "$EVALUATION_SEED"; do
  [[ "$seed_value" =~ ^[0-9]+$ ]] || {
    echo "phase query returned a non-numeric seed" >&2
    exit 2
  }
done

CONTAINER_SHA256="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
RELEASE_TOKEN="${EXPECTED_RELEASE_SHA256:0:12}"
RUN_DIR="$OUTPUT_ROOT/$PHASE_LABEL/dqn/seed_${LEARNING_SEED}/job_${SLURM_ARRAY_JOB_ID:-local}"
require_safe_arc_path "run directory" "$RUN_DIR"
mkdir -p "$OUTPUT_ROOT/$PHASE_LABEL/dqn/seed_${LEARNING_SEED}" "$LEASE_DIR"
OUTPUT_ROOT="$(realpath -e "$OUTPUT_ROOT")"
LEASE_DIR="$(realpath -e "$LEASE_DIR")"
[[ "$LEASE_DIR" == "$OUTPUT_ROOT/.leases" ]] || {
  echo "resolved TB3_LEASES escaped TB3_OUTPUT after directory creation" >&2
  exit 2
}

export APPTAINERENV_PHASE_LABEL="$PHASE_LABEL"
export APPTAINERENV_LEARNING_SEED="$LEARNING_SEED"
export APPTAINERENV_WORLD_SEED="$WORLD_SEED"
export APPTAINERENV_INITIALIZATION_SEED="$INITIALIZATION_SEED"
export APPTAINERENV_DYNAMIC_OBSTACLE_SEED="$DYNAMIC_OBSTACLE_SEED"
export APPTAINERENV_EVALUATION_SEED="$EVALUATION_SEED"
export APPTAINERENV_RUN_DIR="$RUN_DIR"
export APPTAINERENV_LEASE_DIR="$LEASE_DIR"
export APPTAINERENV_CONTAINER_SHA256="$CONTAINER_SHA256"
export APPTAINERENV_JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export APPTAINERENV_SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
export APPTAINERENV_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
export APPTAINERENV_ROS_LOCALHOST_ONLY=1
export APPTAINERENV_TURTLEBOT3_MODEL=burger
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OMP_NUM_THREADS=1

# The SIF hash binds both dependency runtime and embedded source. Recheck it
# immediately before execution; the only writable bind is the result root.
verify_image_identity
assert_only_owned_container_environment

CONTAINER_PID=""
forward_usr1() {
  if [[ -n "$CONTAINER_PID" ]]; then
    kill -USR1 "$CONTAINER_PID" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

set +e
"$APPTAINER_CMD" exec --cleanenv --containall \
  --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" \
  "$IMAGE" bash "$EMBEDDED_REPO/arc/run_dqn_seed.sh" &
CONTAINER_PID=$!
rc=1
while true; do
  wait "$CONTAINER_PID"; rc=$?
  kill -0 "$CONTAINER_PID" 2>/dev/null || break
done
echo "container exited with status $rc; release=$RELEASE_TOKEN"
exit "$rc"
