#!/usr/bin/env bash
# Safe, direct-ARC submission entry point for this standalone DQN package.
#
# Training:
#   bash arc/submit_dqn.sh train <calibration|pilot|controlled> [site options]
# Evaluation:
#   bash arc/submit_dqn.sh eval <calibration|pilot|controlled> <training-array-job-id> [checkpoint-step] [site options]
#
# The wrapper owns --array, --export, --chdir, --output, --error, --dependency
# and the sole batch-script operand. This prevents Slurm from creating its
# stdout/stderr files inside the release tree before the fail-closed
# release-manifest check executes.
set -euo pipefail

fail() {
  echo "DQN SUBMISSION FAIL: $*" >&2
  exit 2
}

usage() {
  sed -n '2,8p' "$0" >&2
  exit 2
}

path_is_within() {
  local child="$1" parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

require_safe_arc_path() {
  local label="$1" value="$2"
  [[ ! "$value" =~ [[:space:]] ]] || fail "$label must not contain whitespace under ROS 2 Foxy: $value"
  [[ "$value" != *","* ]] || fail "$label must not contain a comma because Slurm --export is comma-delimited: $value"
  [[ "$value" != *":"* ]] || fail "$label must not contain a colon because Apptainer bind specifications are colon-delimited: $value"
  [[ "$value" =~ ^/[A-Za-z0-9._/+@-]+$ ]] || fail "$label must be an absolute literal-safe ARC path using only letters, digits, /, ., _, +, @ and -: $value"
}

require_site_option_value() {
  local option="$1" value="$2"
  [[ -n "$value" ]] || fail "$option requires a non-empty value"
  [[ ! "$value" =~ [[:space:]] ]] || fail "$option value must not contain whitespace"
}

[[ $# -ge 2 ]] || usage
MODE="$1"
PHASE_LABEL="$2"
shift 2
[[ "$MODE" == "train" || "$MODE" == "eval" ]] || usage
[[ "$PHASE_LABEL" =~ ^(calibration|pilot|controlled)$ ]] || fail "phase must be calibration, pilot or controlled"

TRAIN_JOB=""
CHECKPOINT_STEP=""
if [[ "$MODE" == "eval" ]]; then
  [[ $# -ge 1 ]] || usage
  TRAIN_JOB="$1"
  shift
  [[ "$TRAIN_JOB" =~ ^[0-9]+$ ]] || fail "training-array-job-id must be numeric"
  if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    CHECKPOINT_STEP="$1"
    shift
  fi
fi

# Accept only exact, attached-value spellings of the site-selection options used
# on ARC. This is intentionally an allowlist: sbatch accepts abbreviated long
# options, bundled short options, `--`, heterogeneous-job `:`, and a replacement
# script operand, so forwarding arbitrary tokens would let a caller bypass the
# wrapper-owned output, array, export, dependency, or script selection.
declare -a SITE_ARGS=()
declare -A SEEN_SITE_OPTIONS=()
for arg in "$@"; do
  case "$arg" in
    --account=*|--partition=*|--qos=*|--reservation=*|--constraint=*)
      option="${arg%%=*}"
      value="${arg#*=}"
      require_site_option_value "$option" "$value"
      [[ -z "${SEEN_SITE_OPTIONS[$option]+present}" ]] || fail "duplicate site option: $option"
      SEEN_SITE_OPTIONS["$option"]=1
      SITE_ARGS+=("$arg")
      ;;
    --)
      fail "caller-supplied -- is forbidden; the wrapper owns the batch-script boundary"
      ;;
    :)
      fail "heterogeneous-job separator ':' is forbidden"
      ;;
    -*)
      fail "unsupported or wrapper-owned sbatch option: $arg"
      ;;
    *)
      fail "unexpected operand; the wrapper owns the batch script: $arg"
      ;;
  esac
done

[[ -n "${TB3_REPO:-}" ]] || fail "TB3_REPO is required"
[[ -n "${TB3_IMAGE:-}" ]] || fail "TB3_IMAGE is required"
[[ -n "${TB3_OUTPUT:-}" ]] || fail "TB3_OUTPUT is required"
[[ -d "$TB3_REPO" ]] || fail "TB3_REPO is not a directory: $TB3_REPO"
[[ -f "$TB3_IMAGE" ]] || fail "TB3_IMAGE is not a file: $TB3_IMAGE"

REPO="$(realpath -e "$TB3_REPO")"
IMAGE="$(realpath -e "$TB3_IMAGE")"
OUTPUT_ROOT="$(realpath -m "$TB3_OUTPUT")"
LOG_ROOT_INPUT="${TB3_SLURM_LOGS:-$OUTPUT_ROOT/slurm/dqn}"
[[ "$LOG_ROOT_INPUT" != *"%"* ]] || fail "TB3_SLURM_LOGS must not contain Slurm percent substitutions: $LOG_ROOT_INPUT"
LOG_ROOT="$(realpath -m "$LOG_ROOT_INPUT")"
[[ "$LOG_ROOT" != *"%"* ]] || fail "resolved TB3_SLURM_LOGS must not contain Slurm percent substitutions: $LOG_ROOT"
LEASE_DIR="$(realpath -m "${TB3_LEASES:-$OUTPUT_ROOT/.leases}")"

require_safe_arc_path "TB3_REPO" "$REPO"
require_safe_arc_path "TB3_IMAGE" "$IMAGE"
require_safe_arc_path "TB3_OUTPUT" "$OUTPUT_ROOT"
require_safe_arc_path "TB3_SLURM_LOGS" "$LOG_ROOT"
require_safe_arc_path "TB3_LEASES" "$LEASE_DIR"
[[ "$LEASE_DIR" == "$OUTPUT_ROOT/.leases" ]] ||
  fail "TB3_LEASES must resolve exactly to TB3_OUTPUT/.leases so it is covered by the sole writable bind"
path_is_within "$OUTPUT_ROOT" "$REPO" && fail "TB3_OUTPUT must be outside TB3_REPO"
path_is_within "$REPO" "$OUTPUT_ROOT" && fail "TB3_REPO and TB3_OUTPUT must not overlap"
path_is_within "$LOG_ROOT" "$REPO" && fail "TB3_SLURM_LOGS must be outside TB3_REPO"
path_is_within "$LEASE_DIR" "$REPO" && fail "TB3_LEASES must be outside TB3_REPO"
path_is_within "$IMAGE" "$REPO" && fail "TB3_IMAGE must be outside TB3_REPO"

[[ "$(tr -d '[:space:]' < "$REPO/ALGORITHM")" == "DQN" ]] || fail "TB3_REPO is not the standalone DQN package"
[[ "$(tr -d '[:space:]' < "$REPO/VERSION")" == "1.1.7" ]] || fail "TB3_REPO is not DQN package version 1.1.7"
[[ "$(stat -c '%A' "$IMAGE")" != *w* ]] || fail "TB3_IMAGE must have all write bits removed; run chmod a-w '$IMAGE'"

case "$PHASE_LABEL" in
  calibration) ARRAY_SPEC="0" ;;
  pilot) ARRAY_SPEC="0-1" ;;
  controlled) ARRAY_SPEC="0-4" ;;
esac

LOG_DIR="$(realpath -m "$LOG_ROOT/$MODE/$PHASE_LABEL")"
[[ "$LOG_DIR" != *"%"* ]] || fail "derived Slurm log directory must not contain Slurm percent substitutions: $LOG_DIR"
require_safe_arc_path "derived Slurm log directory" "$LOG_DIR"
path_is_within "$LOG_DIR" "$REPO" && fail "derived Slurm log directory must be outside TB3_REPO"
path_is_within "$REPO" "$LOG_DIR" && fail "derived Slurm log directory must not contain TB3_REPO"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$LEASE_DIR"
OUTPUT_ROOT="$(realpath -e "$OUTPUT_ROOT")"
LEASE_DIR="$(realpath -e "$LEASE_DIR")"
[[ "$LEASE_DIR" == "$OUTPUT_ROOT/.leases" ]] ||
  fail "resolved TB3_LEASES escaped TB3_OUTPUT after directory creation"
LOG_DIR="$(realpath -e "$LOG_DIR")"
# Recheck after creation so an existing symlink cannot change the validated
# destination between lexical derivation and use.
[[ "$LOG_DIR" != *"%"* ]] || fail "resolved Slurm log directory must not contain Slurm percent substitutions: $LOG_DIR"
path_is_within "$LOG_DIR" "$REPO" && fail "resolved Slurm log directory must be outside TB3_REPO"
path_is_within "$REPO" "$LOG_DIR" && fail "resolved Slurm log directory must not contain TB3_REPO"

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load apptainer 2>/dev/null || module load containers/apptainer 2>/dev/null || true
fi
APPTAINER_CMD="$(command -v apptainer || command -v singularity || true)"
[[ -n "$APPTAINER_CMD" ]] || fail "Apptainer/Singularity is unavailable"
SBATCH_CMD="$(command -v sbatch || true)"
[[ -n "$SBATCH_CMD" ]] || fail "sbatch is unavailable"

# The Slurm client is invoked from a deliberately small host environment.  Its
# absolute path is already resolved, and the batch script loads Apptainer from
# the site module itself.  This prevents caller values such as BASH_ENV,
# LD_PRELOAD, SBATCH_*, SLURM_* or container-runtime controls from becoming a
# second, unaudited input channel.
CONTROLLED_HOST_PATH="/usr/local/bin:/usr/bin:/bin"
ARC_USER="$(id -un)"
ARC_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
[[ "$ARC_USER" =~ ^[A-Za-z0-9._-]+$ ]] || fail "resolved user name contains unsafe characters: $ARC_USER"
[[ -n "$ARC_HOME" && -d "$ARC_HOME" ]] || fail "could not resolve the submitting user's home directory"
require_safe_arc_path "resolved user home" "$(realpath -e "$ARC_HOME")"
ARC_HOME="$(realpath -e "$ARC_HOME")"

# --cleanenv does not disable APPTAINERENV_ injection, ambient bind controls or
# Singularity-compatible control names.  Remove all four prefix families, plus
# shell/loader injection variables, from the login-node authentication process.
declare -a AUTH_ENV_CLEAN=()
while IFS= read -r name; do
  case "$name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*|BASH_ENV|ENV|LD_PRELOAD)
      AUTH_ENV_CLEAN+=("-u" "$name")
      ;;
  esac
done < <(compgen -e)

# Authenticate both copies before Slurm creates a job. The trusted verifier and
# executable release live inside the SIF. The external extraction is mounted
# only read-only at a fixed inspection path and is never used at runtime.
readonly EMBEDDED_REPO=/opt/turtlebot_dqn_random
readonly EXTERNAL_INSPECTION_ROOT=/mnt/tb3-submission-source
IMAGE_SHA256_BEFORE="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
EXTERNAL_RELEASE_SHA256_BEFORE="$(sha256sum "$REPO/RELEASE_MANIFEST.sha256" | cut -d' ' -f1)"
env "${AUTH_ENV_CLEAN[@]}" "$APPTAINER_CMD" exec --cleanenv --containall "$IMAGE" \
  bash "$EMBEDDED_REPO/scripts/verify_package.sh" "$EMBEDDED_REPO"
EMBEDDED_RELEASE_SHA256="$(env "${AUTH_ENV_CLEAN[@]}" "$APPTAINER_CMD" exec --cleanenv --containall "$IMAGE" \
  sha256sum "$EMBEDDED_REPO/RELEASE_MANIFEST.sha256" | cut -d' ' -f1)"
env "${AUTH_ENV_CLEAN[@]}" "$APPTAINER_CMD" exec --cleanenv --containall \
  --bind "$REPO:$EXTERNAL_INSPECTION_ROOT:ro" "$IMAGE" \
  bash "$EMBEDDED_REPO/scripts/verify_package.sh" "$EXTERNAL_INSPECTION_ROOT"
IMAGE_SHA256="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
EXTERNAL_RELEASE_SHA256="$(sha256sum "$REPO/RELEASE_MANIFEST.sha256" | cut -d' ' -f1)"
[[ "$IMAGE_SHA256" == "$IMAGE_SHA256_BEFORE" ]] || fail "TB3_IMAGE changed during submission authentication"
[[ "$EXTERNAL_RELEASE_SHA256" == "$EXTERNAL_RELEASE_SHA256_BEFORE" ]] || fail "external RELEASE_MANIFEST.sha256 changed during submission authentication"
[[ "$EMBEDDED_RELEASE_SHA256" == "$EXTERNAL_RELEASE_SHA256" ]] || fail "TB3_IMAGE embeds a different DQN release than TB3_REPO"

SBATCH_SCRIPT="$REPO/arc/dqn_array.sbatch"
declare -a DEPENDENCY_ARGS=()
if [[ "$MODE" == "eval" ]]; then
  SBATCH_SCRIPT="$REPO/arc/dqn_eval_array.sbatch"
  DEPENDENCY_ARGS+=("--dependency=afterok:$TRAIN_JOB" "--kill-on-invalid-dep=yes")
fi

# Copy the selected batch script once, but take its expected digest from the
# release manifest inside the authenticated image—not from the mutable external
# extraction. Submit the read-only snapshot. A later edit to the external
# package cannot change a queued job. The allocated job also verifies Slurm's
# spooled script bytes against this digest.
BATCH_RELATIVE="${SBATCH_SCRIPT#"$REPO/"}"
EXPECTED_BATCH_SHA256="$(env "${AUTH_ENV_CLEAN[@]}" "$APPTAINER_CMD" exec --cleanenv --containall "$IMAGE" \
  awk -v path="$BATCH_RELATIVE" '$2 == path { print $1 }' "$EMBEDDED_REPO/RELEASE_MANIFEST.sha256")"
[[ "$EXPECTED_BATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "batch script is absent from the authenticated release manifest"
SBATCH_SNAPSHOT="$(mktemp "$LOG_DIR/.dqn-${MODE}-${PHASE_LABEL}.XXXXXX.sbatch")"
cp -- "$SBATCH_SCRIPT" "$SBATCH_SNAPSHOT"
SNAPSHOT_SHA256="$(sha256sum "$SBATCH_SNAPSHOT" | cut -d' ' -f1)"
[[ "$SNAPSHOT_SHA256" == "$EXPECTED_BATCH_SHA256" ]] || fail "batch script changed after authentication; submission refused"
chmod a-w "$SBATCH_SNAPSHOT"
EXPORTS="PHASE_LABEL=$PHASE_LABEL,TB3_SUBMISSION_GUARD=dqn-v1.1.7,TB3_IMAGE=$IMAGE,TB3_OUTPUT=$OUTPUT_ROOT,TB3_LEASES=$LEASE_DIR,TB3_EXPECTED_IMAGE_SHA256=$IMAGE_SHA256,TB3_EXPECTED_RELEASE_SHA256=$EMBEDDED_RELEASE_SHA256,TB3_EXPECTED_BATCH_SCRIPT_SHA256=$SNAPSHOT_SHA256,PATH=$CONTROLLED_HOST_PATH,HOME=$ARC_HOME,USER=$ARC_USER,LOGNAME=$ARC_USER,LANG=C,LC_ALL=C"
if [[ "$MODE" == "eval" ]]; then
  EXPORTS+=",TRAIN_JOB=$TRAIN_JOB"
  [[ -z "$CHECKPOINT_STEP" ]] || EXPORTS+=",CHECKPOINT_STEP=$CHECKPOINT_STEP"
fi

# Detect any post-authentication edit before handing the immutable snapshot to
# Slurm. Even after this check, later external-tree changes are harmless because
# the queued task executes only the SIF-embedded release.
env "${AUTH_ENV_CLEAN[@]}" "$APPTAINER_CMD" exec --cleanenv --containall \
  --bind "$REPO:$EXTERNAL_INSPECTION_ROOT:ro" "$IMAGE" \
  bash "$EMBEDDED_REPO/scripts/verify_package.sh" "$EXTERNAL_INSPECTION_ROOT"
FINAL_EXTERNAL_RELEASE_SHA256="$(sha256sum "$REPO/RELEASE_MANIFEST.sha256" | cut -d' ' -f1)"
[[ "$FINAL_EXTERNAL_RELEASE_SHA256" == "$EMBEDDED_RELEASE_SHA256" ]] || fail "external release identity changed before sbatch"
[[ "$(sha256sum "$IMAGE" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || fail "TB3_IMAGE changed before sbatch"

echo "Submitting standalone DQN $MODE phase=$PHASE_LABEL array=$ARRAY_SPEC"
echo "Slurm logs: $LOG_DIR"
[[ "$MODE" != "eval" ]] || echo "Training dependency: afterok:$TRAIN_JOB"
# Do not use --export=ALL: under Slurm's documented precedence an ambient value
# would override the wrapper's explicit NAME=value.  The sbatch client receives
# only this reviewed host environment, and the job receives only the explicit
# export list plus Slurm/SPANK variables created by the controller.
env -i \
  PATH="$CONTROLLED_HOST_PATH" HOME="$ARC_HOME" USER="$ARC_USER" LOGNAME="$ARC_USER" LANG=C LC_ALL=C \
  "$SBATCH_CMD" \
  "${SITE_ARGS[@]}" \
  --chdir="$LOG_DIR" \
  --output="$LOG_DIR/%x-%A_%a.out" \
  --error="$LOG_DIR/%x-%A_%a.err" \
  --export="$EXPORTS" \
  --array="$ARRAY_SPEC" \
  "${DEPENDENCY_ARGS[@]}" \
  -- \
  "$SBATCH_SNAPSHOT"
