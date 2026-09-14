#!/usr/bin/env bash
# Build the DQN SIF only from a newly materialized, manifest-exact source tree.
# Run on an ARC compute node from any directory; never run on a login node.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

fail() {
  echo "DQN SIF BUILD FAIL: $*" >&2
  exit 2
}

[[ $# -eq 1 ]] || fail "usage: build_sif.sh /absolute/output/image.sif"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$1" == /* ]] || fail "output must be an absolute .sif path"
OUTPUT="$(realpath -m "$1")"
[[ "$OUTPUT" == /* && "$OUTPUT" == *.sif ]] || fail "output must be an absolute .sif path"
[[ ! "$OUTPUT" =~ [[:space:]] ]] || fail "output path must not contain whitespace"
[[ "$OUTPUT" =~ ^/[A-Za-z0-9._/+@-]+$ ]] || fail "output path contains unsupported characters"
[[ ! -e "$OUTPUT" ]] || fail "output already exists; refusing to overwrite: $OUTPUT"
[[ -d "$(dirname "$OUTPUT")" ]] || fail "output parent directory does not exist"
case "$OUTPUT" in
  "$ROOT"|"$ROOT"/*) fail "output SIF must be outside the authenticated source tree" ;;
esac

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load apptainer 2>/dev/null || module load containers/apptainer 2>/dev/null || true
fi
APPTAINER_CMD="$(command -v apptainer || command -v singularity || true)"
[[ -n "$APPTAINER_CMD" ]] || fail "Apptainer/Singularity is unavailable"

STAGING_BASE="$(realpath -e "${TMPDIR:-/tmp}")"
[[ -d "$STAGING_BASE" ]] || fail "temporary directory does not exist: $STAGING_BASE"
[[ ! "$STAGING_BASE" =~ [[:space:]] ]] || fail "temporary directory contains whitespace: $STAGING_BASE"
STAGING_PARENT="$(mktemp -d "$STAGING_BASE/tb3-dqn-v1.1.7.XXXXXX")"
cleanup() {
  [[ -n "${STAGING_PARENT:-}" && -d "$STAGING_PARENT" ]] || return 0
  case "$STAGING_PARENT" in
    "$STAGING_BASE"/tb3-dqn-v1.1.7.*) rm -rf -- "$STAGING_PARENT" ;;
    *) echo "refusing to remove unexpected staging path: $STAGING_PARENT" >&2 ;;
  esac
}
trap cleanup EXIT
STAGED_ROOT="$STAGING_PARENT/TurtleBot_DQN_Random"

bash "$ROOT/scripts/verify_package.sh" "$ROOT"
python3 "$ROOT/scripts/materialize_release.py" \
  --source "$ROOT" --destination "$STAGED_ROOT"
bash "$STAGED_ROOT/scripts/verify_package.sh" "$STAGED_ROOT"

(
  cd "$STAGED_ROOT"
  "$APPTAINER_CMD" build "$OUTPUT" apptainer/tb3_phase1_foxy.def
)
[[ -f "$OUTPUT" ]] || fail "Apptainer returned success without creating the SIF"
echo "built manifest-materialized DQN SIF: $OUTPUT"
echo "next: apptainer test '$OUTPUT'; chmod a-w '$OUTPUT'; sha256sum '$OUTPUT'"
