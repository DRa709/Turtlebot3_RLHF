#!/usr/bin/env bash
# Login-node-safe release verifier. With no argument, inspect the package that
# contains this script. The trusted in-image copy may pass one external package
# root as data; executable verification code always comes from SELF_ROOT.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1

SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 1 )); then
  echo "usage: verify_package.sh [package_root]" >&2
  exit 2
fi
ROOT_INPUT="${1:-$SELF_ROOT}"
ROOT="$(realpath -e "$ROOT_INPUT")"
[[ -d "$ROOT" ]] || {
  echo "FAIL: package root is not a directory: $ROOT" >&2
  exit 2
}

status=0
fail() {
  echo "FAIL: $*" >&2
  status=1
}

echo "== forbidden release paths"
while IFS= read -r -d '' path; do
  fail "forbidden release path: ${path#"$ROOT"/}"
done < <(
  find "$ROOT" -mindepth 1 \
    \( -type d \( \
      -name "__pycache__" -o -name ".pytest_cache" -o -name ".git" -o \
      -name "build" -o -name "install" -o -name "log" -o -name "*.egg-info" \
    \) -prune -print0 \) -o \
    \( -type f \( \
      -name "*.pyc" -o -name "*.pyo" -o -name "*.egg-info" -o \
      -name "__pycache__" -o -name ".pytest_cache" -o -name ".git" -o \
      -name "build" -o -name "install" -o -name "log" -o -name "*.ipynb" \
    \) -print0 \)
)
if (( status != 0 )); then
  echo "verify_package: FAILED" >&2
  exit "$status"
fi

echo "== release manifest"
python3 "$SELF_ROOT/turtlebot3_drl_nav/identity.py" verify-release "$ROOT" ||
  fail "RELEASE_MANIFEST.sha256 does not match the tree"

echo "== shared-layer manifest"
python3 "$SELF_ROOT/turtlebot3_drl_nav/identity.py" verify-shared-layer "$ROOT" ||
  fail "SHARED_LAYER_MANIFEST.sha256 does not match the tree"
if [[ -n "${REFERENCE_SHARED_LAYER_MANIFEST:-}" ]]; then
  cmp -s "$REFERENCE_SHARED_LAYER_MANIFEST" "$ROOT/SHARED_LAYER_MANIFEST.sha256" ||
    fail "shared layer differs from the reference package"
fi

echo "== line endings"
while IFS= read -r -d '' f; do
  if grep -q $'\r' "$f"; then
    fail "CRLF line endings: $f"
  fi
done < <(
  find "$ROOT" -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "*.world" -o -name "*.sbatch" -o -name "*.def" -o -name "*.xml" -o -name "*.cfg" -o -name "*.txt" -o -name "*.csv" \) -not -path "*/.git/*" -print0
)

echo "== shell syntax"
for f in "$ROOT"/scripts/*.sh "$ROOT"/arc/*.sh "$ROOT"/arc/*.sbatch; do
  bash -n "$f" || fail "shell syntax: $f"
done

echo "== python compilation"
PYCACHE_ROOT="$(mktemp -d)"
PYTHONPYCACHEPREFIX="$PYCACHE_ROOT" python3 -m compileall -q \
  "$ROOT/turtlebot3_drl_nav" "$ROOT/scripts" "$ROOT/tests" "$ROOT/launch" "$ROOT/setup.py" >/dev/null ||
  fail "python compilation"
rm -rf -- "$PYCACHE_ROOT"

echo "== frozen scenario list"
python3 "$SELF_ROOT/scripts/make_evaluation_scenarios.py" --package-root "$ROOT" --check ||
  fail "scenario file not reproducible"

echo "== configuration digest"
python3 "$SELF_ROOT/turtlebot3_drl_nav/identity.py" config-digest "$ROOT"

if (( status == 0 )); then
  echo "verify_package: OK"
else
  echo "verify_package: FAILED" >&2
fi
exit $status
