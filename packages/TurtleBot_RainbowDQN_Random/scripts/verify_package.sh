#!/usr/bin/env bash
# Lightweight file verification — the only thing besides job submission that may
# run on an ARC login node. No ROS, no Gazebo, no tests: manifests, line endings,
# shell syntax, Python compilation, forbidden files, frozen scenario file.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
status=0
fail() { echo "FAIL: $*" >&2; status=1; }

echo "== release manifest"
python3 -m turtlebot3_drl_nav.identity verify-release "$ROOT" || fail "RELEASE_MANIFEST.sha256 does not match the tree"
echo "== shared-layer manifest"
python3 -m turtlebot3_drl_nav.identity verify-shared-layer "$ROOT" || fail "SHARED_LAYER_MANIFEST.sha256 does not match the tree"
echo "== forbidden files"
if find . -name "*.ipynb" -not -path "./.git/*" | grep -q .; then fail "notebook files are forbidden"; fi
if find . \( -type d -name "__pycache__" -o -type f -name "*.py[co]" \) -not -path "./.git/*" | grep -q .; then
  fail "Python cache files are forbidden in the release tree"
fi

echo "== line endings"
while IFS= read -r -d '' f; do
  if grep -q $'\r' "$f"; then fail "CRLF line endings: $f"; fi
done < <(find . -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "*.world" -o -name "*.sbatch" -o -name "*.def" -o -name "*.xml" -o -name "*.cfg" -o -name "*.txt" -o -name "*.csv" \) -not -path "./.git/*" -print0)

echo "== shell syntax"
for f in scripts/*.sh arc/*.sh arc/*.sbatch; do
  bash -n "$f" || fail "shell syntax: $f"
done

echo "== executable entry points"
for f in scripts/*.sh scripts/*.py arc/*.sh arc/*.py arc/*.sbatch; do
  [[ -x "$f" ]] || fail "entry point is not executable: $f"
done

echo "== python compilation"
python3 -m compileall -q turtlebot3_drl_nav scripts tests launch setup.py >/dev/null || fail "python compilation"
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "== frozen scenario list"
python3 scripts/make_evaluation_scenarios.py --check || fail "scenario file not reproducible"

echo "== configuration digest"
python3 -m turtlebot3_drl_nav.identity config-digest "$ROOT"

if (( status == 0 )); then echo "verify_package: OK"; else echo "verify_package: FAILED" >&2; fi
exit $status
