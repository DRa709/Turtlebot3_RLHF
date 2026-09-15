#!/usr/bin/env bash
# Mandatory verification tests. Run from a terminal inside the ARC container (or
# a local environment with PyTorch) before any sbatch is authorized. Every test
# must execute; a skip is a failure.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -c "import torch, numpy, yaml; print('torch', torch.__version__, 'numpy', numpy.__version__)"
TEMP_LOG=""
if [[ -n "${TEST_LOG:-}" ]]; then
  LOG_PATH="$TEST_LOG"
else
  TEMP_LOG="$(mktemp)"
  LOG_PATH="$TEMP_LOG"
fi
cleanup() { if [[ -n "$TEMP_LOG" ]]; then rm -f "$TEMP_LOG"; fi; }
trap cleanup EXIT
# These are ordinary package unit tests. Disable site-wide pytest plugins so
# ROS 2 Foxy's legacy launch_testing entry point cannot make the mandatory gate
# depend on the host/container plugin set or an incompatible pytest API.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest -q -rs -p no:cacheprovider tests "$@" | tee "$LOG_PATH"
if grep -Eq '(^|[^0-9])[0-9]+ skipped|SKIPPED' "$LOG_PATH"; then
  echo "skipped tests are not accepted" >&2
  exit 1
fi
