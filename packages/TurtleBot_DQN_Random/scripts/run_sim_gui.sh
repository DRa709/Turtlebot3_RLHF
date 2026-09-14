#!/usr/bin/env bash
# Local convenience: the same simulator with the Gazebo GUI. Never used on ARC.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAZEBO_GUI=true WORLD_SEED="${WORLD_SEED:-7000}" bash "$REPO_ROOT/scripts/run_sim_headless.sh"
