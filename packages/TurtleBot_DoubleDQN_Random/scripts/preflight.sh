#!/usr/bin/env bash
# Preflight inside the ARC container, before the simulator starts. Fails closed.
# Usage: preflight.sh <package_root> <run_dir>
set -euo pipefail
ROOT="$1"; RUN_DIR="$2"
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
[[ ! "$ROOT" =~ [[:space:]] && ! "$RUN_DIR" =~ [[:space:]] ]] || fail "package and run paths must not contain whitespace under ROS 2 Foxy"
[[ "${ROS_DISTRO:-}" == "foxy" ]] || fail "ROS_DISTRO is '${ROS_DISTRO:-unset}', expected foxy"
command -v ros2 >/dev/null || fail "ros2 not on PATH"
command -v gzserver >/dev/null || fail "gzserver not on PATH"
python3 -c "import torch, numpy, yaml, rclpy, gazebo_msgs.srv" || fail "python dependencies missing"
for pkg in gazebo_ros turtlebot3_gazebo turtlebot3_description; do
  ros2 pkg prefix "$pkg" >/dev/null 2>&1 || fail "ROS package $pkg missing"
done
ALGORITHM_CONFIG="$(python3 -c 'import sys; from turtlebot3_drl_nav.launch_config import package_algorithm; print(package_algorithm(sys.argv[1])[1])' "$ROOT")" \
  || fail "could not resolve the package algorithm configuration"
[[ "$ALGORITHM_CONFIG" =~ ^phase1_[a-z0-9]+\.yaml$ ]] || fail "invalid algorithm configuration name"
[[ -f "$ROOT/config/common_environment.yaml" && -f "$ROOT/config/$ALGORITHM_CONFIG" && -f "$ROOT/worlds/phase1_mixed.world" ]] \
  || fail "package files missing"
python3 -m turtlebot3_drl_nav.identity verify-shared-layer "$ROOT" >/dev/null || fail "shared-layer manifest mismatch"
python3 -m turtlebot3_drl_nav.identity verify-release "$ROOT" >/dev/null || fail "release manifest mismatch"
for v in WORLD_SEED INITIALIZATION_SEED DYNAMIC_OBSTACLE_SEED EVALUATION_SEED LEARNING_SEED; do
  [[ "${!v:-}" =~ ^[0-9]+$ ]] || fail "$v is not a non-negative integer"
done
[[ "${PHASE_LABEL:-}" =~ ^(calibration|pilot|controlled)$ ]] || fail "PHASE_LABEL must be calibration, pilot or controlled"
if [[ -e "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
  fail "run directory $RUN_DIR is not empty; runs never append to existing output"
fi
[[ "${CONTAINER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "CONTAINER_SHA256 missing or not a sha256"
echo "preflight OK"
