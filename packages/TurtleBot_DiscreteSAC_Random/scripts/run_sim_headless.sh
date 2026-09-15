#!/usr/bin/env bash
# Start headless Gazebo with the frozen Phase-1 world, spawn the Burger with the
# contact sensor, and start the simulation-time obstacle controller.
# Shared layer. Required environment: WORLD_SEED (Gazebo's own RNG seed).
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${TURTLEBOT3_WS:-}" && -f "${TURTLEBOT3_WS}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${TURTLEBOT3_WS}/install/setup.bash"
elif [[ -f "$HOME/turtlebot3_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/turtlebot3_ws/install/setup.bash"
else
  echo "No built workspace found (TURTLEBOT3_WS or ~/turtlebot3_ws)" >&2
  exit 2
fi
set -u

: "${WORLD_SEED:?WORLD_SEED must be set (Gazebo RNG seed, recorded in the run identity)}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"

TB3_GAZEBO_PREFIX="$(ros2 pkg prefix turtlebot3_gazebo)"
WORLD_SOURCE="${TB3_WORLD:-$REPO_ROOT/worlds/phase1_mixed.world}"
MODEL="$TB3_GAZEBO_PREFIX/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}:$TB3_GAZEBO_PREFIX/share/turtlebot3_gazebo/models"
SIM_RUNTIME_DIR="${SIM_RUNTIME_DIR:-${TMPDIR:-/tmp}/tb3_phase1_sim_${USER:-user}_$$}"
PATCHED_MODEL="$SIM_RUNTIME_DIR/turtlebot3_burger_with_contact.sdf"
WORLD="$SIM_RUNTIME_DIR/phase1_mixed.world"
GAZEBO_PID=""
OBSTACLE_PID=""

# Foxy's gazebo.launch.py tokenizes the world launch argument incorrectly when
# its path contains whitespace and may silently load empty.world. Refuse that
# unsafe path before any Gazebo process starts.
if [[ "$SIM_RUNTIME_DIR" =~ [[:space:]] ]]; then
  echo "SIM_RUNTIME_DIR must not contain whitespace: $SIM_RUNTIME_DIR" >&2
  exit 2
fi

cleanup() {
  for pid in "$OBSTACLE_PID" "$GAZEBO_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "$OBSTACLE_PID" "$GAZEBO_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  pkill -KILL -f "$SIM_RUNTIME_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "$SIM_RUNTIME_DIR"
[[ -f "$WORLD_SOURCE" ]] || { echo "Gazebo world does not exist: $WORLD_SOURCE" >&2; exit 2; }
cp "$WORLD_SOURCE" "$WORLD"
# Keep the packaged 1x real-time rate. The environment pauses physics during
# every policy/optimizer computation, so inference cost cannot advance the MDP.
cmp -s "$WORLD_SOURCE" "$WORLD" || { echo "runtime world differs from the authenticated source" >&2; exit 2; }
grep -q 'libgazebo_ros_state.so' "$WORLD" || { echo "world lacks libgazebo_ros_state.so" >&2; exit 2; }
python3 "$REPO_ROOT/scripts/prepare_contact_model.py" "$MODEL" "$PATCHED_MODEL" --lidar-update-rate 100 --require-lidar
sha256sum "$WORLD" > "$SIM_RUNTIME_DIR/runtime_world.sha256"
sha256sum "$PATCHED_MODEL" > "$SIM_RUNTIME_DIR/runtime_robot_model.sha256"

echo "Starting Gazebo world $WORLD (gui=$GAZEBO_GUI, seed=$WORLD_SEED)"
ros2 launch gazebo_ros gazebo.launch.py gui:="$GAZEBO_GUI" verbose:=false world:="$WORLD" \
  extra_gazebo_args:="--seed $WORLD_SEED" &
GAZEBO_PID=$!
sleep 8

echo "Spawning TurtleBot3 Burger with contact sensor from $PATCHED_MODEL"
ros2 run gazebo_ros spawn_entity.py -entity burger -file "$PATCHED_MODEL" -x 0.0 -y 0.0 -z 0.01

ros2 run turtlebot3_drl_nav dynamic_obstacles --ros-args -p use_sim_time:=true -p control_period:=0.01 &
OBSTACLE_PID=$!

echo "Headless Gazebo is running (pid $GAZEBO_PID); obstacle controller pid $OBSTACLE_PID"
set +e
wait -n "$GAZEBO_PID" "$OBSTACLE_PID"
rc=$?
set -e
if ! kill -0 "$OBSTACLE_PID" 2>/dev/null && kill -0 "$GAZEBO_PID" 2>/dev/null; then
  echo "dynamic-obstacle controller exited while Gazebo was still running" >&2
  exit 3
fi
exit "$rc"
