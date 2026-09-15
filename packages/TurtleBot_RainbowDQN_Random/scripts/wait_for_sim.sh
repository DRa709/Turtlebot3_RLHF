#!/usr/bin/env bash
# Block until the simulator exposes every topic and service the environment
# node depends on, including the simulation clock and the entity-state services.
# Shared layer. Usage: wait_for_sim.sh [timeout_seconds]
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT_SECONDS="${1:-180}"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || TIMEOUT_SECONDS=180
DEADLINE=$((SECONDS + TIMEOUT_SECONDS))
REQUIRED_TOPICS=(/clock /scan /odom /bumper_states /dynamic_obstacle_1/odom /dynamic_obstacle_2/odom /drl/obstacle_status)
REQUIRED_SERVICES=(/reset_world /set_entity_state /get_entity_state /pause_physics /unpause_physics)
while (( SECONDS < DEADLINE )); do
  TOPICS="$(ros2 topic list 2>/dev/null || true)"
  SERVICES="$(ros2 service list 2>/dev/null || true)"
  ok=1
  for t in "${REQUIRED_TOPICS[@]}"; do grep -qx "$t" <<< "$TOPICS" || ok=0; done
  for s in "${REQUIRED_SERVICES[@]}"; do grep -qx "$s" <<< "$SERVICES" || ok=0; done
  if (( ok == 1 )); then
    remaining=$((DEADLINE - SECONDS))
    (( remaining > 0 )) || break
    # Foxy's topic-echo CLI has no portable one-message option. Use a bounded
    # rclpy subscriber that also proves /clock advances and every required
    # topic produces a message rather than merely appearing in the graph.
    python3 "$SCRIPT_DIR/wait_for_topics.py" --timeout "$remaining" "${REQUIRED_TOPICS[@]}"
    exit $?
  fi
  sleep 2
done
echo "Gazebo did not become ready within ${TIMEOUT_SECONDS}s" >&2
echo "topics:" >&2; ros2 topic list >&2 || true
echo "services:" >&2; ros2 service list >&2 || true
exit 1
