#!/usr/bin/env python3
"""Foxy-compatible, bounded liveness probe for the simulator topics.

ROS 2 Foxy's topic echo command has no portable one-message option. This
helper subscribes directly, accepts best-effort publishers, requires one
message from every named topic, and requires simulation time to advance.
"""

import argparse
import sys
import time
from typing import Dict, List, Optional


SUPPORTED_TOPICS = (
    "/clock",
    "/scan",
    "/odom",
    "/bumper_states",
    "/dynamic_obstacle_1/odom",
    "/dynamic_obstacle_2/odom",
    "/drl/obstacle_status",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True, help="wall-clock timeout in seconds")
    parser.add_argument("topics", nargs="+", choices=SUPPORTED_TOPICS)
    args = parser.parse_args(argv)
    if not args.timeout > 0.0:
        parser.error("--timeout must be positive")
    if len(set(args.topics)) != len(args.topics):
        parser.error("topic names must be unique")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    import rclpy
    from gazebo_msgs.msg import ContactsState
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    message_types = {
        "/clock": Clock,
        "/scan": LaserScan,
        "/odom": Odometry,
        "/bumper_states": ContactsState,
        "/dynamic_obstacle_1/odom": Odometry,
        "/dynamic_obstacle_2/odom": Odometry,
        "/drl/obstacle_status": String,
    }
    qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    seen: Dict[str, bool] = {topic: False for topic in args.topics}
    first_clock_ns: Optional[int] = None
    last_clock_ns: Optional[int] = None

    rclpy.init(args=[])
    node = rclpy.create_node("tb3_sim_readiness_probe")
    subscriptions = []

    def callback(topic: str, message: object) -> None:
        nonlocal first_clock_ns, last_clock_ns
        seen[topic] = True
        if topic == "/clock":
            clock = message.clock
            stamp = int(clock.sec) * 1_000_000_000 + int(clock.nanosec)
            if first_clock_ns is None:
                first_clock_ns = stamp
            last_clock_ns = stamp

    try:
        for topic in args.topics:
            subscriptions.append(
                node.create_subscription(message_types[topic], topic, lambda msg, name=topic: callback(name, msg), qos)
            )
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.2, max(0.0, deadline - time.monotonic())))
            clock_progressed = (
                first_clock_ns is not None
                and last_clock_ns is not None
                and last_clock_ns > first_clock_ns
            )
            if all(seen.values()) and clock_progressed:
                print(
                    "Gazebo readiness checks passed "
                    f"(clock_ns {first_clock_ns} -> {last_clock_ns}; every required topic produced a message)"
                )
                return 0
        missing = [topic for topic, received in seen.items() if not received]
        if missing:
            print("required topics without a message: " + ", ".join(missing), file=sys.stderr)
        if first_clock_ns is None:
            print("/clock did not produce a message", file=sys.stderr)
        elif last_clock_ns is None or last_clock_ns <= first_clock_ns:
            print(f"/clock did not advance from {first_clock_ns}", file=sys.stderr)
        return 1
    finally:
        subscriptions.clear()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
