"""Simulation-time controller of the two planar-move obstacles. Shared layer.

The obstacles hold still during every reset transaction and follow, during an
episode, the (sign, offset) schedule the environment node derived from
``dynamic_obstacle_seed`` and the episode index and sent in the ``start``
control message. Time is the node clock (``use_sim_time`` must be true), so the
trajectory in simulation time is a function of the recorded parameters alone,
whatever the real-time factor.
"""

import math
import sys
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import String

from .obstacle_schedule import obstacle_velocity
from .protocol import decode_json, encode_json
from .ros_qos import latched_qos

OBSTACLE_TOPICS = ("/dynamic_obstacle_1/cmd_vel", "/dynamic_obstacle_2/cmd_vel")


class DynamicObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_obstacles")
        self.declare_parameter("control_period", 0.10)
        if not bool(self.get_parameter("use_sim_time").value):
            raise ValueError("use_sim_time must be true for the obstacle schedule")
        period = float(self.get_parameter("control_period").value)
        self.publishers_ = [self.create_publisher(Twist, topic, 10) for topic in OBSTACLE_TOPICS]
        self.status_pub = self.create_publisher(String, "/drl/obstacle_status", latched_qos())
        self.schedule: Optional[Dict[str, object]] = None
        self.last_ready_announce = float("-inf")
        self.create_subscription(String, "/drl/obstacle_control", self._control_callback, 10)
        self.heartbeat_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.005, self._tick, clock=self.heartbeat_clock)
        self._latest_sim_time = 0.0
        self.create_subscription(Odometry, "/odom", self._robot_odom_callback, 10)
        self.create_subscription(Odometry, "/dynamic_obstacle_1/odom", self._obs1_odom_callback, 10)
        self.create_subscription(Odometry, "/dynamic_obstacle_2/odom", self._obs2_odom_callback, 10)
        self.status_pub.publish(String(data=encode_json({"cmd": "ready", "status": "ok"})))
        self.get_logger().info("dynamic obstacles holding; readiness announced")

    def _ack(self, cmd: str, episode_key: int, status: str = "ok", reason: str = "") -> None:
        self.status_pub.publish(String(data=encode_json({
            "cmd": cmd, "episode_key": int(episode_key), "status": status, "reason": reason,
        })))

    def _control_callback(self, msg: String) -> None:
        try:
            payload = decode_json(msg.data)
        except ValueError as error:
            self.get_logger().fatal(f"malformed obstacle control: {error}")
            if rclpy.ok():
                rclpy.shutdown()
            return
        cmd = payload.get("cmd")
        try:
            episode_key = int(payload.get("episode_key", -1))
        except (TypeError, ValueError, OverflowError):
            episode_key = -1
        if cmd == "hold":
            self.schedule = None
            self._publish([(0.0, 0.0), (0.0, 0.0)])
            self._ack("hold", episode_key)
        elif cmd == "start":
            names = list(payload.get("names", []))
            axes = list(payload.get("axes", []))
            signs = list(payload.get("signs", []))
            offsets = list(payload.get("offsets", []))
            numeric = [payload.get("t0"), payload.get("speed"), payload.get("half_period")] + signs + offsets
            try:
                numeric_ok = all(math.isfinite(float(v)) for v in numeric)
                half_period = float(payload["half_period"])
                valid = (
                    episode_key > 0
                    and names == [t.split("/")[1] for t in OBSTACLE_TOPICS]
                    and axes == ["y", "x"]
                    and len(signs) == len(offsets) == 2
                    and numeric_ok
                    and float(payload["speed"]) > 0.0
                    and half_period > 0.0
                    and all(float(v) in (-1.0, 1.0) for v in signs)
                    and all(0.0 <= float(v) < half_period for v in offsets)
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                reason = "obstacle schedule fields do not match the frozen controller contract"
                self.get_logger().error(reason)
                self._ack("start", episode_key, "error", reason)
                return
            self.schedule = payload
            self._tick()
            self._ack("start", episode_key)
        else:
            reason = f"unknown obstacle command {cmd}"
            self.get_logger().error(reason)
            self._ack(str(cmd), episode_key, "error", reason)

    def _robot_odom_callback(self, msg: Odometry) -> None:
        self._update_sim_time(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _obs1_odom_callback(self, msg: Odometry) -> None:
        self._update_sim_time(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _obs2_odom_callback(self, msg: Odometry) -> None:
        self._update_sim_time(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _update_sim_time(self, stamp: float) -> None:
        if stamp > self._latest_sim_time:
            self._latest_sim_time = stamp

    def _now(self) -> float:
        ros_clock = self.get_clock().now().nanoseconds * 1e-9
        if ros_clock < self._latest_sim_time - 1.0:
            self._latest_sim_time = ros_clock
        return max(ros_clock, self._latest_sim_time)

    def _tick(self) -> None:
        if self.schedule is None:
            self._publish([(0.0, 0.0), (0.0, 0.0)])
            now = self._now()
            if now - self.last_ready_announce >= 1.0:
                self.status_pub.publish(String(data=encode_json({"cmd": "ready", "status": "ok"})))
                self.last_ready_announce = now
            return
        now = self._now()
        elapsed = now - float(self.schedule["t0"])
        velocities: List = []
        for index, axis in enumerate(self.schedule["axes"]):
            velocities.append(obstacle_velocity(
                axis, float(self.schedule["signs"][index]), float(self.schedule["offsets"][index]),
                float(self.schedule["speed"]), float(self.schedule["half_period"]), max(elapsed, 0.0),
            ))
        self._publish(velocities)

    def _publish(self, velocities) -> None:
        for publisher, (vx, vy) in zip(self.publishers_, velocities):
            twist = Twist()
            twist.linear.x = float(vx)
            twist.linear.y = float(vy)
            publisher.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DynamicObstacleNode] = None
    exit_code = 0
    try:
        node = DynamicObstacleNode()
        rclpy.spin(node)
    except KeyboardInterrupt:  # launch shutdown delivers SIGINT; an orderly stop is not a failure
        pass
    except Exception as error:  # noqa: BLE001
        print(f"dynamic_obstacles fatal: {error}", file=sys.stderr)
        exit_code = 3
    finally:
        if node is not None:
            node._publish([(0.0, 0.0), (0.0, 0.0)])
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)
