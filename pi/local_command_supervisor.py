#!/usr/bin/env python3
"""
Local Command Supervisor for TurtleBot3 (Raspberry Pi 4 Node).

Role (Section 8):
- Sole autonomous publisher to the base motor command topic (/cmd_vel).
- Pure Python + rclpy (ZERO PyTorch/ML dependencies on Pi).
- Heartbeat / Request expiry watchdog: halts robot within 200 ms if laptop lags/drops.
- Real-time proximity braking: continuously monitors raw LiDAR ranges and halts if
  clearance falls below the hardware stopping threshold (default 0.18 m).
- Sensor staleness watchdog: halts if /scan or /odom drops.
- Latching protective stop state until deliberately re-armed by operator.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class SupervisorConfig:
    command_timeout_sec: float = 0.20
    sensor_timeout_sec: float = 0.25
    hardware_stop_threshold: float = 0.18  # Metres (braking distance margin)
    max_linear_speed: float = 0.15          # Max forward speed (Action 0)
    min_linear_speed: float = -0.05         # Optional slow reverse
    max_angular_speed: float = 1.00         # Max rotation speed (Actions 3, 4)
    use_stamped_vel: bool = False           # True for Jazzy, False for Humble
    cmd_vel_topic: str = "/cmd_vel"
    request_topic: str = "/tb3_sac/cmd_vel_request"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"


class SupervisorState:
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    ACTIVE = "ACTIVE"
    COMMAND_EXPIRED = "COMMAND_EXPIRED"
    SENSOR_STALE = "SENSOR_STALE"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"


class SupervisorEngine:
    """
    Pure Python core logic for the supervisor.
    Decoupled from ROS for deterministic unit testing.
    """

    def __init__(self, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.state = SupervisorState.DISARMED
        self.last_cmd_time = 0.0
        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.latest_min_scan_range = float("inf")
        self.active_seq_id = 0

    def arm(self) -> bool:
        if self.state == SupervisorState.PROTECTIVE_STOP:
            # Re-arming from protective stop requires clearance check
            if self.latest_min_scan_range < self.config.hardware_stop_threshold:
                return False  # Refuse re-arm while obstacle is still in braking envelope
        self.state = SupervisorState.ARMED
        return True

    def disarm(self) -> None:
        self.state = SupervisorState.DISARMED

    def update_scan(self, ranges: Sequence[float], current_time: float,
                    range_min: float = 0.12, range_max: float = 3.5) -> Tuple[bool, float]:
        """
        Updates latest scan ranges and evaluates proximity braking.
        Returns (is_safe, min_valid_range).
        """
        self.last_scan_time = current_time
        valid_ranges = [
            r for r in ranges
            if math.isfinite(r) and (range_min <= r <= range_max)
        ]
        if not valid_ranges:
            # No valid obstacles detected or all out of range
            self.latest_min_scan_range = float("inf")
            return True, float("inf")

        min_r = min(valid_ranges)
        self.latest_min_scan_range = min_r

        if min_r < self.config.hardware_stop_threshold:
            # Emergency braking condition met!
            self.state = SupervisorState.PROTECTIVE_STOP
            return False, min_r

        return True, min_r

    def update_odom(self, current_time: float) -> None:
        self.last_odom_time = current_time

    def process_request(self, req_linear_x: float, req_angular_z: float,
                        seq_id: int, current_time: float) -> Tuple[float, float, str]:
        """
        Processes an incoming velocity request from the laptop.
        Returns (clamped_linear_x, clamped_angular_z, state).
        """
        self.last_cmd_time = current_time
        self.active_seq_id = seq_id

        # 1. State check: Disarmed
        if self.state == SupervisorState.DISARMED:
            return 0.0, 0.0, SupervisorState.DISARMED

        # 2. State check: Latching Protective Stop
        if self.state == SupervisorState.PROTECTIVE_STOP:
            return 0.0, 0.0, SupervisorState.PROTECTIVE_STOP

        # 3. Sensor Staleness check
        if (current_time - self.last_scan_time > self.config.sensor_timeout_sec) or \
           (current_time - self.last_odom_time > self.config.sensor_timeout_sec):
            self.state = SupervisorState.SENSOR_STALE
            return 0.0, 0.0, SupervisorState.SENSOR_STALE

        # 4. Proximity breach check
        if self.latest_min_scan_range < self.config.hardware_stop_threshold:
            self.state = SupervisorState.PROTECTIVE_STOP
            return 0.0, 0.0, SupervisorState.PROTECTIVE_STOP

        # 5. All safety invariants satisfied -> Clamp and forward command
        self.state = SupervisorState.ACTIVE
        clamped_linear = max(self.config.min_linear_speed,
                             min(self.config.max_linear_speed, req_linear_x))
        clamped_angular = max(-self.config.max_angular_speed,
                              min(self.config.max_angular_speed, req_angular_z))
        return clamped_linear, clamped_angular, SupervisorState.ACTIVE

    def tick_watchdog(self, current_time: float) -> Tuple[float, float, str]:
        """
        Evaluates watchdogs when no fresh request has arrived.
        Called periodically (e.g. at 50 Hz).
        """
        if self.state in (SupervisorState.DISARMED, SupervisorState.PROTECTIVE_STOP):
            return 0.0, 0.0, self.state

        # Check command timeout
        if (current_time - self.last_cmd_time) > self.config.command_timeout_sec:
            self.state = SupervisorState.COMMAND_EXPIRED
            return 0.0, 0.0, SupervisorState.COMMAND_EXPIRED

        # Check sensor freshness
        if (current_time - self.last_scan_time) > self.config.sensor_timeout_sec or \
           (current_time - self.last_odom_time) > self.config.sensor_timeout_sec:
            self.state = SupervisorState.SENSOR_STALE
            return 0.0, 0.0, SupervisorState.SENSOR_STALE

        return 0.0, 0.0, self.state


def create_ros_node():
    """
    Instantiates the ROS 2 node wrapper.
    Imported conditionally so unit tests can run in environments without ROS 2.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, TwistStamped
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from std_srvs.srv import SetBool

    class LocalCommandSupervisorNode(Node):
        def __init__(self):
            super().__init__("local_command_supervisor")
            self.config = SupervisorConfig()
            self.engine = SupervisorEngine(self.config)

            # QoS profiles matching TurtleBot3 bringup
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )

            # Publishers
            if self.config.use_stamped_vel:
                self.cmd_pub = self.create_publisher(TwistStamped, self.config.cmd_vel_topic, 10)
            else:
                self.cmd_pub = self.create_publisher(Twist, self.config.cmd_vel_topic, 10)
            
            self.status_pub = self.create_publisher(String, "/tb3_sac/supervisor_status", 10)

            # Subscribers
            self.create_subscription(LaserScan, self.config.scan_topic, self._on_scan, sensor_qos)
            self.create_subscription(Odometry, self.config.odom_topic, self._on_odom, sensor_qos)
            self.create_subscription(Twist, self.config.request_topic, self._on_request, 10)

            # Services
            self.create_service(SetBool, "/tb3_sac/arm", self._on_arm_service)

            # High-rate watchdog timer (50 Hz = 20 ms)
            self.create_timer(0.02, self._on_watchdog_timer)
            self.get_logger().info("Local Command Supervisor initialized (Armed=False). Sole /cmd_vel publisher.")

        def _on_scan(self, msg: LaserScan):
            now = time.monotonic()
            self.engine.update_scan(msg.ranges, now, msg.range_min, msg.range_max)

        def _on_odom(self, msg: Odometry):
            now = time.monotonic()
            self.engine.update_odom(now)

        def _on_request(self, msg: Twist):
            now = time.monotonic()
            vx, wz, state = self.engine.process_request(msg.linear.x, msg.angular.z, 0, now)
            self._publish_cmd(vx, wz)

        def _on_arm_service(self, request, response):
            if request.data:
                ok = self.engine.arm()
                response.success = ok
                response.message = f"Armed={ok}, State={self.engine.state}"
            else:
                self.engine.disarm()
                response.success = True
                response.message = "Disarmed"
            self.get_logger().info(f"Supervisor arm request: {response.message}")
            return response

        def _on_watchdog_timer(self):
            now = time.monotonic()
            vx, wz, state = self.engine.tick_watchdog(now)
            if state != SupervisorState.ACTIVE:
                self._publish_cmd(0.0, 0.0)

            # Periodic status telemetry (1 Hz)
            if int(now * 50) % 50 == 0:
                stat_msg = String()
                stat_msg.data = f"state={self.engine.state};min_scan={self.engine.latest_min_scan_range:.3f}"
                self.status_pub.publish(stat_msg)

        def _publish_cmd(self, linear_x: float, angular_z: float):
            if self.config.use_stamped_vel:
                msg = TwistStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.twist.linear.x = float(linear_x)
                msg.twist.angular.z = float(angular_z)
                self.cmd_pub.publish(msg)
            else:
                msg = Twist()
                msg.linear.x = float(linear_x)
                msg.angular.z = float(angular_z)
                self.cmd_pub.publish(msg)

    return LocalCommandSupervisorNode


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node_cls = create_ros_node()
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.engine.disarm()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
