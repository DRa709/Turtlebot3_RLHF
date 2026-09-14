"""Minimal in-memory stand-ins for the rclpy and ROS message APIs the two nodes
use, so the ROS adapters can be exercised end-to-end without ROS. This is a
mock; it certifies the adapters' logic and message handling, not ROS itself."""

import sys
import types
from typing import Callable, Dict, List, Optional


class _Bus:
    """One process-wide topic bus shared by every stub node."""

    def __init__(self) -> None:
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.publishers: Dict[str, int] = {}
        self.services: Dict[str, Callable] = {}
        self.timers: List["_Timer"] = []
        self.sim_time_ns = 100 * 10**9
        self.wall_time = 0.0
        self.log: List[str] = []
        self.ok = True
        self.latched: Dict[str, object] = {}

    def reset(self) -> None:
        self.__init__()


BUS = _Bus()


# ------------------------------------------------------------- messages

class _Stamp:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _Header:
    def __init__(self) -> None:
        self.stamp = _Stamp()


class _Vec3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Quat:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class _Pose:
    def __init__(self) -> None:
        self.position = _Vec3()
        self.orientation = _Quat()


class _PoseWithCov:
    def __init__(self) -> None:
        self.pose = _Pose()


class Twist:
    def __init__(self) -> None:
        self.linear = _Vec3()
        self.angular = _Vec3()


class Odometry:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _PoseWithCov()


class LaserScan:
    def __init__(self) -> None:
        self.header = _Header()
        self.ranges = []
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0


class _Contact:
    def __init__(self, a: str, b: str) -> None:
        self.collision1_name = a
        self.collision2_name = b


class ContactsState:
    def __init__(self) -> None:
        self.states: List[_Contact] = []


class Float32MultiArray:
    def __init__(self, data=None) -> None:
        self.data = list(data) if data is not None else []


class String:
    def __init__(self, data: str = "") -> None:
        self.data = data


class _EntityState:
    def __init__(self) -> None:
        self.name = ""
        self.pose = _Pose()
        self.twist = Twist()
        self.reference_frame = ""


class SetEntityState:
    class Request:
        def __init__(self) -> None:
            self.state = _EntityState()

    class Response:
        def __init__(self) -> None:
            self.success = False


class GetEntityState:
    class Request:
        def __init__(self) -> None:
            self.name = ""
            self.reference_frame = ""

    class Response:
        def __init__(self) -> None:
            self.header = _Header()
            self.state = _EntityState()
            self.success = False


class Empty:
    class Request:
        pass

    class Response:
        pass


# ---------------------------------------------------------------- rclpy

class _Future:
    def __init__(self, result) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Client:
    def __init__(self, name: str) -> None:
        self.name = name

    def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
        return self.name in BUS.services

    def call_async(self, request) -> _Future:
        return _Future(BUS.services[self.name](request))


class _Publisher:
    def __init__(self, topic: str, latched: bool) -> None:
        self.topic = topic
        self.latched = latched
        BUS.publishers[topic] = BUS.publishers.get(topic, 0) + 1

    def publish(self, msg) -> None:
        if self.latched:
            BUS.latched[self.topic] = msg
        for callback in list(BUS.subscriptions.get(self.topic, [])):
            callback(msg)


class _Timer:
    def __init__(self, period: float, callback: Callable, wall: bool) -> None:
        self.period = period
        self.callback = callback
        self.wall = wall


class _Logger:
    def _log(self, level: str, text: str) -> None:
        BUS.log.append(f"{level}: {text}")

    def info(self, text: str) -> None:
        self._log("INFO", text)

    def warn(self, text: str) -> None:
        self._log("WARN", text)

    def error(self, text: str) -> None:
        self._log("ERROR", text)

    def fatal(self, text: str) -> None:
        self._log("FATAL", text)


class _Parameter:
    def __init__(self, value) -> None:
        self.value = value


class _Time:
    def __init__(self, ns: int) -> None:
        self.nanoseconds = ns


class _NodeClock:
    def now(self) -> _Time:
        return _Time(BUS.sim_time_ns)


class Node:
    overrides: Dict[str, Dict[str, object]] = {}

    def __init__(self, name: str) -> None:
        self._name = name
        self._params: Dict[str, object] = {"use_sim_time": True}
        self._overrides = dict(Node.overrides.get(name, {}))
        self._logger = _Logger()
        self.timers: List[_Timer] = []

    def declare_parameter(self, name: str, value=None):
        self._params[name] = self._overrides.get(name, value)
        return _Parameter(self._params[name])

    def get_parameter(self, name: str) -> _Parameter:
        if name not in self._params:
            raise KeyError(name)
        return _Parameter(self._params[name])

    def create_publisher(self, msg_type, topic: str, qos) -> _Publisher:
        return _Publisher(topic, getattr(qos, "latched", False))

    def create_subscription(self, msg_type, topic: str, callback: Callable, qos) -> None:
        BUS.subscriptions.setdefault(topic, []).append(callback)
        if getattr(qos, "latched", False) and topic in BUS.latched:
            callback(BUS.latched[topic])

    def create_client(self, srv_type, name: str) -> _Client:
        return _Client(name)

    def create_timer(self, period: float, callback: Callable, callback_group=None, clock=None) -> _Timer:
        timer = _Timer(period, callback, wall=clock is not None)
        self.timers.append(timer)
        BUS.timers.append(timer)
        return timer

    def get_clock(self) -> _NodeClock:
        return _NodeClock()

    def get_logger(self) -> _Logger:
        return self._logger

    def count_publishers(self, topic: str) -> int:
        return BUS.publishers.get(topic, 0)

    def destroy_node(self) -> None:
        pass


class QoSProfile:
    def __init__(self, depth=10, reliability=None, durability=None, history=None) -> None:
        self.latched = durability == "TRANSIENT_LOCAL"


class ReliabilityPolicy:
    RELIABLE = "RELIABLE"


class DurabilityPolicy:
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"


class HistoryPolicy:
    KEEP_LAST = "KEEP_LAST"


class ClockType:
    SYSTEM_TIME = "SYSTEM_TIME"
    STEADY_TIME = "STEADY_TIME"


class Clock:
    def __init__(self, clock_type=None) -> None:
        self.clock_type = clock_type


def _rclpy_init(args=None) -> None:
    BUS.ok = True


def _rclpy_ok() -> bool:
    return BUS.ok


def _rclpy_shutdown() -> None:
    BUS.ok = False


def install() -> None:
    """Register the stubs under the real module names (only when ROS is absent)."""
    rclpy = types.ModuleType("rclpy")
    rclpy.init = _rclpy_init
    rclpy.ok = _rclpy_ok
    rclpy.shutdown = _rclpy_shutdown
    rclpy.spin = lambda node: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.QoSProfile, qos_mod.ReliabilityPolicy, qos_mod.DurabilityPolicy, qos_mod.HistoryPolicy = QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    clock_mod = types.ModuleType("rclpy.clock")
    clock_mod.Clock, clock_mod.ClockType = Clock, ClockType
    gz_msg = types.ModuleType("gazebo_msgs.msg")
    gz_msg.ContactsState = ContactsState
    gz_srv = types.ModuleType("gazebo_msgs.srv")
    gz_srv.SetEntityState, gz_srv.GetEntityState = SetEntityState, GetEntityState
    gz = types.ModuleType("gazebo_msgs")
    gz.msg, gz.srv = gz_msg, gz_srv
    geom = types.ModuleType("geometry_msgs.msg")
    geom.Twist = Twist
    nav = types.ModuleType("nav_msgs.msg")
    nav.Odometry = Odometry
    sensor = types.ModuleType("sensor_msgs.msg")
    sensor.LaserScan = LaserScan
    std = types.ModuleType("std_msgs.msg")
    std.Float32MultiArray, std.String = Float32MultiArray, String
    std_srv = types.ModuleType("std_srvs.srv")
    std_srv.Empty = Empty
    modules = {
        "rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos_mod, "rclpy.clock": clock_mod,
        "gazebo_msgs": gz, "gazebo_msgs.msg": gz_msg, "gazebo_msgs.srv": gz_srv,
        "geometry_msgs": types.ModuleType("geometry_msgs"), "geometry_msgs.msg": geom,
        "nav_msgs": types.ModuleType("nav_msgs"), "nav_msgs.msg": nav,
        "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": sensor,
        "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": std,
        "std_srvs": types.ModuleType("std_srvs"), "std_srvs.srv": std_srv,
    }
    for name, module in modules.items():
        sys.modules[name] = module
